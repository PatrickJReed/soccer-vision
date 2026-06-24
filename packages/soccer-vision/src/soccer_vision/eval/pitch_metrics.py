"""Pure metrics for scoring the pitch-keypoint model against labeler ground truth.

Everything here is numpy in / dataclass out, no I/O — so the eval logic itself is
unit-testable (the lesson of anchor_cov, which shipped untested and gave a false
pass). Errors are reported in real-world FEET: x and y are each normalized to [0,1]
independently (x = fraction of width, y = fraction of length), so each axis must be
scaled separately before computing the Euclidean norm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from soccer_vision.pitch.landmarks import NEAR_HALFWAY_IDX as HIDDEN_IDX
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

# Nominal US Soccer 9v9 pitch length: ~68.5 m. Youth fields vary; this is a fixed
# nominal scale so feet errors are interpretable and comparable across retrains.
DEFAULT_PITCH_LENGTH_FT: float = 224.7
DEFAULT_ASPECT_RATIO: float = 1.5  # 9v9 length/width; x is a fraction of width, y of length


def displacement_to_feet(
    disp: NDArray[np.floating],
    *,
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
) -> NDArray[np.float64]:
    """Canonical (dx, dy) displacement(s) -> feet magnitude, scaling x by width
    and y by length (the two pitch axes are independently normalized to [0,1]).
    Accepts shape (2,) or (N, 2); returns scalar-array or (N,) feet."""
    d = np.asarray(disp, dtype=np.float64)
    width_ft = length_ft / aspect_ratio
    scaled = d * np.array([width_ft, length_ft])
    return np.asarray(np.hypot(scaled[..., 0], scaled[..., 1]), dtype=np.float64)


def _apply_h(h: NDArray[np.floating], pts_px: NDArray[np.floating]) -> NDArray[np.float64]:
    """Map (N,2) image pixels -> (N,2) pitch coords through image->pitch H."""
    pts = np.asarray(pts_px, dtype=np.float64)
    homog = np.column_stack([pts, np.ones(len(pts))])
    proj = homog @ np.asarray(h, dtype=np.float64).T
    return np.asarray(proj[:, :2] / proj[:, 2:3], dtype=np.float64)


def keypoint_errors_feet(
    h_gt: NDArray[np.floating],
    gt_kpts: NDArray[np.floating],
    model_kpts: NDArray[np.floating],
    *,
    conf_thr: float = 0.5,
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
) -> dict[int, float]:
    """Per-landmark feet error: map the model's predicted pixel through the GT
    homography into pitch space, compare to the canonical landmark.

    gt_kpts: (21,3) px + visibility (from project_landmarks). model_kpts: (21,3)
    px + confidence. Scores landmarks that are GT-visible, not the hidden idx,
    and predicted with conf >= conf_thr.
    """
    out: dict[int, float] = {}
    for i in range(len(PITCH_LANDMARKS)):
        if i == HIDDEN_IDX or gt_kpts[i, 2] <= 0 or model_kpts[i, 2] < conf_thr:
            continue
        pitch_pred = _apply_h(h_gt, model_kpts[i:i + 1, :2])[0]
        disp = pitch_pred - PITCH_LANDMARKS[i]
        out[i] = float(displacement_to_feet(disp, length_ft=length_ft, aspect_ratio=aspect_ratio))
    return out


def reproj_error_feet(
    h_gt: NDArray[np.floating],
    model_h: NDArray[np.floating],
    gt_kpts: NDArray[np.floating],
    *,
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
) -> float | None:
    """End-to-end check: at each GT-visible landmark's pixel, how far (feet) does
    the MODEL homography place it from the canonical truth? Median over visible
    landmarks. None if no GT-visible landmarks. Catches outlier keypoints that
    wreck the fitted homography even when average keypoint error looks fine.
    """
    idx = [i for i in range(len(PITCH_LANDMARKS)) if i != HIDDEN_IDX and gt_kpts[i, 2] > 0]
    if not idx:
        return None
    px = gt_kpts[idx, :2]
    pitch_via_model = _apply_h(model_h, px)
    disp = pitch_via_model - PITCH_LANDMARKS[idx]
    feet = displacement_to_feet(disp, length_ft=length_ft, aspect_ratio=aspect_ratio)
    return float(np.median(feet))


def labeler_fit_residual_feet(
    h_gt: NDArray[np.floating],
    clicked_px: NDArray[np.floating],
    kp_indices: NDArray[np.integer],
    *,
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
) -> float:
    """Labeler self-consistency: median feet error of its homography on its OWN
    clicked anchors. This is the noise-floor proxy that defines the
    match-the-labeler bar. (clicked_px: (k,2) image pixels; kp_indices: (k,)
    landmark ids for each click.)
    """
    idx = np.asarray(kp_indices)
    pitch_pred = _apply_h(h_gt, np.asarray(clicked_px, dtype=np.float64))
    disp = pitch_pred - PITCH_LANDMARKS[idx]
    feet = displacement_to_feet(disp, length_ft=length_ft, aspect_ratio=aspect_ratio)
    return float(np.median(feet))


@dataclass
class FrameScore:
    frame: int
    per_kp_feet: dict[int, float]
    median_feet: float | None      # None if model produced no scorable keypoints
    reproj_feet: float | None      # None if model H not fittable / no GT-visible
    gt_visible: list[int]
    predicted: list[int]
    matches: bool


@dataclass
class EvalReport:
    n_frames: int                  # scored frames (excludes degenerate GT)
    n_matched: int
    accurate_coverage: float       # n_matched / n_frames  (the headline)
    keypoint_feet_median: float | None
    keypoint_feet_p90: float | None
    reproj_feet_median: float | None
    reproj_feet_p90: float | None
    per_landmark: dict[int, dict[str, float]]  # idx -> {median, p90, detect_rate}
    n_excluded_degenerate: int


def score_frame(
    frame: int,
    h_gt: NDArray[np.floating],
    model_kpts: NDArray[np.floating] | None,
    *,
    frame_size: tuple[int, int],
    match_threshold_feet: float,
    min_match_keypoints: int = 4,
    conf_thr: float = 0.5,
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
) -> FrameScore:
    """Score one benchmark frame. A None / <4-confident model prediction yields a
    not-covered (non-matching) frame, never a skip."""
    from soccer_vision.pitch.autolabel import project_landmarks
    from soccer_vision.pitch.homography import HomographyError, fit_homography

    gt_kpts = project_landmarks(h_gt, PITCH_LANDMARKS, frame_size)
    gt_visible = [i for i in range(len(PITCH_LANDMARKS))
                  if i != HIDDEN_IDX and gt_kpts[i, 2] > 0]

    if model_kpts is None:
        return FrameScore(frame, {}, None, None, gt_visible, [], False)

    errs = keypoint_errors_feet(h_gt, gt_kpts, model_kpts, conf_thr=conf_thr,
                                length_ft=length_ft, aspect_ratio=aspect_ratio)
    predicted = sorted(errs)
    median_feet = float(np.median(list(errs.values()))) if errs else None

    # fit the model homography from its confident keypoints for the reproj check
    reproj_feet: float | None = None
    conf_mask = model_kpts[:, 2] >= conf_thr
    conf_idx = [i for i in np.nonzero(conf_mask)[0] if i != HIDDEN_IDX]
    if len(conf_idx) >= 4:
        try:
            model_h = fit_homography(model_kpts[conf_idx, :2], PITCH_LANDMARKS[conf_idx])
            reproj_feet = reproj_error_feet(h_gt, model_h, gt_kpts, length_ft=length_ft,
                                            aspect_ratio=aspect_ratio)
        except HomographyError:
            reproj_feet = None

    matches = (median_feet is not None and len(errs) >= min_match_keypoints
               and median_feet <= match_threshold_feet)
    return FrameScore(frame, errs, median_feet, reproj_feet, gt_visible, predicted, matches)


def _cond(h: NDArray[np.floating]) -> float:
    return float(np.linalg.cond(np.asarray(h, dtype=np.float64)))


def score_benchmark(
    gt_homographies: dict[int, NDArray[np.floating]],
    model_predictions: dict[int, NDArray[np.floating]],
    *,
    frame_size: tuple[int, int],
    match_threshold_feet: float,
    min_match_keypoints: int = 4,
    conf_thr: float = 0.5,
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    degenerate_cond: float = 1e8,
) -> EvalReport:
    """Score the model over the whole frozen benchmark. Degenerate-GT frames are
    excluded with a count; missing predictions count as not-covered."""
    scores: list[FrameScore] = []
    n_excluded = 0
    for frame in sorted(gt_homographies):
        h_gt = gt_homographies[frame]
        if _cond(h_gt) > degenerate_cond:
            n_excluded += 1
            continue
        scores.append(score_frame(
            frame, h_gt, model_predictions.get(frame), frame_size=frame_size,
            match_threshold_feet=match_threshold_feet,
            min_match_keypoints=min_match_keypoints,
            conf_thr=conf_thr, length_ft=length_ft, aspect_ratio=aspect_ratio))

    n = len(scores)
    n_matched = sum(s.matches for s in scores)
    kp_all = [v for s in scores for v in s.per_kp_feet.values()]
    reproj_all = [s.reproj_feet for s in scores if s.reproj_feet is not None]

    per_landmark: dict[int, dict[str, float]] = {}
    for i in range(len(PITCH_LANDMARKS)):
        if i == HIDDEN_IDX:
            continue
        vals = [s.per_kp_feet[i] for s in scores if i in s.per_kp_feet]
        gt_vis = sum(i in s.gt_visible for s in scores)
        per_landmark[i] = {
            "median": float(np.median(vals)) if vals else float("nan"),
            "p90": float(np.percentile(vals, 90)) if vals else float("nan"),
            "detect_rate": (len(vals) / gt_vis) if gt_vis else float("nan"),
        }

    def _med(x: list[float]) -> float | None:
        return float(np.median(x)) if x else None

    def _p90(x: list[float]) -> float | None:
        return float(np.percentile(x, 90)) if x else None

    return EvalReport(
        n_frames=n,
        n_matched=n_matched,
        accurate_coverage=(n_matched / n) if n else 0.0,
        keypoint_feet_median=_med(kp_all),
        keypoint_feet_p90=_p90(kp_all),
        reproj_feet_median=_med(reproj_all),
        reproj_feet_p90=_p90(reproj_all),
        per_landmark=per_landmark,
        n_excluded_degenerate=n_excluded,
    )
