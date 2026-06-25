"""Three-way head-to-head: free fit vs Engine A vs Engine B on one session.

Pure metrics (coverage, folds, reprojection accuracy in feet) so the comparison
logic is unit-tested; the heavy real-data run lives in
examples/calib_anchor_compare.ipynb (the user runs it and assesses the overlays).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.field_model import METRES_TO_FEET, field_points_3d
from soccer_vision.calib.validate import fold_count
from soccer_vision.pitch.calib_anchor import (
    FramePose,
    calibrate_clicked_frames,
    frame_homography,
    poses_by_click_propagation,
    poses_by_pose_propagation,
)
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import (
    Click,
    FrameFit,
    build_segments,
    cumulative_transforms,
    fit_frame_homographies,
)


@dataclass(frozen=True)
class EngineMetrics:
    """One engine's full-game numbers.

    accuracy here is the REPROJECTION error (feet) of the clicked landmarks under
    the engine's homography at the directly-clicked frames (NOT leave-one-out — a
    uniform, per-engine measure; the held-out calib-model accuracy is Phase-1's
    `leave_one_out_feet`, run alongside in the notebook). `per_frame_median_ft` is
    the per-clicked-frame median, indexed by frame, so the notebook can plot drift
    (error vs frame number).
    """

    n_covered: int
    coverage_fraction: float
    n_folded: int
    median_accuracy_ft: float
    p90_accuracy_ft: float
    per_frame_median_ft: dict[int, float]


def _free_fit_homographies(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    size: tuple[int, int],
    *,
    window: int,
) -> dict[int, NDArray[np.float64]]:
    """The current free fit -> full-pixel image->pitch homographies (export form)."""
    w, h = size
    fits: dict[int, FrameFit] = fit_frame_homographies(
        clicks, transforms, segment_of, PITCH_LANDMARKS, window=window)
    # FrameFit.H is normalized image->pitch; denormalize to full-pixel image->pitch
    # (H_px = H_norm @ diag(1/W,1/H,1); same as labeler.chain.denormalize_homography,
    # inlined to keep pitch/ independent of labeler/).
    s = np.diag([1.0 / w, 1.0 / h, 1.0])
    return {f: np.asarray(fit.H, dtype=np.float64) @ s for f, fit in fits.items()}


def _calib_homographies(poses: Mapping[int, FramePose], k: NDArray[np.floating]) \
        -> dict[int, NDArray[np.float64]]:
    return {f: frame_homography(k, p.rvec, p.tvec) for f, p in poses.items()}


def _accuracy_feet(
    homographies: Mapping[int, NDArray[np.floating]],
    clicks_by_frame: Mapping[int, list[tuple[int, float, float]]],
) -> dict[int, list[float]]:
    """Per CLICKED frame with a homography, the feet error of each clicked landmark.

    A homography is image(px)->pitch[0,1]; apply it to map a clicked pixel to pitch
    [0,1], then scale by field dimensions to get metres, compare to the landmark's
    true metres. Returns {frame: [feet errors]} so the caller can aggregate AND index
    by frame (drift).
    """
    fp = field_points_3d()
    width_m = float(fp[:, 0].max())
    length_m = float(fp[:, 1].max())
    out: dict[int, list[float]] = {}
    for f, obs in clicks_by_frame.items():
        h = homographies.get(f)
        if h is None:
            continue
        h64 = np.asarray(h, dtype=np.float64)
        errs: list[float] = []
        for kp, x_px, y_px in obs:
            if kp >= len(fp):  # out-of-range landmark index -> skip (labeler never emits one)
                continue
            v = h64 @ np.array([x_px, y_px, 1.0])
            pitch = v[:2] / v[2]
            mx, my = pitch[0] * width_m, pitch[1] * length_m
            errs.append(float(np.hypot(mx - fp[kp, 0], my - fp[kp, 1]) * METRES_TO_FEET))
        if errs:
            out[f] = errs
    return out


def compare_engines(
    clicks: Sequence[Click],
    interframe: Mapping[int, NDArray[np.floating]],
    n_frames: int,
    size: tuple[int, int],
    *,
    window: int = 360,
    fold_threshold: int = 16,
    min_points: int = 6,
) -> dict[str, EngineMetrics]:
    """Run all three engines on one session; return per-engine metrics.

    fold_threshold: a covered frame with >= this many in-frame landmarks is counted
    as a fold (a fixed Trace crop never shows the whole field). Accuracy is the
    reprojection error (feet) of the clicked landmarks at the directly-clicked frames
    (uniform per-engine; NOT leave-one-out).
    """
    w, h = size
    segment_of = build_segments(interframe, n_frames)
    transforms = cumulative_transforms(interframe, segment_of)
    clicks_by_frame: dict[int, list[tuple[int, float, float]]] = {}
    for c in clicks:
        clicks_by_frame.setdefault(c.frame, []).append((c.kp_idx, c.x * w, c.y * h))

    k, clicked_poses = calibrate_clicked_frames(clicks, size, min_points=min_points)

    free_h = _free_fit_homographies(clicks, transforms, segment_of, size, window=window)
    a_poses = poses_by_click_propagation(
        clicks, transforms, segment_of, k, size, window=window)
    b_poses = poses_by_pose_propagation(transforms, segment_of, k, clicked_poses, size)
    a_h = _calib_homographies(a_poses, k)
    b_h = _calib_homographies(b_poses, k)

    def _metrics(
        homs: Mapping[int, NDArray[np.floating]],
        folds: Mapping[int, int] | None,
    ) -> EngineMetrics:
        n_cov = len(homs)
        if folds is not None:
            n_fold = sum(1 for fc in folds.values() if fc >= fold_threshold)
        else:
            # free fit: fold via its pitch homography's in-frame landmark count
            # (homs are image->pitch; the pitch-of-image is its inverse)
            n_fold = sum(
                1 for hp in homs.values()
                if fold_count(np.linalg.inv(np.asarray(hp, dtype=np.float64)), size)
                >= fold_threshold)
        per_frame = _accuracy_feet(homs, clicks_by_frame)
        flat = [e for errs in per_frame.values() for e in errs]
        med = float(np.median(flat)) if flat else float("nan")
        p90 = float(np.percentile(flat, 90)) if flat else float("nan")
        per_frame_median = {f: float(np.median(errs)) for f, errs in per_frame.items()}
        return EngineMetrics(
            n_covered=n_cov,
            coverage_fraction=n_cov / n_frames if n_frames else 0.0,
            n_folded=n_fold,
            median_accuracy_ft=med,
            p90_accuracy_ft=p90,
            per_frame_median_ft=per_frame_median,
        )

    return {
        "free_fit": _metrics(free_h, None),
        "engine_a": _metrics(a_h, {f: p.fold_count for f, p in a_poses.items()}),
        "engine_b": _metrics(b_h, {f: p.fold_count for f, p in b_poses.items()}),
    }
