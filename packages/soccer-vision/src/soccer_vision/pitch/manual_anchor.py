"""Manual-anchor point propagation: turn sparse landmark clicks into per-frame
homographies using the fixed camera's frame-to-frame registration chain.

Each click is one (frame, kp_idx, pixel) observation. Because the Trace camera
has no parallax, a click in one frame can be mapped into any other frame in the
same registration-connected segment via cumulative inter-frame transforms. A
frame that accumulates >=4 distinct landmarks (clicked there or propagated in)
gets a homography; the fit's reprojection residual is its confidence.

All homographies map image pixels -> pitch [0,1]^2. Pure: no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.pitch.homography import HomographyError, fit_homography
from soccer_vision.pitch.propagation import HomographyEntry


@dataclass(frozen=True)
class Click:
    """One landmark observation: pixel (x, y) of keypoint kp_idx in a frame."""

    frame: int
    kp_idx: int
    x: float
    y: float


@dataclass(frozen=True, eq=False)
class FrameFit:
    """A fitted per-frame homography plus its quality."""

    H: NDArray[np.floating]
    residual: float        # mean reprojection error in pitch units
    n_points: int          # distinct landmarks used


def build_segments(
    interframe: Mapping[int, NDArray[np.floating]], n_frames: int
) -> dict[int, int]:
    """Assign each frame 0..n_frames-1 a registration-segment id.

    interframe[i] present means frames i and i+1 are linked. A run of consecutive
    linked frames shares a segment id; a missing link starts a new segment.
    """
    seg: dict[int, int] = {}
    cur = 0
    for f in range(n_frames):
        if f == 0:
            seg[f] = 0
        elif (f - 1) in interframe:
            seg[f] = seg[f - 1]
        else:
            cur += 1
            seg[f] = cur
    return seg


def cumulative_transforms(
    interframe: Mapping[int, NDArray[np.floating]], segment_of: Mapping[int, int]
) -> dict[int, NDArray[np.float64]]:
    """M[f] maps frame f pixels -> its segment's reference (first) frame pixels.

    M[start] = I; M[f] = M[f-1] @ inv(interframe[f-1]) within a segment, since
    interframe[f-1] maps f-1 -> f so its inverse maps f -> f-1.
    """
    transforms: dict[int, NDArray[np.float64]] = {}
    for f in sorted(segment_of):
        prev_same = (f - 1) in segment_of and segment_of[f] == segment_of[f - 1]
        if not prev_same:
            transforms[f] = np.eye(3)
        else:
            g = np.asarray(interframe[f - 1], dtype=np.float64)
            transforms[f] = transforms[f - 1] @ np.linalg.inv(g)
    return transforms


def map_point(
    m_src: NDArray[np.floating], m_dst: NDArray[np.floating], x: float, y: float
) -> tuple[float, float]:
    """Map pixel (x, y) from the source frame into the destination frame.

    src -> reference via m_src, reference -> dst via inv(m_dst). Both must be in
    the same segment (caller ensures this).
    """
    ref = np.asarray(m_src, dtype=np.float64) @ np.array([x, y, 1.0])
    dst = np.linalg.inv(np.asarray(m_dst, dtype=np.float64)) @ ref
    return float(dst[0] / dst[2]), float(dst[1] / dst[2])


def _apply(H: NDArray[np.floating], pts: NDArray[np.floating]) -> NDArray[np.float64]:
    """Apply a 3x3 homography to (N, 2) points -> (N, 2)."""
    homog = np.column_stack([pts, np.ones(len(pts))])
    out = (np.asarray(H, dtype=np.float64) @ homog.T).T
    return out[:, :2] / out[:, 2:3]


def fit_frame_homographies(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    landmarks: NDArray[np.floating],
    *,
    window: int,
    min_points: int = 4,
) -> dict[int, FrameFit]:
    """Fit each frame's image->pitch homography from clicks propagated into it.

    For target frame g, gather clicks in the same segment with |click.frame - g|
    <= window, mapped into g's pixels; keep one observation per landmark (nearest
    click frame wins). With >= min_points distinct landmarks, fit a homography and
    record its mean reprojection residual (pitch units).
    """
    fits: dict[int, FrameFit] = {}
    for g in sorted(transforms):
        seg_g = segment_of[g]
        best: dict[int, tuple[int, float, float]] = {}  # kp_idx -> (dist, x, y)
        for c in clicks:
            if segment_of.get(c.frame) != seg_g:
                continue
            dist = abs(c.frame - g)
            if dist > window:
                continue
            x, y = map_point(transforms[c.frame], transforms[g], c.x, c.y)
            if c.kp_idx not in best or dist < best[c.kp_idx][0]:
                best[c.kp_idx] = (dist, x, y)
        if len(best) < min_points:
            continue
        idxs = sorted(best)
        image_pts = np.array([[best[i][1], best[i][2]] for i in idxs], dtype=np.float64)
        pitch_pts = np.asarray(landmarks, dtype=np.float64)[idxs]
        try:
            H = fit_homography(image_pts, pitch_pts)
        except HomographyError:
            continue
        residual = float(np.linalg.norm(_apply(H, image_pts) - pitch_pts, axis=1).mean())
        fits[g] = FrameFit(H=H, residual=residual, n_points=len(idxs))
    return fits


def frame_status(
    fits: Mapping[int, FrameFit], n_frames: int, *, residual_threshold: float = 0.05
) -> dict[int, str]:
    """Per-frame label: green (fit & residual<=thr), yellow (fit & residual>thr),
    red (no fit / <4 landmarks)."""
    status: dict[int, str] = {}
    for f in range(n_frames):
        fit = fits.get(f)
        if fit is None:
            status[f] = "red"
        elif fit.residual <= residual_threshold:
            status[f] = "green"
        else:
            status[f] = "yellow"
    return status


def coverage_fraction(
    fits: Mapping[int, FrameFit], n_frames: int, *, residual_threshold: float = 0.05
) -> float:
    """Fraction of frames that are green (covered & low residual)."""
    if n_frames == 0:
        return 0.0
    green = sum(1 for fit in fits.values() if fit.residual <= residual_threshold)
    return green / n_frames


def to_homography_entries(
    fits: Mapping[int, FrameFit], *, residual_threshold: float = 0.05
) -> dict[int, HomographyEntry]:
    """Green frames -> HomographyEntry(source='manual', confidence from residual)."""
    out: dict[int, HomographyEntry] = {}
    for f, fit in fits.items():
        if fit.residual > residual_threshold:
            continue
        conf = float(np.clip(1.0 - fit.residual / residual_threshold, 0.0, 1.0))
        out[f] = HomographyEntry(np.asarray(fit.H, dtype=np.float64), "manual", conf)
    return out


def clicks_to_keypoints_df(clicks: Sequence[Click]) -> pd.DataFrame:
    """Clicks -> keypoints DataFrame (frame, kp_idx, x_px, y_px, conf=1.0)."""
    df = pd.DataFrame(
        [{"frame": c.frame, "kp_idx": c.kp_idx, "x_px": c.x, "y_px": c.y, "conf": 1.0}
         for c in clicks],
        columns=["frame", "kp_idx", "x_px", "y_px", "conf"],
    )
    return df.astype({"frame": "int64", "kp_idx": "int64", "x_px": "float64",
                      "y_px": "float64", "conf": "float64"})
