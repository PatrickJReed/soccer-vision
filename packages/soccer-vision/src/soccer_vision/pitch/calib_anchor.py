"""Calibration-based per-frame homography engines for a fixed-camera session.

Two engines turn a session's clicks + the registration chain into a calibrated
homography for every frame: A (propagate clicks, then refine_pose per frame) and B
(calibrate clicked frames, then propagate the pose via the chain's recovered camera
rotation). Both reuse the Phase 1/2 calib core, so each frame is a real camera pose
(no fold) solved against the field directly (no homography-chaining drift).

Internally full-pixel image space; emits the labeler's full-pixel image->pitch[0,1]
homography (the export format). Pure: no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.calibrate import (
    CalibError,
    calibrate_camera,
    homography_from_pose,
    pitch_homography,
    refine_pose,
)
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.calib.validate import fold_count
from soccer_vision.pitch.manual_anchor import Click, propagate_clicks


@dataclass(frozen=True, eq=False)
class FramePose:
    """A per-frame calibrated camera pose + quality."""

    rvec: NDArray[np.float64]
    tvec: NDArray[np.float64]
    residual_px: float   # reprojection RMS over the frame's obs (nan if none)
    n_points: int        # point landmarks used (0 if pose-propagated)
    fold_count: int      # landmarks projecting in-frame (slice size, ~6-12)


def frame_homography(
    k: NDArray[np.floating], rvec: NDArray[np.floating], tvec: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Full-pixel image -> pitch-[0,1]^2 homography (the labeler export format)."""
    h_pitch = pitch_homography(homography_from_pose(k, rvec, tvec))  # canon[0,1]^2 -> px
    return np.asarray(np.linalg.inv(h_pitch), dtype=np.float64)


def calibrate_clicked_frames(
    clicks: Sequence[Click],
    size: tuple[int, int],
    *,
    min_points: int = 6,
) -> tuple[NDArray[np.float64], dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]]:
    """Shared focal + per-clicked-frame pose from the DIRECTLY-clicked frames.

    Clicks are normalized [0,1]; converted to full pixel for the calib core. Raises
    CalibError if too few/degenerate clicked views.
    """
    w, h = size
    obs: dict[int, list[tuple[int, float, float]]] = {}
    for c in clicks:
        obs.setdefault(c.frame, []).append((c.kp_idx, c.x * w, c.y * h))
    result = calibrate_camera(obs, size, min_points=min_points)
    return result.K, result.poses


def _reproj_rms_px(
    k: NDArray[np.floating],
    rvec: NDArray[np.floating],
    tvec: NDArray[np.floating],
    point_obs: Sequence[tuple[int, float, float]],
) -> float:
    """Reprojection RMS (px) of point_obs under the pose; nan if no points."""
    if not point_obs:
        return float("nan")
    obj = field_points_3d()[[int(kp) for kp, _, _ in point_obs]].astype(np.float64)
    img = np.array([[x, y] for _, x, y in point_obs], dtype=np.float64)
    proj = cv2.projectPoints(
        obj, rvec, tvec, np.asarray(k, dtype=np.float64),
        np.zeros(5, dtype=np.float64))[0].reshape(-1, 2)
    d = proj - img
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _fold_for_pose(
    k: NDArray[np.floating],
    rvec: NDArray[np.floating],
    tvec: NDArray[np.floating],
    size: tuple[int, int],
) -> int:
    """fold_count for a pose's pitch homography."""
    return fold_count(pitch_homography(homography_from_pose(k, rvec, tvec)), size)


def poses_by_click_propagation(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    k: NDArray[np.floating],
    size: tuple[int, int],
    *,
    window: int,
    min_points: int = 4,
    line_obs: Mapping[int, Sequence[tuple[str, float, float]]] | None = None,
) -> dict[int, FramePose]:
    """Engine A: propagate clicks into each frame, then refine_pose (focal fixed).

    Per target frame: gather window-propagated point landmarks (and any supplied
    pixel-space line_obs for that frame), seed a pose with SQPNP on the propagated
    points, and refine with refine_pose. A frame with < min_points propagated points
    (SQPNP needs them) or a non-converging solve is left uncovered. line_obs are
    pre-propagated, pixel-space lines per frame (3a real run is point-only; line
    propagation is Phase 3b). Returns {frame: FramePose}.
    """
    w, h = size
    k_arr = np.asarray(k, dtype=np.float64)
    fp = field_points_3d()
    propagated = propagate_clicks(clicks, transforms, segment_of, window=window)
    out: dict[int, FramePose] = {}
    for f, kpmap in propagated.items():
        if len(kpmap) < min_points:
            continue
        idxs = sorted(kpmap)
        # propagated positions are normalized -> convert to pixel for the calib core
        point_obs = [(i, kpmap[i][0] * w, kpmap[i][1] * h) for i in idxs]
        lobs = list(line_obs.get(f, [])) if line_obs else []
        obj = fp[idxs].astype(np.float64)
        img = np.array([[x, y] for _, x, y in point_obs], dtype=np.float64)
        ok, rvec0, tvec0 = cv2.solvePnP(obj, img, k_arr, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        try:
            rvec, tvec = refine_pose(
                k_arr, np.asarray(rvec0, dtype=np.float64),
                np.asarray(tvec0, dtype=np.float64), point_obs, lobs)
        except CalibError:
            continue
        out[f] = FramePose(
            rvec=rvec,
            tvec=tvec,
            residual_px=_reproj_rms_px(k_arr, rvec, tvec, point_obs),
            n_points=len(idxs),
            fold_count=_fold_for_pose(k_arr, rvec, tvec, size),
        )
    return out
