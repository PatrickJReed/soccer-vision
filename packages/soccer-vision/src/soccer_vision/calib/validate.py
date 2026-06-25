"""Validation metrics for the camera-calibration core: fold-free and accuracy.

These are pure (given a homography / camera), so the gate logic is unit-tested —
the lesson of the anchor_cov coverage gate that shipped untested.
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.calibrate import homography_from_pose
from soccer_vision.calib.field_model import METRES_TO_FEET, field_points_3d
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS


def fold_count(h_pitch: NDArray[np.floating], frame_size: tuple[int, int]) -> int:
    """How many of the 21 canonical landmarks project IN-FRAME through h_pitch.

    A folding homography pulls the far field in -> count near 21 on a shallow view;
    a physical camera keeps only the visible slice -> count ~6-12.
    """
    w, h = frame_size
    pts = np.column_stack([PITCH_LANDMARKS[:, 0], PITCH_LANDMARKS[:, 1], np.ones(len(PITCH_LANDMARKS))])
    pr = (np.asarray(h_pitch, dtype=np.float64) @ pts.T).T
    wz = pr[:, 2]
    denom = np.where(np.abs(pr[:, 2:3]) < 1e-9, 1e-9, pr[:, 2:3])
    uv = pr[:, :2] / denom
    return int(np.sum((wz > 0) & (uv[:, 0] > 0) & (uv[:, 0] < w) & (uv[:, 1] > 0) & (uv[:, 1] < h)))


def reproj_error_feet(
    h_world: NDArray[np.floating],
    field_pt_3d: NDArray[np.floating],
    clicked_px: tuple[float, float],
) -> float:
    """Map a clicked pixel back to field metres (via inv H) and compare to the true
    field position; error in feet."""
    h_inv = np.linalg.inv(np.asarray(h_world, dtype=np.float64))
    v = h_inv @ np.array([clicked_px[0], clicked_px[1], 1.0])
    fld = v[:2] / v[2]
    return float(np.hypot(fld[0] - field_pt_3d[0], fld[1] - field_pt_3d[1]) * METRES_TO_FEET)


def leave_one_out_feet(
    observations: dict[int, list[tuple[int, float, float]]],
    k: NDArray[np.floating],
    frame_size: tuple[int, int],
    *,
    min_other: int = 4,
) -> dict[int, list[float]]:
    """Held-out accuracy: per frame, fit the pose from all-but-one landmark and
    measure the held-out landmark's reprojection error in feet. Returns
    {kp_idx: [feet errors across frames]}."""
    fp = field_points_3d()
    out: dict[int, list[float]] = defaultdict(list)
    for obs in observations.values():
        seen: dict[int, tuple[float, float]] = {int(kp): (float(x), float(y)) for kp, x, y in obs}
        ids = sorted(seen)
        if len(ids) < min_other + 1:
            continue
        for held in ids:
            others = [i for i in ids if i != held]
            op = fp[others].astype(np.float64)
            ip = np.array([seen[i] for i in others], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(op, ip, np.asarray(k, np.float64), None,
                                          flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                continue
            h_world = homography_from_pose(
                k,
                np.asarray(rvec, dtype=np.float64),
                np.asarray(tvec, dtype=np.float64),
            )
            out[held].append(reproj_error_feet(h_world, fp[held], seen[held]))
    return dict(out)
