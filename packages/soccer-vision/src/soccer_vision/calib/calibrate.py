"""Camera calibration against the known 9v9 field: shared focal + per-frame pose.

A per-frame homography H = K [r1 | r2 | t] comes from a PHYSICAL camera pose, so it
cannot fold the far field into view (the failure of the free per-frame homography);
and each frame is solved directly against the field, so there is no chained-
registration drift.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M


def homography_from_pose(
    k: NDArray[np.floating], rvec: NDArray[np.floating], tvec: NDArray[np.floating]
) -> NDArray[np.float64]:
    """World-metres (X, Y, Z=0) -> pixel homography for a camera (K, rvec, tvec)."""
    rmat, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    cols = np.column_stack([rmat[:, 0], rmat[:, 1], np.asarray(tvec, dtype=np.float64).ravel()])
    return np.asarray(np.asarray(k, dtype=np.float64) @ cols, dtype=np.float64)


def pitch_homography(h_world: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert a world-metres->pixel homography to canonical-[0,1]^2 -> pixel."""
    return np.asarray(np.asarray(h_world, dtype=np.float64) @ np.diag([WIDTH_M, LENGTH_M, 1.0]),
                      dtype=np.float64)
