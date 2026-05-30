"""Canonical pitch landmarks + per-frame homography fitting from keypoints.

PITCH_LANDMARKS maps each pitch-model keypoint index to its canonical pitch
coordinate, normalized to [0, 1]^2. Coordinates are vendored from roboflow's
SoccerPitchConfiguration (sports/configs/soccer.py) — a 12000 x 7000 cm pitch
with 32 vertices, whose order matches the football-pitch-detection keypoint
indices.

Axis convention: roboflow's length axis (x, goal-to-goal) maps to our y, and
roboflow's width axis (y) maps to our x. So landmark = (raw_y / width,
raw_x / length). This makes y the goal-to-goal axis the phase splitter expects
(y < 0.333 = own third, y > 0.667 = opp third).
"""

from __future__ import annotations

from typing import Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.pitch.homography import HomographyError, fit_homography

_PITCH_LENGTH_CM: Final = 12000.0
_PITCH_WIDTH_CM: Final = 7000.0

# roboflow SoccerPitchConfiguration vertices in cm, index order == keypoint order.
_RAW_VERTICES_CM: Final = [
    (0, 0), (0, 1450), (0, 2584), (0, 4416), (0, 5550), (0, 7000),
    (550, 2584), (550, 4416), (1100, 3500),
    (2015, 1450), (2015, 2584), (2015, 4416), (2015, 5550),
    (6000, 0), (6000, 2585), (6000, 4415), (6000, 7000),
    (9985, 1450), (9985, 2584), (9985, 4416), (9985, 5550),
    (10900, 3500), (11450, 2584), (11450, 4416),
    (12000, 0), (12000, 1450), (12000, 2584), (12000, 4416), (12000, 5550), (12000, 7000),
    (5085, 3500), (6915, 3500),
]

# Normalized [0,1]^2 landmarks: (x, y) = (raw_y / width, raw_x / length).
PITCH_LANDMARKS: Final[NDArray[np.float64]] = np.array(
    [(ry / _PITCH_WIDTH_CM, rx / _PITCH_LENGTH_CM) for (rx, ry) in _RAW_VERTICES_CM],
    dtype=np.float64,
)


def build_frame_homographies(
    keypoints: pd.DataFrame,
    *,
    conf_threshold: float = 0.5,
    min_points: int = 4,
) -> dict[int, NDArray[np.floating]]:
    """Fit a per-frame homography from detected pitch keypoints.

    For each frame, keypoints with conf >= conf_threshold and a known kp_idx
    contribute (x_px, y_px) -> PITCH_LANDMARKS[kp_idx] correspondences. Frames
    with at least min_points such correspondences get a homography mapping image
    pixels to canonical pitch coords; other frames are skipped (no entry).

    Returns a sparse {frame: H} dict suitable for smooth_homographies().
    """
    homographies: dict[int, NDArray[np.floating]] = {}
    if keypoints.empty:
        return homographies
    n_landmarks = len(PITCH_LANDMARKS)
    for frame_idx, group in keypoints.groupby("frame", sort=False):
        sel = group[
            (group["conf"] >= conf_threshold)
            & (group["kp_idx"] >= 0)
            & (group["kp_idx"] < n_landmarks)
        ]
        if len(sel) < min_points:
            continue
        image_points = sel[["x_px", "y_px"]].to_numpy(dtype=np.float64)
        pitch_points = PITCH_LANDMARKS[sel["kp_idx"].to_numpy()]
        try:
            homographies[cast(int, frame_idx)] = fit_homography(image_points, pitch_points)
        except HomographyError:
            continue
    return homographies
