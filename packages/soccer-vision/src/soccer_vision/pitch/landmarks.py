"""Canonical youth-9v9 pitch landmarks + per-frame homography fitting.

PITCH_LANDMARKS maps each pitch-model keypoint index to its canonical pitch
coordinate in [0, 1]^2. Coordinates are COMPUTED from a PitchSpec (the same
proportions the metrics layer uses), not vendored from a fixed table, so the
homography target space and the analytics space share one definition.

Axis convention: y = goal-to-goal (the phase splitter expects y < 0.333 = own
third, y > 0.667 = opp third); x = touchline-to-touchline. Each axis is
normalized to [0, 1] independently.

The 21-point schema (see docs/superpowers/specs/2026-06-03-pitch-keypoint-finetune-design.md):
corners (0-3), halfway x touchline far/near (4-5), center mark (6),
center-circle apexes (7-8), own/opp penalty-box corners (9-16), goal-post
bases (17-20). Index 5 (near halfway x touchline) sits under the Trace camera
and is never visible; its slot is reserved for schema regularity.
"""

from __future__ import annotations

from typing import Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.pitch.homography import HomographyError, fit_homography
from soccer_vision.pitch.spec import PitchSpec

# Index of the near halfway x touchline point — directly under the camera, never
# visible, never labeled (kept for schema regularity).
NEAR_HALFWAY_IDX: Final = 5

# Left<->right (x-mirror) keypoint permutation for YOLO-pose fliplr augmentation.
FLIP_IDX: Final[list[int]] = [
    1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19,
]


def youth_landmarks(spec: PitchSpec) -> NDArray[np.float64]:
    """Compute the 21 canonical [0,1]^2 landmark coords from a PitchSpec.

    See module docstring / spec for the index map. y = goal-to-goal,
    x = touchline-to-touchline.
    """
    r = spec.center_circle_radius_frac
    bl = spec.penalty_box_length_frac
    cx_l = 0.5 - spec.penalty_box_width_frac / 2.0
    cx_r = 0.5 + spec.penalty_box_width_frac / 2.0
    gw = spec.goal_width_frac
    pts: list[tuple[float, float]] = [
        (0.0, 0.0),            # 0  corner own-left
        (1.0, 0.0),            # 1  corner own-right
        (0.0, 1.0),            # 2  corner opp-left
        (1.0, 1.0),            # 3  corner opp-right
        (1.0, 0.5),            # 4  halfway x touchline far
        (0.0, 0.5),            # 5  halfway x touchline near (reserved)
        (0.5, 0.5),            # 6  center mark
        (0.5, 0.5 + r),        # 7  center-circle apex far
        (0.5, 0.5 - r),        # 8  center-circle apex near
        (cx_l, bl),            # 9  own box outer-left
        (cx_r, bl),            # 10 own box outer-right
        (cx_l, 0.0),           # 11 own box goalline-left
        (cx_r, 0.0),           # 12 own box goalline-right
        (cx_l, 1.0 - bl),      # 13 opp box outer-left
        (cx_r, 1.0 - bl),      # 14 opp box outer-right
        (cx_l, 1.0),           # 15 opp box goalline-left
        (cx_r, 1.0),           # 16 opp box goalline-right
        (0.5 - gw / 2.0, 0.0), # 17 own goal post left
        (0.5 + gw / 2.0, 0.0), # 18 own goal post right
        (0.5 - gw / 2.0, 1.0), # 19 opp goal post left
        (0.5 + gw / 2.0, 1.0), # 20 opp goal post right
    ]
    return np.array(pts, dtype=np.float64)


PITCH_LANDMARKS: Final[NDArray[np.float64]] = youth_landmarks(PitchSpec.standard_9v9())


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
