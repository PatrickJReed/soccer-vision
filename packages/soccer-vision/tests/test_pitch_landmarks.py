"""Tests for PITCH_LANDMARKS (21-pt youth schema) and build_frame_homographies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from soccer_vision.pitch.landmarks import (
    FLIP_IDX,
    LANDMARK_NAMES,
    NEAR_HALFWAY_IDX,
    PITCH_LANDMARKS,
    build_frame_homographies,
    youth_landmarks,
)
from soccer_vision.pitch.spec import PitchSpec

# Six well-spread, non-collinear indices used across fit tests.
_FIT_IDXS = [0, 3, 6, 11, 16, 19]


def test_landmarks_shape_and_range() -> None:
    assert PITCH_LANDMARKS.shape == (21, 2)
    assert PITCH_LANDMARKS.min() >= 0.0
    assert PITCH_LANDMARKS.max() <= 1.0


def test_corners_at_extremes() -> None:
    assert np.allclose(PITCH_LANDMARKS[0], (0.0, 0.0))
    assert np.allclose(PITCH_LANDMARKS[1], (1.0, 0.0))
    assert np.allclose(PITCH_LANDMARKS[2], (0.0, 1.0))
    assert np.allclose(PITCH_LANDMARKS[3], (1.0, 1.0))


def test_center_mark_is_center() -> None:
    assert np.allclose(PITCH_LANDMARKS[6], (0.5, 0.5))


def test_center_circle_apexes_straddle_center_on_y() -> None:
    r = PitchSpec.standard_9v9().center_circle_radius_frac
    assert np.allclose(PITCH_LANDMARKS[7], (0.5, 0.5 + r))
    assert np.allclose(PITCH_LANDMARKS[8], (0.5, 0.5 - r))


def test_goal_posts_straddle_center_on_x_at_goal_lines() -> None:
    gw = PitchSpec.standard_9v9().goal_width_frac
    assert np.allclose(PITCH_LANDMARKS[17], (0.5 - gw / 2, 0.0))
    assert np.allclose(PITCH_LANDMARKS[18], (0.5 + gw / 2, 0.0))
    assert np.allclose(PITCH_LANDMARKS[19], (0.5 - gw / 2, 1.0))
    assert np.allclose(PITCH_LANDMARKS[20], (0.5 + gw / 2, 1.0))


def test_box_corners_symmetric_about_x_center() -> None:
    assert np.isclose(PITCH_LANDMARKS[9, 0] + PITCH_LANDMARKS[10, 0], 1.0)
    assert np.isclose(PITCH_LANDMARKS[9, 1], PITCH_LANDMARKS[10, 1])


def test_near_halfway_index_constant() -> None:
    assert NEAR_HALFWAY_IDX == 5
    assert np.allclose(PITCH_LANDMARKS[5], (0.0, 0.5))


def test_landmark_names_match_schema() -> None:
    assert len(LANDMARK_NAMES) == len(PITCH_LANDMARKS)
    assert LANDMARK_NAMES[0] == "corner_own_left"
    assert "hidden" in LANDMARK_NAMES[NEAR_HALFWAY_IDX]  # idx 5 flagged never-visible
    assert all(isinstance(n, str) and n for n in LANDMARK_NAMES)


def test_flip_idx_is_an_involution() -> None:
    assert len(FLIP_IDX) == 21
    f = np.array(FLIP_IDX)
    assert np.array_equal(f[f], np.arange(21))


def test_flip_idx_mirrors_x_coordinates() -> None:
    flipped = PITCH_LANDMARKS[np.array(FLIP_IDX)]
    assert np.allclose(flipped[:, 0], 1.0 - PITCH_LANDMARKS[:, 0])
    assert np.allclose(flipped[:, 1], PITCH_LANDMARKS[:, 1])


def test_youth_landmarks_scales_with_spec() -> None:
    wide = youth_landmarks(PitchSpec(goal_width_frac=0.30))
    assert np.isclose(wide[18, 0] - wide[17, 0], 0.30)


def test_youth_landmarks_scales_with_box_depth() -> None:
    deep = youth_landmarks(PitchSpec(penalty_box_length_frac=0.25))
    assert np.isclose(deep[9, 1], 0.25)   # own box outer y = bl
    assert np.isclose(deep[13, 1], 0.75)  # opp box outer y = 1 - bl


def _keypoints_for_identity(frame: int, conf: float = 0.9) -> pd.DataFrame:
    pts = PITCH_LANDMARKS[_FIT_IDXS]
    return pd.DataFrame({
        "frame": [frame] * len(_FIT_IDXS),
        "kp_idx": _FIT_IDXS,
        "x_px": pts[:, 0],
        "y_px": pts[:, 1],
        "conf": [conf] * len(_FIT_IDXS),
    })


def test_build_recovers_known_transform() -> None:
    pitch = PITCH_LANDMARKS[_FIT_IDXS]
    image = pitch * 1000.0 + np.array([50.0, 30.0])
    kp = pd.DataFrame({
        "frame": [7] * len(_FIT_IDXS),
        "kp_idx": _FIT_IDXS,
        "x_px": image[:, 0],
        "y_px": image[:, 1],
        "conf": [0.9] * len(_FIT_IDXS),
    })
    homographies = build_frame_homographies(kp)
    assert set(homographies) == {7}
    pts = np.column_stack([image[:, 0], image[:, 1], np.ones(len(_FIT_IDXS))])
    mapped = (homographies[7] @ pts.T).T
    mapped /= mapped[:, 2:3]
    assert np.allclose(mapped[:, :2], pitch, atol=1e-6)


def test_frames_below_min_points_are_skipped() -> None:
    kp = _keypoints_for_identity(3).head(3)
    assert build_frame_homographies(kp) == {}


def test_low_confidence_keypoints_filtered() -> None:
    kp = _keypoints_for_identity(3, conf=0.1)
    assert build_frame_homographies(kp) == {}


def test_empty_keypoints_returns_empty() -> None:
    empty = pd.DataFrame(columns=["frame", "kp_idx", "x_px", "y_px", "conf"])
    assert build_frame_homographies(empty) == {}


def test_collinear_keypoints_are_skipped() -> None:
    idxs = [0, 1, 2, 3, 4]
    kp = pd.DataFrame({
        "frame": [9] * 5,
        "kp_idx": idxs,
        "x_px": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y_px": [0.0, 1.0, 2.0, 3.0, 4.0],
        "conf": [0.9] * 5,
    })
    assert build_frame_homographies(kp) == {}


def test_multiple_frames_dispatched_independently() -> None:
    good = _keypoints_for_identity(0)
    sparse = _keypoints_for_identity(1).head(3)
    kp = pd.concat([good, sparse], ignore_index=True)
    homographies = build_frame_homographies(kp)
    assert set(homographies) == {0}
