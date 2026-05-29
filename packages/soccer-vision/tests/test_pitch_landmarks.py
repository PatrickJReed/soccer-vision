"""Tests for PITCH_LANDMARKS and build_frame_homographies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS, build_frame_homographies


def test_landmarks_shape_and_range() -> None:
    assert PITCH_LANDMARKS.shape == (32, 2)
    assert PITCH_LANDMARKS.min() >= 0.0
    assert PITCH_LANDMARKS.max() <= 1.0


def _keypoints_for_identity(frame: int, conf: float = 0.9) -> pd.DataFrame:
    """6 well-spread landmarks whose image points equal their pitch coords
    (so the fitted homography is identity)."""
    idxs = [0, 5, 13, 16, 24, 29]
    pts = PITCH_LANDMARKS[idxs]
    return pd.DataFrame({
        "frame": [frame] * len(idxs),
        "kp_idx": idxs,
        "x_px": pts[:, 0],
        "y_px": pts[:, 1],
        "conf": [conf] * len(idxs),
    })


def test_build_recovers_known_transform() -> None:
    idxs = [0, 5, 13, 16, 24, 29]
    pitch = PITCH_LANDMARKS[idxs]
    image = pitch * 1000.0 + np.array([50.0, 30.0])  # known affine
    kp = pd.DataFrame({
        "frame": [7] * len(idxs),
        "kp_idx": idxs,
        "x_px": image[:, 0],
        "y_px": image[:, 1],
        "conf": [0.9] * len(idxs),
    })
    homographies = build_frame_homographies(kp)
    assert set(homographies) == {7}
    pts = np.column_stack([image[:, 0], image[:, 1], np.ones(len(idxs))])
    mapped = (homographies[7] @ pts.T).T
    mapped /= mapped[:, 2:3]
    assert np.allclose(mapped[:, :2], pitch, atol=1e-6)


def test_frames_below_min_points_are_skipped() -> None:
    kp = _keypoints_for_identity(3).head(3)  # only 3 points
    assert build_frame_homographies(kp) == {}


def test_low_confidence_keypoints_filtered() -> None:
    kp = _keypoints_for_identity(3, conf=0.1)  # all below default 0.5
    assert build_frame_homographies(kp) == {}


def test_empty_keypoints_returns_empty() -> None:
    empty = pd.DataFrame(columns=["frame", "kp_idx", "x_px", "y_px", "conf"])
    assert build_frame_homographies(empty) == {}
