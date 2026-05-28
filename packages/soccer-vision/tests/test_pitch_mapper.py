"""Tests for PitchMapper transform."""

from __future__ import annotations

import numpy as np
import pandas as pd
from soccer_vision.pitch.mapper import PitchMapper


def test_identity_transform_passes_coords_through() -> None:
    detections = pd.DataFrame({
        "frame": [0, 0, 1],
        "x_px": [0.2, 0.4, 0.6],
        "y_px": [0.3, 0.5, 0.7],
    })
    identity = np.eye(3)
    homographies = {0: identity, 1: identity}
    mapper = PitchMapper()
    out = mapper.transform(detections, homographies)
    assert np.allclose(out["x_pitch"].to_numpy(), detections["x_px"].to_numpy())
    assert np.allclose(out["y_pitch"].to_numpy(), detections["y_px"].to_numpy())


def test_translation_homography_applied() -> None:
    detections = pd.DataFrame({
        "frame": [0],
        "x_px": [0.5],
        "y_px": [0.5],
    })
    # H translates by (-0.1, -0.2)
    H = np.array([
        [1.0, 0.0, -0.1],
        [0.0, 1.0, -0.2],
        [0.0, 0.0, 1.0],
    ])
    mapper = PitchMapper()
    out = mapper.transform(detections, {0: H})
    assert abs(out["x_pitch"].iloc[0] - 0.4) < 1e-6
    assert abs(out["y_pitch"].iloc[0] - 0.3) < 1e-6


def test_missing_homography_emits_nan() -> None:
    detections = pd.DataFrame({
        "frame": [0, 1],
        "x_px": [0.5, 0.5],
        "y_px": [0.5, 0.5],
    })
    homographies = {0: np.eye(3)}  # frame 1 missing
    mapper = PitchMapper()
    out = mapper.transform(detections, homographies)
    assert not pd.isna(out["x_pitch"].iloc[0])
    assert pd.isna(out["x_pitch"].iloc[1])
