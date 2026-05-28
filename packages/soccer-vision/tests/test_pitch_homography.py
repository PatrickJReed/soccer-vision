"""Tests for pitch homography fitting."""

from __future__ import annotations

import numpy as np
import pytest
from soccer_vision.pitch.homography import HomographyError, fit_homography


def test_identity_homography_from_unit_square() -> None:
    """Mapping the unit square to itself returns ~identity homography."""
    img_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    pitch_pts = img_pts.copy()
    H = fit_homography(img_pts, pitch_pts)
    H_norm = H / H[2, 2]
    assert np.allclose(H_norm, np.eye(3), atol=1e-6)


def test_translation_homography() -> None:
    """Image points shifted by +10 in x should produce a translation H."""
    img_pts = np.array([[10, 0], [11, 0], [11, 1], [10, 1]], dtype=np.float32)
    pitch_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H = fit_homography(img_pts, pitch_pts)
    pt = np.array([10.5, 0.5, 1.0])
    out = H @ pt
    out /= out[2]
    assert abs(out[0] - 0.5) < 1e-4
    assert abs(out[1] - 0.5) < 1e-4


def test_too_few_points_raises() -> None:
    img_pts = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
    pitch_pts = img_pts.copy()
    with pytest.raises(HomographyError, match="at least 4"):
        fit_homography(img_pts, pitch_pts)


def test_mismatched_shapes_raises() -> None:
    img_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    pitch_pts = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
    with pytest.raises(HomographyError, match="same number"):
        fit_homography(img_pts, pitch_pts)
