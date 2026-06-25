"""Tests for the 9v9 field model in metres."""

from __future__ import annotations

import numpy as np
from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M, field_points_3d


def test_field_points_shape_and_plane() -> None:
    pts = field_points_3d()
    assert pts.shape == (21, 3)
    assert np.allclose(pts[:, 2], 0.0)  # planar, Z=0


def test_field_points_corners_in_metres() -> None:
    pts = field_points_3d()
    assert np.allclose(pts[0], [0.0, 0.0, 0.0])                 # corner_own_left  (0,0)
    assert np.allclose(pts[3], [WIDTH_M, LENGTH_M, 0.0])        # corner_opp_right (1,1)
    assert np.allclose(pts[1], [WIDTH_M, 0.0, 0.0])             # corner_own_right (1,0)
