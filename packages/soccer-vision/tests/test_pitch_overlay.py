"""Tests for clipped_polyline helper in pitch_overlay."""

from __future__ import annotations

import numpy as np
from soccer_vision.viz.pitch_overlay import clipped_polyline


def test_clipped_polyline_keeps_in_frame_points() -> None:
    h_pitch_to_px = np.array([[1000.0, 0, 200], [0, 1000.0, 200], [0, 0, 1.0]])
    pts = np.array([[0.1, 0.1], [0.15, 0.15]])   # -> (300,300),(350,350), in frame
    out = clipped_polyline(h_pitch_to_px, pts, size=(1920, 1080), margin=80)
    assert len(out) == 2
    assert all(-80 <= x <= 1920 + 80 and -80 <= y <= 1080 + 80 for x, y in out)


def test_clipped_polyline_drops_behind_camera() -> None:
    h_neg = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])  # forces w<0
    out = clipped_polyline(h_neg, np.array([[0.1, 0.1]]), size=(1920, 1080), margin=80)
    assert out == []


def test_clipped_polyline_drops_offscreen() -> None:
    h = np.array([[100000.0, 0, 0], [0, 100000.0, 0], [0, 0, 1.0]])  # maps far off-frame
    out = clipped_polyline(h, np.array([[0.5, 0.5]]), size=(1920, 1080), margin=80)
    assert out == []
