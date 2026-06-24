"""Tests for the reprojected-pitch accuracy overlay."""

from __future__ import annotations

import numpy as np
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.viz.pitch_overlay import draw_reprojected_pitch, reproject_landmarks

_W, _H = 1920, 1080
_SCALE = np.array([_W, _H], dtype=float)


def test_reproject_landmarks_round_trip() -> None:
    # image_points = pitch landmarks scaled to pixels (a valid homography); a
    # subset of >=4 must recover H so EVERY landmark reprojects to its px spot.
    idx = np.array([0, 3, 13, 16, 19, 20])
    image_points = PITCH_LANDMARKS[idx] * _SCALE
    out = reproject_landmarks(image_points, idx)
    assert out is not None
    expected = PITCH_LANDMARKS * _SCALE
    assert np.allclose(out, expected, atol=1e-3)


def test_reproject_landmarks_too_few_points() -> None:
    idx = np.array([0, 3, 13])
    image_points = PITCH_LANDMARKS[idx] * _SCALE
    assert reproject_landmarks(image_points, idx) is None


def test_draw_reprojected_pitch_marks_and_lines() -> None:
    idx = np.array([0, 3, 13, 16, 19, 20])
    image_points = PITCH_LANDMARKS[idx] * _SCALE
    frame = np.zeros((_H, _W, 3), dtype=np.uint8)
    out, fit_ok = draw_reprojected_pitch(frame, image_points, idx)
    assert fit_ok
    assert out.shape == frame.shape
    assert int((out != frame).sum()) > 0  # something was drawn
    # the input frame must be untouched (copy, not in-place)
    assert int(frame.sum()) == 0


def test_draw_reprojected_pitch_no_fit_still_draws_dots() -> None:
    idx = np.array([0, 3, 13])  # < 4 -> no homography
    image_points = PITCH_LANDMARKS[idx] * _SCALE
    frame = np.zeros((_H, _W, 3), dtype=np.uint8)
    out, fit_ok = draw_reprojected_pitch(frame, image_points, idx)
    assert not fit_ok
    assert int((out != frame).sum()) > 0  # dots still drawn
