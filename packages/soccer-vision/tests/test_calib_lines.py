"""Phase 2 tests: field-lines registry, line residual, line-constrained pose refine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from soccer_vision.calib.calibrate import homography_from_pose, line_residual
from soccer_vision.calib.field_model import (
    FIELD_LINES,
    LENGTH_M,
    WIDTH_M,
    field_line_3d,
    field_points_3d,
)

_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)


def _look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV (world->camera) rvec, tvec for a camera at `eye` looking at `target`."""
    eye_a, target_a, up_a = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    fwd = target_a - eye_a
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_a)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rmat = np.vstack([right, down, fwd])
    rvec, _ = cv2.Rodrigues(rmat)
    return rvec, (-rmat @ eye_a).reshape(3, 1)


def test_field_line_3d_near_touchline_endpoints() -> None:
    p1, p2 = field_line_3d("near_touchline")
    fp = field_points_3d()
    assert np.allclose(p1, fp[0])  # corner_own_left
    assert np.allclose(p2, fp[2])  # corner_opp_left
    assert p1[2] == 0.0 and p2[2] == 0.0  # on the Z=0 plane


def test_field_line_3d_midline_spans_full_width() -> None:
    # midline = (5, 4): halfway_near (x=0) -> halfway_far (x=WIDTH), both at y=L/2
    p1, p2 = field_line_3d("midline")
    assert np.allclose(p1, [0.0, LENGTH_M / 2, 0.0])
    assert np.allclose(p2, [WIDTH_M, LENGTH_M / 2, 0.0])


def test_field_lines_registry_has_five_named_lines() -> None:
    assert set(FIELD_LINES) == {
        "near_touchline", "far_touchline", "own_goal_line", "opp_goal_line", "midline",
    }


def test_field_line_3d_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        field_line_3d("not_a_line")


def test_line_residual_zero_on_the_line() -> None:
    # A pixel that is the projection of a point ON near_touchline (x=0) must have ~0
    # distance to the projected near_touchline.
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    h = homography_from_pose(_K, rvec, tvec)
    p1, p2 = field_line_3d("near_touchline")
    on_line_world = np.array([0.0, 30.0, 0.0])  # x=0 -> on near_touchline
    px = cv2.projectPoints(on_line_world.reshape(1, 3), rvec, tvec, _K, np.zeros(5))[0].reshape(2)
    d = line_residual(h, p1, p2, (float(px[0]), float(px[1])))
    assert d < 0.01


def test_line_residual_equals_perpendicular_offset() -> None:
    # Take an on-line pixel, then offset it perpendicular to the projected line by a
    # known number of pixels; the residual should equal that offset.
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    h = homography_from_pose(_K, rvec, tvec)
    p1, p2 = field_line_3d("near_touchline")
    a = h @ np.array([p1[0], p1[1], 1.0])
    a = a[:2] / a[2]
    b = h @ np.array([p2[0], p2[1], 1.0])
    b = b[:2] / b[2]
    ell = np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])
    n = np.array([ell[0], ell[1]]) / np.hypot(ell[0], ell[1])  # unit normal
    on_line = (a + b) / 2.0
    offset_px = 7.5
    clicked = on_line + offset_px * n
    d = line_residual(h, p1, p2, (float(clicked[0]), float(clicked[1])))
    assert abs(d - offset_px) < 1e-6


def test_line_residual_horizontal_line_known_distance() -> None:
    # Non-circular sanity check: identity homography + two endpoints with equal y
    # give a horizontal image line at y=540; a click 10px above it is 10px away.
    # No cross-product/normal reconstruction here -> catches a wrong-formula bug.
    h = np.eye(3)
    p1 = np.array([0.0, 540.0, 0.0])
    p2 = np.array([1000.0, 540.0, 0.0])
    d = line_residual(h, p1, p2, (500.0, 550.0))
    assert abs(d - 10.0) < 1e-9


def test_line_residual_finite_when_endpoints_offscreen() -> None:
    # near_touchline endpoints (corners) project far off-screen in a midfield view,
    # but the residual must still be a correct finite distance.
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    h = homography_from_pose(_K, rvec, tvec)
    p1, p2 = field_line_3d("near_touchline")
    d = line_residual(h, p1, p2, (500.0, 500.0))
    assert np.isfinite(d) and d >= 0.0


def test_line_residual_degenerate_endpoints_returns_zero() -> None:
    # A homography that collapses both endpoints to the same pixel -> degenerate line.
    p1, p2 = field_line_3d("near_touchline")
    h = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 200.0], [0.0, 0.0, 1.0]])  # maps everything to (100,200)
    d = line_residual(h, p1, p2, (300.0, 400.0))
    assert d == 0.0
