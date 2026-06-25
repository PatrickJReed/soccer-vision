"""Phase 2 tests: field-lines registry, line residual, line-constrained pose refine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from soccer_vision.calib.calibrate import (
    CalibError,
    homography_from_pose,
    line_residual,
    refine_pose,
)
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


def _project_ids(
    rvec: np.ndarray, tvec: np.ndarray, ids: list[int], k: np.ndarray = _K
) -> dict[int, tuple[float, float]]:
    fp = field_points_3d()
    px = cv2.projectPoints(fp[ids], rvec, tvec, k, np.zeros(5))[0].reshape(-1, 2)
    return {i: (float(px[n, 0]), float(px[n, 1])) for n, i in enumerate(ids)}


def test_refine_pose_too_few_constraints_raises() -> None:
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    # 2 points (4 residuals) + 1 line (1 residual) = 5 < 6
    pts = _project_ids(rvec, tvec, [1, 3])
    point_obs = [(i, x, y) for i, (x, y) in pts.items()]
    line_obs = [("near_touchline", 500.0, 500.0)]
    with pytest.raises(CalibError):
        refine_pose(_K, rvec, tvec, point_obs, line_obs)


def test_refine_pose_bad_point_index_raises() -> None:
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    # 3 points (6 residuals >= min) but kp_idx 99 is out of range -> CalibError, not IndexError
    point_obs = [(1, 100.0, 100.0), (3, 200.0, 200.0), (99, 300.0, 300.0)]
    with pytest.raises(CalibError):
        refine_pose(_K, rvec, tvec, point_obs, [])


def _inframe(px: np.ndarray, w: int = 1920, h: int = 1080) -> bool:
    return bool(0 < px[0] < w and 0 < px[1] < h)


def test_refine_pose_roundtrip_recovers_true_pose() -> None:
    # Noiseless full obs from a known pose; seed a perturbed pose; refine must return
    # to the truth (point reprojection residual ~0).
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    ids = [1, 3, 4, 6, 7, 8, 10, 12, 14, 16]
    pts = _project_ids(rvec, tvec, ids)
    point_obs = [(i, x, y) for i, (x, y) in pts.items()]
    # one midline line click: a pixel on the midline (y=L/2), in frame
    on_line = cv2.projectPoints(
        np.array([[22.0, 34.25, 0.0]]), rvec, tvec, _K, np.zeros(5))[0].reshape(2)
    line_obs = [("midline", float(on_line[0]), float(on_line[1]))]
    rvec0 = rvec + np.array([[0.03], [-0.02], [0.01]])
    tvec0 = tvec + np.array([[1.5], [-1.0], [0.8]])
    rr, tt = refine_pose(_K, rvec0, tvec0, point_obs, line_obs)
    # reprojection of the clicked points under the recovered pose is ~exact
    proj = cv2.projectPoints(field_points_3d()[ids], rr, tt, _K, np.zeros(5))[0].reshape(-1, 2)
    truth = np.array([pts[i] for i in ids])
    assert np.max(np.linalg.norm(proj - truth, axis=1)) < 0.5


def test_line_constraint_tightens_underconstrained_midline() -> None:
    # THE GATE (Patrick's complaint #1: the midline's NEAR part is never labeled,
    # only the far end). A Trace-like midfield sideline view. Every clickable point
    # landmark in view sits at x >= ~22 m (centre line and the far half) -- the NEAR
    # half of the visible midline (x ~ 3..20 m) has NO point support, so points-only
    # must EXTRAPOLATE it. Under click noise that extrapolation swings; a single
    # midline line click in the near region pins it. Averaged over fixed noise draws,
    # points+line beats points-only on a held-out near-midline point. All clicks are
    # IN FRAME (a realistic labeler click), unlike the near touchline which projects
    # below the frame bottom in this geometry.
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 40.0, 0.0))
    fp = field_points_3d()
    allpx = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    # realistic: click EVERY in-frame landmark (idx 5 is hidden under the camera)
    ids = [i for i in range(21) if i != 5 and _inframe(allpx[i])]
    clean = {i: (float(allpx[i, 0]), float(allpx[i, 1])) for i in ids}
    # visible span of the midline (y = L/2); held-out + line-click both in its NEAR,
    # point-free region, at DIFFERENT points (so the line must generalise, not memo)
    y_mid = 34.25
    xs = np.linspace(0.0, 45.7, 80)
    mlpx = cv2.projectPoints(
        np.column_stack([xs, np.full(80, y_mid), np.zeros(80)]), rvec, tvec, _K, np.zeros(5)
    )[0].reshape(-1, 2)
    vis_x = sorted(float(x) for x, px in zip(xs, mlpx, strict=True) if _inframe(px))
    x_held, x_click = vis_x[0] + 3.0, vis_x[0] + 8.0
    held_world = np.array([[x_held, y_mid, 0.0]])
    held_px = cv2.projectPoints(held_world, rvec, tvec, _K, np.zeros(5))[0].reshape(2)
    line_px = cv2.projectPoints(
        np.array([[x_click, y_mid, 0.0]]), rvec, tvec, _K, np.zeros(5))[0].reshape(2)
    assert _inframe(held_px) and _inframe(line_px)  # realistic in-frame clicks

    rng = np.random.default_rng(2)
    err_points_only: list[float] = []
    err_points_line: list[float] = []
    for _ in range(10):
        noisy = {i: (x + rng.normal(0, 2.0), y + rng.normal(0, 2.0)) for i, (x, y) in clean.items()}
        point_obs = [(i, x, y) for i, (x, y) in noisy.items()]
        line_obs = [("midline", float(line_px[0]), float(line_px[1]))]
        # Both arms seed from the TRUE pose, so the test isolates the line
        # constraint's benefit from any seed-recovery confound (if anything this
        # favours the points-only arm).
        ra, ta = refine_pose(_K, rvec, tvec, point_obs, [])
        rb, tb = refine_pose(_K, rvec, tvec, point_obs, line_obs)
        pa = cv2.projectPoints(held_world, ra, ta, _K, np.zeros(5))[0].reshape(2)
        pb = cv2.projectPoints(held_world, rb, tb, _K, np.zeros(5))[0].reshape(2)
        err_points_only.append(float(np.linalg.norm(pa - held_px)))
        err_points_line.append(float(np.linalg.norm(pb - held_px)))

    mean_only = float(np.mean(err_points_only))
    mean_line = float(np.mean(err_points_line))
    # the line constraint must MATERIALLY tighten the under-constrained near midline.
    # Prototyped value with this seed: ratio ~0.77 (≈23% lower held-out error).
    assert mean_line < mean_only, f"line did not help: only={mean_only:.1f} line={mean_line:.1f}"
    assert mean_line < 0.85 * mean_only, f"line gain too small: only={mean_only:.1f} line={mean_line:.1f}"
