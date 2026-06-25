"""Tests for camera calibration: homography helpers + calibrate_camera."""

from __future__ import annotations

import cv2
import numpy as np
from soccer_vision.calib.calibrate import (
    CalibError,
    CalibResult,
    calibrate_camera,
    homography_from_pose,
    pitch_homography,
)
from soccer_vision.calib.field_model import field_points_3d


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV (world->camera) rvec, tvec for a camera at `eye` looking at `target`."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    rmat = np.vstack([right, down, fwd])
    rvec, _ = cv2.Rodrigues(rmat)
    return rvec, (-rmat @ eye).reshape(3, 1)


_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]])


def test_homography_from_pose_matches_projectpoints() -> None:
    rvec, tvec = _look_at(np.array([20.0, -10, 12]), np.array([22.0, 34, 0]), np.array([0.0, 0, 1]))
    fp = field_points_3d()
    H = homography_from_pose(_K, rvec, tvec)
    proj = (H @ np.column_stack([fp[:, 0], fp[:, 1], np.ones(len(fp))]).T).T
    proj = proj[:, :2] / proj[:, 2:3]
    truth, _ = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))
    assert np.allclose(proj, truth.reshape(-1, 2), atol=1e-6)


def test_pitch_homography_scales_canonical() -> None:
    from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
    rvec, tvec = _look_at(np.array([20.0, -10, 12]), np.array([22.0, 34, 0]), np.array([0.0, 0, 1]))
    Hw = homography_from_pose(_K, rvec, tvec)
    Hp = pitch_homography(Hw)
    canon = np.column_stack([PITCH_LANDMARKS[:, 0], PITCH_LANDMARKS[:, 1], np.ones(21)])
    a = (Hp @ canon.T).T
    a = a[:, :2] / a[:, 2:3]
    fp = field_points_3d()
    b = (Hw @ np.column_stack([fp[:, 0], fp[:, 1], np.ones(21)]).T).T
    b = b[:, :2] / b[:, 2:3]
    assert np.allclose(a, b, atol=1e-6)


def _observations(
    eyes: list[tuple[float, float, float]],
    target: tuple[float, float, float] = (22.85, 34.25, 0.0),
    k: np.ndarray = _K,
    w: int = 1920,
    h: int = 1080,
) -> dict[int, list[tuple[int, float, float]]]:
    fp = field_points_3d()
    obs: dict[int, list[tuple[int, float, float]]] = {}
    for i, e in enumerate(eyes):
        rvec, tvec = _look_at(np.array(e, float), np.array(target, float), np.array([0.0, 0, 1]))
        px, _ = cv2.projectPoints(fp, rvec, tvec, k, np.zeros(5))
        px = px.reshape(-1, 2)
        ids = [j for j in range(21) if j != 5 and 0 < px[j, 0] < w and 0 < px[j, 1] < h]
        if len(ids) >= 6:
            obs[i] = [(j, float(px[j, 0]), float(px[j, 1])) for j in ids]
    return obs


# 8 elevated, varied viewpoints -> the whole field projects in-frame; well-conditioned focal
_EYES: list[tuple[float, float, float]] = [
    (8.0, 4.0, 70.0), (33.0, 14.0, 80.0), (18.0, 59.0, 75.0), (40.0, 44.0, 85.0),
    (23.0, -1.0, 90.0), (3.0, 34.0, 78.0), (31.0, 64.0, 82.0), (38.0, 19.0, 88.0),
]


def test_calibrate_recovers_known_focal() -> None:
    obs = _observations(_EYES)
    res = calibrate_camera(obs, (1920, 1080), min_points=6)
    assert isinstance(res, CalibResult)
    assert abs(res.K[0, 0] - 1400.0) < 30.0          # focal recovered (~2%)
    assert max(res.rms_px.values()) < 1.0            # near-perfect reprojection (noiseless)
    # the recovered homography reprojects the field to the same pixels as the pose
    fp = field_points_3d()
    f0 = res.frames[0]
    Hpix = (res.homography(f0) @ np.column_stack([fp[:, 0], fp[:, 1], np.ones(21)]).T).T
    Hpix = Hpix[:, :2] / Hpix[:, 2:3]
    rv, tv = res.poses[f0]
    truth, _ = cv2.projectPoints(fp, rv, tv, res.K, np.zeros(5))
    assert np.allclose(Hpix, truth.reshape(-1, 2), atol=1.0)


def test_calibrate_rejects_outlier_view() -> None:
    obs = _observations(_EYES)
    # add a garbage view (random pixels) that should be rejected
    obs[99] = [(j, 5.0 * j, 7.0 * j) for j in range(0, 12)]
    res = calibrate_camera(obs, (1920, 1080), min_points=6, rms_reject_px=20.0)
    assert res.n_excluded >= 1
    assert 99 not in res.frames
    assert abs(res.K[0, 0] - 1400.0) < 40.0


def test_calibrate_too_few_views_raises() -> None:
    obs = _observations(_EYES[:2])
    try:
        calibrate_camera(obs, (1920, 1080), min_points=6)
    except CalibError as e:
        assert "view" in str(e).lower()
    else:
        raise AssertionError("expected CalibError")
