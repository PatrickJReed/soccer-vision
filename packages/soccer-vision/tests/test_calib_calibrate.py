"""Tests for camera calibration: homography helpers + calibrate_camera."""

from __future__ import annotations

import cv2
import numpy as np
from soccer_vision.calib.calibrate import homography_from_pose, pitch_homography
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
