"""Tests for the calibration validation metrics."""

from __future__ import annotations

import cv2
import numpy as np
from soccer_vision.calib.calibrate import homography_from_pose, pitch_homography
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.calib.validate import fold_count, leave_one_out_feet, reproj_error_feet


def _look_at(
    eye: tuple[float, ...] | list[float],
    target: tuple[float, ...] | list[float],
    up: tuple[float, ...] | list[float] = (0.0, 0, 1),
) -> tuple[np.ndarray, np.ndarray]:
    eye_a = np.asarray(eye, float)
    target_a = np.asarray(target, float)
    up_a = np.asarray(up, float)
    fwd = target_a - eye_a
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_a)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rmat = np.vstack([right, down, fwd])
    rvec, _ = cv2.Rodrigues(rmat)
    return rvec, (-rmat @ eye_a).reshape(3, 1)


_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]])


def test_fold_count_full_field_view() -> None:
    # high overhead view -> most of the 21 landmarks in-frame
    rvec, tvec = _look_at((23.0, 4, 80), (22.85, 34.25, 0))
    Hp = pitch_homography(homography_from_pose(_K, rvec, tvec))
    assert fold_count(Hp, (1920, 1080)) >= 15


def test_reproj_error_feet_zero_for_exact() -> None:
    rvec, tvec = _look_at((20.0, -8, 12), (22.85, 34.25, 0))
    Hw = homography_from_pose(_K, rvec, tvec)
    fp = field_points_3d()
    px, _ = cv2.projectPoints(fp[3:4], rvec, tvec, _K, np.zeros(5))
    px_flat = px.reshape(2)
    err = reproj_error_feet(Hw, fp[3], (float(px_flat[0]), float(px_flat[1])))
    assert err < 0.1  # exact projection -> ~0 ft


def test_leave_one_out_feet_perfect_projection() -> None:
    fp = field_points_3d()
    rvec, tvec = _look_at((20.0, -8, 12), (22.85, 34.25, 0))
    px, _ = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))
    px = px.reshape(-1, 2)
    ids = [0, 1, 3, 9, 13, 14, 16, 4]
    obs = {0: [(i, float(px[i, 0]), float(px[i, 1])) for i in ids]}
    errs = leave_one_out_feet(obs, _K, (1920, 1080), min_other=4)
    allv = [v for vals in errs.values() for v in vals]
    assert allv and max(allv) < 1.0  # perfect data -> sub-foot held-out error
