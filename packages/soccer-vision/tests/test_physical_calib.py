import cv2
import numpy as np
from numpy.typing import NDArray
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.pitch.manual_anchor import Click
from soccer_vision.pitch.physical_calib import solve_session

SIZE = (1920, 1080)


def _project(
    fp_ids: list[int],
    K: NDArray[np.float64],
    rvec: NDArray[np.float64],
    tvec: NDArray[np.float64],
    size: tuple[int, int],
) -> list[tuple[int, float, float]]:
    fp = field_points_3d()
    img = cv2.projectPoints(fp[fp_ids], rvec, tvec, K, None)[0].reshape(-1, 2)
    w, h = size
    return [(i, x / w, y / h) for i, (x, y) in zip(fp_ids, img, strict=True)]


def _synthetic_clicks(
    size: tuple[int, int],
) -> tuple[list[Click], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    K = np.array([[1460.0, 0, size[0] / 2], [0, 1460.0, size[1] / 2], [0, 0, 1]])
    rvec = np.array([[1.2], [0.0], [0.0]])
    tvec = np.array([[-22.0], [-3.0], [40.0]])
    ids = [0, 1, 4, 9, 10, 13, 14]
    proj = _project(ids, K, rvec, tvec, size)
    return [Click(frame=100, kp_idx=i, x=x, y=y) for i, x, y in proj], K, rvec, tvec


def test_solve_session_recovers_anchor() -> None:
    clicks, _K, _rvec, _tvec = _synthetic_clicks(SIZE)
    calib = solve_session(clicks, [], SIZE, {100: np.eye(3)})
    assert calib.is_anchor(100)
    H = calib.frame_homography(100)
    assert H is not None
    from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
    for c in clicks:
        q = H @ np.array([c.x, c.y, 1.0])
        q = q[:2] / q[2]
        assert np.linalg.norm(q - PITCH_LANDMARKS[c.kp_idx]) < 0.02
    assert calib.frame_homography(999) is None
