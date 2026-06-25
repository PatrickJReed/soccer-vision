"""Smoke test for the three-way engine comparison harness."""

from __future__ import annotations

import cv2
import numpy as np
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.pitch.calib_compare import EngineMetrics, compare_engines
from soccer_vision.pitch.manual_anchor import Click


def _look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    e, t, u = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    fwd = t - e
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, u)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rmat = np.vstack([right, down, fwd])
    rvec, _ = cv2.Rodrigues(rmat)
    return rvec, (-rmat @ e).reshape(3, 1)


def test_compare_engines_returns_three_engine_metrics() -> None:
    _K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)
    fp = field_points_3d()
    eyes = {0: (8.0, 4, 70), 2: (33, 14, 80), 4: (18, 59, 75), 6: (40, 44, 85)}
    clicks: list[Click] = []
    for f, e in eyes.items():
        rvec, tvec = _look_at(e, (22.85, 34.25, 0.0))
        px = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080:
                clicks.append(Click(f, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080))
    interframe = {i: np.eye(3) for i in range(6)}
    result = compare_engines(clicks, interframe, 7, (1920, 1080), window=360)
    assert set(result) == {"free_fit", "engine_a", "engine_b"}
    for m in result.values():
        assert isinstance(m, EngineMetrics)
        assert m.n_covered >= 0 and m.n_folded >= 0
        assert 0.0 <= m.coverage_fraction <= 1.0
        assert isinstance(m.per_frame_median_ft, dict)
