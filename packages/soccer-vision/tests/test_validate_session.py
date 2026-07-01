"""Tests for the physical-calibration acceptance gate CLI (run_gate)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from soccer_vision.calib.field_model import LENGTH_M, field_points_3d
from soccer_vision.labeler.chain import normalize_homography, save_chain
from soccer_vision.pitch.validate_session import run_gate

_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)
SIZE = (1920, 1080)


def _look_at(eye: Any, target: Any) -> tuple[Any, NDArray[np.float64]]:
    e, t, u = np.asarray(eye, float), np.asarray(target, float), np.array([0.0, 0.0, 1.0])
    f = t - e
    f /= np.linalg.norm(f)
    r = np.cross(f, u)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    rvec, _ = cv2.Rodrigues(np.vstack([r, d, f]))
    return rvec, (-np.vstack([r, d, f]) @ e).reshape(3, 1)


def _write_session(tmp_path: Path, n: int = 12) -> tuple[Path, Path]:
    anchors = sorted({0, (n - 1) // 3, 2 * (n - 1) // 3, n - 1})
    poses = {f: _look_at((-8.0, 34.0, 9.0), (22.85, 34.0 + dy, 0.0))
             for f, dy in enumerate(np.linspace(-10, 10, n))}
    interframe: dict[int, NDArray[np.float64]] = {}
    for i in range(n - 1):
        ri, _ = cv2.Rodrigues(poses[i][0])
        rj, _ = cv2.Rodrigues(poses[i + 1][0])
        g = _K @ rj @ np.linalg.inv(ri) @ np.linalg.inv(_K)
        interframe[i] = normalize_homography(g, SIZE)
    chain_path = tmp_path / "chain.npz"
    save_chain(chain_path, interframe, n, SIZE)

    fp = field_points_3d()
    near = np.array([[0.0, y, 0.0] for y in np.linspace(5.0, LENGTH_M - 5.0, 3)])
    clicks: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    for f in anchors:
        px = cv2.projectPoints(fp, poses[f][0], poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080:
                clicks.append({"frame": f, "kp_idx": j,
                               "x": float(px[j, 0]) / 1920, "y": float(px[j, 1]) / 1080})
        npx = cv2.projectPoints(near, poses[f][0], poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        for x, y in npx:
            lines.append({"frame": f, "line_id": "near_touchline",
                          "x": float(x) / 1920, "y": float(y) / 1080})
    clicks_path = tmp_path / "clip.clicks.json"
    clicks_path.write_text(json.dumps({"clicks": clicks, "line_clicks": lines}))
    return chain_path, clicks_path


def test_run_gate_passes_on_clean_session(tmp_path: Path) -> None:
    chain_path, clicks_path = _write_session(tmp_path)
    report = run_gate(chain_path, clicks_path)
    assert report is not None
    assert report.fg_n > 0 and report.prop_n > 0
    assert report.fg_median_ft <= 5.0 and report.fg_p90_ft <= 12.0
    assert report.prop_median_ft <= 5.0
    assert report.passed_numeric


def test_run_gate_none_when_chain_missing(tmp_path: Path) -> None:
    assert run_gate(tmp_path / "nope.npz", tmp_path / "x.json") is None
