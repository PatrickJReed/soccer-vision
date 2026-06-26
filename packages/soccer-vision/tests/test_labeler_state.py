"""Tests for the calibrated LabelerState backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.labeler.chain import normalize_homography
from soccer_vision.labeler.state import LabelerState
from soccer_vision.pitch.manual_anchor import Click

_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)


def _look_at(
    eye: Any,
    target: Any,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[Any, NDArray[np.float64]]:
    e, t, u = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    f = t - e
    f /= np.linalg.norm(f)
    r = np.cross(f, u)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    rvec, _ = cv2.Rodrigues(np.vstack([r, d, f]))
    return rvec, (-np.vstack([r, d, f]) @ e).reshape(3, 1)


def _pan_session(
    n: int = 9,
) -> tuple[dict[int, NDArray[np.float64]], dict[int, Any], list[Click]]:
    center = (-8.0, 34.0, 9.0)
    poses = {f: _look_at(center, (22.85, 34.0 + dy, 0.0))
             for f, dy in enumerate(np.linspace(-10, 10, n))}
    interframe: dict[int, NDArray[np.float64]] = {}
    for i in range(n - 1):
        ri, _ = cv2.Rodrigues(poses[i][0])
        rj, _ = cv2.Rodrigues(poses[i + 1][0])
        g = _K @ rj @ np.linalg.inv(ri) @ np.linalg.inv(_K)
        interframe[i] = normalize_homography(g, (1920, 1080))
    fp = field_points_3d()
    clicks: list[Click] = []
    for f in (0, 4, 8):
        px = cv2.projectPoints(fp, poses[f][0], poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080:
                clicks.append(Click(f, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080))
    return interframe, poses, clicks


def test_labeler_bootstraps_focal_and_covers_all_frames() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    st.add_clicks(clicks)
    assert st._calibrated
    assert st.coverage() > 0.8
    cf = st.frame_homography(2)  # UNCLICKED frame covered via propagation
    assert cf is not None and cf.H.shape == (3, 3)


def test_labeler_uncalibrated_before_three_anchors() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    st.add_clicks([c for c in clicks if c.frame in (0, 4)])  # only 2 anchors
    assert not st._calibrated
    assert st.frame_homography(2) is None  # bootstrap gap


def test_labeler_flags_outlier_click() -> None:
    interframe, _poses, clicks = _pan_session(9)
    clicks = [c if not (c.frame == 4 and c.kp_idx == 6)
              else Click(c.frame, c.kp_idx, c.x + 0.25, c.y) for c in clicks]
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    st.add_clicks(clicks)
    assert st._outliers.get(4) == [6]   # the mislabel flagged session-level


def test_labeler_add_line_click_refits_and_persists(tmp_path: Path) -> None:
    interframe, _poses, clicks = _pan_session(9)
    sidecar = tmp_path / "s.json"
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080),
                      window=360, autosave_path=sidecar)
    st.add_clicks(clicks)                     # bootstrap on points
    _cf_before = st.frame_homography(4)
    st.add_line_click(4, "midline", 0.5, 0.5)
    assert len(st.line_clicks) == 1
    assert st.frame_homography(4) is not None  # still covered (refine ran)
    # sidecar carries both
    import json
    data = json.loads(sidecar.read_text())
    assert data["line_clicks"] == [{"frame": 4, "line_id": "midline", "x": 0.5, "y": 0.5}]
    assert len(data["clicks"]) == len(clicks)


def test_labeler_remove_last_pops_line_then_point() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    st.add_clicks(clicks)
    st.add_line_click(4, "near_touchline", 0.1, 0.9)
    n_pts = len(st.clicks)
    st.remove_last()                          # pops the line click (added last)
    assert len(st.line_clicks) == 0 and len(st.clicks) == n_pts
    st.remove_last()                          # now pops a point
    assert len(st.clicks) == n_pts - 1


def test_labeler_export_writes_line_clicks_parquet(tmp_path: Path) -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    st.add_clicks(clicks)
    st.add_line_click(4, "midline", 0.5, 0.5)
    st.export(tmp_path)
    import pandas as pd
    df = pd.read_parquet(tmp_path / "line_clicks.parquet")
    assert list(df.columns) == ["frame", "line_id", "x_px", "y_px"]
    assert df.iloc[0]["line_id"] == "midline"
    assert abs(df.iloc[0]["x_px"] - 0.5 * 1920) < 1e-6
