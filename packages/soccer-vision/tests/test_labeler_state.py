"""Tests for the calibrated LabelerState backend (physical per-frame engine)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from soccer_vision.calib.field_model import LENGTH_M, field_points_3d
from soccer_vision.labeler.chain import normalize_homography
from soccer_vision.labeler.state import CalibFrame, LabelerState
from soccer_vision.pitch.manual_anchor import Click, LineClick

_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)
SIZE = (1920, 1080)


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


def _spread_anchors(n: int) -> list[int]:
    """Anchor frames spread across the whole clip (full pan diversity for the shared focal)."""
    return sorted({0, (n - 1) // 3, 2 * (n - 1) // 3, n - 1})


def _pan_session(
    n: int = 9, anchors: list[int] | None = None
) -> tuple[dict[int, NDArray[np.float64]], dict[int, Any], list[Click]]:
    """A fixed-centre camera PANNING across the field (rotation-only, no parallax): the
    physically-consistent session the physical engine expects. Anchors are spread across the
    clip so the >= 3-diverse-view shared-focal calibration is well posed."""
    anchors = _spread_anchors(n) if anchors is None else anchors
    center = (-8.0, 34.0, 9.0)
    poses = {f: _look_at(center, (22.85, 34.0 + dy, 0.0))
             for f, dy in enumerate(np.linspace(-10, 10, n))}
    interframe: dict[int, NDArray[np.float64]] = {}
    for i in range(n - 1):
        ri, _ = cv2.Rodrigues(poses[i][0])
        rj, _ = cv2.Rodrigues(poses[i + 1][0])
        g = _K @ rj @ np.linalg.inv(ri) @ np.linalg.inv(_K)
        interframe[i] = normalize_homography(g, SIZE)
    fp = field_points_3d()
    clicks: list[Click] = []
    for f in anchors:
        px = cv2.projectPoints(fp, poses[f][0], poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080:
                clicks.append(Click(f, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080))
    return interframe, poses, clicks


def _near_tl_clicks(poses: dict[int, Any], frames: list[int], n: int = 3) -> list[LineClick]:
    """Near-touchline (x=0) line clicks for the given frames, projected from each pose."""
    obj = np.array([[0.0, y, 0.0] for y in np.linspace(5.0, LENGTH_M - 5.0, n)])
    out: list[LineClick] = []
    for f in frames:
        px = cv2.projectPoints(obj, poses[f][0], poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        for x, y in px:
            out.append(LineClick(f, "near_touchline", float(x) / 1920, float(y) / 1080))
    return out


# ---- calibration + propagation ----
def test_bootstraps_and_covers_whole_segment() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        st.add_clicks(clicks)
        assert st._calibrated
        # every frame (clicked anchor or propagated within the gap guard) has a homography
        assert all(st.frame_homography(f) is not None for f in range(9))
        anchors = _spread_anchors(9)
        assert st.frame_homography(anchors[0]).is_anchor          # type: ignore[union-attr]
        unclicked = next(f for f in range(9) if f not in anchors)
        assert not st.frame_homography(unclicked).is_anchor       # type: ignore[union-attr]
    finally:
        st.stop_worker()


def test_uncalibrated_below_three_diverse_frames() -> None:
    # The physical model needs >= 3 diverse views for the shared focal; 2 clicked frames
    # cannot calibrate -> no anchors, no homography.
    interframe, _poses, clicks = _pan_session(9)
    two_frames = [c for c in clicks if c.frame in _spread_anchors(9)[:2]]
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        st.add_clicks(two_frames)
        assert not st._calibrated
        assert st.frame_homography(1) is None
    finally:
        st.stop_worker()


# ---- coverage-graded status ----
def test_anchor_is_yellow_without_near_touchline() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        st.add_clicks(clicks)           # points only, no near-touchline
        st.wait_idle(timeout=10)
        assert "green" not in st.status_list()   # foreground unverified -> never green
    finally:
        st.stop_worker()


def test_anchor_is_green_with_near_touchline() -> None:
    interframe, poses, clicks = _pan_session(9)
    anchors = _spread_anchors(9)
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        st.add_clicks(clicks)
        st.add_line_clicks(_near_tl_clicks(poses, anchors))
        st.wait_idle(timeout=10)
        greens = [f for f in range(9) if st._status_of(f) == "green"]
        assert set(greens) == set(anchors)   # exactly the anchors pass the foreground check
    finally:
        st.stop_worker()


# ---- export honesty ----
def test_export_writes_only_green_frames(tmp_path: Path) -> None:
    import pandas as pd

    interframe, poses, clicks = _pan_session(9)
    anchors = _spread_anchors(9)
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        st.add_clicks(clicks)
        st.add_line_clicks(_near_tl_clicks(poses, anchors))
        st.wait_idle(timeout=10)
        greens = [f for f in range(9) if st._status_of(f) == "green"]
        st.export(tmp_path)
        hdf = pd.read_parquet(tmp_path / "homographies.parquet")
        assert sorted(hdf["frame"].tolist()) == sorted(greens)   # only green frames
        assert (hdf["confidence"] == 1.0).all()
    finally:
        st.stop_worker()


def test_export_gate_is_status_green(tmp_path: Path) -> None:
    # Unit-level: export writes exactly the frames whose CalibFrame.status == "green".
    import pandas as pd

    interframe, _poses, _clicks = _pan_session(9)
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        h = np.eye(3)
        with st._lock:
            st._fits = {
                0: CalibFrame(H=h, status="green", is_anchor=True, residual=None, n_points=6),
                1: CalibFrame(H=h, status="yellow", is_anchor=False, residual=None, n_points=0),
                2: CalibFrame(H=h, status="red", is_anchor=False, residual=None, n_points=0),
            }
        st.export(tmp_path)
        hdf = pd.read_parquet(tmp_path / "homographies.parquet")
        assert hdf["frame"].tolist() == [0]
    finally:
        st.stop_worker()


def test_status_of_reflects_calibframe_status() -> None:
    interframe, _poses, _clicks = _pan_session(9)
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        with st._lock:
            st._fits = {
                0: CalibFrame(np.eye(3), "green", True, None, 6),
                1: CalibFrame(np.eye(3), "yellow", False, None, 0),
            }
        assert st._status_of(0) == "green"
        assert st._status_of(1) == "yellow"
        assert st._status_of(2) == "red"   # no fit -> red
    finally:
        st.stop_worker()


# ---- responsiveness / concurrency wiring (unchanged behaviour) ----
def test_add_click_is_nonblocking_and_defers_to_worker() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe, 40, size=SIZE)
    try:
        st.add_clicks(clicks[:-1])    # bootstrap synchronously (bulk), worker then idle
        st.wait_idle(timeout=10)
        last = clicks[-1]
        assert st.frame_homography(last.frame) is not None   # instant cached fit
        st.add_click(last.frame, last.kp_idx, last.x, last.y)
        assert st.pending() > 0                              # deferred, no sync solve
        assert st.frame_homography(last.frame) is not None   # cached overlay still served
        st.wait_idle(timeout=10)
        assert st.pending() == 0
        after = st.frame_homography(last.frame)
        assert after is not None and after.H.shape == (3, 3)
    finally:
        st.stop_worker()


def test_async_refit_matches_synchronous_full_recompute() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe, 40, size=SIZE, line_band=60)
    try:
        for c in clicks:
            st.add_click(c.frame, c.kp_idx, c.x, c.y)
        st.wait_idle(timeout=10)
        fits_async = {f: cf.H.copy() for f, cf in st._fits.items()}
        ref = st._compute_dirty(sorted(st._transforms), lambda: False)
        assert ref is not None
        expected = {f: cf.H for f, cf in ref.items() if cf is not None}
        assert set(fits_async) == set(expected)
        for f in expected:
            np.testing.assert_allclose(fits_async[f], expected[f], atol=1e-6)
    finally:
        st.stop_worker()


def test_pending_drains_to_zero() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe, 40, size=SIZE)
    try:
        for c in clicks:
            st.add_click(c.frame, c.kp_idx, c.x, c.y)
        st.wait_idle(timeout=10)
        assert st.pending() == 0
    finally:
        st.stop_worker()


def test_concurrent_edits_during_refit_are_safe() -> None:
    # Hammer the mutators without wait_idle so the worker is mid-solve (it snapshots clicks
    # under the lock, then runs solve_session off the lock) while clicks are appended /
    # nudged / popped. Snapshot + locked mutations must not raise or corrupt _seq/_fits.
    interframe, poses, clicks = _pan_session(40)
    fp = field_points_3d()
    real: dict[tuple[int, int], tuple[float, float]] = {}
    for f in _spread_anchors(40):
        px = cv2.projectPoints(fp, poses[f][0], poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080:
                real[(f, j)] = (float(px[j, 0]) / 1920, float(px[j, 1]) / 1080)
    keys = sorted(real)
    st = LabelerState(interframe, 40, size=SIZE)
    try:
        st.add_clicks(clicks)
        for r in range(200):
            f, kp = keys[r % len(keys)]
            x, y = real[(f, kp)]
            st.add_click(f, kp, x, y)
            if r % 3 == 0:
                st.nudge_click(f, kp, x, y)
            if r % 7 == 0:
                st.remove_last()
        st.wait_idle(timeout=30)
        assert st.pending() == 0
        assert len(st._seq) == len(st.clicks) + len(st.line_clicks)
        for cf in st._fits.values():
            assert cf.H.shape == (3, 3)
    finally:
        st.stop_worker()


def test_does_not_autoflag_outliers_in_live_path() -> None:
    # solve_session drops per-frame point outliers internally, but the live path does not
    # record them in st._outliers (that is the recalibrate-only path).
    interframe, _poses, clicks = _pan_session(9)
    clicks = [c if not (c.frame == _spread_anchors(9)[1] and c.kp_idx == 6)
              else Click(c.frame, c.kp_idx, c.x + 0.25, c.y) for c in clicks]
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        st.add_clicks(clicks)
        assert st._calibrated
        assert st._outliers == {}
    finally:
        st.stop_worker()


# ---- line clicks: refit, persistence, band scoping ----
def test_add_line_click_refits_and_persists(tmp_path: Path) -> None:
    import json

    interframe, _poses, clicks = _pan_session(9)
    sidecar = tmp_path / "s.json"
    st = LabelerState(interframe, 9, size=SIZE, autosave_path=sidecar)
    try:
        st.add_clicks(clicks)
        st.add_line_click(4, "midline", 0.5, 0.5)
        assert len(st.line_clicks) == 1
        assert st.frame_homography(4) is not None
        data = json.loads(sidecar.read_text())
        assert data["line_clicks"] == [{"frame": 4, "line_id": "midline", "x": 0.5, "y": 0.5}]
        assert len(data["clicks"]) == len(clicks)
    finally:
        st.stop_worker()


def test_remove_last_pops_line_then_point() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        st.add_clicks(clicks)
        st.add_line_click(4, "near_touchline", 0.1, 0.9)
        n_pts = len(st.clicks)
        st.remove_last()
        assert len(st.line_clicks) == 0 and len(st.clicks) == n_pts
        st.remove_last()
        assert len(st.clicks) == n_pts - 1
    finally:
        st.stop_worker()


def test_export_writes_line_clicks_parquet(tmp_path: Path) -> None:
    import pandas as pd

    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        st.add_clicks(clicks)
        st.add_line_click(4, "midline", 0.5, 0.5)
        st.export(tmp_path)
        df = pd.read_parquet(tmp_path / "line_clicks.parquet")
        assert list(df.columns) == ["frame", "line_id", "x_px", "y_px"]
        assert df.iloc[0]["line_id"] == "midline"
        assert abs(df.iloc[0]["x_px"] - 0.5 * 1920) < 1e-6
    finally:
        st.stop_worker()


def test_line_obs_scoped_to_line_band() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe, 9, size=SIZE, line_band=1)
    try:
        st.add_clicks(clicks)
        st.line_clicks.append(LineClick(frame=4, line_id="midline", x=0.5, y=0.5))
        obs = st._line_obs(list(range(9)))
        carrying = sorted(f for f, lst in obs.items() if lst)
        assert carrying == [3, 4, 5]
    finally:
        st.stop_worker()


# ---- summary / bulk wiring ----
def test_status_summary_makes_one_status_pass() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe, 40, size=SIZE)
    try:
        st.add_clicks(clicks)
        st.wait_idle(timeout=10)
        calls = {"n": 0}
        real = st._status_of

        def counting(f: int) -> str:
            calls["n"] += 1
            return real(f)

        st._status_of = counting  # type: ignore[method-assign]
        coverage, buckets, bucket_size = st.status_summary()
        assert calls["n"] == st.n_frames
        assert 0.0 <= coverage <= 1.0
        assert isinstance(buckets, list) and isinstance(bucket_size, int)
    finally:
        st.stop_worker()


def test_bulk_add_clicks_keeps_lists_in_lockstep_with_worker_live() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe, 40, size=SIZE)
    try:
        st.add_clicks(clicks)
        before = len(st.clicks)
        st.add_clicks(clicks)
        st.wait_idle(timeout=10)
        assert len(st.clicks) == before + len(clicks)
        assert len(st._seq) == len(st.clicks) + len(st.line_clicks)
    finally:
        st.stop_worker()
