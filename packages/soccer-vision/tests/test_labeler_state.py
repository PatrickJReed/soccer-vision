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
from soccer_vision.pitch.global_calib import OPP_END_IDX, OWN_END_IDX
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click, LineClick

_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)

SIZE = (1920, 1080)


def _identity_chain(n: int) -> dict[int, NDArray[np.float64]]:
    """All frames linked by identity inter-frame transforms (one segment, no motion)."""
    return {i: np.eye(3) for i in range(n - 1)}


def _clicks_for(
    frame: int, ids: list[int], h_pitch_to_img: NDArray[np.float64]
) -> list[tuple[int, int, float, float]]:
    """Project each landmark id through a pitch->image_norm homography into clicks."""
    out: list[tuple[int, int, float, float]] = []
    for kp in ids:
        p = h_pitch_to_img @ np.array([*PITCH_LANDMARKS[kp], 1.0])
        p = p[:2] / p[2]
        out.append((frame, kp, float(p[0]), float(p[1])))
    return out


def test_single_end_session_is_never_green_but_two_ended_is() -> None:
    # A narrow Trace-like view: only the OWN half lands in [0,1]^2 (pitch_y in [0,0.5]
    # -> img_y in [0,1]); the opp half projects off-screen, so fold stays in the
    # physical range. A single-ended session is honestly YELLOW (never green); the
    # session only goes green once clicks at BOTH ends constrain the one global H.
    h = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])  # pitch -> img_norm
    st = LabelerState(_identity_chain(10), 10, size=SIZE)
    try:
        for f, kp, x, y in _clicks_for(0, OWN_END_IDX, h):  # own end only, frame 0
            st.add_click(f, kp, x, y)
        st.wait_idle(timeout=10)
        assert "green" not in st.status_list()        # single-ended -> yellow at best
        for f, kp, x, y in _clicks_for(5, OPP_END_IDX, h):  # opp end, frame 5
            st.add_click(f, kp, x, y)
        st.wait_idle(timeout=10)
        assert "green" in st.status_list()            # both ends constrain the global H
    finally:
        st.stop_worker()


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


def test_labeler_bootstraps_and_covers_all_segment_frames() -> None:
    # The global solve fits ONE homography for the segment, so EVERY frame in it
    # (clicked or not) gets an overlay — no per-frame propagation gap. (Green
    # coverage needs a planar-crop session; the synthetic rotation pan is fold-
    # incompatible, so green is exercised by the identity-chain test above.)
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    st.add_clicks(clicks)
    assert st._calibrated
    assert all(st.frame_homography(f) is not None for f in range(9))
    cf = st.frame_homography(2)  # UNCLICKED frame covered by the global H
    assert cf is not None and cf.H.shape == (3, 3)


def test_labeler_uncalibrated_below_min_points() -> None:
    # Calibration now needs only >= 4 pooled clicks (a fittable homography), not the
    # old >= 3 anchor frames; below that there is no homography.
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    st.add_clicks(clicks[:3])  # fewer than 4 points -> nothing fittable
    assert not st._calibrated
    assert st.frame_homography(2) is None  # no global H yet


def test_labeler_does_not_autoflag_outliers_in_live_path() -> None:
    # Outlier flagging needed the focal K and is dropped from the live path; a
    # mislabeled click is simply tolerated and never recorded in _outliers.
    interframe, _poses, clicks = _pan_session(9)
    clicks = [c if not (c.frame == 4 and c.kp_idx == 6)
              else Click(c.frame, c.kp_idx, c.x + 0.25, c.y) for c in clicks]
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    st.add_clicks(clicks)
    assert st._calibrated        # still solves a global homography despite the mislabel
    assert st._outliers == {}    # no live outlier flagging (RANSAC path, not K-based)


def test_labeler_add_line_click_refits_and_persists(tmp_path: Path) -> None:
    interframe, _poses, clicks = _pan_session(9)
    sidecar = tmp_path / "s.json"
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080),
                      window=360, autosave_path=sidecar)
    st.add_clicks(clicks)                     # bootstrap on points
    _cf_before = st.frame_homography(4)
    st.add_line_click(4, "midline", 0.5, 0.5)
    assert len(st.line_clicks) == 1
    # still covered by the global H (line obs is stored but unused by the point solve)
    assert st.frame_homography(4) is not None
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


def test_line_obs_scoped_to_line_band() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080),
                      window=360, line_band=1)   # band of +/-1 frame
    st.add_clicks(clicks)
    st.line_clicks.append(LineClick(frame=4, line_id="midline", x=0.5, y=0.5))
    # _line_obs over all frames: only frames within +/-1 of frame 4 carry the line obs
    obs = st._line_obs(list(range(9)))
    carrying = sorted(f for f, lst in obs.items() if lst)
    assert carrying == [3, 4, 5]


def test_add_click_fits_current_frame_synchronously() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080), window=360)
    try:
        # bootstrap on all but the last click, synchronously (bulk)
        st.add_clicks(clicks[:-1])
        last = clicks[-1]
        st.add_click(last.frame, last.kp_idx, last.x, last.y)
        # the clicked frame is fit on return, BEFORE the background worker drains
        assert st.frame_homography(last.frame) is not None
    finally:
        st.stop_worker()


def test_async_refit_matches_synchronous_full_recompute() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080),
                      window=360, line_band=60)
    try:
        for c in clicks:
            st.add_click(c.frame, c.kp_idx, c.x, c.y)
        st.wait_idle(timeout=10)
        fits_async = {f: cf.H.copy() for f, cf in st._fits.items()}
        # synchronous full recompute on the SAME calibrated state (same clicks)
        ref = st._compute_dirty(sorted(st._transforms), lambda: False)
        assert ref is not None
        expected = {f: cf.H for f, cf in ref.items() if cf is not None}
        assert set(fits_async) == set(expected)
        for f in expected:  # np is imported at module top (used by _pan_session/_K)
            np.testing.assert_allclose(fits_async[f], expected[f], atol=1e-6)
    finally:
        st.stop_worker()


def test_pending_drains_to_zero() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080), window=360)
    try:
        for c in clicks:
            st.add_click(c.frame, c.kp_idx, c.x, c.y)
        st.wait_idle(timeout=10)
        assert st.pending() == 0
    finally:
        st.stop_worker()


def test_concurrent_edits_during_refit_are_safe() -> None:
    # Hammer the mutators from the main thread WITHOUT wait_idle between them, so the
    # background worker is mid-compute (iterating the clicks snapshot) while clicks are
    # appended / replaced / popped. With the snapshot copied and the mutations locked
    # this must not raise (RuntimeError: list changed size during iteration) or corrupt
    # the _seq/_fits invariants. Without the fix it trips intermittently.
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080), window=360)
    try:
        st.add_clicks(clicks)  # bootstrap -> calibrated, worker live
        for r in range(200):
            f = r % 40
            st.add_click(f, r % 6, 0.30 + 0.001 * (r % 100), 0.40)
            if r % 3 == 0:
                st.nudge_click(f, r % 6, 0.50, 0.50)
            if r % 7 == 0:
                st.remove_last()
        st.wait_idle(timeout=20)
        # invariants hold once everything settles
        assert st.pending() == 0
        assert len(st._seq) == len(st.clicks) + len(st.line_clicks)
        for cf in st._fits.values():
            assert cf.H.shape == (3, 3)  # no torn / partial fit
    finally:
        st.stop_worker()


def test_state_legacy_params_wired_and_clean_session_fits() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080), window=360)
    try:
        # legacy gated-engine knobs are accepted-but-unused; still exposed as attributes
        assert (st.seed_size, st.gate_px, st.gap_dist) == (6, 60.0, 180)
        st.add_clicks(clicks)
        st.wait_idle(timeout=10)
        # clean pan: clicked frames are covered, span both ends, with a tight in-sample
        # global-fit residual (normalized px, not the old pixel scale).
        for c in sorted({cl.frame for cl in clicks})[:3]:
            cf = st.frame_homography(c)
            assert cf is not None and cf.two_ended
            assert np.isfinite(cf.residual) and cf.residual < 1.0
    finally:
        st.stop_worker()


def test_global_model_covers_whole_segment_regardless_of_gap_dist() -> None:
    # gap_dist is a legacy accepted-but-unused knob: the global homography covers EVERY
    # frame in the segment, so a tight gap_dist no longer reds far frames (no windowing).
    interframe, _poses, clicks = _pan_session(40)
    wide = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080),
                        window=360, gap_dist=180)
    tight = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080),
                         window=360, gap_dist=3)
    try:
        wide.add_clicks(clicks)
        wide.wait_idle(timeout=10)
        tight.add_clicks(clicks)
        tight.wait_idle(timeout=10)
        covered_wide = {f for f in range(40) if wide.frame_homography(f) is not None}
        covered_tight = {f for f in range(40) if tight.frame_homography(f) is not None}
        # the whole (single) segment is covered either way — gap_dist is inert now
        assert covered_tight == covered_wide == set(range(40))
    finally:
        wide.stop_worker()
        tight.stop_worker()
