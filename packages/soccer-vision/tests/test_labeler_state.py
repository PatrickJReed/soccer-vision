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


def test_export_skips_non_green_and_blocks_until_idle(tmp_path: Path) -> None:
    # The honest export gate is whole-field GREEN (two-ended + plausible fold), NOT the
    # old in-sample residual gate. A single-ended (yellow) session has a tiny residual,
    # so the old gate would export it as a "sky" frame; the green gate must export nothing.
    import pandas as pd

    h = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])  # pitch -> img_norm
    st = LabelerState(_identity_chain(10), 10, size=SIZE)
    try:
        for f, kp, x, y in _clicks_for(0, OWN_END_IDX, h):  # own end only -> yellow
            st.add_click(f, kp, x, y)
        st.wait_idle(timeout=10)
        assert "green" not in st.status_list()
        st.export(tmp_path)
        hdf = pd.read_parquet(tmp_path / "homographies.parquet")
        assert len(hdf) == 0  # nothing green -> nothing exported (no sky homographies)

        # Add opp-end clicks: the segment becomes two-ended -> green frames -> exported.
        for f, kp, x, y in _clicks_for(5, OPP_END_IDX, h):
            st.add_click(f, kp, x, y)
        st.wait_idle(timeout=10)
        green = [f for f in range(10) if st._status_of(f) == "green"]
        assert green  # both ends now constrain the global H -> some frames green
        st.export(tmp_path)
        hdf = pd.read_parquet(tmp_path / "homographies.parquet")
        assert len(hdf) == len(green)  # exactly the green frames are exported
        assert (hdf["confidence"] > 0).all()
    finally:
        st.stop_worker()


def test_two_ended_session_is_green_and_reprojects_clicked_landmark() -> None:
    # A two-ended clicked session over an identity chain: own-end clicks at frame 0 +
    # opp-end clicks at frame 5 constrain the ONE global homography across the whole
    # field -> green frames, and the bundle H reprojects a clicked landmark back onto its
    # canonical pitch coords within a small pitch error (in-sample, so ~exact).
    h = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])  # pitch -> img_norm
    st = LabelerState(_identity_chain(10), 10, size=SIZE)
    try:
        own = _clicks_for(0, OWN_END_IDX, h)
        opp = _clicks_for(5, OPP_END_IDX, h)
        for f, kp, x, y in own + opp:
            st.add_click(f, kp, x, y)
        st.wait_idle(timeout=10)
        assert st._status_of(0) == "green"  # two-ended segment + plausible fold -> green
        cf = st.frame_homography(0)
        assert cf is not None and cf.two_ended
        # the bundle H (normalized image -> pitch) maps a clicked own-end landmark back
        # onto its canonical pitch coords
        _f, kp, x, y = own[0]
        v = cf.H @ np.array([x, y, 1.0])
        pitch = v[:2] / v[2]
        assert float(np.linalg.norm(pitch - PITCH_LANDMARKS[kp])) < 1e-3
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
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080))
    st.add_clicks(clicks)
    assert st._calibrated
    assert all(st.frame_homography(f) is not None for f in range(9))
    cf = st.frame_homography(2)  # UNCLICKED frame covered by the global H
    assert cf is not None and cf.H.shape == (3, 3)


def test_labeler_uncalibrated_below_min_points() -> None:
    # Calibration now needs only >= 4 pooled clicks (a fittable homography), not the
    # old >= 3 anchor frames; below that there is no homography.
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080))
    st.add_clicks(clicks[:3])  # fewer than 4 points -> nothing fittable
    assert not st._calibrated
    assert st.frame_homography(2) is None  # no global H yet


def test_labeler_does_not_autoflag_outliers_in_live_path() -> None:
    # Outlier flagging needed the focal K and is dropped from the live path; a
    # mislabeled click is simply tolerated and never recorded in _outliers.
    interframe, _poses, clicks = _pan_session(9)
    clicks = [c if not (c.frame == 4 and c.kp_idx == 6)
              else Click(c.frame, c.kp_idx, c.x + 0.25, c.y) for c in clicks]
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080))
    st.add_clicks(clicks)
    assert st._calibrated        # still solves a global homography despite the mislabel
    assert st._outliers == {}    # no live outlier flagging (RANSAC path, not K-based)


def test_labeler_add_line_click_refits_and_persists(tmp_path: Path) -> None:
    interframe, _poses, clicks = _pan_session(9)
    sidecar = tmp_path / "s.json"
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080),
                      autosave_path=sidecar)
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
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080))
    st.add_clicks(clicks)
    st.add_line_click(4, "near_touchline", 0.1, 0.9)
    n_pts = len(st.clicks)
    st.remove_last()                          # pops the line click (added last)
    assert len(st.line_clicks) == 0 and len(st.clicks) == n_pts
    st.remove_last()                          # now pops a point
    assert len(st.clicks) == n_pts - 1


def test_labeler_export_writes_line_clicks_parquet(tmp_path: Path) -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080))
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
                      line_band=1)   # band of +/-1 frame
    st.add_clicks(clicks)
    st.line_clicks.append(LineClick(frame=4, line_id="midline", x=0.5, y=0.5))
    # _line_obs over all frames: only frames within +/-1 of frame 4 carry the line obs
    obs = st._line_obs(list(range(9)))
    carrying = sorted(f for f, lst in obs.items() if lst)
    assert carrying == [3, 4, 5]


def test_add_click_fits_current_frame_synchronously() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080))
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
                      line_band=60)
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
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080))
    try:
        for c in clicks:
            st.add_click(c.frame, c.kp_idx, c.x, c.y)
        st.wait_idle(timeout=10)
        assert st.pending() == 0
    finally:
        st.stop_worker()


def test_concurrent_edits_during_refit_are_safe() -> None:
    # Hammer the mutators from the main thread WITHOUT wait_idle between them, so the
    # background worker is mid-compute (it copies the clicks under the lock, then runs the
    # field-anchored bundle solve OFF the lock) while clicks are appended / replaced /
    # popped. With the snapshot copied and the mutations locked this must not raise
    # (RuntimeError: list changed size during iteration) or corrupt the _seq/_fits
    # invariants. The hammered clicks are REAL projected landmarks on a handful of frames
    # so each bundle solve converges fast: garbage clicks force the least_squares to
    # max_nfev (~10 s per solve), and the lock/snapshot safety this exercises is
    # independent of the clicks' accuracy. (The off-lock bundle solve is ~tens of ms vs
    # the old microsecond DLT, so the concurrent-mutation race window is now WIDER.)
    interframe, poses, clicks = _pan_session(40)
    fp = field_points_3d()
    real: dict[tuple[int, int], tuple[float, float]] = {}  # (frame, kp) -> norm click
    for f in range(6):
        px = cv2.projectPoints(fp, poses[f][0], poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080:
                real[(f, j)] = (float(px[j, 0]) / 1920, float(px[j, 1]) / 1080)
    keys = sorted(real)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080))
    try:
        st.add_clicks(clicks)  # bootstrap -> calibrated, worker live
        for r in range(200):
            f, kp = keys[r % len(keys)]
            x, y = real[(f, kp)]
            st.add_click(f, kp, x, y)
            if r % 3 == 0:
                st.nudge_click(f, kp, x, y)
            if r % 7 == 0:
                st.remove_last()
        st.wait_idle(timeout=30)
        # invariants hold once everything settles
        assert st.pending() == 0
        assert len(st._seq) == len(st.clicks) + len(st.line_clicks)
        for cf in st._fits.values():
            assert cf.H.shape == (3, 3)  # no torn / partial fit
    finally:
        st.stop_worker()


def test_clean_session_fits_two_ended_with_tight_residual() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080))
    try:
        st.add_clicks(clicks)
        st.wait_idle(timeout=10)
        # clean pan: clicked frames are covered, span both ends, with a tight in-sample
        # bundle-fit reprojection residual (pitch units).
        for c in sorted({cl.frame for cl in clicks})[:3]:
            cf = st.frame_homography(c)
            assert cf is not None and cf.two_ended
            assert np.isfinite(cf.residual) and cf.residual < 1.0
    finally:
        st.stop_worker()


def test_global_model_covers_whole_segment() -> None:
    # The field-anchored bundle fits one homography per segment, so EVERY frame in the
    # segment (clicked or not) gets an overlay — there is no per-frame windowing gap.
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080))
    try:
        st.add_clicks(clicks)
        st.wait_idle(timeout=10)
        covered = {f for f in range(40) if st.frame_homography(f) is not None}
        assert covered == set(range(40))  # whole (single) segment covered
    finally:
        st.stop_worker()
