"""Tests for LabelerState: click handling, coverage, and parquet export."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from soccer_vision.labeler.state import LabelerState, clicks_from_keypoints_parquet
from soccer_vision.pipeline import homographies_from_parquet
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click

_SCALE = 1000.0
_IDXS = [0, 3, 6, 11, 16, 19]


def _state(n: int = 6) -> LabelerState:
    interframe = {i: np.eye(3) for i in range(n - 1)}
    return LabelerState(interframe=interframe, n_frames=n, size=(1920, 1080), window=10)


def test_add_click_updates_coverage() -> None:
    st = _state()
    assert st.coverage() == 0.0
    for f, idx in enumerate(_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        st.add_click(frame=f, kp_idx=idx, x=float(px), y=float(py))
    assert st.coverage() > 0.0
    assert st.frame_homography(3) is not None


def test_remove_last_click() -> None:
    st = _state()
    st.add_click(0, 0, 1.0, 2.0)
    assert len(st.clicks) == 1
    st.remove_last()
    assert len(st.clicks) == 0


def test_status_list_length_matches_frames() -> None:
    st = _state(5)
    assert len(st.status_list()) == 5
    assert set(st.status_list()) <= {"green", "yellow", "red"}


def test_export_writes_both_parquets(tmp_path: Path) -> None:
    st = _state()
    for f, idx in enumerate(_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        st.add_click(frame=f, kp_idx=idx, x=float(px), y=float(py))
    st.export(tmp_path)
    kp = pd.read_parquet(tmp_path / "keypoints.parquet")
    hom = pd.read_parquet(tmp_path / "homographies.parquet")
    assert list(kp.columns) == ["frame", "kp_idx", "x_px", "y_px", "conf"]
    assert len(kp) == len(_IDXS)
    assert "source" in hom.columns and (hom["source"] == "manual").all()


def test_add_clicks_bulk_matches_individual() -> None:
    individual = _state()
    bulk = _state()
    clicks = []
    for f, idx in enumerate(_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        individual.add_click(frame=f, kp_idx=idx, x=float(px), y=float(py))
        clicks.append(Click(frame=f, kp_idx=idx, x=float(px), y=float(py)))
    bulk.add_clicks(clicks)
    assert bulk.coverage() == individual.coverage()
    assert bulk.status_list() == individual.status_list()


def test_resume_round_trip_restores_coverage(tmp_path: Path) -> None:
    st = _state()
    for f, idx in enumerate(_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        st.add_click(frame=f, kp_idx=idx, x=float(px), y=float(py))
    st.export(tmp_path)
    resumed = clicks_from_keypoints_parquet(tmp_path / "keypoints.parquet", (1920, 1080))
    st2 = _state()
    st2.add_clicks(resumed)
    assert len(st2.clicks) == len(st.clicks)
    assert st2.coverage() == st.coverage()
    assert st2.status_list() == st.status_list()


def test_export_homography_maps_pixels_to_pitch(tmp_path: Path) -> None:
    st = LabelerState(
        interframe={i: np.eye(3) for i in range(5)},
        n_frames=6, size=(1920, 1080), window=10,
    )
    # identity chain => normalized image coords == pitch coords; click each
    # landmark at its NORMALIZED pitch position.
    for f, idx in enumerate(_IDXS):
        nx, ny = PITCH_LANDMARKS[idx]
        st.add_click(frame=f, kp_idx=idx, x=float(nx), y=float(ny))
    st.export(tmp_path)
    entries = homographies_from_parquet(tmp_path / "homographies.parquet")
    h = entries[3].H
    # landmark 3 = pitch (1,1); on a 1920x1080 frame that is pixel (1920, 1080)
    out = h @ np.array([1920.0, 1080.0, 1.0])
    out = out[:2] / out[2]
    assert np.allclose(out, PITCH_LANDMARKS[3], atol=1e-6)


def _full_recompute_reference(st: LabelerState) -> LabelerState:
    """Fresh state replaying st's clicks via bulk load (full recompute)."""
    ref = LabelerState(
        interframe={i: np.eye(3) for i in range(st.n_frames - 1)},
        n_frames=st.n_frames, size=st.size, window=st.window,
    )
    ref.add_clicks(list(st.clicks))
    return ref


def _assert_fits_equal(a: LabelerState, b: LabelerState) -> None:
    fa = {f: a.frame_homography(f) for f in range(a.n_frames)}
    fb = {f: b.frame_homography(f) for f in range(b.n_frames)}
    keys_a = {f for f, v in fa.items() if v is not None}
    keys_b = {f for f, v in fb.items() if v is not None}
    assert keys_a == keys_b
    for f in keys_a:
        va, vb = fa[f], fb[f]
        assert va is not None and vb is not None
        assert np.allclose(va.H, vb.H)
        assert np.isclose(va.residual, vb.residual)


def test_incremental_equals_full_after_mutation_sequence() -> None:
    rng = random.Random(0)
    st = _state(n=40)
    for _step in range(30):
        action = rng.random()
        if action < 0.6 or not st.clicks:
            idx = rng.choice(_IDXS)
            px, py = PITCH_LANDMARKS[idx] * _SCALE
            st.add_click(frame=rng.randrange(40), kp_idx=idx,
                         x=float(px) + rng.random(), y=float(py) + rng.random())
        elif action < 0.8:
            st.remove_last()
        else:
            c = st.clicks[rng.randrange(len(st.clicks))]
            st.nudge_click(c.frame, c.kp_idx, c.x + 0.5, c.y + 0.5)
    _assert_fits_equal(st, _full_recompute_reference(st))


def test_remove_last_clears_lost_fits() -> None:
    st = _state()
    for f, idx in enumerate(_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        st.add_click(frame=f, kp_idx=idx, x=float(px), y=float(py))
    assert st.frame_homography(3) is not None
    for _ in range(3):
        st.remove_last()
    # only 3 landmarks remain -> no frame can fit
    assert all(st.frame_homography(f) is None for f in range(st.n_frames))


def test_bulk_add_chunked_matches_unchunked() -> None:
    st_small_chunk = _state(n=20)
    st_default = _state(n=20)
    clicks = []
    for f, idx in enumerate(_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        clicks.append(Click(frame=f * 3, kp_idx=idx, x=float(px), y=float(py)))
    st_small_chunk.add_clicks(clicks, chunk=4)
    st_default.add_clicks(clicks)
    _assert_fits_equal(st_small_chunk, st_default)


def test_nudge_click_moves_most_recent_match() -> None:
    st = _state()
    st.add_click(0, 0, 0.1, 0.1)
    st.add_click(0, 0, 0.2, 0.2)   # duplicate (frame, kp_idx)
    assert st.nudge_click(0, 0, 0.9, 0.9) is True
    assert np.isclose(st.clicks[1].x, 0.9)   # most recent moved
    assert np.isclose(st.clicks[0].x, 0.1)   # older untouched
    assert st.nudge_click(5, 7, 0.5, 0.5) is False
