"""Tests for LabelerState: click handling, coverage, and parquet export."""

from __future__ import annotations

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
