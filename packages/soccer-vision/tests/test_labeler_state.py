"""Tests for LabelerState: click handling, coverage, and parquet export."""

from __future__ import annotations

import numpy as np
import pandas as pd
from soccer_vision.labeler.state import LabelerState
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

_SCALE = 1000.0
_IDXS = [0, 3, 6, 11, 16, 19]


def _state(n: int = 6) -> LabelerState:
    interframe = {i: np.eye(3) for i in range(n - 1)}
    return LabelerState(interframe=interframe, n_frames=n, window=10)


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


def test_export_writes_both_parquets(tmp_path) -> None:
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
