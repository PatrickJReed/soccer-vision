"""Tests for smooth_possession (30-frame modal smoothing, preserving contested)."""

from __future__ import annotations

import pandas as pd
from soccer_vision.phase.possession import smooth_possession


def test_removes_single_frame_flicker() -> None:
    s = pd.Series(["own", "opp", "own", "own", "own"], index=[0, 1, 2, 3, 4])
    out = smooth_possession(s, window_frames=3)
    assert out.tolist() == ["own", "own", "own", "own", "own"]


def test_preserves_contested() -> None:
    s = pd.Series(["own", "contested", "own", "own", "own"], index=[0, 1, 2, 3, 4])
    out = smooth_possession(s, window_frames=3)
    assert out.iloc[1] == "contested"


def test_small_window_is_noop() -> None:
    s = pd.Series(["own", "opp"], index=[0, 1])
    out = smooth_possession(s, window_frames=1)
    assert out.tolist() == ["own", "opp"]


def test_preserves_index_and_name() -> None:
    s = pd.Series(["own", "opp", "own"], index=[10, 11, 12], name="possession_state")
    out = smooth_possession(s, window_frames=3)
    assert list(out.index) == [10, 11, 12]
    assert out.name == "possession_state"


def test_contested_neighbors_do_not_flip_clear_frame() -> None:
    # 'own' frame flanked by a contested scramble keeps 'own' (contested excluded from the vote).
    s = pd.Series(["own", "contested", "contested", "contested", "contested"], index=[0, 1, 2, 3, 4])
    out = smooth_possession(s, window_frames=5)
    assert out.iloc[0] == "own"


def test_boundary_frames_use_clipped_window() -> None:
    s = pd.Series(["opp", "own", "own", "own", "own"], index=[0, 1, 2, 3, 4])
    out = smooth_possession(s, window_frames=5)
    # frame 0's window is the clipped [0:3] = opp, own, own -> mode 'own'
    assert out.iloc[0] == "own"
