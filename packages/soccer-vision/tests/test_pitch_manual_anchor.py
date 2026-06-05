"""Tests for the manual-anchor point-propagation core."""

from __future__ import annotations

import numpy as np
from soccer_vision.pitch.manual_anchor import Click, FrameFit, build_segments


def test_click_and_framefit_fields() -> None:
    c = Click(frame=3, kp_idx=0, x=10.0, y=20.0)
    assert (c.frame, c.kp_idx, c.x, c.y) == (3, 0, 10.0, 20.0)
    f = FrameFit(H=np.eye(3), residual=0.01, n_points=5)
    assert f.n_points == 5 and f.residual == 0.01
    assert np.array_equal(f.H, np.eye(3))


def test_segments_single_connected_run() -> None:
    interframe = {0: np.eye(3), 1: np.eye(3), 2: np.eye(3)}  # links 0-1-2-3
    seg = build_segments(interframe, n_frames=4)
    assert seg == {0: 0, 1: 0, 2: 0, 3: 0}


def test_segments_split_on_missing_link() -> None:
    interframe = {0: np.eye(3), 2: np.eye(3)}  # link 0-1, gap at 1-2, link 2-3
    seg = build_segments(interframe, n_frames=4)
    assert seg == {0: 0, 1: 0, 2: 1, 3: 1}


def test_segments_all_isolated() -> None:
    seg = build_segments({}, n_frames=3)
    assert seg == {0: 0, 1: 1, 2: 2}
