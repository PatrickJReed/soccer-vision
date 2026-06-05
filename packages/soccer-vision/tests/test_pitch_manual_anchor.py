"""Tests for the manual-anchor point-propagation core."""

from __future__ import annotations

import numpy as np
from soccer_vision.pitch.manual_anchor import (
    Click,
    FrameFit,
    build_segments,
    cumulative_transforms,
    map_point,
)


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


def test_cumulative_identity_chain() -> None:
    interframe = {0: np.eye(3), 1: np.eye(3)}
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    for f in range(3):
        assert np.allclose(M[f], np.eye(3))


def test_cumulative_translation_chain() -> None:
    # each frame shifts +10px in x relative to the previous (i -> i+1).
    g = np.eye(3)
    g[0, 2] = 10.0
    interframe = {0: g, 1: g}
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    # M[f] maps frame f -> reference(0). frame 2 is +20px from frame 0, so
    # mapping a point back to ref subtracts 20.
    assert np.allclose(M[2] @ np.array([20.0, 0.0, 1.0]), [0.0, 0.0, 1.0])


def test_cumulative_resets_per_segment() -> None:
    g = np.eye(3)
    g[0, 2] = 10.0
    interframe = {0: g}  # link 0-1 only; frame 2 is a new segment
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    assert np.allclose(M[2], np.eye(3))  # segment start -> identity


def test_map_point_through_translation() -> None:
    g = np.eye(3)
    g[0, 2] = 10.0
    interframe = {0: g, 1: g}
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    # a point at x=5 in frame 0 appears at x=25 in frame 2 (camera moved +20).
    x, y = map_point(M[0], M[2], 5.0, 0.0)
    assert np.isclose(x, 25.0) and np.isclose(y, 0.0)
