"""Phase 3a tests: shared propagation + calibration registration engines."""

from __future__ import annotations

import cv2  # noqa: F401  # used by the calib-engine tests added in later Phase-3a tasks
import numpy as np
from soccer_vision.pitch.manual_anchor import (
    Click,
    build_segments,
    cumulative_transforms,
    propagate_clicks,
)


def test_propagate_clicks_carries_a_click_along_an_identity_chain() -> None:
    # Identity inter-frame transforms -> a click at frame 0 propagates UNCHANGED to
    # every frame within the window, for its own landmark.
    interframe = {i: np.eye(3) for i in range(5)}  # frames 0..5 linked
    seg = build_segments(interframe, 6)
    transforms = cumulative_transforms(interframe, seg)
    clicks = [Click(frame=0, kp_idx=3, x=0.4, y=0.6)]
    prop = propagate_clicks(clicks, transforms, seg, window=10)
    assert prop[2][3] == (0.4, 0.6)  # frame 2, landmark 3
    assert prop[5][3] == (0.4, 0.6)
    # window is INCLUSIVE (|Δframe| <= window): frame 1 (distance 1) is in, frame 5 is out
    prop_small = propagate_clicks(clicks, transforms, seg, window=1)
    assert prop_small[1][3] == (0.4, 0.6)  # boundary: distance == window, included
    assert 3 not in prop_small.get(5, {})


def test_propagate_clicks_respects_segments() -> None:
    # A gap (missing link at 2) splits segments; a click in segment 0 does not reach
    # segment 1.
    interframe = {0: np.eye(3), 1: np.eye(3), 3: np.eye(3)}  # link missing at 2
    seg = build_segments(interframe, 5)
    transforms = cumulative_transforms(interframe, seg)
    clicks = [Click(frame=0, kp_idx=1, x=0.5, y=0.5)]
    prop = propagate_clicks(clicks, transforms, seg, window=10)
    assert 1 in prop[1]   # same segment
    assert 4 not in prop  # frame 4 is a different segment -> never receives the click
