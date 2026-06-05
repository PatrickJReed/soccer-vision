"""Manual-anchor point propagation: turn sparse landmark clicks into per-frame
homographies using the fixed camera's frame-to-frame registration chain.

Each click is one (frame, kp_idx, pixel) observation. Because the Trace camera
has no parallax, a click in one frame can be mapped into any other frame in the
same registration-connected segment via cumulative inter-frame transforms. A
frame that accumulates >=4 distinct landmarks (clicked there or propagated in)
gets a homography; the fit's reprojection residual is its confidence.

All homographies map image pixels -> pitch [0,1]^2. Pure: no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Click:
    """One landmark observation: pixel (x, y) of keypoint kp_idx in a frame."""

    frame: int
    kp_idx: int
    x: float
    y: float


@dataclass(frozen=True, eq=False)
class FrameFit:
    """A fitted per-frame homography plus its quality."""

    H: NDArray[np.floating]
    residual: float        # mean reprojection error in pitch units
    n_points: int          # distinct landmarks used


def build_segments(
    interframe: Mapping[int, NDArray[np.floating]], n_frames: int
) -> dict[int, int]:
    """Assign each frame 0..n_frames-1 a registration-segment id.

    interframe[i] present means frames i and i+1 are linked. A run of consecutive
    linked frames shares a segment id; a missing link starts a new segment.
    """
    seg: dict[int, int] = {}
    cur = 0
    for f in range(n_frames):
        if f == 0:
            seg[f] = 0
        elif (f - 1) in interframe:
            seg[f] = seg[f - 1]
        else:
            cur += 1
            seg[f] = cur
    return seg
