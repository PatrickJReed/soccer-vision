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


def cumulative_transforms(
    interframe: Mapping[int, NDArray[np.floating]], segment_of: Mapping[int, int]
) -> dict[int, NDArray[np.float64]]:
    """M[f] maps frame f pixels -> its segment's reference (first) frame pixels.

    M[start] = I; M[f] = M[f-1] @ inv(interframe[f-1]) within a segment, since
    interframe[f-1] maps f-1 -> f so its inverse maps f -> f-1.
    """
    transforms: dict[int, NDArray[np.float64]] = {}
    for f in sorted(segment_of):
        prev_same = (f - 1) in segment_of and segment_of[f] == segment_of[f - 1]
        if not prev_same:
            transforms[f] = np.eye(3)
        else:
            g = np.asarray(interframe[f - 1], dtype=np.float64)
            transforms[f] = transforms[f - 1] @ np.linalg.inv(g)
    return transforms


def map_point(
    m_src: NDArray[np.floating], m_dst: NDArray[np.floating], x: float, y: float
) -> tuple[float, float]:
    """Map pixel (x, y) from the source frame into the destination frame.

    src -> reference via m_src, reference -> dst via inv(m_dst). Both must be in
    the same segment (caller ensures this).
    """
    ref = np.asarray(m_src, dtype=np.float64) @ np.array([x, y, 1.0])
    dst = np.linalg.inv(np.asarray(m_dst, dtype=np.float64)) @ ref
    return float(dst[0] / dst[2]), float(dst[1] / dst[2])
