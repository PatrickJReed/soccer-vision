"""Homography propagation: bridge no-landmark frames by registering them to anchors.

Trace is a no-parallax virtual-PTZ crop of a fixed camera, so consecutive frames
are related by a global homography recoverable from static background features.
We chain those inter-frame homographies from the nearest landmark anchors on both
sides of a gap and blend them, lifting pitch-homography coverage without labeling.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

# Image-space reference grid (normalized 0..1) used to measure forward/backward
# disagreement; scaled to the frame size at use.
_GRID = np.array([(x, y) for x in (0.2, 0.5, 0.8) for y in (0.2, 0.5, 0.8)], dtype=np.float64)


@dataclass(frozen=True)
class HomographyEntry:
    """A frame's homography plus where it came from."""

    H: NDArray[np.floating]
    source: str           # "anchor" | "propagated"
    confidence: float     # 1.0 for anchors; runtime estimate for propagated


def register(
    img_src: NDArray[np.uint8],
    img_dst: NDArray[np.uint8],
    mask_src: NDArray[np.uint8],
    mask_dst: NDArray[np.uint8],
    *,
    n_features: int = 3000,
    min_inliers: int = 12,
) -> NDArray[np.floating] | None:
    """Homography mapping img_src pixels -> img_dst pixels from masked ORB features.

    Returns None when too few features/matches are found (e.g. blurred or blank
    frames). Masks are uint8 (255 = use, 0 = ignore — e.g. over moving players).
    """
    orb = cv2.ORB_create(n_features)  # type: ignore[attr-defined]
    gs = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gd = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    ks, ds = orb.detectAndCompute(gs, mask_src)
    kd, dd = orb.detectAndCompute(gd, mask_dst)
    if ds is None or dd is None or len(ds) < min_inliers or len(dd) < min_inliers:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(ds, dd)
    if len(matches) < min_inliers:
        return None
    src: NDArray[np.float32] = np.array(
        [ks[m.queryIdx].pt for m in matches], dtype=np.float32
    )
    dst: NDArray[np.float32] = np.array(
        [kd[m.trainIdx].pt for m in matches], dtype=np.float32
    )
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return None if H is None else H.astype(np.float64)


def blend_homographies(
    h_a: NDArray[np.floating], h_b: NDArray[np.floating], w_a: float
) -> NDArray[np.floating]:
    """Weighted element-wise blend of two homographies, normalized so H[2,2]=1.

    Element-wise blending is an approximation (homographies are not a vector
    space); it is accurate for the small inter-frame motions we blend here.
    """
    blended = w_a * h_a + (1.0 - w_a) * h_b
    if abs(blended[2, 2]) > 1e-9:
        blended = blended / blended[2, 2]
    return blended


def _map_points(H: NDArray[np.floating], pts: NDArray[np.floating]) -> NDArray[np.floating]:
    homog = np.column_stack([pts, np.ones(len(pts))])
    out = (H @ homog.T).T
    return out[:, :2] / out[:, 2:3]


def disagreement_confidence(
    h_fwd: NDArray[np.floating],
    h_bwd: NDArray[np.floating],
    *,
    tau: float = 0.10,
    frame_size: tuple[int, int] = (1920, 1080),
) -> float:
    """Confidence in [0,1] from how much the two chains disagree (pitch-units).

    Both Hs map the same reference grid into pitch space; their mean separation is
    a runtime estimate of propagation error. confidence = clamp(1 - disagree/tau).

    Pass the actual frame size; the (1920, 1080) default suits 1080p clips like the
    bake-off clip. The disagreement magnitude (and thus tau's calibration) depends
    on it.
    """
    grid_px = _GRID * np.array(frame_size, dtype=np.float64)
    disagree = float(
        np.linalg.norm(
            _map_points(h_fwd, grid_px) - _map_points(h_bwd, grid_px), axis=1
        ).mean()
    )
    return float(np.clip(1.0 - disagree / tau, 0.0, 1.0))
