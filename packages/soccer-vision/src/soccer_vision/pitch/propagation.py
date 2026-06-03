"""Homography propagation: bridge no-landmark frames by registering them to anchors.

Trace is a no-parallax virtual-PTZ crop of a fixed camera, so consecutive frames
are related by a global homography recoverable from static background features.
We chain those inter-frame homographies from the nearest landmark anchors on both
sides of a gap and blend them, lifting pitch-homography coverage without labeling.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
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


_PLAYER_CLASSES = ("player", "goalkeeper", "referee")


def _frame_mask(boxes: pd.DataFrame, frame: int, shape: tuple[int, int]) -> NDArray[np.uint8]:
    """255 on static background, 0 over player/ref boxes (dilated)."""
    mask = np.full(shape, 255, np.uint8)
    sel = boxes[(boxes["frame"] == frame) & boxes["class"].isin(_PLAYER_CLASSES)]
    for _, r in sel.iterrows():
        cv2.rectangle(
            mask,
            (int(r["bbox_x1"]) - 12, int(r["bbox_y1"]) - 12),
            (int(r["bbox_x2"]) + 12, int(r["bbox_y2"]) + 12),
            0, -1,
        )
    return mask


def _chain(
    anchor: int,
    targets: list[int],
    read_frame: Callable[[int], NDArray[np.uint8] | None],
    boxes: pd.DataFrame,
    H_anchor: NDArray[np.floating],
    n_features: int,
    min_inliers: int,
) -> dict[int, NDArray[np.floating]]:
    """Chain consecutive registrations from `anchor` over `targets` (ordered, adjacent).

    Returns {frame: pitch_H} for each reached frame; stops at the first failure.
    """
    out: dict[int, NDArray[np.floating]] = {}
    prev_img = read_frame(anchor)
    if prev_img is None:
        return out
    shape = prev_img.shape[:2]
    W = np.eye(3)                                  # maps anchor pixels -> current pixels
    prev_frame = anchor
    for f in targets:
        cur = read_frame(f)
        if cur is None:
            break
        G = register(prev_img, cur, _frame_mask(boxes, prev_frame, shape),
                     _frame_mask(boxes, f, shape), n_features=n_features, min_inliers=min_inliers)
        if G is None:
            break
        W = G @ W
        out[f] = H_anchor @ np.linalg.inv(W)       # pixel_f -> pitch
        prev_img, prev_frame = cur, f
    return out


def propagate_homographies(
    anchors: Mapping[int, NDArray[np.floating]],
    read_frame: Callable[[int], NDArray[np.uint8] | None],
    player_boxes: pd.DataFrame,
    *,
    max_gap: int = 25,
    disagreement_tau: float = 0.10,
    n_features: int = 3000,
    min_inliers: int = 12,
) -> dict[int, HomographyEntry]:
    """Bridge no-landmark gaps between anchors via bidirectional chaining.

    Each anchor keeps source='anchor', confidence=1.0. For each gap <= max_gap,
    chain forward from the left anchor and backward from the right anchor, blend by
    distance, and set confidence from forward/backward disagreement. Frames reached
    by neither chain are absent. Edge gaps (before first / after last anchor) are
    not bridged in v1.
    """
    out: dict[int, HomographyEntry] = {
        f: HomographyEntry(np.asarray(H, dtype=np.float64), "anchor", 1.0)
        for f, H in anchors.items()
    }
    keys = sorted(anchors)
    if not keys:
        return out

    # Frame size (width, height) for the disagreement metric; default 1080p if unreadable.
    frame_size = (1920, 1080)
    for f in keys:
        img = read_frame(f)
        if img is not None:
            frame_size = (int(img.shape[1]), int(img.shape[0]))
            break

    for a, b in itertools.pairwise(keys):
        gap = b - a - 1
        if gap < 1 or gap > max_gap:
            continue
        inner = list(range(a + 1, b))
        fwd = _chain(a, inner, read_frame, player_boxes, np.asarray(anchors[a], np.float64),
                     n_features, min_inliers)
        bwd = _chain(b, inner[::-1], read_frame, player_boxes, np.asarray(anchors[b], np.float64),
                     n_features, min_inliers)
        for t in inner:
            hf, hb = fwd.get(t), bwd.get(t)
            if hf is not None and hb is not None:
                w_f = (b - t) / (b - a)
                out[t] = HomographyEntry(
                    blend_homographies(hf, hb, w_f), "propagated",
                    disagreement_confidence(hf, hb, tau=disagreement_tau, frame_size=frame_size),
                )
            elif hf is not None:
                out[t] = HomographyEntry(hf, "propagated", max(0.0, 1.0 - (t - a) / (max_gap + 1)))
            elif hb is not None:
                out[t] = HomographyEntry(hb, "propagated", max(0.0, 1.0 - (b - t) / (max_gap + 1)))
    return out
