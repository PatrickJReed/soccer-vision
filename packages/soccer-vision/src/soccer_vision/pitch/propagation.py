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
from typing import Any

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Image-space reference grid (normalized 0..1) used to measure forward/backward
# disagreement; scaled to the frame size at use.
_GRID = np.array([(x, y) for x in (0.2, 0.5, 0.8) for y in (0.2, 0.5, 0.8)], dtype=np.float64)


@dataclass(frozen=True)
class HomographyEntry:
    """A frame's homography plus where it came from.

    confidence is 1.0 for anchors and a runtime estimate for propagated frames
    (which can legitimately reach 0.0 at maximum disagreement). Use ``source`` —
    not confidence — to detect absence: absent frames are reported as 'none'.
    """

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
    interframe: Mapping[int, NDArray[np.floating]],
    H_anchor: NDArray[np.floating],
) -> dict[int, NDArray[np.floating]]:
    """Compose precomputed inter-frame homographies from `anchor` over adjacent `targets`.

    `interframe[i]` maps frame i pixels -> frame i+1 pixels. `targets` is the ordered
    adjacent sequence (ascending for a forward chain, descending for backward). Returns
    {frame: pixel->pitch H}; stops at the first missing inter-frame homography.
    """
    out: dict[int, NDArray[np.floating]] = {}
    W = np.eye(3)                                  # maps anchor pixels -> current pixels
    prev = anchor
    for f in targets:
        if f == prev + 1:
            step = interframe.get(prev)            # prev -> f
        elif f == prev - 1:
            g = interframe.get(f)                  # f -> prev
            step = np.linalg.inv(g) if g is not None else None  # prev -> f
        else:
            break                                  # non-adjacent target (should not happen)
        if step is None:
            break
        W = step @ W
        out[f] = H_anchor @ np.linalg.inv(W)       # pixel_f -> pitch
        prev = f
    return out


def propagate_homographies(
    anchors: Mapping[int, NDArray[np.floating]],
    interframe: Mapping[int, NDArray[np.floating]],
    *,
    max_gap: int = 25,
    disagreement_tau: float = 0.10,
    frame_size: tuple[int, int] = (1920, 1080),
) -> dict[int, HomographyEntry]:
    """Bridge no-landmark gaps between anchors by composing inter-frame homographies.

    `interframe[i]` maps frame i pixels -> frame i+1 pixels (from
    compute_interframe_homographies). Each anchor keeps source='anchor', confidence=1.0.
    For each gap <= max_gap, chain forward from the left anchor and backward from the
    right anchor, blend by distance, and set confidence from forward/backward
    disagreement. Frames reached by neither chain are absent. Edge gaps are not bridged
    in v1. Pure: no video I/O.
    """
    out: dict[int, HomographyEntry] = {
        f: HomographyEntry(np.asarray(H, dtype=np.float64), "anchor", 1.0)
        for f, H in anchors.items()
    }
    keys = sorted(anchors)
    for a, b in itertools.pairwise(keys):
        gap = b - a - 1
        if gap < 1 or gap > max_gap:
            continue
        inner = list(range(a + 1, b))
        fwd = _chain(a, inner, interframe, np.asarray(anchors[a], np.float64))
        bwd = _chain(b, inner[::-1], interframe, np.asarray(anchors[b], np.float64))
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


def _orb_downscaled(
    img: NDArray[np.uint8], mask: NDArray[np.uint8], downscale: float, n_features: int
) -> tuple[list[Any], NDArray[Any] | None]:
    """ORB keypoints+descriptors on a downscaled copy (keypoints in DOWNSCALED coords)."""
    if downscale != 1.0:
        small = cv2.resize(img, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        smask = cv2.resize(mask, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_NEAREST)
    else:
        small, smask = img, mask
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(n_features)  # type: ignore[attr-defined]
    return orb.detectAndCompute(gray, smask)  # type: ignore[no-any-return]


def _homography_from_descriptors(
    kp_a: list[Any], d_a: NDArray[Any] | None, kp_b: list[Any], d_b: NDArray[Any] | None,
    downscale: float, min_inliers: int,
) -> NDArray[np.floating] | None:
    """Match descriptors -> homography (downscaled px), rescaled to FULL-res px->px."""
    if d_a is None or d_b is None or len(d_a) < min_inliers or len(d_b) < min_inliers:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d_a, d_b)
    if len(matches) < min_inliers:
        return None
    src = np.array([kp_a[m.queryIdx].pt for m in matches], dtype=np.float32)
    dst = np.array([kp_b[m.trainIdx].pt for m in matches], dtype=np.float32)
    g_small, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if g_small is None:
        return None
    s = np.diag([downscale, downscale, 1.0])            # full px -> small px
    s_inv = np.diag([1.0 / downscale, 1.0 / downscale, 1.0])
    return (s_inv @ g_small @ s).astype(np.float64)     # full px -> full px


def compute_interframe_homographies(
    read_frame: Callable[[int], NDArray[np.uint8] | None],
    needed_pairs: set[int],
    player_boxes: pd.DataFrame,
    *,
    downscale: float = 0.5,
    n_features: int = 3000,
    min_inliers: int = 12,
) -> dict[int, NDArray[np.floating]]:
    """Register every needed consecutive frame pair in ONE ascending pass.

    `needed_pairs` is the set of indices i for which interframe[i] (frame i -> i+1) is
    wanted. Each frame is read once (read_frame is called in ascending order, so the
    caller can decode sequentially) and ORB'd once on a `downscale` copy; the resulting
    homography is rescaled back to full-resolution pixels. Returns {i: full-res G[i]}.
    """
    interframe: dict[int, NDArray[np.floating]] = {}
    if not needed_pairs:
        return interframe
    frames = sorted(needed_pairs | {i + 1 for i in needed_pairs})
    prev_idx: int | None = None
    prev_kp: list[Any] = []
    prev_d: NDArray[Any] | None = None
    for idx in frames:
        img = read_frame(idx)
        if img is None:
            prev_idx = None
            continue
        mask = _frame_mask(player_boxes, idx, img.shape[:2])
        kp, d = _orb_downscaled(img, mask, downscale, n_features)
        if prev_idx == idx - 1 and (idx - 1) in needed_pairs:
            g = _homography_from_descriptors(prev_kp, prev_d, kp, d, downscale, min_inliers)
            if g is not None:
                interframe[idx - 1] = g
        prev_idx, prev_kp, prev_d = idx, kp, d
    return interframe
