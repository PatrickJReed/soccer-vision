"""Drift-free pitch calibration by direct frame->representative registration (Slice 2).

Each frame gets its pitch homography from ONE registration to its view's labeled
representative (compose H_rep_pitch @ G_frame_to_rep) instead of composing the long
inter-frame chain, so error does not accumulate with temporal distance. Output is a
drop-in homographies.parquet. See docs/superpowers/specs/2026-07-06-view-registration-design.md.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

DEFAULT_N_FEATURES = 3000
DEFAULT_MIN_INLIERS = 12
DEFAULT_TOP_K = 2
_FULL_CONF_INLIERS = 50.0    # inliers at/above which confidence saturates to 1.0


def compose_pitch_homography(
    H_rep_pitch: NDArray[np.floating], G_frame_to_rep: NDArray[np.floating],
) -> NDArray[np.float64]:
    """frame_px -> pitch, from rep_px->pitch and frame_px->rep_px. Normalized so H[2,2]=1."""
    H = np.asarray(H_rep_pitch, np.float64) @ np.asarray(G_frame_to_rep, np.float64)
    if abs(H[2, 2]) > 1e-9:
        H = H / H[2, 2]
    return H


def register_to_best_rep(
    frame_kp: list[Any], frame_desc: NDArray[Any] | None,
    rep_kps: list[list[Any]], rep_descs: list[NDArray[Any] | None], *,
    min_inliers: int = DEFAULT_MIN_INLIERS, top_k: int = DEFAULT_TOP_K,
) -> tuple[int, NDArray[np.float64], int] | None:
    """Register a frame to its best-matching representative.

    Ranks reps by cross-checked match count, runs RANSAC findHomography on the top_k,
    and returns (best_rep_index, G frame_px->rep_px, n_inliers) with the most inliers
    (>= min_inliers), or None if none qualify.
    """
    if frame_desc is None or len(frame_desc) < min_inliers:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ranked: list[tuple[int, int, Sequence[Any]]] = []
    for i, rd in enumerate(rep_descs):
        if rd is None or len(rd) < min_inliers:
            continue
        m = matcher.match(frame_desc, rd)
        if len(m) >= min_inliers:
            ranked.append((len(m), i, m))
    ranked.sort(key=lambda x: x[0], reverse=True)
    best: tuple[int, NDArray[np.float64], int] | None = None
    for _, i, m in ranked[:top_k]:
        src = np.array([frame_kp[x.queryIdx].pt for x in m], dtype=np.float32)
        dst = np.array([rep_kps[i][x.trainIdx].pt for x in m], dtype=np.float32)
        G, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if G is None:
            continue
        n_in = int(mask.sum())
        if n_in >= min_inliers and (best is None or n_in > best[2]):
            best = (i, G.astype(np.float64), n_in)
    return best
