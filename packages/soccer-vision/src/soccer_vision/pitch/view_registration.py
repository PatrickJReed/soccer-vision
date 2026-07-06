"""Drift-free pitch calibration by direct frame->representative registration (Slice 2).

Each frame gets its pitch homography from ONE registration to its view's labeled
representative (compose H_rep_pitch @ G_frame_to_rep) instead of composing the long
inter-frame chain, so error does not accumulate with temporal distance. Output is a
drop-in homographies.parquet. See docs/superpowers/specs/2026-07-06-view-registration-design.md.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.labeler.view_digest import ViewDigest, _read_frames
from soccer_vision.pipeline import homographies_to_parquet
from soccer_vision.pitch.propagation import HomographyEntry, _frame_mask

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


@dataclass(frozen=True)
class RegisteredCalib:
    """Every frame's f_px->pitch homography, plus which view it registered through."""

    homographies: dict[int, HomographyEntry]   # frame -> (H f_px->pitch, source, confidence)
    rep_of: dict[int, int]                      # frame -> view id it registered through
    stats: dict[str, Any]


def rep_homographies_from_parquet(
    homographies_parquet: str | Path, representatives: dict[int, int],
) -> dict[int, NDArray[np.float64]]:
    """Extract the labeled full-px->pitch H at each rep's frame index (skip reps absent)."""
    df = pd.read_parquet(homographies_parquet)
    by_frame = {int(r.frame): np.array([[r.h00, r.h01, r.h02], [r.h10, r.h11, r.h12],  # type: ignore[arg-type]
                                        [r.h20, r.h21, r.h22]], dtype=np.float64)
                for r in df.itertuples()}
    return {int(v): by_frame[int(f)] for v, f in representatives.items() if int(f) in by_frame}


def _orb_full(img: NDArray[np.uint8], mask: NDArray[np.uint8] | None,
              n_features: int) -> tuple[list[Any], NDArray[Any] | None]:
    o = cv2.ORB_create(n_features)  # type: ignore[attr-defined]
    return o.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)  # type: ignore[no-any-return]


def register_clip(
    video_path: str | Path, digest: ViewDigest, rep_homographies: dict[int, NDArray[np.floating]],
    *, player_boxes: pd.DataFrame | None = None, frames: list[int] | None = None,
    n_features: int = DEFAULT_N_FEATURES, min_inliers: int = DEFAULT_MIN_INLIERS,
    cache_dir: str | Path | None = None,
) -> RegisteredCalib:
    """Register every frame to its best labeled representative -> drift-free f_px->pitch H.

    `cache_dir` is accepted but reserved (unused in v1); do not rely on caching yet.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frames is None:
        frames = list(range(max(n_frames, 1)))

    view_ids = sorted(rep_homographies)                      # candidate views with a labeled H
    rep_frames = [digest.representatives[v] for v in view_ids]
    rep_H = [np.asarray(rep_homographies[v], np.float64) for v in view_ids]
    rep_frame_to_view = {digest.representatives[v]: v for v in view_ids}

    rep_imgs = _read_frames(video_path, rep_frames)

    def mask_for(idx: int, img: NDArray[np.uint8] | None) -> NDArray[np.uint8] | None:
        if player_boxes is None or img is None:
            return None
        return _frame_mask(player_boxes, idx, img.shape[:2])

    rep_kps: list[list[Any]] = []
    rep_descs: list[NDArray[Any] | None] = []
    for rf, im in zip(rep_frames, rep_imgs, strict=True):
        kp, d = (_orb_full(im, mask_for(rf, im), n_features) if im is not None else ([], None))
        rep_kps.append(kp)
        rep_descs.append(d)

    homographies: dict[int, HomographyEntry] = {}
    rep_of: dict[int, int] = {}
    inliers: list[int] = []
    cap = cv2.VideoCapture(str(video_path))
    pos = 0
    try:
        for f in frames:
            if f < pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                pos = f
            while pos < f:
                if not cap.grab():
                    break
                pos += 1
            ok, img = cap.read()
            pos += 1
            if f in rep_frame_to_view:                       # rep passes through with labeled H
                v = rep_frame_to_view[f]
                homographies[f] = HomographyEntry(
                    np.asarray(rep_homographies[v], np.float64), "rep", 1.0)
                rep_of[f] = v
                continue
            if not ok:
                continue                                     # gap
            kp, d = _orb_full(img, mask_for(f, img), n_features)  # type: ignore[arg-type]
            res = register_to_best_rep(kp, d, rep_kps, rep_descs, min_inliers=min_inliers)
            if res is None:
                continue                                     # gap
            idx, G, n_in = res
            hf = compose_pitch_homography(rep_H[idx], G)
            conf = float(min(1.0, n_in / _FULL_CONF_INLIERS))
            homographies[f] = HomographyEntry(hf, "registered", conf)
            rep_of[f] = view_ids[idx]
            inliers.append(n_in)
    finally:
        cap.release()

    n_reg = sum(1 for e in homographies.values() if e.source == "registered")
    stats: dict[str, Any] = {
        "n_frames": len(frames), "n_registered": n_reg,
        "n_rep": sum(1 for e in homographies.values() if e.source == "rep"),
        "n_gap": len(frames) - len(homographies),
        "coverage": len(homographies) / max(len(frames), 1),
        "median_inliers": float(np.median(inliers)) if inliers else 0.0,
        "per_view_counts": {int(v): int(sum(1 for x in rep_of.values() if x == v))
                            for v in view_ids},
    }
    return RegisteredCalib(homographies=homographies, rep_of=rep_of, stats=stats)


def write_homographies(calib: RegisteredCalib, out_path: str | Path) -> None:
    """Write the registered homographies to a drop-in homographies.parquet."""
    homographies_to_parquet(calib.homographies, Path(out_path))
