"""Drift-free pitch calibration by direct frame->representative registration (Slice 2).

Each frame gets its pitch homography from ONE registration to its view's labeled
representative (compose H_rep_pitch @ G_frame_to_rep) instead of composing the long
inter-frame chain, so error does not accumulate with temporal distance. Output is a
drop-in homographies.parquet. See docs/superpowers/specs/2026-07-06-view-registration-design.md.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    video_path: str | Path, digest: ViewDigest,
    rep_homographies: Mapping[int, NDArray[np.floating]],
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


def cross_registration_error(
    video_path: str | Path, digest: ViewDigest,
    rep_homographies: Mapping[int, NDArray[np.floating]], *,
    n_features: int = DEFAULT_N_FEATURES, min_inliers: int = DEFAULT_MIN_INLIERS,
) -> pd.DataFrame:
    """Drift-free proof: register each labeled rep R to every OTHER rep R', compose, and
    measure the mean pitch reprojection error of the 4 image corners vs R's own manual H.
    Returns rows (rep_R, rep_Rprime, temporal_dist=|frame_R-frame_R'|, corner_err_pitch)."""
    video_path = Path(video_path)
    view_ids = sorted(rep_homographies)
    rep_frames = {v: digest.representatives[v] for v in view_ids}
    imgs = dict(zip(view_ids, _read_frames(video_path, [rep_frames[v] for v in view_ids]),
                    strict=True))
    kp: dict[int, list[Any]] = {}
    desc: dict[int, NDArray[Any] | None] = {}
    for v in view_ids:
        im = imgs[v]
        kp[v], desc[v] = (_orb_full(im, None, n_features) if im is not None else ([], None))
    first_img = next((im for im in imgs.values() if im is not None), None)
    if first_img is None:
        return pd.DataFrame(columns=["rep_R", "rep_Rprime", "temporal_dist", "corner_err_pitch"])
    h, w = first_img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)

    def project(hm: NDArray[np.floating], pts: NDArray[np.floating]) -> NDArray[np.float64]:
        p = (np.asarray(hm, np.float64) @ np.column_stack([pts, np.ones(len(pts))]).T).T
        return np.asarray(p[:, :2] / p[:, 2:3], dtype=np.float64)

    cols = ["rep_R", "rep_Rprime", "temporal_dist", "corner_err_pitch"]
    rows: list[dict[str, Any]] = []
    for r in view_ids:
        truth = project(rep_homographies[r], corners)
        for rp in view_ids:
            if rp == r:
                continue
            res = register_to_best_rep(kp[r], desc[r], [kp[rp]], [desc[rp]], min_inliers=min_inliers)
            if res is None:
                err = float("nan")                       # views don't overlap -> no proof point
            else:
                _, g, _n = res
                hc = compose_pitch_homography(rep_homographies[rp], g)
                err = float(np.linalg.norm(project(hc, corners) - truth, axis=1).mean())
            rows.append({"rep_R": r, "rep_Rprime": rp,
                         "temporal_dist": abs(rep_frames[r] - rep_frames[rp]),
                         "corner_err_pitch": err})
    return pd.DataFrame(rows, columns=cols)


def _digest_from_json(path: str | Path) -> ViewDigest:
    import json
    d = json.loads(Path(path).read_text())
    reps = {int(k): int(v) for k, v in d["representatives"].items()}
    view_of = {int(k): int(v) for k, v in d.get("view_of", {}).items()}
    return ViewDigest(sample_frames=sorted(view_of), view_of=view_of,
                      representatives=reps, similarity=np.zeros((1, 1)))


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Drift-free view-registration calibration (Slice 2)")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--digest-json", required=True, type=Path)
    ap.add_argument("--rep-homographies", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--boxes", type=Path, default=None)
    ap.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(argv)
    out = args.out or (args.video.parent / "view_registration_out")
    Path(out).mkdir(parents=True, exist_ok=True)
    digest = _digest_from_json(args.digest_json)
    rep_h = rep_homographies_from_parquet(args.rep_homographies, digest.representatives)
    boxes = pd.read_parquet(args.boxes) if args.boxes else None
    calib = register_clip(args.video, digest, rep_h, player_boxes=boxes, min_inliers=args.min_inliers)
    write_homographies(calib, Path(out) / "homographies.parquet")
    print("view-registration:", {k: calib.stats[k]
                                  for k in ("coverage", "n_registered", "n_gap", "median_inliers")})
    if args.validate:
        df = cross_registration_error(args.video, digest, rep_h, min_inliers=args.min_inliers)
        if not df.empty:
            med = df["temporal_dist"].median()
            near = df[df.temporal_dist <= med]["corner_err_pitch"].mean()
            far = df[df.temporal_dist > med]["corner_err_pitch"].mean()
            print(f"cross-reg corner error (pitch units): median "
                  f"{df['corner_err_pitch'].median():.2f}")
            print(f"  near-rep pairs mean {near:.2f}  vs  far-rep pairs mean {far:.2f} "
                  f"(flat => drift-free)")
    print(f"wrote homographies.parquet -> {out}")


if __name__ == "__main__":
    main()
