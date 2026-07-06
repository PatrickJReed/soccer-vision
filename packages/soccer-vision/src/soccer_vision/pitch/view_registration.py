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


def _nondegenerate(
    G: NDArray[np.floating], *, det_lo: float = 0.02, det_hi: float = 50.0,
) -> bool:
    """Reject degenerate / extreme-scale homographies (a cheap sanity gate on |det(G[:2,:2])|).

    NOTE: an earlier version also rejected near-collinear inlier point clouds (eigenvalue-ratio
    gate). Measured on real oceanside footage that dropped ~60% of GOOD frames: broadcast
    background features legitimately concentrate in a horizontal band (the pitch recedes to a
    thin band; foreground grass is textureless), yet still register with ~1350 inliers and tight
    within-view consistency (spread shrank when they were included). The within-view consistency
    validation is the real garbage detector; masked footage (more line-dominated) may warrant a
    confidence-based conditioning signal later — tracked as a follow-up, not a hard reject.
    """
    det = abs(float(np.linalg.det(np.asarray(G, np.float64)[:2, :2])))
    return det_lo <= det <= det_hi


def register_to_best_rep(
    frame_kp: list[Any], frame_desc: NDArray[Any] | None,
    rep_kps: list[list[Any]], rep_descs: list[NDArray[Any] | None], *,
    min_inliers: int = DEFAULT_MIN_INLIERS, top_k: int = DEFAULT_TOP_K,
) -> tuple[int, NDArray[np.float64], int] | None:
    """Register a frame to its best-matching representative.

    Ranks reps by cross-checked match count, runs RANSAC findHomography on the top_k,
    and returns (best_rep_index, G frame_px->rep_px, n_inliers) with the most inliers
    (>= min_inliers) whose homography is non-degenerate (sane determinant), or None if none
    qualify.
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
        Gf = G.astype(np.float64)
        inlier_mask = mask.ravel().astype(bool)
        n_in = int(inlier_mask.sum())
        if n_in < min_inliers:
            continue
        if not _nondegenerate(Gf):
            continue                              # degenerate / extreme-scale H -> reject
        if best is None or n_in > best[2]:
            best = (i, Gf, n_in)
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


def homographies_dict_from_parquet(path: str | Path) -> dict[int, NDArray[np.float64]]:
    """Load a homographies.parquet (frame->3x3 H) into a dict, e.g. a chain-propagation run."""
    df = pd.read_parquet(path)
    return {int(r.frame): np.array([[r.h00, r.h01, r.h02], [r.h10, r.h11, r.h12],  # type: ignore[arg-type]
                                    [r.h20, r.h21, r.h22]], dtype=np.float64)
            for r in df.itertuples()}


def _project_point(H: NDArray[np.floating], p: NDArray[np.floating]) -> NDArray[np.float64]:
    """Project a homogeneous image point p=(x,y,1) to pitch (x,y) via H."""
    q = np.asarray(H, np.float64) @ p
    return np.asarray(q[:2] / q[2], dtype=np.float64)


def view_consistency(
    calib: RegisteredCalib, *, ref_px: tuple[float, float] = (960.0, 540.0),
    chain: Mapping[int, NDArray[np.floating]] | None = None,
) -> pd.DataFrame:
    """Within-view drift-free proof: for each view, project a reference image point to pitch
    via every frame assigned to that view and measure the spread around the per-view median.
    A drift-free method keeps this spread small and INDEPENDENT of the view's temporal span
    (the same camera pose recurs across the clip). Optionally compares the same spread computed
    from `chain` (frame->H) homographies for contrast.

    Returns one row per view with >=2 frames:
      view, n_frames, temporal_span (max-min frame), reg_spread (mean L2 pitch dist to the
      per-view median projected point), chain_spread (same via `chain`, NaN if unavailable).

    "Spread" is the mean over the view's frames of the L2 distance from each frame's projected
    point to the view's component-wise median projected point.
    """
    p = np.array([ref_px[0], ref_px[1], 1.0])

    # view id -> frames assigned to it that have an accepted registered/rep homography.
    frames_by_view: dict[int, list[int]] = {}
    for f, v in calib.rep_of.items():
        e = calib.homographies.get(f)
        if e is None or e.source not in ("registered", "rep"):
            continue
        frames_by_view.setdefault(int(v), []).append(int(f))

    def spread(frames: list[int], src: Mapping[int, NDArray[np.floating]]) -> float:
        pts = [_project_point(src[f], p) for f in frames if f in src]
        if len(pts) < 2:
            return float("nan")
        arr = np.asarray(pts, np.float64)
        med = np.median(arr, axis=0)                       # component-wise median
        return float(np.linalg.norm(arr - med, axis=1).mean())

    rows: list[dict[str, Any]] = []
    for v in sorted(frames_by_view):
        frames = sorted(frames_by_view[v])
        if len(frames) < 2:
            continue                                       # no spread defined
        reg_src = {f: calib.homographies[f].H for f in frames}
        rows.append({
            "view": v,
            "n_frames": len(frames),
            "temporal_span": frames[-1] - frames[0],
            "reg_spread": spread(frames, reg_src),
            "chain_spread": spread(frames, chain) if chain is not None else float("nan"),
        })
    cols = ["view", "n_frames", "temporal_span", "reg_spread", "chain_spread"]
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
    ap.add_argument("--chain-homographies", type=Path, default=None,
                    help="Optional chain-propagation homographies.parquet for drift contrast.")
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
        chain = homographies_dict_from_parquet(args.chain_homographies) \
            if args.chain_homographies else None
        vc = view_consistency(calib, chain=chain)
        n_views = len(vc)
        print(f"view-consistency: {n_views} views with >=2 frames")
        if n_views == 0:
            print("  insufficient multi-frame views to measure within-view drift")
        else:
            print(f"  registration within-view spread (pitch units): "
                  f"median {vc['reg_spread'].median():.3f}  max {vc['reg_spread'].max():.3f}")
            span_range = vc["temporal_span"].max() - vc["temporal_span"].min()
            if n_views >= 3 and span_range > 0:
                med_span = vc["temporal_span"].median()
                lo = vc[vc.temporal_span <= med_span]
                hi = vc[vc.temporal_span > med_span]
                corr = float(vc["reg_spread"].corr(vc["temporal_span"].astype(float)))
                print(f"  reg_spread vs temporal_span: corr={corr:.2f}  |  "
                      f"short-span (n={len(lo)}) mean {lo['reg_spread'].mean():.3f}  vs  "
                      f"long-span (n={len(hi)}) mean {hi['reg_spread'].mean():.3f} "
                      f"(flat => drift-free)")
            else:
                print("  insufficient views to claim flatness; raw per-view table:")
                print(vc.to_string(index=False))
            if chain is not None and vc["chain_spread"].notna().any():
                print(f"  chain within-view spread (pitch units): "
                      f"median {vc['chain_spread'].median():.3f}  "
                      f"max {vc['chain_spread'].max():.3f} "
                      f"(registration should be smaller)")
    print(f"wrote homographies.parquet -> {out}")


if __name__ == "__main__":
    main()
