"""Classical view-digest — cluster a video's frames into distinct camera VIEWS.

A fixed broadcast camera pans back and forth over one field, so the number of
distinct fields-of-view is far smaller than the number of frames (measured: ~13
views over 2700 oceanside frames). This module recovers that structure with no
GPU: ORB self-similarity over sampled frames -> agglomerative clustering into
views -> one representative frame per view. The representatives are the small set
of frames a human labels to cover the whole clip, and they double as the
selection of training frames for the later learned-embedding model.

Pipeline (pure, unit-tested): frame_descriptors -> similarity_matrix ->
cluster_views -> medoid_representatives, orchestrated by digest_frames. Video
decode, plotting and the CLI are the only I/O.

See docs/superpowers/2026-07-02-revisit-structure-findings.md and
docs/superpowers/specs/2026-07-02-view-digest-design.md.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]

from soccer_vision.labeler.chain import _video_hash

# ORB descriptor byte-count; two frames of the same view share static field/stand
# structure, so a large fraction of their features match at a small Hamming distance.
DEFAULT_STRIDE = 25
DEFAULT_N_FEATURES = 1200
DEFAULT_DOWNSCALE = 0.5
# Finer (lower) threshold splits more; that is the safe bias for calibration transfer —
# under-clustering assigns a representative's homography to frames it does not fit, while
# over-clustering only costs a few extra labels. 0.5 reproduces the ~13-view oceanside
# finding; Slice 2's fit-check will let us auto-tune this per clip.
DEFAULT_DIST_THRESHOLD = 0.5
DEFAULT_MIN_MATCH_DIST = 48
DEFAULT_MIN_KEYPOINTS = 10

_Descriptor = NDArray[np.uint8] | None


@dataclass(frozen=True)
class ViewDigest:
    """The clustering of a clip into views.

    sample_frames:   the sampled frame indices (rows/cols of `similarity`).
    view_of:         sampled frame index -> view id (0-based, by first appearance).
    representatives: view id -> the medoid frame index for that view (label these).
    similarity:      NxN sampled-frame ORB match-fraction matrix (diagnostic).
    """

    sample_frames: list[int]
    view_of: dict[int, int]
    representatives: dict[int, int]
    similarity: NDArray[np.float64]

    @property
    def n_views(self) -> int:
        return len(self.representatives)


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------


def frame_descriptors(
    frames: list[NDArray[np.uint8] | None],
    *,
    n_features: int = DEFAULT_N_FEATURES,
    downscale: float = DEFAULT_DOWNSCALE,
    masks: list[NDArray[np.uint8] | None] | None = None,
) -> tuple[list[_Descriptor], list[int]]:
    """ORB descriptors + keypoint counts for each frame.

    Frames may be BGR or already-gray; `masks` (aligned to `frames`, at each frame's
    original resolution, 0 = ignore) suppress moving players so clustering keys on
    the static camera pose. A None frame yields (None, 0).
    """
    orb = cv2.ORB_create(n_features)  # type: ignore[attr-defined]
    descriptors: list[_Descriptor] = []
    counts: list[int] = []
    for i, frame in enumerate(frames):
        if frame is None:
            descriptors.append(None)
            counts.append(0)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if downscale != 1.0:
            gray = cv2.resize(gray, None, fx=downscale, fy=downscale)
        mask = None
        if masks is not None:
            m = masks[i]
            if m is not None:
                mask = (m if m.shape[:2] == gray.shape[:2]
                        else cv2.resize(m, (gray.shape[1], gray.shape[0]),
                                        interpolation=cv2.INTER_NEAREST))
        keypoints, desc = orb.detectAndCompute(gray, mask)
        descriptors.append(desc)
        counts.append(len(keypoints) if keypoints is not None else 0)
    return descriptors, counts


def _pair_match_fraction(
    da: _Descriptor,
    db: _Descriptor,
    ca: int,
    cb: int,
    *,
    min_match_dist: int = DEFAULT_MIN_MATCH_DIST,
    min_keypoints: int = DEFAULT_MIN_KEYPOINTS,
) -> float:
    """Fraction of cross-checked ORB matches (dist < min_match_dist) / min(keypoints)."""
    if da is None or db is None or ca < min_keypoints or cb < min_keypoints:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    good = sum(1 for m in matcher.match(da, db) if m.distance < min_match_dist)
    return good / max(1, min(ca, cb))


def similarity_matrix(
    descriptors: list[_Descriptor],
    keypoint_counts: list[int],
    *,
    min_match_dist: int = DEFAULT_MIN_MATCH_DIST,
    min_keypoints: int = DEFAULT_MIN_KEYPOINTS,
) -> NDArray[np.float64]:
    """Symmetric NxN view-similarity = fraction of cross-checked ORB matches.

    similarity[a,b] = (#matches with Hamming dist < min_match_dist) /
    min(keypoints_a, keypoints_b). Frames with too few keypoints score 0. The
    diagonal is 1.0 (a frame is identical to itself).
    """
    n = len(descriptors)
    sim = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(a + 1, n):
            score = _pair_match_fraction(
                descriptors[a], descriptors[b],
                keypoint_counts[a], keypoint_counts[b],
                min_match_dist=min_match_dist, min_keypoints=min_keypoints)
            sim[a, b] = sim[b, a] = score
    np.fill_diagonal(sim, 1.0)
    return sim


def _relabel_by_first_appearance(labels: NDArray[np.int_]) -> NDArray[np.int_]:
    """Remap arbitrary cluster ids to 0-based ids ordered by first occurrence."""
    remap: dict[int, int] = {}
    out = np.empty(len(labels), dtype=np.int_)
    for i, raw in enumerate(labels.tolist()):
        if raw not in remap:
            remap[raw] = len(remap)
        out[i] = remap[raw]
    return out


def cluster_views(
    similarity: NDArray[np.float64],
    *,
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
) -> NDArray[np.int_]:
    """Agglomerative (average-linkage) clustering of frames into views.

    Distance = 1 - similarity / max(off-diagonal similarity), cut at
    `dist_threshold`. Returns a 0-based view id per row, ordered by first
    appearance. n<=1 is a single view.
    """
    sim = np.asarray(similarity, dtype=np.float64)
    n = sim.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int_)
    if n == 1:
        return np.zeros(1, dtype=np.int_)
    off = sim.copy()
    np.fill_diagonal(off, 0.0)
    smax = max(float(off.max()), 1e-9)
    dist = 1.0 - sim / smax
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, None)
    z = linkage(squareform(dist, checks=False), method="average")
    raw = fcluster(z, t=dist_threshold, criterion="distance")
    return _relabel_by_first_appearance(np.asarray(raw, dtype=np.int_))


def medoid_representatives(
    labels: NDArray[np.int_],
    similarity: NDArray[np.float64],
    sample_frames: list[int],
) -> dict[int, int]:
    """view id -> the frame index most similar to the rest of its view (the medoid)."""
    sim = np.asarray(similarity, dtype=np.float64)
    label_list = labels.tolist()
    reps: dict[int, int] = {}
    for view in sorted(set(label_list)):
        members = [i for i, v in enumerate(label_list) if v == view]
        if len(members) == 1:
            reps[view] = sample_frames[members[0]]
            continue
        best_i, best_score = members[0], -1.0
        for i in members:
            score = float(sum(sim[i, j] for j in members if j != i))
            if score > best_score:
                best_i, best_score = i, score
        reps[view] = sample_frames[best_i]
    return reps


def _digest_from_similarity(
    sim: NDArray[np.float64], sample_frames: list[int], dist_threshold: float,
) -> ViewDigest:
    labels = cluster_views(sim, dist_threshold=dist_threshold)
    view_of = {sample_frames[i]: int(labels[i]) for i in range(len(sample_frames))}
    reps = medoid_representatives(labels, sim, sample_frames)
    return ViewDigest(sample_frames=list(sample_frames), view_of=view_of,
                      representatives=reps, similarity=sim)


def digest_frames(
    frames: list[NDArray[np.uint8] | None],
    sample_frames: list[int],
    *,
    n_features: int = DEFAULT_N_FEATURES,
    downscale: float = DEFAULT_DOWNSCALE,
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
    min_match_dist: int = DEFAULT_MIN_MATCH_DIST,
    masks: list[NDArray[np.uint8] | None] | None = None,
) -> ViewDigest:
    """Cluster already-decoded frames into views (the pure pipeline, no video I/O)."""
    descriptors, counts = frame_descriptors(
        frames, n_features=n_features, downscale=downscale, masks=masks)
    sim = similarity_matrix(descriptors, counts, min_match_dist=min_match_dist)
    return _digest_from_similarity(sim, sample_frames, dist_threshold)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------


def _read_frames(video_path: Path, indices: list[int]) -> list[NDArray[np.uint8] | None]:
    """Decode the requested (ascending) frame indices, seeking forward by grab()."""
    cap = cv2.VideoCapture(str(video_path))
    frames: list[NDArray[np.uint8] | None] = []
    pos = 0
    try:
        for idx in indices:
            if idx < pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                pos = idx
            while pos < idx:
                if not cap.grab():
                    break
                pos += 1
            ok, frame = cap.read()
            pos += 1
            frames.append(frame if ok else None)  # type: ignore[arg-type]
    finally:
        cap.release()
    return frames


def _build_masks(
    player_boxes: pd.DataFrame, indices: list[int], size: tuple[int, int],
) -> list[NDArray[np.uint8] | None]:
    """Per-sampled-frame masks (255 keep, 0 = player box) at original resolution."""
    w, h = size
    by_frame = {f: g for f, g in player_boxes.groupby("frame")}
    masks: list[NDArray[np.uint8] | None] = []
    for idx in indices:
        mask = np.full((h, w), 255, dtype=np.uint8)
        boxes = by_frame.get(idx)
        if boxes is not None:
            for _, row in boxes.iterrows():
                x1 = int(max(0, row["bbox_x1"]))
                y1 = int(max(0, row["bbox_y1"]))
                x2 = int(min(w, row["bbox_x2"]))
                y2 = int(min(h, row["bbox_y2"]))
                mask[y1:y2, x1:x2] = 0
        masks.append(mask)
    return masks


def _sim_cache_path(
    video_path: Path, cache_dir: Path, stride: int, n_features: int,
    downscale: float, min_match_dist: int,
) -> Path:
    key = f"{_video_hash(video_path)}_s{stride}_n{n_features}_d{downscale}_m{min_match_dist}"
    return cache_dir / f"viewdigest_{key}.npz"


def compute_view_digest(
    video_path: str | Path,
    *,
    stride: int = DEFAULT_STRIDE,
    n_features: int = DEFAULT_N_FEATURES,
    downscale: float = DEFAULT_DOWNSCALE,
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
    min_match_dist: int = DEFAULT_MIN_MATCH_DIST,
    player_boxes: pd.DataFrame | None = None,
    cache_dir: str | Path | None = None,
) -> ViewDigest:
    """Sample a video every `stride` frames, cluster the samples into views.

    The (expensive) ORB self-similarity matrix is cached per (video, stride,
    n_features, downscale, min_match_dist), so re-clustering at a different
    `dist_threshold` is instant. Player masking (when `player_boxes` is given)
    bypasses the cache.
    """
    video_path = Path(video_path)
    probe = cv2.VideoCapture(str(video_path))
    n_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    probe.release()

    sample_frames = list(range(0, max(n_frames, 1), stride))
    cache_dir = Path(cache_dir or (video_path.parent / ".sv_labeler_cache"))
    cache_path = _sim_cache_path(
        video_path, cache_dir, stride, n_features, downscale, min_match_dist)

    sim: NDArray[np.float64] | None = None
    if player_boxes is None and cache_path.exists():
        data = np.load(cache_path)
        if data["sample_frames"].tolist() == sample_frames:
            sim = np.asarray(data["similarity"], dtype=np.float64)

    if sim is None:
        frames = _read_frames(video_path, sample_frames)
        masks = (_build_masks(player_boxes, sample_frames, (width, height))
                 if player_boxes is not None else None)
        descriptors, counts = frame_descriptors(
            frames, n_features=n_features, downscale=downscale, masks=masks)
        sim = similarity_matrix(descriptors, counts, min_match_dist=min_match_dist)
        if player_boxes is None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path,
                     sample_frames=np.array(sample_frames, dtype=np.int64),
                     similarity=sim)

    return _digest_from_similarity(sim, sample_frames, dist_threshold)


# ---------------------------------------------------------------------------
# Render + persist
# ---------------------------------------------------------------------------


def render_digest(
    digest: ViewDigest, video_path: str | Path, out_dir: str | Path,
) -> list[Path]:
    """Write views_montage.png, similarity.png and view_digest.json; return paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # montage of representative frames
    rep_items = sorted(digest.representatives.items())
    rep_frames = _read_frames(video_path, [f for _, f in rep_items])
    tile_h, tile_w = 180, 240
    cols = max(1, int(np.ceil(np.sqrt(len(rep_items)))))
    rows = max(1, int(np.ceil(len(rep_items) / cols)))
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 20, dtype=np.uint8)
    for k, ((view, frame), img) in enumerate(zip(rep_items, rep_frames, strict=True)):
        if img is None:
            continue
        thumb = cv2.resize(img, (tile_w, tile_h))
        r, c = divmod(k, cols)
        y0, x0 = r * tile_h, c * tile_w
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = thumb
        cv2.putText(canvas, f"view {view} / f{frame}", (x0 + 6, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    montage_path = out_dir / "views_montage.png"
    cv2.imwrite(str(montage_path), canvas)

    # similarity heatmap
    sim_path = out_dir / "similarity.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    sf = digest.sample_frames
    extent = ((float(sf[0]), float(sf[-1]), float(sf[0]), float(sf[-1]))
              if len(sf) > 1 else None)
    im = ax.imshow(digest.similarity, cmap="magma", origin="lower", extent=extent)
    ax.set_xlabel("frame")
    ax.set_ylabel("frame")
    ax.set_title(f"View self-similarity — {digest.n_views} views\n"
                 "off-diagonal bands = revisited views")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(sim_path, dpi=120)
    plt.close(fig)

    # machine-readable digest
    json_path = out_dir / "view_digest.json"
    stride = (sf[1] - sf[0]) if len(sf) > 1 else None
    json_path.write_text(json.dumps({
        "n_views": digest.n_views,
        "stride": stride,
        "sample_frames": sf,
        "representatives": {str(v): f for v, f in digest.representatives.items()},
        "view_of": {str(f): v for f, v in digest.view_of.items()},
    }, indent=2))

    return [montage_path, sim_path, json_path]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Classical view-digest (annotation-scaling Slice 1)")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    ap.add_argument("--downscale", type=float, default=DEFAULT_DOWNSCALE)
    ap.add_argument("--dist-threshold", type=float, default=DEFAULT_DIST_THRESHOLD)
    ap.add_argument("--min-match-dist", type=int, default=DEFAULT_MIN_MATCH_DIST)
    ap.add_argument("--boxes", type=Path, default=None,
                    help="optional trajectories parquet for player masking")
    args = ap.parse_args(argv)

    out = args.out or (args.video.parent / "view_digest_out")
    player_boxes = pd.read_parquet(args.boxes) if args.boxes else None
    digest = compute_view_digest(
        args.video, stride=args.stride, n_features=args.n_features,
        downscale=args.downscale, dist_threshold=args.dist_threshold,
        min_match_dist=args.min_match_dist, player_boxes=player_boxes)
    render_digest(digest, args.video, out)

    reps = sorted(digest.representatives.items())
    span = {v: sum(1 for vv in digest.view_of.values() if vv == v)
            for v in digest.representatives}
    print(f"sampled frames: {len(digest.sample_frames)} (stride {args.stride})")
    print(f"distinct views: {digest.n_views}")
    print("label these representative frames:")
    for view, frame in reps:
        print(f"  view {view:2d}: frame {frame:6d}  ({span[view]} sampled frames in view)")
    print(f"\nwrote montage + heatmap + json -> {out}")


if __name__ == "__main__":
    main()
