"""View-dataset exporter — turn a video + its ViewDigest into a Colab training manifest.

Slice 1.5 of the annotation-scaling sub-project. Given the small set of camera VIEWS
recovered by ``view_digest`` (Slice 1), assign every densely-sampled frame a ``view_id``
pseudo-label by nearest-representative ORB match, then export a manifest the Colab model
consumes for BOTH a contrastive/metric-learning embedding (same-view frames = positives)
and a masked-autoencoder (mask players, reconstruct the field).

Manifest-first: the default export writes no image files — a compact parquet manifest plus
a self-describing JSON sidecar, and Colab decodes frames on demand from the same ALL-INTRA
mp4. An optional ``materialize`` escape hatch writes a portable ImageFolder tree. Same house
style as ``view_digest``: pure, unit-tested core + thin video I/O + ``.npz`` cache + CLI.

This module sits beside and imports from ``view_digest.py`` (distinct from the top-level
``dataset_export.py`` YOLO-pose exporter). See
docs/superpowers/specs/2026-07-02-view-dataset-exporter-design.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.labeler.chain import _video_hash
from soccer_vision.labeler.view_digest import (
    DEFAULT_DOWNSCALE,
    DEFAULT_MIN_KEYPOINTS,
    DEFAULT_MIN_MATCH_DIST,
    DEFAULT_N_FEATURES,
    ViewDigest,
    _build_masks,
    _Descriptor,
    _pair_match_fraction,
    _read_frames,
    compute_view_digest,
    frame_descriptors,
)

DEFAULT_ASSIGN_STRIDE = 5
DEFAULT_AMBIGUITY_MARGIN = 0.05
DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_VAL_FRAC = 0.1
DEFAULT_CHUNK = 512
SCHEMA_VERSION = 1


def cross_match_fractions(
    query_descriptors: list[_Descriptor],
    query_counts: list[int],
    ref_descriptors: list[_Descriptor],
    ref_counts: list[int],
    *,
    min_match_dist: int = DEFAULT_MIN_MATCH_DIST,
    min_keypoints: int = DEFAULT_MIN_KEYPOINTS,
) -> NDArray[np.float64]:
    """(Q,R) match-fraction of each query frame vs each representative — same metric as
    similarity_matrix, rectangular (no NxN over queries)."""
    q, r = len(query_descriptors), len(ref_descriptors)
    out = np.zeros((q, r), dtype=np.float64)
    for i in range(q):
        for j in range(r):
            out[i, j] = _pair_match_fraction(
                query_descriptors[i], ref_descriptors[j],
                query_counts[i], ref_counts[j],
                min_match_dist=min_match_dist, min_keypoints=min_keypoints)
    return out


def assign_nearest_view(
    match: NDArray[np.float64],
    ref_view_ids: Sequence[int],
) -> pd.DataFrame:
    """Assign each query row (frame) to its nearest reference VIEW.

    ``match`` is (Q, R); ``ref_view_ids`` aligns to the columns. For each query row:
      - best column ``j* = argmax(row)`` (numpy default: first max on ties);
        ``confidence = row[j*]``.
      - ``view_id_raw = ref_view_ids[j*]`` when ``confidence > 0`` else ``-1``.
      - ``view_second`` = the view of the highest-scoring column whose view differs
        from the best view; ``-1`` if no such column exists or ``confidence == 0``.
      - ``margin = confidence - max(row over other-view columns)``; if there are no
        other-view columns or ``confidence == 0``, ``margin = confidence``.

    Returns a DataFrame indexed ``0..Q-1`` with columns
    ``view_id_raw:int32, view_second:int32, confidence:float32, margin:float32``.
    """
    view_ids = np.asarray(ref_view_ids, dtype=np.int64)
    q = match.shape[0]
    view_id_raw = np.full(q, -1, dtype=np.int32)
    view_second = np.full(q, -1, dtype=np.int32)
    confidence = np.zeros(q, dtype=np.float32)
    margin = np.zeros(q, dtype=np.float32)

    for i in range(q):
        row = match[i]
        best_col = int(np.argmax(row))
        conf = float(row[best_col])
        confidence[i] = conf
        if conf <= 0.0:
            # unassigned row: raw -1, second -1, margin = conf (0.0)
            margin[i] = conf
            continue
        best_view = int(view_ids[best_col])
        view_id_raw[i] = best_view
        other = view_ids != best_view
        if bool(other.any()):
            other_best = float(row[other].max())
            other_best_col = int(np.flatnonzero(other)[int(np.argmax(row[other]))])
            view_second[i] = int(view_ids[other_best_col])
            margin[i] = conf - other_best
        else:
            margin[i] = conf
    return pd.DataFrame(
        {
            "view_id_raw": view_id_raw,
            "view_second": view_second,
            "confidence": confidence,
            "margin": margin,
        }
    )


def smooth_view_sequence(
    view_ids: Sequence[int],
    *,
    window: int = DEFAULT_SMOOTH_WINDOW,
) -> NDArray[np.int_]:
    """Odd sliding-window majority (mode) vote centered on each index.

    The window is truncated at the sequence edges. ``window <= 1`` returns the input
    unchanged. On a tie for the mode the ORIGINAL center label is kept when it is among
    the tied winners, else the smallest winning label (deterministic). ``half = window //
    2``, so an even ``window`` behaves like the next odd size (e.g. 4 acts like 5).
    """
    seq = np.asarray(view_ids, dtype=np.int_)
    n = seq.shape[0]
    if window <= 1 or n == 0:
        return seq.copy()
    half = window // 2
    out = seq.copy()
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        counts = Counter(int(v) for v in seq[lo:hi])
        top = max(counts.values())
        winners = {v for v, c in counts.items() if c == top}
        # tie (or lone winner): keep the original center label when it is among winners,
        # else fall back to the smallest winning label (deterministic)
        out[i] = seq[i] if int(seq[i]) in winners else min(winners)
    return out


def assign_splits(
    manifest: pd.DataFrame,
    *,
    val_frac: float = DEFAULT_VAL_FRAC,
    policy: str = "per_view_tail",
    holdout_views: set[int] | None = None,
) -> pd.DataFrame:
    """Return a copy of ``manifest`` with an added ``split`` column ("train"/"val").

    - ``per_view_tail``: within each ``view_id`` group sorted by ``frame`` ascending,
      the last ``ceil(val_frac * n_group)`` rows -> "val", the rest -> "train".
    - ``holdout_views``: rows whose ``view_id`` is in ``holdout_views`` -> "val",
      else "train".

    Deterministic; the returned frame keeps the input row order (assigned by index).
    Note: a view with a single frame (``ceil(val_frac*1) == 1``) lands entirely in "val"
    with no train row — unavoidable for a 1-frame view and a non-issue under dense
    per-view sampling.
    """
    out = manifest.copy()
    split = pd.Series("train", index=out.index, dtype=object)

    if policy == "per_view_tail":
        for _view, group in out.groupby("view_id", sort=False):
            ordered = group.sort_values("frame", kind="stable")
            n_val = math.ceil(val_frac * len(ordered))
            if n_val > 0:
                val_idx = ordered.index[len(ordered) - n_val:]
                split.loc[val_idx] = "val"
    elif policy == "holdout_views":
        hold = holdout_views or set()
        split.loc[out["view_id"].isin(hold)] = "val"
    else:
        raise ValueError(f"unknown split policy: {policy!r}")

    out["split"] = split
    return out


def build_manifest(
    query_frames: Sequence[int],
    match: NDArray[np.float64],
    ref_view_ids: Sequence[int],
    keypoint_counts: Sequence[int],
    *,
    game: str,
    fps: float,
    n_boxes: Sequence[int] | None = None,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    val_frac: float = DEFAULT_VAL_FRAC,
    split_policy: str = "per_view_tail",
) -> pd.DataFrame:
    """Assemble the training manifest from precomputed cross-view match scores (pure).

    ``match`` is (Q, R) — each query frame vs each reference representative — with
    ``query_frames``, ``keypoint_counts`` (and optional ``n_boxes``) aligned to the Q
    rows and ``ref_view_ids`` aligned to the R columns. Each frame is assigned its
    nearest view (``assign_nearest_view``), the per-frame ``view_id_raw`` sequence is
    temporally smoothed (``smooth_view_sequence``) into ``view_id`` while the raw label
    is retained, and rows are split train/val (``assign_splits``).

    ``ambiguous`` flags frames whose across-view ``margin`` is below ``ambiguity_margin``.
    ``view_key`` and ``weight`` use the SMOOTHED ``view_id`` and the ``confidence``
    respectively. Rows are sorted by ``frame`` ascending BEFORE smoothing (so the
    temporal neighborhood is correct even for unsorted input) and index-reset, with a
    fixed column order matching the manifest schema. Deterministic; no video I/O.
    """
    # Sort every Q-aligned input by frame up front: smoothing is order-sensitive, so the
    # temporal neighborhood must be in frame order regardless of how the caller ordered rows.
    frame = np.asarray(query_frames, dtype=np.int64)
    order = np.argsort(frame, kind="stable")
    frame = frame[order]
    match = np.asarray(match, dtype=np.float64)[order]
    n_kp = np.asarray(keypoint_counts, dtype=np.int32)[order]
    boxes = (
        np.zeros(len(frame), dtype=np.int32)
        if n_boxes is None
        else np.asarray(n_boxes, dtype=np.int32)[order]
    )

    base = assign_nearest_view(match, ref_view_ids)
    view_id = smooth_view_sequence(
        base["view_id_raw"].tolist(), window=smooth_window
    ).astype(np.int32)
    conf = base["confidence"].to_numpy().astype(np.float32)
    margin = base["margin"].to_numpy()

    df = pd.DataFrame(
        {
            "game": pd.Series([game] * len(frame), dtype=object),
            "frame": frame,
            "t_seconds": frame.astype(np.float64) / fps,
            "view_id": view_id,
            "view_id_raw": base["view_id_raw"].to_numpy().astype(np.int32),
            "view_second": base["view_second"].to_numpy().astype(np.int32),
            "view_key": pd.Series([f"{game}:{v}" for v in view_id], dtype=object),
            "confidence": conf,
            "weight": conf,
            "margin": margin.astype(np.float32),
            "ambiguous": margin < ambiguity_margin,
            "n_keypoints": n_kp,
            "n_boxes": boxes,
        }
    )

    df = assign_splits(df, val_frac=val_frac, policy=split_policy)
    df = df.sort_values("frame", kind="stable").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Streaming video orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewAssignment:
    """A video's densely-sampled per-frame view assignment plus provenance.

    manifest:        the per-query-frame training manifest (see build_manifest).
    representatives: view id -> the medoid frame index labelled for that view.
    meta:            JSON-serializable provenance (schema, video probe, params, digest)
                     that the Task-5 sidecar writer consumes.
    """

    manifest: pd.DataFrame
    representatives: dict[int, int]
    meta: dict[str, Any]

    @property
    def n_frames(self) -> int:
        return len(self.manifest)

    @property
    def n_views(self) -> int:
        # exclude the -1 "unassigned" sentinel so it is not counted as a real view
        vid = self.manifest["view_id"]
        return int(vid[vid != -1].nunique())

    @property
    def n_ambiguous(self) -> int:
        return int(self.manifest["ambiguous"].sum())

    @property
    def switch_rate(self) -> float:
        # fraction of consecutive (frame-sorted) rows where view_id changes
        v = self.manifest.sort_values("frame")["view_id"].to_numpy()
        if len(v) < 2:
            return 0.0
        return float((v[1:] != v[:-1]).mean())

    def materialize(
        self,
        out_dir: str | Path,
        *,
        video_path: str | Path,
        image_downscale: float = 0.5,
        jpeg_quality: int = 90,
        boxes: pd.DataFrame | None = None,
    ) -> list[Path]:
        """Write a portable ImageFolder tree by re-decoding the manifest's frames.

        Escape hatch off the manifest-first default: for every manifest row, decode the
        frame (optionally player-masked + downscaled) and write a JPEG under
        ``out_dir/frames/<split>/view_XX/<game>_f<frame>.jpg``; when ``boxes`` is given
        the keep-mask is also written under ``out_dir/masks/<split>/view_XX/...png``.
        Returns the list of written IMAGE paths (all under ``out_dir``). Pure
        re-materialization: no scoring, deterministic from the manifest + video.

        The reader is given ``expected_video_hash`` explicitly (a fresh in-memory
        manifest may carry no ``video_hash`` attr) so the wrong-video guard still fires.
        """
        out_dir = Path(out_dir)
        reader = ViewFrameReader(
            video_path, self.manifest, boxes=boxes, downscale=image_downscale,
            expected_video_hash=self.meta["video"]["video_hash"])
        written: list[Path] = []
        try:
            for i in range(len(reader)):
                img, keep_mask, row = reader.read(i)
                split = str(row["split"])
                view_id = int(row["view_id"])
                game = str(row["game"])
                frame_idx = int(row["frame"])
                view_dir = f"view_{view_id:02d}"
                stem = f"{game}_f{frame_idx:06d}"
                img_path = out_dir / "frames" / split / view_dir / f"{stem}.jpg"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                written.append(img_path)
                if keep_mask is not None:
                    mask_path = out_dir / "masks" / split / view_dir / f"{stem}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(mask_path), keep_mask)
        finally:
            reader.close()
        return written


def _reps_fingerprint(rep_frames: Sequence[int], ref_view_ids: Sequence[int]) -> str:
    """Short stable hash of the representative frame indices + their view ids."""
    payload = repr((tuple(int(f) for f in rep_frames), tuple(int(v) for v in ref_view_ids)))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _viewassign_cache_path(
    video_path: Path,
    cache_dir: Path,
    assign_stride: int,
    n_features: int,
    downscale: float,
    min_match_dist: int,
    min_keypoints: int,
    reps_fingerprint: str,
) -> Path:
    # Key on the video CONTENT hash (path+size+mtime, like compute_view_digest) + scoring
    # params + reps, so a re-encode at the same path AUTO-INVALIDATES the cache instead of
    # serving a stale match matrix into training data.
    key = (
        f"{_video_hash(video_path)}|{assign_stride}|{n_features}|{downscale}"
        f"|{min_match_dist}|{min_keypoints}|{reps_fingerprint}"
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return cache_dir / (
        f"viewassign_{digest}_s{assign_stride}_n{n_features}_d{downscale}"
        f"_m{min_match_dist}_k{min_keypoints}_{reps_fingerprint}.npz"
    )


def build_view_assignment(
    video_path: str | Path,
    digest: ViewDigest,
    *,
    game: str | None = None,
    assign_stride: int = DEFAULT_ASSIGN_STRIDE,
    n_features: int = DEFAULT_N_FEATURES,
    downscale: float = DEFAULT_DOWNSCALE,
    min_match_dist: int = DEFAULT_MIN_MATCH_DIST,
    min_keypoints: int = DEFAULT_MIN_KEYPOINTS,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    val_frac: float = DEFAULT_VAL_FRAC,
    split_policy: str = "per_view_tail",
    player_boxes: pd.DataFrame | None = None,
    chunk: int = DEFAULT_CHUNK,
    cache_dir: str | Path | None = None,
) -> ViewAssignment:
    """Densely sample a video and assign each frame its nearest digest VIEW.

    Streams the query frames in ``chunk``-sized batches (decode -> optional player
    masking -> ORB descriptors -> rectangular match vs the digest representatives),
    discarding pixels between chunks so memory stays flat over a full game. The
    (expensive) (Q, R) match block is cached in an ``.npz`` keyed by the resolved video
    path + scoring params + a fingerprint of the representatives (NOT file content),
    so a re-run is instant and survives an incidental content change. Player masking
    bypasses the cache (the match depends on the masks). Returns a ``ViewAssignment``
    wrapping the manifest, the labelled representatives, and JSON-safe provenance.
    """
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()

    query_frames = list(range(0, max(n_frames, 1), assign_stride))

    rep_items = sorted(digest.representatives.items())
    ref_view_ids = [v for v, _ in rep_items]
    rep_frames = [f for _, f in rep_items]

    reps_fingerprint = _reps_fingerprint(rep_frames, ref_view_ids)
    cache_dir = Path(cache_dir or (video_path.parent / ".sv_labeler_cache"))
    cache_path = _viewassign_cache_path(
        video_path, cache_dir, assign_stride, n_features, downscale,
        min_match_dist, min_keypoints, reps_fingerprint)

    match: NDArray[np.float64] | None = None
    keypoint_counts: NDArray[np.int32] = np.zeros(0, dtype=np.int32)

    if player_boxes is None and cache_path.exists():
        data = np.load(cache_path)
        # Belt-and-suspenders (like compute_view_digest): the content-hashed key already
        # guarantees same pixels, but re-verify the sampled frames before trusting it.
        if data["query_frames"].tolist() == query_frames:
            match = np.asarray(data["match"], dtype=np.float64)
            keypoint_counts = np.asarray(data["keypoint_counts"], dtype=np.int32)

    if match is None:
        # Decode representatives only on a cache miss (the fingerprint above needs just
        # their frame indices + view ids, not their pixels).
        rep_imgs = _read_frames(video_path, rep_frames)
        rep_desc, rep_counts = frame_descriptors(
            rep_imgs, n_features=n_features, downscale=downscale)
        match_rows: list[NDArray[np.float64]] = []
        kp_all: list[int] = []
        # One persistent capture, forward-read across ALL chunks. query_frames is ascending
        # (a strided range), so reads never rewind — O(n_frames) grabs total, not O(chunks x
        # chunk_start). Only one chunk of pixels is held at a time (bounded memory).
        cap = cv2.VideoCapture(str(video_path))
        pos = 0
        try:
            for start in range(0, len(query_frames), chunk):
                chunk_frames = query_frames[start:start + chunk]
                imgs: list[NDArray[np.uint8] | None] = []
                for idx in chunk_frames:
                    if idx < pos:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        pos = idx
                    while pos < idx:
                        if not cap.grab():
                            break
                        pos += 1
                    ok, frame = cap.read()
                    pos += 1
                    imgs.append(frame if ok else None)  # type: ignore[arg-type]
                masks = (_build_masks(player_boxes, chunk_frames, (width, height))
                         if player_boxes is not None else None)
                descs, counts = frame_descriptors(
                    imgs, n_features=n_features, downscale=downscale, masks=masks)
                m = cross_match_fractions(
                    descs, counts, rep_desc, rep_counts,
                    min_match_dist=min_match_dist, min_keypoints=min_keypoints)
                match_rows.append(m)
                kp_all.extend(counts)
        finally:
            cap.release()
        match = (np.concatenate(match_rows, axis=0) if match_rows
                 else np.zeros((0, len(ref_view_ids)), dtype=np.float64))
        keypoint_counts = np.asarray(kp_all, dtype=np.int32)
        if player_boxes is None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                query_frames=np.array(query_frames, dtype=np.int64),
                match=match,
                keypoint_counts=keypoint_counts,
                ref_view_ids=np.array(ref_view_ids, dtype=np.int64),
                video_hash=_video_hash(video_path))

    if player_boxes is not None:
        counts_by_frame = player_boxes.groupby("frame").size()
        n_boxes: list[int] | None = [int(counts_by_frame.get(f, 0)) for f in query_frames]
    else:
        n_boxes = None

    manifest = build_manifest(
        query_frames, match, ref_view_ids, keypoint_counts.tolist(),
        game=game or video_path.stem, fps=fps, n_boxes=n_boxes,
        ambiguity_margin=ambiguity_margin, smooth_window=smooth_window,
        val_frac=val_frac, split_policy=split_policy)

    # A cache hit means identical pixels (content-hashed key), so the live hash is correct.
    provenance_hash = _video_hash(video_path)
    representatives = {int(v): int(f) for v, f in digest.representatives.items()}
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "video": {
            "path": str(video_path),
            "abspath": str(video_path.resolve()),
            "video_hash": provenance_hash,
            "n_frames": int(n_frames),
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
        },
        "params": {
            "assign_stride": int(assign_stride),
            "n_features": int(n_features),
            "downscale": float(downscale),
            "min_match_dist": int(min_match_dist),
            "min_keypoints": int(min_keypoints),
            "ambiguity_margin": float(ambiguity_margin),
            "smooth_window": int(smooth_window),
            "val_frac": float(val_frac),
            "split_policy": str(split_policy),
            "chunk": int(chunk),
        },
        "digest": {
            "n_views": int(digest.n_views),
            "representatives": representatives,
        },
        "boxes_source": None,
    }

    return ViewAssignment(manifest=manifest, representatives=representatives, meta=meta)


# ---------------------------------------------------------------------------
# Persist + Colab reader
# ---------------------------------------------------------------------------


def write_export(
    assignment: ViewAssignment,
    out_dir: str | Path,
    *,
    video_path: str | Path,
) -> list[Path]:
    """Persist the manifest-first export: a parquet manifest + a self-describing JSON.

    Writes no image files (that is ``materialize``'s job). The sidecar is a copy of
    ``assignment.meta`` augmented with a ``dataset_fingerprint`` (a stable hash of the
    scoring params) and a ``stats`` block computed from the manifest, all cast to plain
    Python / JSON-safe types. Returns ``[parquet_path, json_path]``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = assignment.manifest

    stats: dict[str, Any] = {
        "n_frames": len(m),
        "per_view_counts": {
            str(int(cast(int, v))): int(c)
            for v, c in m["view_id"].value_counts().sort_index().items()
        },
        "n_ambiguous": int(m["ambiguous"].sum()),
        "n_train": int((m["split"] == "train").sum()),
        "n_val": int((m["split"] == "val").sum()),
        "confidence_median": float(m["confidence"].median()) if len(m) else 0.0,
        "margin_median": float(m["margin"].median()) if len(m) else 0.0,
        "switch_rate": float(assignment.switch_rate),
    }
    dataset_fingerprint = hashlib.sha1(
        json.dumps(assignment.meta["params"], sort_keys=True).encode()).hexdigest()[:16]

    sidecar = dict(assignment.meta)
    sidecar["dataset_fingerprint"] = dataset_fingerprint
    sidecar["stats"] = stats

    parquet_path = out_dir / "view_dataset.parquet"
    json_path = out_dir / "view_dataset.json"
    m.to_parquet(parquet_path, index=False)
    json_path.write_text(json.dumps(sidecar, indent=2))
    return [parquet_path, json_path]


def load_manifest(export_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read back a ``write_export`` bundle -> ``(manifest, meta)``.

    Attaches the sidecar's ``video_hash`` to ``df.attrs`` so a downstream
    ``ViewFrameReader`` can verify it decodes the RIGHT video for this manifest.
    """
    export_dir = Path(export_dir)
    df = pd.read_parquet(export_dir / "view_dataset.parquet")
    meta: dict[str, Any] = json.loads((export_dir / "view_dataset.json").read_text())
    df.attrs["video_hash"] = meta["video"]["video_hash"]
    return df, meta


class ViewFrameReader:
    """Colab-side on-demand decoder — yields ``(frame, keep_mask, row)`` per manifest row.

    Keeps the manifest-first export small: instead of shipping images, the trainer opens
    the same ALL-INTRA mp4 and decodes each frame as needed. Guards against decoding the
    WRONG video (a re-encode / different clip) by comparing ``_video_hash`` to the
    expected hash (from ``expected_video_hash`` or ``manifest.attrs['video_hash']``).
    """

    def __init__(
        self,
        video_path: str | Path,
        manifest: pd.DataFrame,
        *,
        boxes: pd.DataFrame | None = None,
        downscale: float = 1.0,
        expected_video_hash: str | None = None,
    ) -> None:
        # Guard against decoding the WRONG video for a manifest. The expected hash comes
        # from the caller or from manifest.attrs (set by load_manifest).
        expected = expected_video_hash or manifest.attrs.get("video_hash")
        if expected is not None and _video_hash(Path(video_path)) != expected:
            raise ValueError(
                "video_hash mismatch: this manifest was built for a different video")
        self._manifest = manifest.reset_index(drop=True)
        self._boxes = boxes
        self._downscale = downscale
        self._cap = cv2.VideoCapture(str(video_path))
        self._pos = 0
        # (width,height) for masks: probe from the capture
        self._size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920,
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080,
        )

    def __enter__(self) -> ViewFrameReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._manifest)

    def read(
        self, i: int,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8] | None, pd.Series]:
        row = self._manifest.iloc[i]
        frame_idx = int(row["frame"])
        # forward-grab fast path + seek fallback (mirror view_digest._read_frames)
        if frame_idx < self._pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self._pos = frame_idx
        while self._pos < frame_idx:
            if not self._cap.grab():
                break
            self._pos += 1
        ok, frame = self._cap.read()
        self._pos += 1
        if not ok:
            raise IndexError(f"could not decode frame {frame_idx}")
        keep_mask: NDArray[np.uint8] | None = None
        if self._boxes is not None:
            masks = _build_masks(self._boxes, [frame_idx], self._size)
            keep_mask = masks[0]
        if self._downscale != 1.0:
            frame = cv2.resize(frame, None, fx=self._downscale, fy=self._downscale)
            if keep_mask is not None:
                keep_mask = cv2.resize(  # type: ignore[assignment]
                    keep_mask, (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
        return frame, keep_mask, row  # type: ignore[return-value]

    def close(self) -> None:
        self._cap.release()


# --- Using the export in Colab ---------------------------------------------
# from soccer_vision.labeler.view_dataset import load_manifest, ViewFrameReader
# m, meta = load_manifest("view_dataset_out/")
# with ViewFrameReader("oceanside_clip.mp4", m,
#                       expected_video_hash=meta["video"]["video_hash"]) as reader:
#     # contrastive: positives share row.view_key (or view_id); loss weight = row.weight
#     # MAE: frame, keep_mask, _row = reader.read(i); mask players via keep_mask when present
#     frame, keep_mask, row = reader.read(0)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m soccer_vision.labeler.view_dataset",
        description="Export a video + its ViewDigest into a manifest-first view dataset.")
    p.add_argument("--video", type=Path, required=True,
                   help="input mp4 (ideally ALL-INTRA)")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: <video>.parent/view_dataset_out)")
    p.add_argument("--game", type=str, default=None,
                   help="game/clip id for view_key (default: video stem)")
    p.add_argument("--digest-json", type=Path, default=None,
                   help="reuse a render_digest view_digest.json instead of recomputing")
    p.add_argument("--stride", type=int, default=25,
                   help="DIGEST sampling stride for compute_view_digest")
    p.add_argument("--dist-threshold", type=float, default=0.5,
                   help="clustering distance threshold for compute_view_digest")
    p.add_argument("--assign-stride", type=int, default=DEFAULT_ASSIGN_STRIDE,
                   help="dense per-frame assignment stride")
    p.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW,
                   help="temporal majority-vote window for view smoothing")
    p.add_argument("--ambiguity-margin", type=float, default=DEFAULT_AMBIGUITY_MARGIN,
                   help="flag frames whose across-view margin is below this")
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                   help="per-view temporal-tail validation fraction")
    p.add_argument("--boxes", type=Path, default=None,
                   help="optional player trajectories parquet (enables masking)")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="dir for the .npz similarity/match caches")
    p.add_argument("--materialize", action="store_true",
                   help="also write a portable ImageFolder tree of decoded frames")
    p.add_argument("--image-downscale", type=float, default=0.5,
                   help="downscale factor for materialized JPEGs")
    p.add_argument("--jpeg-quality", type=int, default=90,
                   help="JPEG quality for materialized frames")
    return p


def _digest_from_json(path: Path) -> ViewDigest:
    """Rebuild a ViewDigest from a render_digest ``view_digest.json``.

    Only ``.representatives`` (and the derived ``.n_views``) are needed by the exporter,
    so ``similarity`` is a zero placeholder and ``view_of`` is restored for completeness.
    """
    data = json.loads(path.read_text())
    representatives = {int(k): int(v) for k, v in data["representatives"].items()}
    view_of = {int(k): int(v) for k, v in data.get("view_of", {}).items()}
    sample_frames = [int(f) for f in data.get("sample_frames", sorted(view_of))]
    n = len(sample_frames)
    return ViewDigest(
        sample_frames=sample_frames, view_of=view_of, representatives=representatives,
        similarity=np.zeros((n, n), dtype=np.float64))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: video (+ optional digest json / boxes) -> manifest-first export."""
    args = _build_arg_parser().parse_args(argv)
    video: Path = args.video
    out: Path = args.out or (video.parent / "view_dataset_out")

    if args.digest_json is not None:
        digest = _digest_from_json(args.digest_json)
    else:
        digest = compute_view_digest(
            video, stride=args.stride, dist_threshold=args.dist_threshold,
            cache_dir=args.cache_dir)

    player_boxes = pd.read_parquet(args.boxes) if args.boxes else None

    va = build_view_assignment(
        video, digest, game=args.game, assign_stride=args.assign_stride,
        smooth_window=args.smooth_window, ambiguity_margin=args.ambiguity_margin,
        val_frac=args.val_frac, player_boxes=player_boxes, cache_dir=args.cache_dir)

    write_export(va, out, video_path=video)
    if args.materialize:
        va.materialize(
            out / "frames_export", video_path=video,
            image_downscale=args.image_downscale, jpeg_quality=args.jpeg_quality,
            boxes=player_boxes)

    split_counts = va.manifest["split"].value_counts()
    n_train = int(split_counts.get("train", 0))
    n_val = int(split_counts.get("val", 0))
    print(
        f"view-dataset export -> {out}\n"
        f"  n_frames={va.n_frames}  n_views={va.n_views}  "
        f"switch_rate={va.switch_rate:.4f}  n_ambiguous={va.n_ambiguous}\n"
        f"  train={n_train}  val={n_val}")


if __name__ == "__main__":
    main()
