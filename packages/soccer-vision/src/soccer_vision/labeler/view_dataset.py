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

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.labeler.view_digest import (
    DEFAULT_MIN_KEYPOINTS,
    DEFAULT_MIN_MATCH_DIST,
    _Descriptor,
    _pair_match_fraction,
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
    respectively. The result is sorted by ``frame`` ascending and index-reset, with a
    fixed column order matching the manifest schema. Deterministic; no video I/O.
    """
    base = assign_nearest_view(match, ref_view_ids)
    view_id = smooth_view_sequence(
        base["view_id_raw"].tolist(), window=smooth_window
    ).astype(np.int32)

    frame = np.asarray(query_frames, dtype=np.int64)
    n_kp = np.asarray(keypoint_counts, dtype=np.int32)
    boxes = (
        np.zeros(len(frame), dtype=np.int32)
        if n_boxes is None
        else np.asarray(n_boxes, dtype=np.int32)
    )
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
            "confidence": base["confidence"].to_numpy().astype(np.float32),
            "weight": base["confidence"].to_numpy().astype(np.float32),
            "margin": margin.astype(np.float32),
            "ambiguous": margin < ambiguity_margin,
            "n_keypoints": n_kp,
            "n_boxes": boxes,
        }
    )

    df = assign_splits(df, val_frac=val_frac, policy=split_policy)
    df = df.sort_values("frame", kind="stable").reset_index(drop=True)
    return df
