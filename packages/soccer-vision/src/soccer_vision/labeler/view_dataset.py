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
