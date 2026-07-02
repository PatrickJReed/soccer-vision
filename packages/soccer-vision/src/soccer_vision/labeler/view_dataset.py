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

import numpy as np
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
