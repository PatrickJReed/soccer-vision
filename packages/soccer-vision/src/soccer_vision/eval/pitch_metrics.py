"""Pure metrics for scoring the pitch-keypoint model against labeler ground truth.

Everything here is numpy in / dataclass out, no I/O — so the eval logic itself is
unit-testable (the lesson of anchor_cov, which shipped untested and gave a false
pass). Errors are reported in real-world FEET: both pitch axes are fractions of
pitch length, so a canonical Euclidean distance scales by one constant.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

# Nominal US Soccer 9v9 pitch length: ~68.5 m. Youth fields vary; this is a fixed
# nominal scale so feet errors are interpretable and comparable across retrains.
DEFAULT_PITCH_LENGTH_FT: float = 224.7
HIDDEN_IDX: int = 5  # under-camera landmark, never ground-truth-visible


def canonical_to_feet(
    distance: float | NDArray[np.floating],
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
) -> float | NDArray[np.floating]:
    """Convert a canonical-pitch distance (fraction of length) to feet."""
    return distance * length_ft


def _apply_h(h: NDArray[np.floating], pts_px: NDArray[np.floating]) -> NDArray[np.float64]:
    """Map (N,2) image pixels -> (N,2) pitch coords through image->pitch H."""
    pts = np.asarray(pts_px, dtype=np.float64)
    homog = np.column_stack([pts, np.ones(len(pts))])
    proj = homog @ np.asarray(h, dtype=np.float64).T
    return np.asarray(proj[:, :2] / proj[:, 2:3], dtype=np.float64)


def keypoint_errors_feet(
    h_gt: NDArray[np.floating],
    gt_kpts: NDArray[np.floating],
    model_kpts: NDArray[np.floating],
    *,
    conf_thr: float = 0.5,
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
) -> dict[int, float]:
    """Per-landmark feet error: map the model's predicted pixel through the GT
    homography into pitch space, compare to the canonical landmark.

    gt_kpts: (21,3) px + visibility (from project_landmarks). model_kpts: (21,3)
    px + confidence. Scores landmarks that are GT-visible, not the hidden idx,
    and predicted with conf >= conf_thr.
    """
    out: dict[int, float] = {}
    for i in range(len(PITCH_LANDMARKS)):
        if i == HIDDEN_IDX or gt_kpts[i, 2] <= 0 or model_kpts[i, 2] < conf_thr:
            continue
        pitch_pred = _apply_h(h_gt, model_kpts[i:i + 1, :2])[0]
        d = float(np.hypot(*(pitch_pred - PITCH_LANDMARKS[i])))
        out[i] = float(canonical_to_feet(d, length_ft))
    return out
