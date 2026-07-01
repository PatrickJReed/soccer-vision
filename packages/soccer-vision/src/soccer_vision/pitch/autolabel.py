"""Project canonical pitch landmarks into image pixels to seed keypoint labels.

DEFERRED ML — the Phase 3.5b active-learning loop is deferred. NOTE project_landmarks
itself is NOT orphaned: it is used LIVE by eval (eval/pitch_metrics.score_frame) and by
viz/pitch_overlay — do not delete it.

The active-learning loop (Phase 3.5b) runs the current pitch model to get sparse
anchors, propagates homographies into neighboring frames (pitch/propagation.py),
then uses this module to project the canonical landmarks BACK into each frame as
proposed keypoint labels for human correction in Roboflow.

Homographies map image pixels -> pitch [0,1]^2 (as produced by
build_frame_homographies / propagate_homographies); projecting landmarks INTO the
image therefore applies the inverse. Pure: no model or video I/O.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from soccer_vision.pitch.propagation import HomographyEntry


def project_landmarks(
    H: NDArray[np.floating],
    landmarks: NDArray[np.floating],
    frame_size: tuple[int, int],
) -> NDArray[np.float64]:
    """Project canonical landmarks (pitch [0,1]^2) into image pixels via inv(H).

    Parameters
    ----------
    H
        3x3 homography mapping image pixels -> pitch coords.
    landmarks
        (N, 2) canonical pitch coordinates.
    frame_size
        (width, height) in pixels.

    Returns
    -------
    (N, 3) array of (x_px, y_px, visible). visible = 2.0 if the projected point
    lands inside the frame, else 0.0 (and x_px, y_px are zeroed). A singular or
    non-invertible H yields all-invisible rows.
    """
    width, height = frame_size
    n = len(landmarks)
    out = np.zeros((n, 3), dtype=np.float64)
    try:
        h_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    except np.linalg.LinAlgError:
        return out
    homog = np.column_stack([landmarks, np.ones(n)])
    proj = (h_inv @ homog.T).T
    w = proj[:, 2]
    valid = np.abs(w) > 1e-9
    px = np.zeros(n)
    py = np.zeros(n)
    px[valid] = proj[valid, 0] / w[valid]
    py[valid] = proj[valid, 1] / w[valid]
    in_frame = valid & (px >= 0) & (px <= width) & (py >= 0) & (py <= height)
    out[in_frame, 0] = px[in_frame]
    out[in_frame, 1] = py[in_frame]
    out[in_frame, 2] = 2.0
    return out


def propose_labels(
    homographies: Mapping[int, HomographyEntry],
    landmarks: NDArray[np.floating],
    frame_size: tuple[int, int],
    *,
    min_confidence: float = 0.5,
) -> dict[int, NDArray[np.float64]]:
    """Project landmarks for every frame whose homography confidence is high enough.

    Anchors have confidence 1.0; propagated frames carry a runtime estimate. Frames
    below min_confidence are dropped (their proposals would be too noisy to correct
    cheaply). Returns {frame: (N, 3) projected keypoints}.

    Boundary semantics: frames where ``confidence >= min_confidence`` are kept
    (equal-to-threshold is included); frames where ``confidence < min_confidence``
    are dropped.
    """
    out: dict[int, NDArray[np.float64]] = {}
    for frame, entry in homographies.items():
        if entry.confidence < min_confidence:
            continue
        out[frame] = project_landmarks(entry.H, landmarks, frame_size)
    return out


def to_yolo_pose_line(
    keypoints: NDArray[np.floating],
    frame_size: tuple[int, int],
    *,
    class_id: int = 0,
    bbox_pad: float = 0.01,
) -> str:
    """Format one YOLO-pose label line for a single pitch instance.

    Layout: ``class cx cy w h  x1 y1 v1  ... xN yN vN`` with all coords normalized
    to [0,1] by frame_size. The bounding box is the tight box around the visible
    keypoints (padded by bbox_pad, clamped to the frame). Invisible keypoints are
    written as ``0 0 0``. If no keypoints are visible the box covers the full frame.

    Visibility convention: keypoints from ``project_landmarks`` carry visibility 0
    (absent/out-of-frame) or 2 (visible in-frame). Any positive visibility value is
    passed through to the label line as-is (YOLO/COCO: 1 = occluded, 2 = visible);
    only visibility 0 produces the ``0 0 0`` placeholder.
    """
    width, height = frame_size
    vis = keypoints[:, 2] > 0
    if vis.any():
        xs = keypoints[vis, 0]
        ys = keypoints[vis, 1]
        x1 = max(0.0, xs.min() / width - bbox_pad)
        y1 = max(0.0, ys.min() / height - bbox_pad)
        x2 = min(1.0, xs.max() / width + bbox_pad)
        y2 = min(1.0, ys.max() / height + bbox_pad)
    else:
        x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1
    tokens: list[str] = [str(class_id), _f(cx), _f(cy), _f(bw), _f(bh)]
    for row in keypoints:
        v = float(row[2])
        if v > 0:
            tokens += [_f(float(row[0]) / width), _f(float(row[1]) / height), str(int(v))]
        else:
            tokens += ["0", "0", "0"]
    return " ".join(tokens)


def _f(value: float) -> str:
    """Compact fixed-precision float for label files."""
    return f"{value:.6f}"
