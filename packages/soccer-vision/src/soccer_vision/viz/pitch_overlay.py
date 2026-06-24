"""Reproject a fitted pitch homography back onto a frame, to eyeball accuracy.

Coverage (anchor_cov) only asks whether enough confident keypoints fit *a*
homography; it does not say the homography is *accurate*. This overlay closes
that gap visually: draw the model's detected keypoints (dots) and the pitch
lines reprojected through the inverse homography (image = H^-1 · pitch). If the
green dots sit on the markings and the orange lines trace the painted pitch, the
homography is accurate — not just present.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.pitch.homography import HomographyError, fit_homography
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

# canonical pitch edges (landmark index pairs) — the set the labeler overlays.
_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 3), (3, 2), (2, 0), (4, 6), (9, 10), (11, 12), (9, 11),
    (10, 12), (13, 14), (15, 16), (13, 15), (14, 16), (17, 18), (19, 20),
)


def reproject_landmarks(
    image_points: NDArray[np.floating],
    kp_indices: NDArray[np.integer],
) -> NDArray[np.floating] | None:
    """Fit image->pitch H from the correspondences and reproject ALL canonical
    landmarks back to image pixels (via H^-1). Returns a (21, 2) px array, or
    None if fewer than 4 points or the fit fails.
    """
    pts = np.asarray(image_points, dtype=np.float64)
    idx = np.asarray(kp_indices)
    if len(pts) < 4:
        return None
    try:
        h = fit_homography(pts, PITCH_LANDMARKS[idx])
    except HomographyError:
        return None
    h_inv = np.linalg.inv(h)
    canon = np.column_stack([PITCH_LANDMARKS, np.ones(len(PITCH_LANDMARKS))])
    proj = canon @ h_inv.T
    return np.asarray(proj[:, :2] / proj[:, 2:3], dtype=np.float64)


def draw_reprojected_pitch(
    frame: NDArray[np.uint8],
    image_points: NDArray[np.floating],
    kp_indices: NDArray[np.integer],
) -> tuple[NDArray[np.uint8], bool]:
    """Draw detected keypoints (green dots) and, if a homography fits, the
    reprojected pitch lines (orange). Returns (annotated_copy, fit_ok). Marker
    sizes scale with frame width so they survive contact-sheet downscaling.
    """
    out = np.array(frame, dtype=np.uint8).copy()
    w = out.shape[1]
    radius = max(4, w // 240)
    thick = max(2, w // 640)

    for (x, y), _idx in zip(image_points, kp_indices, strict=True):
        cv2.circle(out, (int(x), int(y)), radius, (60, 220, 120), -1)
        cv2.circle(out, (int(x), int(y)), radius, (0, 0, 0), 1)

    landmarks = reproject_landmarks(image_points, kp_indices)
    if landmarks is None:
        return out, False
    for a, b in _EDGES:
        pa = (int(landmarks[a][0]), int(landmarks[a][1]))
        pb = (int(landmarks[b][0]), int(landmarks[b][1]))
        cv2.line(out, pa, pb, (40, 200, 255), thick)
    return out, True
