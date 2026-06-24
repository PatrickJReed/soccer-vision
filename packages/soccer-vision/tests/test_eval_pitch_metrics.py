"""Synthetic-ground-truth tests for the pitch-model eval metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from soccer_vision.eval.pitch_metrics import (
    DEFAULT_PITCH_LENGTH_FT,
    canonical_to_feet,
    keypoint_errors_feet,
)
from soccer_vision.pitch.autolabel import project_landmarks
from soccer_vision.pitch.homography import fit_homography
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS


def test_canonical_to_feet_scalar() -> None:
    # both pitch axes are fractions of length, so a 0.1 canonical distance is
    # 0.1 * length_ft feet.
    assert canonical_to_feet(0.1) == DEFAULT_PITCH_LENGTH_FT * 0.1


def test_canonical_to_feet_array() -> None:
    out = canonical_to_feet(np.array([0.0, 0.5, 1.0]), length_ft=200.0)
    assert np.allclose(out, [0.0, 100.0, 200.0])


_W, _H = 1920, 1080


def _gt_homography() -> np.ndarray:
    # map the full frame to the pitch region [0,0.6]x[0,1.0]; image->pitch.
    img = np.array([[0, 0], [_W, 0], [_W, _H], [0, _H]], dtype=float)
    pitch = np.array([[0.0, 0.0], [0.6, 0.0], [0.6, 1.0], [0.0, 1.0]])
    return fit_homography(img, pitch)


def _pitch_to_px(h_gt: np.ndarray, pt: np.ndarray) -> NDArray[np.float64]:
    inv: NDArray[np.float64] = np.linalg.inv(h_gt)
    v: NDArray[np.float64] = inv @ np.array([pt[0], pt[1], 1.0])
    return np.asarray(v[:2] / v[2], dtype=np.float64)


def test_keypoint_errors_feet_known_offset() -> None:
    h_gt = _gt_homography()
    gt = project_landmarks(h_gt, PITCH_LANDMARKS, (_W, _H))  # (21,3) px+vis
    # model = perfect, except landmark 0 nudged by a known pitch offset.
    # NOTE: plan used j=3 but corner_opp_right=(1.0,1.0) is outside [0,0.6]x
    # frame coverage -> gt_kpts[3,2]==0 -> KeyError. Use j=0 (corner_own_left,
    # pitch=(0,0), visible and offset to (0.02,0) stays in frame).
    model = gt.copy()
    model[:, 2] = 2.0  # treat 'visible' column as confidence >= thr
    offset = np.array([0.02, 0.0])  # canonical -> 0.02 * length feet
    j = 0
    model[j, :2] = _pitch_to_px(h_gt, PITCH_LANDMARKS[j] + offset)
    errs = keypoint_errors_feet(h_gt, gt, model, conf_thr=0.5)
    assert abs(errs[j] - 0.02 * 224.7) < 0.1
    for i, v in errs.items():
        if i != j:
            assert v < 0.01  # perfect elsewhere


def test_keypoint_errors_feet_skips_hidden_and_lowconf() -> None:
    h_gt = _gt_homography()
    gt = project_landmarks(h_gt, PITCH_LANDMARKS, (_W, _H))
    model = gt.copy()
    model[:, 2] = 2.0
    model[7, 2] = 0.1  # below conf threshold -> excluded
    errs = keypoint_errors_feet(h_gt, gt, model, conf_thr=0.5)
    assert 5 not in errs  # hidden idx never scored
    assert 7 not in errs  # low confidence excluded
