"""Tests for drift-free view-registration calibration (Slice 2)."""
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from soccer_vision.pitch.view_registration import compose_pitch_homography, register_to_best_rep

H, W = 240, 320


def _pattern(seed: int) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 30, np.uint8)
    for _ in range(60):
        x1, x2 = sorted(rng.integers(0, W, size=2).tolist())
        y1, y2 = sorted(rng.integers(0, H, size=2).tolist())
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        else:
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
    return img


def _orb(img):
    o = cv2.ORB_create(3000)  # type: ignore[attr-defined]
    return o.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)


def test_compose_pitch_homography() -> None:
    H_rep = np.array([[2.0, 0, 1], [0, 3.0, 2], [0, 0, 1]])   # rep_px -> pitch
    G = np.array([[1.0, 0, 5], [0, 1.0, 7], [0, 0, 1]])        # frame_px -> rep_px
    Hf = compose_pitch_homography(H_rep, G)
    p = Hf @ np.array([0, 0, 1.0])
    p = p[:2] / p[2]
    assert np.allclose(p, [11.0, 23.0])
    assert abs(Hf[2, 2] - 1.0) < 1e-9


def test_register_to_best_rep_picks_matching_view() -> None:
    rep_a = _pattern(1)
    rep_b = _pattern(99)
    M = np.array([[1, 0, 8.0], [0, 1, -5.0], [0, 0, 1]])
    frame = cv2.warpPerspective(rep_a, M, (W, H))
    fkp, fdesc = _orb(frame)
    akp, adesc = _orb(rep_a)
    bkp, bdesc = _orb(rep_b)
    out = register_to_best_rep(fkp, fdesc, [akp, bkp], [adesc, bdesc], min_inliers=12)
    assert out is not None
    idx, G, n_in = out
    assert idx == 0 and n_in >= 12
    p = G @ np.array([8.0, 0.0, 1.0])
    p = p[:2] / p[2]
    assert np.linalg.norm(p - np.array([0.0, 5.0])) < 3.0


def test_register_to_best_rep_no_match_returns_none() -> None:
    frame = _pattern(3)
    fkp, fdesc = _orb(frame)
    blank = np.full((H, W, 3), 30, np.uint8)
    bkp, bdesc = _orb(blank)
    assert register_to_best_rep(fkp, fdesc, [bkp], [bdesc], min_inliers=12) is None
