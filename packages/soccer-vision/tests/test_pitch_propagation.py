"""Tests for the homography-propagation leaf helpers."""

from __future__ import annotations

import cv2
import numpy as np
from soccer_vision.pitch.propagation import (
    HomographyEntry,
    blend_homographies,
    disagreement_confidence,
    register,
)


def _textured_image(seed: int = 0) -> np.ndarray:
    """A BGR image with strong, repeatable corner features for ORB."""
    rng = np.random.default_rng(seed)
    gray = (rng.random((400, 600)) * 60).astype(np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(20, 560)), int(rng.integers(20, 360))
        cv2.rectangle(gray, (x, y), (x + 18, y + 18), int(rng.integers(80, 255)), -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_register_recovers_translation() -> None:
    base = _textured_image()
    M = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, 15.0], [0.0, 0.0, 1.0]])
    warped = cv2.warpPerspective(base, M, (600, 400))
    full = np.full((400, 600), 255, np.uint8)

    G = register(base, warped, full, full)   # maps base pixels -> warped pixels
    assert G is not None
    p = np.array([300.0, 200.0, 1.0])
    q = G @ p
    q /= q[2]
    assert abs(q[0] - 325.0) < 3.0 and abs(q[1] - 215.0) < 3.0


def test_register_returns_none_on_blank_frames() -> None:
    blank = np.zeros((400, 600, 3), np.uint8)
    full = np.full((400, 600), 255, np.uint8)
    assert register(blank, blank, full, full) is None


def test_blend_is_weighted_average_normalized() -> None:
    h1 = np.eye(3)
    h2 = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # w1=1 -> exactly h1; w1=0 -> exactly h2; w1=0.5 -> midpoint translation 5
    assert np.allclose(blend_homographies(h1, h2, 1.0), h1)
    assert np.allclose(blend_homographies(h1, h2, 0.0), h2)
    mid = blend_homographies(h1, h2, 0.5)
    assert np.isclose(mid[2, 2], 1.0)
    assert np.isclose(mid[0, 2], 5.0)


def test_disagreement_confidence_clamped_to_zero() -> None:
    h = np.eye(3)
    assert disagreement_confidence(h, h, tau=0.1) == 1.0   # identical -> 1.0
    far = np.array([[1.0, 0.0, 500.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert disagreement_confidence(h, far, tau=0.1) == 0.0  # huge disagree -> clamped 0


def test_disagreement_confidence_linear_region() -> None:
    # pixel->pitch Hs (divide by 1000); a 0.05 pitch-unit shift with tau=0.10 -> conf 0.5
    h1 = np.diag([1 / 1000.0, 1 / 1000.0, 1.0])
    h2 = h1.copy()
    h2[0, 2] += 0.05
    c = disagreement_confidence(h1, h2, tau=0.10, frame_size=(1920, 1080))
    assert abs(c - 0.5) < 0.05


def test_homography_entry_fields() -> None:
    e = HomographyEntry(np.eye(3), "anchor", 1.0)
    assert e.source == "anchor" and e.confidence == 1.0 and e.H.shape == (3, 3)
