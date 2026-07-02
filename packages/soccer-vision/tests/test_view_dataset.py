"""Tests for the view-dataset exporter (annotation-scaling Slice 1.5)."""
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from soccer_vision.labeler.view_dataset import cross_match_fractions
from soccer_vision.labeler.view_digest import (
    _pair_match_fraction,
    frame_descriptors,
    similarity_matrix,
)

H, W = 240, 320


def _pattern(seed: int) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 30, np.uint8)
    for _ in range(45):
        x1, x2 = sorted(rng.integers(0, W, size=2).tolist())
        y1, y2 = sorted(rng.integers(0, H, size=2).tolist())
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        else:
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
    return img


def test_pair_match_fraction_matches_similarity_matrix() -> None:
    descs, counts = frame_descriptors([_pattern(0), _pattern(1), _pattern(0)])
    s = similarity_matrix(descs, counts)
    assert abs(_pair_match_fraction(descs[0], descs[1], counts[0], counts[1]) - s[0, 1]) < 1e-12
    assert abs(_pair_match_fraction(descs[0], descs[2], counts[0], counts[2]) - s[0, 2]) < 1e-12


def test_cross_match_fractions_parity_with_similarity_matrix() -> None:
    frames = [_pattern(5), _pattern(1), _pattern(2), _pattern(3)]
    q, refs = frames[:1], frames[1:]
    qd, qc = frame_descriptors(q)
    rd, rc = frame_descriptors(refs)
    cross = cross_match_fractions(qd, qc, rd, rc)          # (1, 3)
    full = similarity_matrix(*frame_descriptors(q + refs))  # (4, 4)
    assert cross.shape == (1, 3)
    assert np.allclose(cross[0], full[0, 1:], atol=1e-12)


def test_cross_match_fractions_low_keypoints_zero() -> None:
    qd, qc = frame_descriptors([_pattern(0)])
    rd, _rc = frame_descriptors([_pattern(1)])
    cross = cross_match_fractions(qd, qc, rd, [3], min_keypoints=10)  # ref "blind"
    assert cross[0, 0] == 0.0
