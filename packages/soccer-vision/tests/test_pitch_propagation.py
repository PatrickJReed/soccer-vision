"""Tests for the homography-propagation leaf helpers."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from soccer_vision.pitch.propagation import (
    HomographyEntry,
    blend_homographies,
    compute_interframe_homographies,
    disagreement_confidence,
    propagate_homographies,
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


def test_frame_mask_blanks_player_boxes() -> None:
    from soccer_vision.pitch.propagation import _frame_mask

    boxes = pd.DataFrame([
        {"frame": 0, "bbox_x1": 100, "bbox_y1": 100, "bbox_x2": 140, "bbox_y2": 180, "class": "player"},
    ])
    mask = _frame_mask(boxes, 0, (400, 600))   # shape (rows=400, cols=600)
    assert mask[140, 120] == 0     # inside the dilated player box -> ignored by ORB
    assert mask[10, 10] == 255     # background -> used


def _pan_interframe(n_pairs: int, dx: float = 4.0) -> dict[int, np.ndarray]:
    """interframe[i] maps frame i pixels -> i+1 pixels: a constant +dx translation."""
    G = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return {i: G for i in range(n_pairs)}


def _pan_anchor_H(frame: int, dx: float = 4.0) -> np.ndarray:
    """pixel->pitch for a frame panned by frame*dx: undo pan, then /1000."""
    undo = np.array([[1.0, 0.0, -frame * dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return np.diag([1 / 1000.0, 1 / 1000.0, 1.0]) @ undo


def test_propagation_bridges_gap_within_window() -> None:
    anchors = {0: _pan_anchor_H(0), 10: _pan_anchor_H(10)}
    interframe = _pan_interframe(10)                       # G[0..9]
    out = propagate_homographies(anchors, interframe, max_gap=15)

    assert out[0].source == "anchor" and out[10].source == "anchor"
    assert out[5].source == "propagated"
    p = np.array([300.0, 200.0, 1.0])
    got = out[5].H @ p
    got = got[:2] / got[2]
    exp = _pan_anchor_H(5) @ p
    exp = exp[:2] / exp[2]
    assert np.linalg.norm(got - exp) < 1e-9


def test_gap_beyond_max_is_not_bridged() -> None:
    anchors = {0: _pan_anchor_H(0), 10: _pan_anchor_H(10)}
    out = propagate_homographies(anchors, _pan_interframe(10), max_gap=4)  # gap 9 > 4
    assert set(out) == {0, 10}


def test_one_sided_when_interframe_missing() -> None:
    anchors = {0: _pan_anchor_H(0), 10: _pan_anchor_H(10)}
    interframe = _pan_interframe(10)
    del interframe[1]                  # break forward after frame 1; backward stops before frame 1
    out = propagate_homographies(anchors, interframe, max_gap=15)
    assert out[1].source == "propagated"   # forward reaches frame 1 (via G[0])
    assert out[9].source == "propagated"   # backward reaches frame 9 (via inv(G[9]))


def test_empty_anchors_returns_empty() -> None:
    assert propagate_homographies({}, {}, max_gap=15) == {}


def test_anchors_have_unit_confidence() -> None:
    anchors = {0: _pan_anchor_H(0), 10: _pan_anchor_H(10)}
    out = propagate_homographies(anchors, _pan_interframe(10), max_gap=15)
    assert out[0].confidence == 1.0 and out[10].confidence == 1.0
    assert 0.0 <= out[5].confidence <= 1.0


def _textured(seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = (rng.random((400, 600)) * 60).astype(np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(20, 560)), int(rng.integers(20, 360))
        cv2.rectangle(g, (x, y), (x + 18, y + 18), int(rng.integers(80, 255)), -1)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def test_compute_interframe_recovers_known_pan_downscaled() -> None:
    base = _textured()
    dx = 6

    def shift(f: int) -> np.ndarray:
        M = np.array([[1.0, 0.0, float(f * dx)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return cv2.warpPerspective(base, M, (600, 400))

    frames = {f: shift(f) for f in range(4)}

    def read_frame(i):
        return frames.get(i)

    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    interframe = compute_interframe_homographies(
        read_frame, needed_pairs={0, 1, 2}, player_boxes=boxes, downscale=0.5,
    )
    assert set(interframe) == {0, 1, 2}
    # G[1] maps frame1 px -> frame2 px: a +dx translation, recovered at FULL resolution.
    p = np.array([300.0, 200.0, 1.0])
    q = interframe[1] @ p
    q /= q[2]
    assert abs(q[0] - (300 + dx)) < 2.0 and abs(q[1] - 200.0) < 2.0


def test_compute_interframe_empty_pairs() -> None:
    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    assert compute_interframe_homographies(lambda i: None, set(), boxes) == {}
