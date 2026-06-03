"""Tests for the homography-propagation leaf helpers."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from soccer_vision.pitch.propagation import (
    HomographyEntry,
    blend_homographies,
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


def _shift_frame(base: np.ndarray, dx: int) -> np.ndarray:
    M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return cv2.warpPerspective(base, M, (base.shape[1], base.shape[0]))


def _scene(n_frames: int, pan_per_frame: int = 4):
    """A panning clip: frame f is base shifted by f*pan. read_frame + anchors map
    each frame's pixels back to a fixed 'pitch' = base-frame pixel coords / 1000."""
    base = _textured_image(1)
    frames = {f: _shift_frame(base, f * pan_per_frame) for f in range(n_frames)}

    def read_frame(f: int):
        return frames.get(f)

    def anchor_H(f: int) -> np.ndarray:
        undo = np.array([[1.0, 0.0, -f * pan_per_frame], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        scale = np.diag([1 / 1000.0, 1 / 1000.0, 1.0])
        return scale @ undo

    return read_frame, anchor_H, frames


def test_propagation_bridges_gap_within_window() -> None:
    read_frame, anchor_H, _ = _scene(11)
    anchors = {0: anchor_H(0), 10: anchor_H(10)}     # gap of 9 frames between
    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    out = propagate_homographies(anchors, read_frame, boxes, max_gap=15)

    assert out[0].source == "anchor" and out[10].source == "anchor"
    assert 5 in out and out[5].source == "propagated"
    truth = anchor_H(5)
    p = np.array([300.0, 200.0])
    got = (out[5].H @ np.array([p[0], p[1], 1.0]))
    got = got[:2] / got[2]
    exp = (truth @ np.array([p[0], p[1], 1.0]))
    exp = exp[:2] / exp[2]
    assert np.linalg.norm(got - exp) < 0.02   # < 0.02 pitch-units


def test_gap_beyond_max_is_not_bridged() -> None:
    read_frame, anchor_H, _ = _scene(11)
    anchors = {0: anchor_H(0), 10: anchor_H(10)}
    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    out = propagate_homographies(anchors, read_frame, boxes, max_gap=4)   # gap 9 > 4
    assert set(out) == {0, 10}                # only anchors, nothing bridged


def test_unbridged_frames_absent_and_anchors_confident() -> None:
    read_frame, anchor_H, _ = _scene(11)
    anchors = {0: anchor_H(0), 10: anchor_H(10)}
    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    out = propagate_homographies(anchors, read_frame, boxes, max_gap=15)
    assert out[0].confidence == 1.0
    assert 0.0 <= out[5].confidence <= 1.0


def test_empty_anchors_returns_empty() -> None:
    read_frame, _, _ = _scene(3)
    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    assert propagate_homographies({}, read_frame, boxes, max_gap=15) == {}


def test_one_sided_coverage_when_a_chain_breaks() -> None:
    # Frame 2 is unregisterable (blank) -> breaks the forward chain after frame 1
    # and the backward chain after frame 3. Frame 1 is reached forward-only, frame 9
    # backward-only, frame 2 by neither (spec: either chain can bridge a frame).
    read_frame, anchor_H, _ = _scene(11)
    blank = np.zeros((400, 600, 3), np.uint8)

    def gated(f: int):
        return blank if f == 2 else read_frame(f)

    anchors = {0: anchor_H(0), 10: anchor_H(10)}
    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    out = propagate_homographies(anchors, gated, boxes, max_gap=15)

    assert 2 not in out                                   # reachable by neither chain
    assert 1 in out and out[1].source == "propagated"     # forward-only reach
    assert 9 in out and out[9].source == "propagated"     # backward-only reach


def test_frame_mask_blanks_player_boxes() -> None:
    from soccer_vision.pitch.propagation import _frame_mask

    boxes = pd.DataFrame([
        {"frame": 0, "bbox_x1": 100, "bbox_y1": 100, "bbox_x2": 140, "bbox_y2": 180, "class": "player"},
    ])
    mask = _frame_mask(boxes, 0, (400, 600))   # shape (rows=400, cols=600)
    assert mask[140, 120] == 0     # inside the dilated player box -> ignored by ORB
    assert mask[10, 10] == 255     # background -> used
