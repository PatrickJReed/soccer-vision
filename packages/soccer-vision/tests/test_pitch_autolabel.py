"""Tests for pitch.autolabel — projecting canonical landmarks into image pixels."""

from __future__ import annotations

import numpy as np
from soccer_vision.pitch.autolabel import (
    project_landmarks,
    propose_labels,
    to_yolo_pose_line,
)
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.propagation import HomographyEntry

# H maps image -> pitch. Use pitch = image / 1000, so image = pitch * 1000.
# inv(H) maps pitch -> image.
_H_IMG_TO_PITCH = np.diag([1.0 / 1000.0, 1.0 / 1000.0, 1.0])
_FRAME = (1920, 1080)  # (width, height)


def test_project_landmarks_shape_and_columns() -> None:
    out = project_landmarks(_H_IMG_TO_PITCH, PITCH_LANDMARKS, _FRAME)
    assert out.shape == (21, 3)  # x_px, y_px, visible


def test_in_frame_landmarks_marked_visible_with_correct_pixels() -> None:
    out = project_landmarks(_H_IMG_TO_PITCH, PITCH_LANDMARKS, _FRAME)
    # corner 0 -> pitch (0,0) -> image (0,0): in frame, visible
    assert out[0, 2] == 2.0
    assert np.allclose(out[0, :2], (0.0, 0.0))
    # center mark idx 6 -> pitch (0.5,0.5) -> image (500,500): in frame
    assert out[6, 2] == 2.0
    assert np.allclose(out[6, :2], (500.0, 500.0))


def test_out_of_frame_landmarks_marked_not_visible() -> None:
    # corner 3 -> pitch (1,1) -> image (1000,1000). Shrink frame so it's outside.
    out = project_landmarks(_H_IMG_TO_PITCH, PITCH_LANDMARKS, (800, 800))
    assert out[3, 2] == 0.0
    assert np.allclose(out[3, :2], (0.0, 0.0))  # zeroed when not visible


def test_singular_homography_returns_all_invisible() -> None:
    singular = np.zeros((3, 3))
    out = project_landmarks(singular, PITCH_LANDMARKS, _FRAME)
    assert out.shape == (21, 3)
    assert np.all(out[:, 2] == 0.0)


def test_propose_labels_filters_low_confidence() -> None:
    homs = {
        0: HomographyEntry(_H_IMG_TO_PITCH, "anchor", 1.0),
        1: HomographyEntry(_H_IMG_TO_PITCH, "propagated", 0.2),
    }
    out = propose_labels(homs, PITCH_LANDMARKS, _FRAME, min_confidence=0.5)
    assert set(out) == {0}  # frame 1 dropped (conf 0.2 < 0.5)


def test_propose_labels_keeps_high_confidence_propagated() -> None:
    homs = {1: HomographyEntry(_H_IMG_TO_PITCH, "propagated", 0.9)}
    out = propose_labels(homs, PITCH_LANDMARKS, _FRAME, min_confidence=0.5)
    assert set(out) == {1}
    assert out[1].shape == (21, 3)


def test_to_yolo_pose_line_format() -> None:
    kpts = project_landmarks(_H_IMG_TO_PITCH, PITCH_LANDMARKS, _FRAME)
    line = to_yolo_pose_line(kpts, _FRAME, class_id=0)
    parts = line.split()
    # class + bbox(4) + 21*(x,y,v)=63  -> 68 tokens
    assert len(parts) == 1 + 4 + 21 * 3
    assert parts[0] == "0"
    # bbox normalized to [0,1]
    bbox = np.array(parts[1:5], dtype=float)
    assert bbox.min() >= 0.0
    assert bbox.max() <= 1.0
    # keypoint triplets: x,y normalized to [0,1]; visibility in {0, 2}
    kp = np.array(parts[5:], dtype=float).reshape(21, 3)
    assert kp[:, :2].min() >= 0.0
    assert kp[:, :2].max() <= 1.0
    assert set(np.unique(kp[:, 2]).tolist()) <= {0.0, 2.0}


def test_to_yolo_pose_line_invisible_keypoints_are_zero() -> None:
    kpts = project_landmarks(_H_IMG_TO_PITCH, PITCH_LANDMARKS, (800, 800))
    line = to_yolo_pose_line(kpts, (800, 800), class_id=0)
    parts = line.split()
    # corner 3 (idx 3) is invisible -> its triplet is 0 0 0
    base = 1 + 4 + 3 * 3
    assert parts[base : base + 3] == ["0", "0", "0"]
