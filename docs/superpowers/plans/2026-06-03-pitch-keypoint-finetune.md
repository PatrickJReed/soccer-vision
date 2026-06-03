# Phase 3.5b — Pitch-keypoint Fine-tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the codebase support for fine-tuning the pitch-keypoint model on Trace footage — a 21-point youth-9v9 landmark schema with PitchSpec-derived canonical coords, a propagation-projected auto-labeling engine, fine-tuned-pitch-weights wiring, and the Colab notebooks Patrick runs to label/train/gate.

**Architecture:** The canonical landmark coordinates are computed from `PitchSpec` (one source of truth shared with metrics). A new pure module `pitch/autolabel.py` projects canonical landmarks back into image pixels through per-frame homographies (anchors + 3.5a propagation) to produce YOLO-pose pre-labels for the active-learning loop. `RoboflowBackend` is rewired to download fine-tuned pitch weights from a GitHub release. Four notebooks drive the GPU/labeling work.

**Tech Stack:** Python 3, numpy, pandas, OpenCV, ultralytics YOLOv8-pose, pytest, mypy strict, ruff. uv workspace; package at `packages/soccer-vision/`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `packages/soccer-vision/src/soccer_vision/pitch/spec.py` | Dimensionless pitch proportions | Modify: add `goal_width_frac` |
| `packages/soccer-vision/src/soccer_vision/pitch/landmarks.py` | Canonical 21-pt landmarks + per-frame homography fit | Modify: `youth_landmarks()`, new `PITCH_LANDMARKS`, `FLIP_IDX`, index-group constants |
| `packages/soccer-vision/src/soccer_vision/pitch/autolabel.py` | Project canonical landmarks → image pixels → YOLO-pose pre-labels | Create |
| `packages/soccer-vision/src/soccer_vision/tracking/roboflow.py` | Backend + weight registry | Modify: `PITCH_V1_URL`, `WEIGHTS["pitch"]` → url, `pitch_weights_path` arg |
| `packages/soccer-vision/tests/test_pitch_spec.py` | PitchSpec tests | Modify |
| `packages/soccer-vision/tests/test_pitch_landmarks.py` | Landmarks tests | Modify (21-pt schema) |
| `packages/soccer-vision/tests/test_pitch_autolabel.py` | Autolabel tests | Create |
| `packages/soccer-vision/tests/test_tracking_roboflow.py` | Backend/weights tests | Modify |
| `examples/extract_pitch_frames.ipynb` | Stratified frame sampling from training games | Create |
| `examples/finetune_pitch.ipynb` | Train YOLOv8-pose v0→vN | Create |
| `examples/autolabel_pitch.ipynb` | Drive autolabel.py → Roboflow pre-annotations | Create |
| `examples/acceptance_pitch.ipynb` | Held-out multi-field coverage gate | Create |

**Note on landmark axis convention (carried from the existing code and spec):**
`PITCH_LANDMARKS[i] = (x, y)` in `[0,1]^2` where **y = goal-to-goal** (phase
splitter uses `y<0.333`/`y>0.667`) and **x = touchline-to-touchline**. The pitch
length axis is the longer one; aspect_ratio = length/width = 1.5, so the field is
1.5× longer (in y) than wide (in x). All coordinates normalized so each axis
spans `[0,1]` independently (the existing 32-pt table normalizes each axis to
[0,1] independently too — `raw_y/width`, `raw_x/length`).

---

## Task 1: Add `goal_width_frac` to PitchSpec

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/spec.py`
- Test: `packages/soccer-vision/tests/test_pitch_spec.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pitch_spec.py`:

```python
def test_standard_9v9_has_goal_width_frac() -> None:
    spec = PitchSpec.standard_9v9()
    # US Soccer 9v9: ~6.4 m goal on a ~45.7 m wide field -> ~0.14 of width.
    assert 0.10 < spec.goal_width_frac < 0.20


def test_fifa_11v11_has_goal_width_frac() -> None:
    spec = PitchSpec.fifa_11v11()
    # 7.32 m goal on a 68 m wide field -> ~0.108 of width.
    assert 0.08 < spec.goal_width_frac < 0.14
```

Ensure the file imports `PitchSpec` (it already does). 

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_spec.py -v`
Expected: FAIL — `AttributeError: 'PitchSpec' object has no attribute 'goal_width_frac'`

- [ ] **Step 3: Add the field**

In `spec.py`, add to the dataclass body (after `center_circle_radius_frac`):

```python
    goal_width_frac: float = 0.140
```

And in `fifa_11v11()` add the override inside the `cls(...)` call:

```python
            goal_width_frac=0.108,
```

`goal_width_frac` is "goal mouth width as a fraction of pitch **width**" (the
x-axis). `standard_9v9()` uses the default 0.140.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_spec.py -v`
Expected: PASS (all tests in file)

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/spec.py packages/soccer-vision/tests/test_pitch_spec.py
git commit -m "feat(pitch): add goal_width_frac to PitchSpec"
```

---

## Task 2: 21-point youth landmark schema (PitchSpec-derived)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/landmarks.py`
- Test: `packages/soccer-vision/tests/test_pitch_landmarks.py`

This replaces the 32-pt FIFA table with a 21-pt youth table computed from a
`PitchSpec`. `build_frame_homographies` is unchanged (it indexes
`PITCH_LANDMARKS[kp_idx]` generically).

**Index map (21 points), with canonical (x, y) in [0,1]^2:**

```
0  corner: x=0, y=0           (own-goal-line, left touchline)
1  corner: x=1, y=0           (own-goal-line, right touchline)
2  corner: x=0, y=1           (opp-goal-line, left touchline)
3  corner: x=1, y=1           (opp-goal-line, right touchline)
4  halfway x touchline far:  x=1, y=0.5
5  halfway x touchline near: x=0, y=0.5   (RESERVED — under camera, never labeled)
6  center mark:              x=0.5, y=0.5
7  center-circle apex far:   x=0.5, y=0.5 + r   (toward opp goal)
8  center-circle apex near:  x=0.5, y=0.5 - r
9  own box: outer-left:      x=cx_l, y=bl
10 own box: outer-right:     x=cx_r, y=bl
11 own box: goalline-left:   x=cx_l, y=0
12 own box: goalline-right:  x=cx_r, y=0
13 opp box: outer-left:      x=cx_l, y=1-bl
14 opp box: outer-right:     x=cx_r, y=1-bl
15 opp box: goalline-left:   x=cx_l, y=1
16 opp box: goalline-right:  x=cx_r, y=1
17 own goal post left:       x=0.5 - gw/2, y=0
18 own goal post right:      x=0.5 + gw/2, y=0
19 opp goal post left:       x=0.5 - gw/2, y=1
20 opp goal post right:      x=0.5 + gw/2, y=1
```

where `r = center_circle_radius_frac` (as a fraction of length, the y-axis),
`bl = penalty_box_length_frac` (fraction of length from the goal line),
`cx_l = 0.5 - penalty_box_width_frac/2`, `cx_r = 0.5 + penalty_box_width_frac/2`,
`gw = goal_width_frac`. "left/right" are in canonical x; "own/opp" in canonical y.

**FLIP_IDX** (left↔right mirror for horizontal-flip augmentation; index `i` maps
to the keypoint it becomes when the image is mirrored in x):

```
[1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19]
```

(corners swap L/R, halfway far/near swap, center+apexes map to themselves, each
box's left/right corners swap, goal posts swap.)

- [ ] **Step 1: Write the failing tests**

Replace the body of `tests/test_pitch_landmarks.py` import line and
`test_landmarks_shape_and_range`, and the `idxs` lists used by helpers, with the
following. Full new file:

```python
"""Tests for PITCH_LANDMARKS (21-pt youth schema) and build_frame_homographies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from soccer_vision.pitch.landmarks import (
    FLIP_IDX,
    NEAR_HALFWAY_IDX,
    PITCH_LANDMARKS,
    build_frame_homographies,
    youth_landmarks,
)
from soccer_vision.pitch.spec import PitchSpec

# Six well-spread, non-collinear indices used across fit tests.
_FIT_IDXS = [0, 3, 6, 11, 16, 19]


def test_landmarks_shape_and_range() -> None:
    assert PITCH_LANDMARKS.shape == (21, 2)
    assert PITCH_LANDMARKS.min() >= 0.0
    assert PITCH_LANDMARKS.max() <= 1.0


def test_corners_at_extremes() -> None:
    assert np.allclose(PITCH_LANDMARKS[0], (0.0, 0.0))
    assert np.allclose(PITCH_LANDMARKS[1], (1.0, 0.0))
    assert np.allclose(PITCH_LANDMARKS[2], (0.0, 1.0))
    assert np.allclose(PITCH_LANDMARKS[3], (1.0, 1.0))


def test_center_mark_is_center() -> None:
    assert np.allclose(PITCH_LANDMARKS[6], (0.5, 0.5))


def test_center_circle_apexes_straddle_center_on_y() -> None:
    r = PitchSpec.standard_9v9().center_circle_radius_frac
    assert np.allclose(PITCH_LANDMARKS[7], (0.5, 0.5 + r))
    assert np.allclose(PITCH_LANDMARKS[8], (0.5, 0.5 - r))


def test_goal_posts_straddle_center_on_x_at_goal_lines() -> None:
    gw = PitchSpec.standard_9v9().goal_width_frac
    assert np.allclose(PITCH_LANDMARKS[17], (0.5 - gw / 2, 0.0))
    assert np.allclose(PITCH_LANDMARKS[18], (0.5 + gw / 2, 0.0))
    assert np.allclose(PITCH_LANDMARKS[19], (0.5 - gw / 2, 1.0))
    assert np.allclose(PITCH_LANDMARKS[20], (0.5 + gw / 2, 1.0))


def test_box_corners_symmetric_about_x_center() -> None:
    # own box outer-left (9) and outer-right (10) mirror across x=0.5
    assert np.isclose(PITCH_LANDMARKS[9, 0] + PITCH_LANDMARKS[10, 0], 1.0)
    assert np.isclose(PITCH_LANDMARKS[9, 1], PITCH_LANDMARKS[10, 1])


def test_near_halfway_index_constant() -> None:
    assert NEAR_HALFWAY_IDX == 5
    assert np.allclose(PITCH_LANDMARKS[5], (0.0, 0.5))


def test_flip_idx_is_an_involution() -> None:
    assert len(FLIP_IDX) == 21
    f = np.array(FLIP_IDX)
    assert np.array_equal(f[f], np.arange(21))  # applying twice is identity


def test_flip_idx_mirrors_x_coordinates() -> None:
    flipped = PITCH_LANDMARKS[np.array(FLIP_IDX)]
    # mirroring x then re-mirroring landmark order recovers the table
    assert np.allclose(flipped[:, 0], 1.0 - PITCH_LANDMARKS[:, 0])
    assert np.allclose(flipped[:, 1], PITCH_LANDMARKS[:, 1])


def test_youth_landmarks_scales_with_spec() -> None:
    wide = youth_landmarks(PitchSpec(goal_width_frac=0.30))
    assert np.isclose(wide[18, 0] - wide[17, 0], 0.30)


def _keypoints_for_identity(frame: int, conf: float = 0.9) -> pd.DataFrame:
    pts = PITCH_LANDMARKS[_FIT_IDXS]
    return pd.DataFrame({
        "frame": [frame] * len(_FIT_IDXS),
        "kp_idx": _FIT_IDXS,
        "x_px": pts[:, 0],
        "y_px": pts[:, 1],
        "conf": [conf] * len(_FIT_IDXS),
    })


def test_build_recovers_known_transform() -> None:
    pitch = PITCH_LANDMARKS[_FIT_IDXS]
    image = pitch * 1000.0 + np.array([50.0, 30.0])
    kp = pd.DataFrame({
        "frame": [7] * len(_FIT_IDXS),
        "kp_idx": _FIT_IDXS,
        "x_px": image[:, 0],
        "y_px": image[:, 1],
        "conf": [0.9] * len(_FIT_IDXS),
    })
    homographies = build_frame_homographies(kp)
    assert set(homographies) == {7}
    pts = np.column_stack([image[:, 0], image[:, 1], np.ones(len(_FIT_IDXS))])
    mapped = (homographies[7] @ pts.T).T
    mapped /= mapped[:, 2:3]
    assert np.allclose(mapped[:, :2], pitch, atol=1e-6)


def test_frames_below_min_points_are_skipped() -> None:
    kp = _keypoints_for_identity(3).head(3)
    assert build_frame_homographies(kp) == {}


def test_low_confidence_keypoints_filtered() -> None:
    kp = _keypoints_for_identity(3, conf=0.1)
    assert build_frame_homographies(kp) == {}


def test_empty_keypoints_returns_empty() -> None:
    empty = pd.DataFrame(columns=["frame", "kp_idx", "x_px", "y_px", "conf"])
    assert build_frame_homographies(empty) == {}


def test_collinear_keypoints_are_skipped() -> None:
    idxs = [0, 1, 2, 3, 4]
    kp = pd.DataFrame({
        "frame": [9] * 5,
        "kp_idx": idxs,
        "x_px": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y_px": [0.0, 1.0, 2.0, 3.0, 4.0],
        "conf": [0.9] * 5,
    })
    assert build_frame_homographies(kp) == {}


def test_multiple_frames_dispatched_independently() -> None:
    good = _keypoints_for_identity(0)
    sparse = _keypoints_for_identity(1).head(3)
    kp = pd.concat([good, sparse], ignore_index=True)
    homographies = build_frame_homographies(kp)
    assert set(homographies) == {0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_landmarks.py -v`
Expected: FAIL — `ImportError` for `FLIP_IDX`/`NEAR_HALFWAY_IDX`/`youth_landmarks`.

- [ ] **Step 3: Rewrite landmarks.py head**

Replace everything in `landmarks.py` from the module docstring through the
`PITCH_LANDMARKS = ...` definition (i.e. lines 1–44, up to but NOT including
`def build_frame_homographies`) with:

```python
"""Canonical youth-9v9 pitch landmarks + per-frame homography fitting.

PITCH_LANDMARKS maps each pitch-model keypoint index to its canonical pitch
coordinate in [0, 1]^2. Coordinates are COMPUTED from a PitchSpec (the same
proportions the metrics layer uses), not vendored from a fixed table, so the
homography target space and the analytics space share one definition.

Axis convention: y = goal-to-goal (the phase splitter expects y < 0.333 = own
third, y > 0.667 = opp third); x = touchline-to-touchline. Each axis is
normalized to [0, 1] independently.

The 21-point schema (see docs/superpowers/specs/2026-06-03-pitch-keypoint-finetune-design.md):
corners (0-3), halfway x touchline far/near (4-5), center mark (6),
center-circle apexes (7-8), own/opp penalty-box corners (9-16), goal-post
bases (17-20). Index 5 (near halfway x touchline) sits under the Trace camera
and is never visible; its slot is reserved for schema regularity.
"""

from __future__ import annotations

from typing import Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.pitch.homography import HomographyError, fit_homography
from soccer_vision.pitch.spec import PitchSpec

# Index of the near halfway x touchline point — directly under the camera, never
# visible, never labeled (kept for schema regularity).
NEAR_HALFWAY_IDX: Final = 5

# Left<->right (x-mirror) keypoint permutation for YOLO-pose fliplr augmentation.
FLIP_IDX: Final[list[int]] = [
    1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19,
]


def youth_landmarks(spec: PitchSpec) -> NDArray[np.float64]:
    """Compute the 21 canonical [0,1]^2 landmark coords from a PitchSpec.

    See module docstring / spec for the index map. y = goal-to-goal,
    x = touchline-to-touchline.
    """
    r = spec.center_circle_radius_frac
    bl = spec.penalty_box_length_frac
    cx_l = 0.5 - spec.penalty_box_width_frac / 2.0
    cx_r = 0.5 + spec.penalty_box_width_frac / 2.0
    gw = spec.goal_width_frac
    pts: list[tuple[float, float]] = [
        (0.0, 0.0),            # 0  corner own-left
        (1.0, 0.0),            # 1  corner own-right
        (0.0, 1.0),            # 2  corner opp-left
        (1.0, 1.0),            # 3  corner opp-right
        (1.0, 0.5),            # 4  halfway x touchline far
        (0.0, 0.5),            # 5  halfway x touchline near (reserved)
        (0.5, 0.5),            # 6  center mark
        (0.5, 0.5 + r),        # 7  center-circle apex far
        (0.5, 0.5 - r),        # 8  center-circle apex near
        (cx_l, bl),            # 9  own box outer-left
        (cx_r, bl),            # 10 own box outer-right
        (cx_l, 0.0),           # 11 own box goalline-left
        (cx_r, 0.0),           # 12 own box goalline-right
        (cx_l, 1.0 - bl),      # 13 opp box outer-left
        (cx_r, 1.0 - bl),      # 14 opp box outer-right
        (cx_l, 1.0),           # 15 opp box goalline-left
        (cx_r, 1.0),           # 16 opp box goalline-right
        (0.5 - gw / 2.0, 0.0), # 17 own goal post left
        (0.5 + gw / 2.0, 0.0), # 18 own goal post right
        (0.5 - gw / 2.0, 1.0), # 19 opp goal post left
        (0.5 + gw / 2.0, 1.0), # 20 opp goal post right
    ]
    return np.array(pts, dtype=np.float64)


PITCH_LANDMARKS: Final[NDArray[np.float64]] = youth_landmarks(PitchSpec.standard_9v9())
```

Leave `build_frame_homographies` (the rest of the file) untouched.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_landmarks.py -v`
Expected: PASS (all)

- [ ] **Step 5: Run the full pitch suite + typecheck (catch fallout)**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_mapper.py tests/test_pitch_filter.py tests/test_pipeline_homographies.py -v && uv run mypy src/soccer_vision/pitch/landmarks.py`
Expected: PASS / no type errors. If `test_pitch_mapper.py` or `test_pitch_filter.py` hardcoded 32-pt indices, fix those references to use `_FIT_IDXS`-style valid indices (0-20) — repeat the index-fix in this step rather than leaving it.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/landmarks.py packages/soccer-vision/tests/test_pitch_landmarks.py
git commit -m "feat(pitch): 21-pt youth landmark schema derived from PitchSpec + FLIP_IDX"
```

---

## Task 3: `autolabel.py` — propagation-projected pre-labels

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/autolabel.py`
- Test: `packages/soccer-vision/tests/test_pitch_autolabel.py`

Pure module. Given per-frame homographies (image→pitch) it projects the canonical
landmarks back into image pixels (via `inv(H)`), keeps the in-frame ones, and
formats YOLO-pose label lines for the active-learning loop. No video/model I/O —
the notebook supplies the homographies (from `build_frame_homographies` +
`propagate_homographies`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pitch_autolabel.py`:

```python
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
    # corner 3 -> pitch (1,1) -> image (1000,1000): y=1000 in 1080 frame -> in.
    # Shrink the frame so (1000,1000) falls outside.
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
    # all coords normalized into [0,1]
    coords = np.array(parts[1:], dtype=float)
    assert coords.min() >= 0.0
    assert coords.max() <= 1.0


def test_to_yolo_pose_line_invisible_keypoints_are_zero() -> None:
    kpts = project_landmarks(_H_IMG_TO_PITCH, PITCH_LANDMARKS, (800, 800))
    line = to_yolo_pose_line(kpts, (800, 800), class_id=0)
    parts = line.split()
    # corner 3 (idx 3) is invisible -> its triplet is 0 0 0
    base = 1 + 4 + 3 * 3
    assert parts[base : base + 3] == ["0", "0", "0"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_autolabel.py -v`
Expected: FAIL — `ModuleNotFoundError: soccer_vision.pitch.autolabel`

- [ ] **Step 3: Implement autolabel.py**

Create `src/soccer_vision/pitch/autolabel.py`:

```python
"""Project canonical pitch landmarks into image pixels to seed keypoint labels.

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
    for x_px, y_px, v in keypoints:
        if v > 0:
            tokens += [_f(x_px / width), _f(y_px / height), str(int(v))]
        else:
            tokens += ["0", "0", "0"]
    return " ".join(tokens)


def _f(value: float) -> str:
    """Compact fixed-precision float for label files."""
    return f"{value:.6f}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_autolabel.py -v`
Expected: PASS (all 9)

- [ ] **Step 5: Typecheck + lint**

Run: `cd packages/soccer-vision && uv run mypy src/soccer_vision/pitch/autolabel.py && uv run ruff check src/soccer_vision/pitch/autolabel.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/autolabel.py packages/soccer-vision/tests/test_pitch_autolabel.py
git commit -m "feat(pitch): autolabel — project canonical landmarks to YOLO-pose pre-labels"
```

---

## Task 4: Wire fine-tuned pitch weights into RoboflowBackend

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/tracking/roboflow.py`
- Test: `packages/soccer-vision/tests/test_tracking_roboflow.py`

Mirror the ball-v1 wiring: publish `pitch_yolov8_v1.pt` as a GitHub release asset,
download via URL, allow a local override.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_tracking_roboflow.py`:

```python
def test_pitch_weights_use_release_url() -> None:
    from soccer_vision.tracking.roboflow import PITCH_V1_URL, WEIGHTS

    kind, locator, filename = WEIGHTS["pitch"]
    assert kind == "url"
    assert locator == PITCH_V1_URL
    assert filename == "pitch_yolov8_v1.pt"
    assert PITCH_V1_URL.endswith("pitch_yolov8_v1.pt")
    assert "releases/download/pitch-v1/" in PITCH_V1_URL


def test_pitch_weights_path_override_missing_raises(tmp_path) -> None:
    from soccer_vision.tracking.roboflow import RoboflowBackend

    missing = tmp_path / "nope.pt"
    try:
        RoboflowBackend(pitch_weights_path=missing)
    except FileNotFoundError as e:
        assert "pitch_weights_path" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_pitch_weights_path_override_accepted(tmp_path) -> None:
    from soccer_vision.tracking.roboflow import RoboflowBackend

    w = tmp_path / "custom_pitch.pt"
    w.write_bytes(b"stub")
    backend = RoboflowBackend(pitch_weights_path=w, detect_pitch=True)
    assert backend.pitch_weights_path == w
```

(If `test_tracking_roboflow.py` lacks the `tmp_path` import style, note pytest
provides `tmp_path` as a fixture argument — no import needed.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_tracking_roboflow.py -k pitch -v`
Expected: FAIL — `ImportError: PITCH_V1_URL` / `TypeError: unexpected keyword argument 'pitch_weights_path'`

- [ ] **Step 3: Add PITCH_V1_URL + flip the registry entry**

In `roboflow.py`, after the `BALL_V1_URL` block (around line 36), add:

```python
# Direct download URL for the fine-tuned pitch-keypoint detector (Phase 3.5b),
# published as a GitHub release asset. The asset filename must match the URL tail.
PITCH_V1_URL: Final = (
    "https://github.com/PatrickJReed/soccer-vision/releases/download/"
    "pitch-v1/pitch_yolov8_v1.pt"
)
```

Change the `WEIGHTS["pitch"]` entry from the gdrive baseline to:

```python
    "pitch":  ("url",    PITCH_V1_URL, "pitch_yolov8_v1.pt"),
```

(Leave the old gdrive id in a trailing comment for provenance, mirroring the ball
note: `# roboflow baseline pitch model lived at gdrive id 1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf.`)

- [ ] **Step 4: Add the `pitch_weights_path` constructor arg**

In `RoboflowBackend.__init__`, add the parameter (after `ball_max_gap_frames`,
before `detect_pitch`):

```python
        pitch_weights_path: Path | None = None,
```

And in the body, after the `ball_weights_path` validation block:

```python
        if pitch_weights_path is not None and not pitch_weights_path.exists():
            raise FileNotFoundError(f"pitch_weights_path does not exist: {pitch_weights_path}")
        self.pitch_weights_path: Path | None = pitch_weights_path
```

Then in `_run_pipeline`, where the pitch model is loaded
(`pitch_model = YOLO(str(weight_paths["pitch"])).to(device=device)`), prefer the
override:

```python
            pitch_weights = self.pitch_weights_path or weight_paths["pitch"]
            pitch_model = YOLO(str(pitch_weights)).to(device=device)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_tracking_roboflow.py -v`
Expected: PASS (all). Then full suite + typecheck:

Run: `cd packages/soccer-vision && uv run pytest && uv run mypy src/`
Expected: PASS / no type errors.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/tracking/roboflow.py packages/soccer-vision/tests/test_tracking_roboflow.py
git commit -m "feat(tracking): download fine-tuned pitch-v1 weights; add pitch_weights_path override"
```

---

## Task 5: `extract_pitch_frames.ipynb` — stratified frame sampling

**Files:**
- Create: `examples/extract_pitch_frames.ipynb`

Notebook (no TDD; mirror the structure of `examples/finetune_ball.ipynb`). Patrick
runs this per training game to dump candidate frames for Roboflow.

- [ ] **Step 1: Create the notebook with these cells**

Use the same Colab-badge markdown header style as `finetune_ball.ipynb` (link to
`examples/extract_pitch_frames.ipynb`). Cells:

**MD cell:**
```
# Pitch-frame extraction (Phase 3.5b)
Samples frames from a Trace game for pitch-keypoint labeling in Roboflow.
Stratified to over-sample hard frames (end-zone pans, shadow boundaries).
Do NOT run on the bake-off clip or held-out games — training games only.
```

**Code cell (params):**
```python
from pathlib import Path
import cv2

VIDEO = Path("/content/game.mp4")   # a TRAINING game, not the bake-off clip
OUT = Path("/content/frames"); OUT.mkdir(exist_ok=True)
STRIDE = 30                         # ~1 fps base sampling at 30fps
MAX_FRAMES = 400
```

**Code cell (sampling — uniform base + simple hard-case oversample):**
```python
cap = cv2.VideoCapture(str(VIDEO))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
saved = 0
for i in range(0, n, STRIDE):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        continue
    # crude pan-position proxy: column of brightest green mass (field center).
    # Frames where the field center is near an edge are end-zone pans -> keep all;
    # otherwise keep every other one to avoid easy-center-frame overrepresentation.
    g = frame[:, :, 1].astype("int32")
    col_energy = g.sum(axis=0)
    center_col = int(col_energy.argmax())
    edge = center_col < frame.shape[1] * 0.3 or center_col > frame.shape[1] * 0.7
    if not edge and (i // STRIDE) % 2 == 1:
        continue
    cv2.imwrite(str(OUT / f"f{i:06d}.jpg"), frame)
    saved += 1
    if saved >= MAX_FRAMES:
        break
cap.release()
print("saved", saved, "frames to", OUT)
```

**Code cell (zip for download):**
```python
import shutil
shutil.make_archive("/content/pitch_frames", "zip", OUT)
from google.colab import files
files.download("/content/pitch_frames.zip")
```

- [ ] **Step 2: Sanity-check the notebook is valid JSON**

Run: `python3 -c "import json; json.load(open('examples/extract_pitch_frames.ipynb'))" && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add examples/extract_pitch_frames.ipynb
git commit -m "docs(examples): extract_pitch_frames notebook (stratified sampling)"
```

---

## Task 6: `finetune_pitch.ipynb` — train YOLOv8-pose

**Files:**
- Create: `examples/finetune_pitch.ipynb`

- [ ] **Step 1: Create the notebook with these cells**

Colab-badge header linking `examples/finetune_pitch.ipynb`. Cells:

**MD cell:**
```
# Pitch-keypoint fine-tune (Phase 3.5b)
Trains YOLOv8-pose on the 21-point youth landmark schema. Output:
runs/pose/pitch_v0/weights/best.pt -> publish as the pitch-v1 release asset.
kpt_shape and flip_idx MUST match soccer_vision.pitch.landmarks.
```

**Code cell (install + roboflow download):**
```python
!pip install -q "ultralytics>=8.2" roboflow
import os
from google.colab import userdata
os.environ['ROBOFLOW_API_KEY'] = userdata.get('ROBOFLOW_API_KEY')
from pathlib import Path
WORK = Path("/content/work"); WORK.mkdir(exist_ok=True)
%cd /content/work

from roboflow import Roboflow
rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
# TODO when running: replace workspace/project/version
project = rf.workspace("YOUR_WORKSPACE").project("pitch_v1")
dataset = project.version(1).download("yolov8")
print("Dataset at:", dataset.location)
```

**MD cell (flip_idx note):**
```
IMPORTANT: data.yaml must declare kpt_shape: [21, 3] and
flip_idx: [1,0,3,2,5,4,6,7,8,10,9,12,11,14,13,16,15,18,17,20,19]
(matches soccer_vision.pitch.landmarks.FLIP_IDX). Without flip_idx, fliplr
augmentation corrupts left/right keypoints. Patch the exported data.yaml if
Roboflow omits it.
```

**Code cell (ensure kpt_shape/flip_idx in data.yaml):**
```python
import yaml
dy = Path(dataset.location) / "data.yaml"
cfg = yaml.safe_load(dy.read_text())
cfg["kpt_shape"] = [21, 3]
cfg["flip_idx"] = [1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19]
dy.write_text(yaml.safe_dump(cfg))
print(cfg)
```

**Code cell (train — mirror ball recipe):**
```python
from ultralytics import YOLO
model = YOLO("yolov8s-pose.pt")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100, imgsz=1280, batch=8, patience=20,
    mosaic=1.0, mixup=0.15, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    scale=0.5, fliplr=0.5,
    project=".", name="pitch_v0",
)
```

**Code cell (val + download):**
```python
metrics = model.val()
print("pose mAP50:", metrics.pose.map50)
from google.colab import files
files.download("/content/work/pitch_v0/weights/best.pt")
```

- [ ] **Step 2: Validate JSON**

Run: `python3 -c "import json; json.load(open('examples/finetune_pitch.ipynb'))" && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add examples/finetune_pitch.ipynb
git commit -m "docs(examples): finetune_pitch notebook (YOLOv8-pose, 21-pt schema)"
```

---

## Task 7: `autolabel_pitch.ipynb` — drive the proposal engine

**Files:**
- Create: `examples/autolabel_pitch.ipynb`

Runs the current pitch model on a game, fits anchors, propagates, projects
landmarks via `pitch/autolabel.py`, writes YOLO-pose label files + an overlay
preview, zips for Roboflow re-import / correction.

- [ ] **Step 1: Create the notebook with these cells**

Colab-badge header linking `examples/autolabel_pitch.ipynb`. Cells:

**MD cell:**
```
# Pitch auto-label proposals (Phase 3.5b active-learning loop)
Predict (pose-vN) -> fit anchors -> propagate homographies -> project canonical
landmarks back into frames as YOLO-pose pre-labels. Correct these in Roboflow.
Pre-labels inherit propagation error (a few px) — review before training on them.
```

**Code cell (install package + params):**
```python
!pip install -q "git+https://github.com/PatrickJReed/soccer-vision.git#subdirectory=packages/soccer-vision[roboflow]"
from pathlib import Path
VIDEO = Path("/content/game.mp4")     # a training game
WEIGHTS = Path("/content/pitch_v0.pt")  # current pose model
OUT = Path("/content/autolabels"); OUT.mkdir(exist_ok=True)
MIN_CONF = 0.5
```

**Code cell (detect + keypoints via the backend):**
```python
from soccer_vision.tracking.roboflow import RoboflowBackend
backend = RoboflowBackend(detect_pitch=True, pitch_weights_path=WEIGHTS)
trajectories, keypoints = backend.process_with_pitch(VIDEO)
print("frames with keypoints:", keypoints['frame'].nunique())
```

**Code cell (anchors -> propagate):**
```python
import cv2
from soccer_vision.pitch.landmarks import build_frame_homographies, PITCH_LANDMARKS
from soccer_vision.pitch.propagation import (
    compute_interframe_homographies, propagate_homographies,
)

anchors = build_frame_homographies(keypoints)
cap = cv2.VideoCapture(str(VIDEO))
def read_frame(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    return f if ok else None

keys = sorted(anchors)
needed = {i for a, b in zip(keys, keys[1:]) for i in range(a, b)}
interframe = compute_interframe_homographies(read_frame, needed, trajectories)
homs = propagate_homographies(anchors, interframe, max_gap=45)
print("homographies (anchor+propagated):", len(homs))
```

**Code cell (project -> YOLO-pose labels):**
```python
from soccer_vision.pitch.autolabel import propose_labels, to_yolo_pose_line

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
proposals = propose_labels(homs, PITCH_LANDMARKS, (w, h), min_confidence=MIN_CONF)
img_dir = OUT / "images"; lbl_dir = OUT / "labels"
img_dir.mkdir(exist_ok=True); lbl_dir.mkdir(exist_ok=True)
for frame, kpts in proposals.items():
    img = read_frame(frame)
    if img is None:
        continue
    cv2.imwrite(str(img_dir / f"f{frame:06d}.jpg"), img)
    (lbl_dir / f"f{frame:06d}.txt").write_text(to_yolo_pose_line(kpts, (w, h)))
cap.release()
print("wrote", len(proposals), "pre-labeled frames")
```

**Code cell (overlay preview for spot-check):**
```python
import matplotlib.pyplot as plt
sample = sorted(proposals)[len(proposals)//2]
img = cv2.cvtColor(cv2.imread(str(img_dir / f"f{sample:06d}.jpg")), cv2.COLOR_BGR2RGB)
for x, y, v in proposals[sample]:
    if v > 0:
        cv2.circle(img, (int(x), int(y)), 6, (255, 0, 0), -1)
plt.figure(figsize=(12, 7)); plt.imshow(img); plt.title(f"frame {sample}"); plt.axis("off")
```

**Code cell (zip):**
```python
import shutil
shutil.make_archive("/content/autolabels", "zip", OUT)
from google.colab import files
files.download("/content/autolabels.zip")
```

- [ ] **Step 2: Validate JSON**

Run: `python3 -c "import json; json.load(open('examples/autolabel_pitch.ipynb'))" && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add examples/autolabel_pitch.ipynb
git commit -m "docs(examples): autolabel_pitch notebook (propagation-projected pre-labels)"
```

---

## Task 8: `acceptance_pitch.ipynb` — held-out multi-field coverage gate

**Files:**
- Create: `examples/acceptance_pitch.ipynb`

Runs the full pipeline on each held-out clip and reports the gate metrics from the
spec, split by unseen-field vs unseen-time.

- [ ] **Step 1: Create the notebook with these cells**

Colab-badge header linking `examples/acceptance_pitch.ipynb`. Cells:

**MD cell:**
```
# Pitch-v1 acceptance gate (Phase 3.5b)
Runs the pipeline on the held-out multi-field set (bake-off clip + clips from
>=2 other games/fields, >=1 entirely unseen in training).
Gates (aggregate): anchor coverage >= 40%, combined coverage >= 65%,
held-out reproj error <= 0.05 on every clip. Reports per-clip + per-field,
split unseen-field vs unseen-time.
```

**Code cell (install + clip registry):**
```python
!pip install -q "git+https://github.com/PatrickJReed/soccer-vision.git#subdirectory=packages/soccer-vision[roboflow]"
from pathlib import Path
WEIGHTS = Path("/content/pitch_v1.pt")
# Each clip: (path, field_id, split) where split in {"unseen_field","unseen_time"}.
CLIPS = [
    (Path("/content/bakeoff_clip.mp4"), "bakeoff", "unseen_field"),
    (Path("/content/heldout_fieldA.mp4"), "A", "unseen_field"),
    (Path("/content/heldout_fieldB.mp4"), "B", "unseen_time"),
]
```

**Code cell (per-clip coverage):**
```python
import cv2
from soccer_vision.tracking.roboflow import RoboflowBackend
from soccer_vision.pitch.landmarks import build_frame_homographies
from soccer_vision.pitch.propagation import (
    compute_interframe_homographies, propagate_homographies,
)

backend = RoboflowBackend(detect_pitch=True, pitch_weights_path=WEIGHTS)
rows = []
for path, field, split in CLIPS:
    traj, kps = backend.process_with_pitch(path)
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def read_frame(i, _cap=cap):
        _cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = _cap.read(); return f if ok else None
    anchors = build_frame_homographies(kps)
    keys = sorted(anchors)
    needed = {i for a, b in zip(keys, keys[1:]) for i in range(a, b)}
    interframe = compute_interframe_homographies(read_frame, needed, traj)
    homs = propagate_homographies(anchors, interframe, max_gap=45)
    cap.release()
    anchor_cov = len(anchors) / n
    combined_cov = len(homs) / n
    rows.append(dict(field=field, split=split, frames=n,
                     anchor_cov=anchor_cov, combined_cov=combined_cov))
    print(f"{field:8s} [{split}] anchor={anchor_cov:.1%} combined={combined_cov:.1%}")
```

**Code cell (aggregate gate verdict):**
```python
import pandas as pd
df = pd.DataFrame(rows)
agg_anchor = df["anchor_cov"].mean()
agg_combined = df["combined_cov"].mean()
print(f"AGG anchor={agg_anchor:.1%} (gate >=40%): {'PASS' if agg_anchor>=0.40 else 'FAIL'}")
print(f"AGG combined={agg_combined:.1%} (gate >=65%): {'PASS' if agg_combined>=0.65 else 'FAIL'}")
print("\nBy split:")
print(df.groupby('split')[['anchor_cov','combined_cov']].mean())
print("\nWorst field (anchor):", df.loc[df['anchor_cov'].idxmin(), 'field'])
```

**MD cell (held-out reproj + keypoint accuracy):**
```
Reprojection error and keypoint accuracy reuse the 3.5a held-out probe
(examples/colab_homography_propagation.ipynb) and a small labeled slice per
clip. Report median per-keypoint pixel error + per-landmark detection rate,
split unseen-field vs unseen-time. Reproj error must stay <= 0.05 on every clip.
```

- [ ] **Step 2: Validate JSON**

Run: `python3 -c "import json; json.load(open('examples/acceptance_pitch.ipynb'))" && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add examples/acceptance_pitch.ipynb
git commit -m "docs(examples): acceptance_pitch notebook (held-out multi-field gate)"
```

---

## Final verification

- [ ] **Run the full test suite + typecheck + lint**

Run: `cd packages/soccer-vision && uv run pytest && uv run mypy src/ && uv run ruff check src/ tests/`
Expected: all green.

- [ ] **Confirm all four notebooks are valid JSON**

Run: `for nb in extract_pitch_frames finetune_pitch autolabel_pitch acceptance_pitch; do python3 -c "import json; json.load(open('examples/$nb.ipynb'))" && echo "$nb OK"; done`
Expected: four `OK` lines.

---

## What's left to Patrick (out of plan scope — GPU/manual)

These are the human/GPU steps the code above enables; they are NOT subagent tasks:

1. Reserve held-out games/fields; sample training frames (`extract_pitch_frames.ipynb`).
2. Label the ~100-frame seed in Roboflow (21-pt skeleton, define the flip pairs).
3. Train pose-v0 (`finetune_pitch.ipynb`).
4. Run the active-learning loop (`autolabel_pitch.ipynb` → correct in Roboflow → retrain) toward 600+ frames.
5. Publish `pitch_yolov8_v1.pt` as the `pitch-v1` GitHub release asset (so `PITCH_V1_URL` resolves).
6. Run `acceptance_pitch.ipynb`; if gates pass, the wiring from Task 4 makes `detect_pitch=True` use the new weights automatically.

Early-bailout signal (from the spec): if anchor coverage stalls below ~25% after
a couple of active-learning rounds, reassess rather than grind more labels.
