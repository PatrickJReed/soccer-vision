# Calibration Global-Homography Rework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the labeler's per-frame independent SQPNP calibration with one global image→pitch homography per registration segment (`H_f = H_global ∘ T_f`), with an honest whole-field status/export gate and a held-out cross-end validation that is the acceptance bar.

**Architecture:** A new pure module `pitch/global_calib.py` pools every click across all frames — lifting each to its segment's reference frame via the existing cumulative-transform reduced to a 2D translation — and fits ONE homography per segment. Own-end and opp-end clicks (from different frames) land at different reference positions and jointly constrain that single homography, so no frame is under-constrained and there is no chained-error accumulation. `LabelerState` is rewired to this module behind its existing `_compute_*` seam; the HTTP/UI/export contracts and `pipeline.assemble_from_homographies` are unchanged.

**Tech Stack:** Python 3.11, numpy, OpenCV (`cv2.findHomography` RANSAC), pytest. uv workspace; run tests with `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-06-26-calibration-global-homography-design.md`. Two refinements made during planning (confirm if undesired):
1. **Live green criterion** is "segment has clicks at BOTH ends" (cheap, per-segment) rather than a per-frame held-out check; the held-out cross-end validation (Task 6) is the offline proof that justifies this criterion.
2. **Per-frame outlier flagging is dropped from the live path** — `cv2.findHomography(method=RANSAC)` rejects outlier clicks in the global fit, making `flag_outlier_clicks` (which needs the now-removed focal K) redundant live. `flag_outlier_clicks`/`_robust_sqpnp` are NOT deleted (spec §8 retains them); they are simply not wired into the global path.

**Coordinate conventions (verified against the code):**
- Clicks are normalized `[0,1]` (`Click.x`, `Click.y`).
- `LabelerState._transforms[f]` = `M[f]`: maps frame-`f` normalized pixels → its segment's reference-frame normalized pixels (`cumulative_transforms`, manual_anchor.py:76).
- `CalibFrame.H` is **normalized** image→pitch[0,1]; export denormalizes via `denormalize_homography` (state.py:365).
- `_to_norm = diag([w,h,1])`, and `h_norm = h_px @ _to_norm` (state.py:96,129), so `h_px = h_norm @ diag(1/w,1/h,1)`.
- `fold_count(h_pitch, size)` (calib/validate.py:20) takes a **pitch[0,1]→full-pixel** homography and counts the 21 landmarks landing in-frame; pass `inv(h_px)`.
- `PITCH_LANDMARKS` (landmarks.py:103): 21×2 canonical `[0,1]²`, y=goal-to-goal (own goal y=0, opp goal y=1). `NEAR_HALFWAY_IDX=5` is the hidden under-camera point.

---

## File Structure

- **Create** `packages/soccer-vision/src/soccer_vision/pitch/global_calib.py` — pure global-homography solver + frame emit + cross-end held-out validation. One responsibility: turn clicks+transforms into one homography per segment and per-frame H.
- **Create** `packages/soccer-vision/tests/test_pitch_global_calib.py` — unit tests for the above.
- **Modify** `packages/soccer-vision/src/soccer_vision/labeler/state.py` — rewire `_compute_poses`→global solve; whole-segment dirty marking; honest `_status_of`; export blocks-until-idle + honest gate; `CalibFrame` gains `two_ended`.
- **Modify** `packages/soccer-vision/src/soccer_vision/labeler/server.py` — expose `residual_px_threshold` (kept as a displayed diagnostic) and the status to the frontend (minimal).
- **Modify** `packages/soccer-vision/src/soccer_vision/labeler/static/app.js` — per-frame readout colour no longer hard-codes `0.05`.
- **Create** `packages/soccer-vision/tests/test_labeler_state.py` additions / **Modify** existing labeler tests that assert the residual gate.
- **Modify (Task 7, gated)** `pitch/calib_anchor.py`, `pitch/calib_compare.py`, and their tests — delete the dead per-frame engines.

---

## Task 1: Pure global-homography solver (`pitch/global_calib.py`)

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/global_calib.py`
- Test: `packages/soccer-vision/tests/test_pitch_global_calib.py`

- [ ] **Step 1: Write the failing test for `solve_global` round-trip recovery (single-end frames)**

```python
# packages/soccer-vision/tests/test_pitch_global_calib.py
from __future__ import annotations

import numpy as np

from soccer_vision.pitch.global_calib import (
    GlobalCalib,
    cross_end_holdout,
    solve_global,
)
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click

SIZE = (1920, 1080)


def _norm(pt_px: np.ndarray) -> np.ndarray:
    return pt_px / np.array([SIZE[0], SIZE[1]])


def _build_session():
    """A synthetic fixed-camera session: one known global homography (pitch->image_ref),
    each frame a pure 2D translation of the reference, and each frame clicks ONLY the
    landmarks that fall on 'its' end (so no single frame sees the whole field)."""
    # Ground-truth pitch[0,1] -> reference-image (normalized) homography.
    # A gentle perspective so the test is non-trivial but invertible.
    H_pitch_to_ref = np.array(
        [[0.6, 0.0, 0.2],
         [0.0, 0.9, 0.05],
         [0.0, 0.15, 1.0]],
        dtype=np.float64,
    )
    H_ref_to_pitch = np.linalg.inv(H_pitch_to_ref)

    # Per-frame normalized 2D offsets (frame pixel = ref pixel + offset_f).
    offsets = {0: np.array([0.0, 0.0]), 1: np.array([0.30, 0.0]),
               2: np.array([0.55, 0.0]), 3: np.array([-0.20, 0.0])}
    # M[f] maps frame-f -> ref, i.e. a pure translation by +offset_f.
    transforms = {
        f: np.array([[1.0, 0.0, off[0]], [0.0, 1.0, off[1]], [0.0, 0.0, 1.0]])
        for f, off in offsets.items()
    }
    segment_of = {f: 0 for f in offsets}

    own_end = [i for i in range(len(PITCH_LANDMARKS)) if PITCH_LANDMARKS[i, 1] < 0.5 and i != 5]
    opp_end = [i for i in range(len(PITCH_LANDMARKS)) if PITCH_LANDMARKS[i, 1] > 0.5 and i != 5]

    def click_for(frame: int, kp: int) -> Click:
        # true reference pixel for this landmark
        ref = H_pitch_to_ref @ np.array([*PITCH_LANDMARKS[kp], 1.0])
        ref = ref[:2] / ref[2]
        frame_pt = ref + offsets[frame]  # frame pixel = ref + offset
        return Click(frame=frame, kp_idx=kp, x=float(frame_pt[0]), y=float(frame_pt[1]))

    clicks = [click_for(0, kp) for kp in own_end]          # frame 0 sees own end only
    clicks += [click_for(2, kp) for kp in opp_end]         # frame 2 sees opp end only
    return clicks, transforms, segment_of, H_ref_to_pitch, own_end, opp_end


def test_solve_global_recovers_homography_and_projects_unclicked_end():
    clicks, transforms, segment_of, H_ref_to_pitch, own_end, opp_end = _build_session()
    gc = solve_global(clicks, transforms, segment_of, SIZE)
    assert isinstance(gc, GlobalCalib)
    assert set(gc.h_by_segment) == {0}

    # Frame 0 clicked ONLY the own end; its H must still project the OPP end correctly
    # (this is the "lines in the sky" regression the old per-frame engine failed).
    H0 = gc.frame_homography(0)
    assert H0 is not None
    for kp in opp_end:
        ref = np.linalg.inv(H_ref_to_pitch) @ np.array([*PITCH_LANDMARKS[kp], 1.0])
        ref = ref[:2] / ref[2]
        frame_pt = ref + transforms[0][:2, 2]  # offset_0 = 0, but keep general
        proj = H0 @ np.array([frame_pt[0], frame_pt[1], 1.0])
        proj = proj[:2] / proj[2]
        assert np.allclose(proj, PITCH_LANDMARKS[kp], atol=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_global_calib.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'soccer_vision.pitch.global_calib'`.

- [ ] **Step 3: Implement `pitch/global_calib.py`**

```python
# packages/soccer-vision/src/soccer_vision/pitch/global_calib.py
"""Global-homography calibration for a fixed (Trace virtual-PTZ) camera.

The whole session is one canvas: every frame is a 2D crop of it, so there is ONE
image_reference -> pitch[0,1] homography per registration segment. Each click is
lifted to its segment's reference frame (via the cumulative transform reduced to a
2D translation) and ALL clicks are fit jointly, so own-end and opp-end clicks from
different frames constrain a single homography (no per-frame under-constraint, no
chained-error accumulation). Per-frame H_f = H_global @ T_f. Pure: no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from soccer_vision.pitch.homography import HomographyError, fit_homography
from soccer_vision.pitch.landmarks import NEAR_HALFWAY_IDX, PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click

# Landmark index sets per field end (own goal at y=0, opp at y=1; midline y=0.5
# and the hidden under-camera point are excluded).
OWN_END_IDX: list[int] = [
    i for i in range(len(PITCH_LANDMARKS))
    if PITCH_LANDMARKS[i, 1] < 0.5 and i != NEAR_HALFWAY_IDX
]
OPP_END_IDX: list[int] = [
    i for i in range(len(PITCH_LANDMARKS))
    if PITCH_LANDMARKS[i, 1] > 0.5 and i != NEAR_HALFWAY_IDX
]

_CENTER = np.array([0.5, 0.5, 1.0])  # normalized image centre


def _translation_of(m: NDArray[np.floating]) -> NDArray[np.float64]:
    """2D translation M induces at the image centre (normalized). M maps frame -> ref,
    so this is the offset to ADD to a frame click to reach the reference frame."""
    mapped = np.asarray(m, dtype=np.float64) @ _CENTER
    mapped = mapped[:2] / mapped[2]
    return np.asarray(mapped - _CENTER[:2], dtype=np.float64)


def _translation_matrix(offset: NDArray[np.floating]) -> NDArray[np.float64]:
    return np.array([[1.0, 0.0, float(offset[0])],
                     [0.0, 1.0, float(offset[1])],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _apply_h(h: NDArray[np.floating], pts: NDArray[np.floating]) -> NDArray[np.float64]:
    """Apply a 3x3 homography to (N,2) points -> (N,2)."""
    homog = np.column_stack([np.asarray(pts, dtype=np.float64), np.ones(len(pts))])
    out = (np.asarray(h, dtype=np.float64) @ homog.T).T
    return np.asarray(out[:, :2] / out[:, 2:3], dtype=np.float64)


@dataclass(frozen=True, eq=False)
class GlobalCalib:
    """One image_ref -> pitch[0,1] homography per segment + per-frame 2D offsets."""

    h_by_segment: dict[int, NDArray[np.float64]]   # segment -> H_global (norm ref -> pitch)
    offsets: dict[int, NDArray[np.float64]]        # frame -> (dx, dy) normalized
    segment_of: dict[int, int]
    rms_by_segment: dict[int, float]               # diagnostic: in-sample reproj RMS (norm px)
    n_by_segment: dict[int, int]                   # clicks used per segment

    def frame_homography(self, frame: int) -> NDArray[np.float64] | None:
        """Normalized image_frame -> pitch[0,1] homography, or None if uncalibrated."""
        seg = self.segment_of.get(frame)
        if seg is None or seg not in self.h_by_segment:
            return None
        off = self.offsets.get(frame)
        if off is None:
            return None
        return np.asarray(self.h_by_segment[seg] @ _translation_matrix(off), dtype=np.float64)


def solve_global(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    size: tuple[int, int],
    *,
    min_points: int = 4,
) -> GlobalCalib:
    """Fit one image_ref -> pitch homography per segment from all clicks pooled.

    Each click is lifted to its segment's reference frame via the cumulative
    transform reduced to a 2D translation, then cv2.findHomography (RANSAC) fits
    the segment's homography over the union of its clicks (RANSAC rejects outliers).
    """
    offsets = {f: _translation_of(m) for f, m in transforms.items()}
    img_by_seg: dict[int, list[list[float]]] = {}
    pitch_by_seg: dict[int, list[NDArray[np.float64]]] = {}
    for c in clicks:
        if c.kp_idx == NEAR_HALFWAY_IDX:
            continue
        seg = segment_of.get(c.frame)
        off = offsets.get(c.frame)
        if seg is None or off is None:
            continue
        img_by_seg.setdefault(seg, []).append([c.x + float(off[0]), c.y + float(off[1])])
        pitch_by_seg.setdefault(seg, []).append(PITCH_LANDMARKS[c.kp_idx])

    h_by_seg: dict[int, NDArray[np.float64]] = {}
    rms_by_seg: dict[int, float] = {}
    n_by_seg: dict[int, int] = {}
    for seg, img_list in img_by_seg.items():
        if len(img_list) < min_points:
            continue
        img = np.asarray(img_list, dtype=np.float64)
        pitch = np.asarray(pitch_by_seg[seg], dtype=np.float64)
        try:
            h = fit_homography(img, pitch)
        except HomographyError:
            continue
        proj = _apply_h(h, img)
        h_by_seg[seg] = np.asarray(h, dtype=np.float64)
        rms_by_seg[seg] = float(np.sqrt(np.mean(np.sum((proj - pitch) ** 2, axis=1))))
        n_by_seg[seg] = len(img)

    return GlobalCalib(h_by_seg, offsets, dict(segment_of), rms_by_seg, n_by_seg)


def two_ended_segments(
    clicks: Sequence[Click], segment_of: Mapping[int, int]
) -> set[int]:
    """Segments whose clicks include BOTH an own-end and an opp-end landmark — i.e. the
    global homography is constrained across the whole field, so per-frame H is trustworthy."""
    own = set(OWN_END_IDX)
    opp = set(OPP_END_IDX)
    seen_own: set[int] = set()
    seen_opp: set[int] = set()
    for c in clicks:
        seg = segment_of.get(c.frame)
        if seg is None:
            continue
        if c.kp_idx in own:
            seen_own.add(seg)
        elif c.kp_idx in opp:
            seen_opp.add(seg)
    return seen_own & seen_opp


@dataclass(frozen=True)
class HoldoutReport:
    """Leave-one-frame-out cross-end accuracy in feet."""

    median_ft: float
    p90_ft: float
    own_median_ft: float | None
    opp_median_ft: float | None
    n: int


def cross_end_holdout(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    size: tuple[int, int],
    *,
    length_ft: float = 224.7,
    aspect_ratio: float = 1.5,
) -> HoldoutReport | None:
    """Leave-one-clicked-frame-out validation: for each clicked frame, refit the global
    homography from the OTHER frames' clicks and measure how far (feet) this frame's
    clicks land from their canonical pitch coords. Reports overall + per-end medians.
    None if there are <2 clicked frames. This is the acceptance bar (Task 6)."""
    from soccer_vision.eval.pitch_metrics import displacement_to_feet

    own = set(OWN_END_IDX)
    clicked_frames = sorted({c.frame for c in clicks})
    if len(clicked_frames) < 2:
        return None

    feet: list[float] = []
    own_feet: list[float] = []
    opp_feet: list[float] = []
    for f in clicked_frames:
        rest = [c for c in clicks if c.frame != f]
        held = [c for c in clicks if c.frame == f and c.kp_idx != NEAR_HALFWAY_IDX]
        if not rest or not held:
            continue
        gc = solve_global(rest, transforms, segment_of, size)
        h = gc.frame_homography(f)
        if h is None:
            continue
        for c in held:
            pred = _apply_h(h, np.array([[c.x, c.y]]))[0]
            disp = pred - PITCH_LANDMARKS[c.kp_idx]
            ft = float(displacement_to_feet(disp, length_ft=length_ft, aspect_ratio=aspect_ratio))
            feet.append(ft)
            (own_feet if c.kp_idx in own else opp_feet).append(ft)

    if not feet:
        return None
    return HoldoutReport(
        median_ft=float(np.median(feet)),
        p90_ft=float(np.percentile(feet, 90)),
        own_median_ft=float(np.median(own_feet)) if own_feet else None,
        opp_median_ft=float(np.median(opp_feet)) if opp_feet else None,
        n=len(feet),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_global_calib.py -x -q`
Expected: PASS.

- [ ] **Step 5: Add the remaining unit tests (two-ended, holdout, degenerate)**

```python
# append to packages/soccer-vision/tests/test_pitch_global_calib.py
from soccer_vision.pitch.global_calib import two_ended_segments


def test_two_ended_segments_detects_both_ends():
    clicks, transforms, segment_of, *_ = _build_session()
    assert two_ended_segments(clicks, segment_of) == {0}
    # own end only -> not two-ended
    own_only = [c for c in clicks if c.kp_idx in
                {i for i in range(len(PITCH_LANDMARKS)) if PITCH_LANDMARKS[i, 1] < 0.5}]
    assert two_ended_segments(own_only, segment_of) == set()


def test_cross_end_holdout_is_small_for_consistent_session():
    clicks, transforms, segment_of, *_ = _build_session()
    report = cross_end_holdout(clicks, transforms, segment_of, SIZE)
    assert report is not None
    # synthetic data is exactly planar+translation, so leave-one-frame-out should
    # reconstruct each end to within a few feet.
    assert report.median_ft < 5.0


def test_solve_global_skips_segment_with_too_few_points():
    transforms = {0: np.eye(3), 1: np.eye(3)}
    segment_of = {0: 0, 1: 0}
    clicks = [Click(0, 0, 0.1, 0.1), Click(0, 1, 0.2, 0.2)]  # only 2 points
    gc = solve_global(clicks, transforms, segment_of, SIZE)
    assert gc.h_by_segment == {}
    assert gc.frame_homography(0) is None
```

- [ ] **Step 6: Run all global_calib tests**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_global_calib.py -q`
Expected: PASS (4 tests). Then `uv run ruff check packages/soccer-vision/src/soccer_vision/pitch/global_calib.py` and `uv run mypy` — both clean.

- [ ] **Step 7: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/global_calib.py \
        packages/soccer-vision/tests/test_pitch_global_calib.py
git commit -m "feat(pitch): global_calib — one homography per segment from all clicks"
```

---

## Task 2: Honest per-frame status helper

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/global_calib.py`
- Test: `packages/soccer-vision/tests/test_pitch_global_calib.py`

The status rule (whole-field honest): **red** if no H or `fold_count` out of the physical range; **green** if fold-OK and the frame's segment is two-ended; **yellow** if fold-OK but single-ended.

- [ ] **Step 1: Write the failing test**

```python
# append to test_pitch_global_calib.py
from soccer_vision.pitch.global_calib import frame_status, fold_of_norm


def test_fold_of_norm_counts_in_frame_landmarks():
    clicks, transforms, segment_of, *_ = _build_session()
    gc = solve_global(clicks, transforms, segment_of, SIZE)
    fold = fold_of_norm(gc.frame_homography(0), SIZE)
    assert 0 <= fold <= len(PITCH_LANDMARKS)


def test_frame_status_green_yellow_red():
    clicks, transforms, segment_of, *_ = _build_session()
    gc = solve_global(clicks, transforms, segment_of, SIZE)
    two_ended = two_ended_segments(clicks, segment_of)
    # frame 0: plausible H + two-ended segment -> green
    assert frame_status(gc.frame_homography(0), SIZE, segment_two_ended=True) == "green"
    # plausible H but single-ended -> yellow
    assert frame_status(gc.frame_homography(0), SIZE, segment_two_ended=False) == "yellow"
    # no H -> red
    assert frame_status(None, SIZE, segment_two_ended=True) == "red"
    # a wildly-folding H (maps everything off-frame) -> red
    sky = np.array([[1e-6, 0, 0], [0, 1e-6, 0], [0, 0, 1.0]])
    assert frame_status(sky, SIZE, segment_two_ended=True) == "red"
    assert two_ended == {0}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_global_calib.py::test_frame_status_green_yellow_red -q`
Expected: FAIL with `ImportError: cannot import name 'frame_status'`.

- [ ] **Step 3: Implement `fold_of_norm` + `frame_status`**

```python
# add near the top imports of global_calib.py:
from soccer_vision.calib.validate import fold_count

# add to global_calib.py:
FOLD_MIN: int = 4    # a real narrow Trace view shows ~6-12 of 21; <FOLD_MIN = sky/off-frame
FOLD_MAX: int = 15   # folding pulls the far field in toward 21


def fold_of_norm(h_norm: NDArray[np.floating] | None, size: tuple[int, int]) -> int:
    """fold_count for a NORMALIZED image->pitch homography (converts to the pitch->pixel
    form fold_count expects). 0 if h_norm is None or singular."""
    if h_norm is None:
        return 0
    w, h = size
    h_px = np.asarray(h_norm, dtype=np.float64) @ np.diag([1.0 / w, 1.0 / h, 1.0])
    try:
        h_pitch_to_px = np.linalg.inv(h_px)
    except np.linalg.LinAlgError:
        return 0
    return fold_count(h_pitch_to_px, size)


def frame_status(
    h_norm: NDArray[np.floating] | None,
    size: tuple[int, int],
    *,
    segment_two_ended: bool,
    fold_min: int = FOLD_MIN,
    fold_max: int = FOLD_MAX,
) -> str:
    """red = no H or implausible whole-field projection; green = plausible AND the
    segment saw both ends (homography constrained across the field); yellow = plausible
    but single-ended (honestly under-constrained)."""
    if h_norm is None:
        return "red"
    fold = fold_of_norm(h_norm, size)
    if not fold_min <= fold <= fold_max:
        return "red"
    return "green" if segment_two_ended else "yellow"
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_global_calib.py -q`
Expected: PASS (6 tests). Then ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/global_calib.py \
        packages/soccer-vision/tests/test_pitch_global_calib.py
git commit -m "feat(pitch): honest whole-field frame_status (fold + two-ended), not residual"
```

---

## Task 3: Rewire `LabelerState` to the global solve

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py`
- Test: `packages/soccer-vision/tests/test_labeler_state.py`

Key behavioural changes: (a) `_compute_*` returns `CalibFrame`s built from the global solve; (b) `_affected(frame)` marks the WHOLE segment dirty (a click changes that segment's global H everywhere), not a ±window; (c) `_status_of` uses `frame_status`; (d) `CalibFrame` gains `two_ended`; (e) bootstrap no longer needs the focal K.

- [ ] **Step 1: Write the failing test — single-end frame is NOT green; both-ends session goes green**

```python
# add to packages/soccer-vision/tests/test_labeler_state.py
import numpy as np
from soccer_vision.labeler.state import LabelerState
from soccer_vision.pitch.global_calib import OPP_END_IDX, OWN_END_IDX
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

SIZE = (1920, 1080)


def _identity_chain(n):
    # all frames linked by identity inter-frame transforms (one segment, zero motion)
    return {i: np.eye(3) for i in range(n - 1)}


def _clicks_for(frame, ids, H_pitch_to_img_norm):
    out = []
    for kp in ids:
        p = H_pitch_to_img_norm @ np.array([*PITCH_LANDMARKS[kp], 1.0])
        p = p[:2] / p[2]
        out.append((frame, kp, float(p[0]), float(p[1])))
    return out


def test_single_end_session_is_never_green_but_two_ended_is():
    H = np.array([[0.5, 0.0, 0.25], [0.0, 0.5, 0.25], [0.0, 0.0, 1.0]])
    st = LabelerState(_identity_chain(10), 10, size=SIZE)
    # own-end clicks only, on frame 0
    for f, kp, x, y in _clicks_for(0, OWN_END_IDX, H):
        st.add_click(f, kp, x, y)
    st.wait_idle()
    assert "green" not in st.status_list()      # single-ended -> yellow at best
    # now add opp-end clicks on frame 5 -> segment becomes two-ended
    for f, kp, x, y in _clicks_for(5, OPP_END_IDX, H):
        st.add_click(f, kp, x, y)
    st.wait_idle()
    assert "green" in st.status_list()          # both ends constrain the global H
    st.stop_worker()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/soccer-vision/tests/test_labeler_state.py::test_single_end_session_is_never_green_but_two_ended_is -q`
Expected: FAIL (current code green-gates on residual, so single-end frames go green).

- [ ] **Step 3: Edit `state.py` — imports + `CalibFrame`**

Replace the calib_anchor import block (state.py:24-30) with:

```python
from soccer_vision.pitch.global_calib import (
    GlobalCalib,
    frame_status,
    fold_of_norm,
    solve_global,
    two_ended_segments,
)
```

Add `two_ended` to `CalibFrame` (state.py:42-49):

```python
@dataclass(frozen=True, eq=False)
class CalibFrame:
    """A calibrated per-frame result in the labeler's normalized space."""

    H: NDArray[np.float64]  # NORMALIZED image -> pitch[0,1] (frontend overlay)
    residual: float         # in-sample global-fit RMS (norm px) — diagnostic only
    n_points: int           # clicks in this frame's segment
    fold_count: int
    two_ended: bool         # the segment saw both field ends (drives green vs yellow)
```

- [ ] **Step 4: Edit `state.py` — replace `_compute_poses` + drop the K bootstrap**

Replace `_try_bootstrap` (state.py:107-124), `_calib_frame` (126-131), and `_compute_poses` (140-155) with:

```python
    def _try_bootstrap(self) -> bool:
        """Calibrated once any segment has >= 4 clicks (a homography is fittable).
        No focal/K: the global model is a plain image->pitch homography."""
        if self._calibrated:
            return True
        with self._lock:
            clicks = list(self._active_clicks())
        gc = solve_global(clicks, self._transforms, self._segment_of, self.size)
        if not gc.h_by_segment:
            return False
        with self._lock:
            self._calibrated = True
        return True

    def _compute_poses(self, frames: Sequence[int]) -> dict[int, CalibFrame]:
        """Solve the global homography (per segment) from ALL clicks, then emit a
        CalibFrame for each requested frame. Snapshots inputs under the lock and solves
        off the lock. The global solve is cheap (one DLT per segment), so recomputing it
        per chunk is acceptable."""
        with self._lock:
            if not self._calibrated:
                return {}
            clicks = list(self._active_clicks())
        gc = solve_global(clicks, self._transforms, self._segment_of, self.size)
        two_ended = two_ended_segments(clicks, self._segment_of)
        out: dict[int, CalibFrame] = {}
        for f in frames:
            h_norm = gc.frame_homography(f)
            if h_norm is None:
                continue
            seg = self._segment_of.get(f)
            out[f] = CalibFrame(
                H=h_norm,
                residual=gc.rms_by_segment.get(seg, float("nan")) if seg is not None else float("nan"),
                n_points=gc.n_by_segment.get(seg, 0) if seg is not None else 0,
                fold_count=fold_of_norm(h_norm, self.size),
                two_ended=(seg in two_ended),
            )
        return out
```

Note: `_active_clicks` no longer depends on K/outliers; leave it returning `self.clicks` (the `_outliers` map stays empty since `flag_outlier_clicks` is no longer wired in — RANSAC in `fit_homography` rejects outliers). `self._K`, `self._outliers` remain as unused fields for now; Task 7 removes them.

- [ ] **Step 5: Edit `state.py` — worker type + `_compute_dirty`/`_apply_fits`/`_refit_one`**

`_compute_poses` now returns `CalibFrame`, so the worker generic and applier change. Update the worker field (state.py:97-98):

```python
        self._worker: RefitWorker[CalibFrame | None] = RefitWorker(
            self._compute_dirty, self._apply_fits)
```

Update `_compute_dirty` return type and body (state.py:157-172) to use `CalibFrame`:

```python
    def _compute_dirty(
        self, frames: Sequence[int], is_cancelled: Callable[[], bool]
    ) -> dict[int, CalibFrame | None] | None:
        out: dict[int, CalibFrame | None] = {}
        ordered = list(frames)
        for i in range(0, len(ordered), self._refit_chunk):
            if is_cancelled():
                return None
            chunk = ordered[i:i + self._refit_chunk]
            solved = self._compute_poses(chunk)
            for f in chunk:
                out[f] = solved.get(f)
        return out
```

Update `_apply_fits` (state.py:174-181) to store `CalibFrame` directly (no `_calib_frame`):

```python
    def _apply_fits(self, results: dict[int, CalibFrame | None]) -> None:
        with self._lock:
            for f, cf in results.items():
                if cf is None:
                    self._fits.pop(f, None)
                else:
                    self._fits[f] = cf
```

Remove the now-unused `FramePose` import and `calibrate_clicked_frames`/`flag_outlier_clicks`/`frame_homography`/`poses_by_gated_propagation` imports from state.py:24-30 (already replaced in Step 3). Remove the `from soccer_vision.calib.calibrate import CalibError` import if no longer referenced.

- [ ] **Step 6: Edit `state.py` — whole-segment dirty marking + honest status**

Replace `_affected` (state.py:222-226):

```python
    def _affected(self, frame: int) -> list[int]:
        """A click changes the global homography for its ENTIRE segment, so every frame
        in that segment must be recomputed (not just a window)."""
        seg = self._segment_of.get(frame)
        return [f for f in range(self.n_frames) if self._segment_of.get(f) == seg]
```

Replace `_status_of` (state.py:319-324):

```python
    def _status_of(self, f: int) -> str:
        with self._lock:
            cf = self._fits.get(f)
        if cf is None:
            return "red"
        return frame_status(cf.H, self.size, segment_two_ended=cf.two_ended)
```

- [ ] **Step 7: Run the new test + the full labeler suite**

Run: `uv run pytest packages/soccer-vision/tests/test_labeler_state.py -q`
Expected: the new test PASSES. Other tests in this file that asserted residual-based green/`n_points`/`fold` from the SQPNP path will need updating — fix their expectations to the global model (e.g. assertions on `cf.residual < 60` become assertions on status/`two_ended`). Update each failing assertion to match the new semantics; do not weaken a test to pass — re-derive the expected value.

- [ ] **Step 8: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/state.py \
        packages/soccer-vision/tests/test_labeler_state.py
git commit -m "feat(labeler): LabelerState uses the global-homography solve + honest status"
```

---

## Task 4: Honest export — block until idle, gate on whole-field green

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py`
- Test: `packages/soccer-vision/tests/test_labeler_state.py`

- [ ] **Step 1: Write the failing test**

```python
# add to test_labeler_state.py
def test_export_skips_non_green_and_blocks_until_idle(tmp_path):
    H = np.array([[0.5, 0.0, 0.25], [0.0, 0.5, 0.25], [0.0, 0.0, 1.0]])
    st = LabelerState(_identity_chain(10), 10, size=SIZE)
    for f, kp, x, y in _clicks_for(0, OWN_END_IDX, H):   # single-ended -> yellow
        st.add_click(f, kp, x, y)
    st.wait_idle()
    st.export(tmp_path)
    st.stop_worker()
    import pandas as pd
    h = pd.read_parquet(tmp_path / "homographies.parquet")
    assert len(h) == 0   # nothing green -> nothing exported (no sky homographies)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/soccer-vision/tests/test_labeler_state.py::test_export_skips_non_green_and_blocks_until_idle -q`
Expected: FAIL — current export gates on `residual <= threshold`, so single-end (low-residual) frames are exported.

- [ ] **Step 3: Edit `export` (state.py:350-366)**

```python
    def export(self, out_dir: Path) -> None:
        # Block until the background worker has fully drained — never write a partial set.
        while self.pending() > 0:
            self.wait_idle()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        w, h = self.size
        px_clicks = [Click(c.frame, c.kp_idx, c.x * w, c.y * h) for c in self.clicks]
        clicks_to_keypoints_df(px_clicks).to_parquet(out / "keypoints.parquet", index=False)
        entries: dict[int, HomographyEntry] = {}
        for f in range(self.n_frames):
            with self._lock:  # single locked read: no green-then-missing TOCTOU window
                cf = self._fits.get(f)
            if cf is None or self._status_of(f) != "green":  # only export trustworthy frames
                continue
            # Confidence: 1.0 for a two-ended green frame (whole-field constrained); the
            # in-sample RMS is a soft penalty, not the gate.
            conf = float(np.clip(1.0 - cf.residual / max(self.residual_px_threshold, 1e-9), 0.0, 1.0)) \
                if np.isfinite(cf.residual) else 1.0
            entries[f] = HomographyEntry(
                denormalize_homography(cf.H, self.size), "manual", conf)
        homographies_to_parquet(entries, out / "homographies.parquet")
        if self.line_clicks:
            pd.DataFrame(
                [{"frame": lc.frame, "line_id": lc.line_id, "x_px": lc.x * w, "y_px": lc.y * h}
                 for lc in self.line_clicks],
                columns=["frame", "line_id", "x_px", "y_px"],
            ).to_parquet(out / "line_clicks.parquet", index=False)
```

Note: `_status_of` takes the lock internally; calling it here (outside the `with self._lock` block) avoids RLock re-entrancy concerns since `RLock` is re-entrant on the same thread anyway, but to be safe read `cf` first (as shown) and compute status from `cf` directly if preferred. Keeping `self._status_of(f)` is correct because `RLock` is reentrant.

- [ ] **Step 4: Run to verify it passes + full labeler suite**

Run: `uv run pytest packages/soccer-vision/tests/test_labeler_state.py packages/soccer-vision/tests/test_labeler_server.py -q`
Expected: PASS (update any export test asserting the old conf/gate).

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/state.py \
        packages/soccer-vision/tests/test_labeler_state.py
git commit -m "fix(labeler): export blocks until idle and gates on whole-field green"
```

---

## Task 5: Frontend residual readout colour (stop hard-coding 0.05)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/server.py`
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/static/app.js`
- Test: `packages/soccer-vision/tests/test_labeler_server.py`

- [ ] **Step 1: Write the failing test (server exposes the threshold)**

```python
# add to test_labeler_server.py — assert /api/state includes residual_px_threshold
def test_state_payload_includes_residual_threshold(make_client):
    client = make_client()  # use the existing fixture/helper in this file
    state = client.get_json("/api/state")
    assert "residual_px_threshold" in state
```

If the test file has no `make_client` helper, mirror the existing pattern used by the other `/api/state` tests in `test_labeler_server.py` (reuse whatever client/handler fixture they use).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/soccer-vision/tests/test_labeler_server.py -k residual_threshold -q`
Expected: FAIL (`KeyError`/assert).

- [ ] **Step 3: Add `residual_px_threshold` to the `/api/state` payload**

In `server.py` `_state_payload` (~lines 54-66) add to the returned dict:

```python
        "residual_px_threshold": state.residual_px_threshold,
```

- [ ] **Step 4: Fix the frontend colour (app.js ~line 95)**

Replace the hard-coded `0.05`:

```javascript
// was: const ok = fh.residual <= 0.05;
const thr = (state && state.residual_px_threshold) || 60.0;
const ok = fh.residual <= thr;
```

(Use whatever variable currently holds the `/api/state` JSON in `app.js`; if the readout render path doesn't already have it, store the threshold once when `/api/state` is fetched.)

- [ ] **Step 5: Run server tests + commit**

Run: `uv run pytest packages/soccer-vision/tests/test_labeler_server.py -q`
Expected: PASS.

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/server.py \
        packages/soccer-vision/src/soccer_vision/labeler/static/app.js \
        packages/soccer-vision/tests/test_labeler_server.py
git commit -m "fix(labeler): per-frame readout colour uses the server threshold, not 0.05"
```

---

## Task 6: Cross-end validation run — THE ACCEPTANCE BAR (gate before cleanup)

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/validate_session.py` (a tiny CLI wrapper)
- Test: `packages/soccer-vision/tests/test_pitch_global_calib.py` (already covers `cross_end_holdout`)

- [ ] **Step 1: Add a CLI that loads a labeler session and prints the held-out report**

```python
# packages/soccer-vision/src/soccer_vision/pitch/validate_session.py
"""CLI: measure the global-homography held-out cross-end accuracy for a labeler session.

Usage:
  uv run python -m soccer_vision.pitch.validate_session \
      --chain ~/sv-labeler/.sv_labeler_cache/<hash>.npz \
      --clicks ~/sv-labeler/.sv_labeler_cache/training_clip.clicks.json \
      --width 1920 --height 1080 --n-frames 2700
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from soccer_vision.labeler.chain import load_chain  # existing chain loader
from soccer_vision.labeler.state import clicks_from_sidecar
from soccer_vision.pitch.global_calib import cross_end_holdout
from soccer_vision.pitch.manual_anchor import build_segments, cumulative_transforms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", type=Path, required=True)
    ap.add_argument("--clicks", type=Path, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--n-frames", type=int, required=True)
    args = ap.parse_args()

    interframe = load_chain(args.chain)          # {i: 3x3} normalized inter-frame
    clicks = clicks_from_sidecar(args.clicks)
    segment_of = build_segments(interframe, args.n_frames)
    transforms = cumulative_transforms(interframe, segment_of)
    report = cross_end_holdout(clicks, transforms, segment_of, (args.width, args.height))
    if report is None:
        print("Not enough clicked frames for a held-out report.")
        return
    print(f"leave-one-frame-out cross-end error (feet):")
    print(f"  overall median={report.median_ft:.2f}  p90={report.p90_ft:.2f}  n={report.n}")
    print(f"  own-end median={report.own_median_ft}  opp-end median={report.opp_median_ft}")
    print(f"  ACCEPTANCE (<1.5m ~= 4.92 ft): {'PASS' if report.median_ft < 4.92 else 'FAIL'}")


if __name__ == "__main__":
    main()
```

Note: confirm the exact name of the chain loader in `labeler/chain.py` (e.g. `load_chain`); if it differs, use the actual function. If clicks live in a different sidecar than the JSON (e.g. an exported `keypoints.parquet`), use `clicks_from_keypoints_parquet` instead.

- [ ] **Step 2: Run it on the real training session (HUMAN STEP — Patrick runs this)**

Run (paths from the handoff doc):
```bash
cd packages/soccer-vision
uv run python -m soccer_vision.pitch.validate_session \
  --chain ~/sv-labeler/.sv_labeler_cache/ef2546eaddd5e6fc.npz \
  --clicks ~/sv-labeler/.sv_labeler_cache/training_clip.clicks.json \
  --width 1920 --height 1080 --n-frames 2700
```
Expected: an overall median in feet. **Acceptance bar: overall median < 1.5 m (~4.92 ft).**

- [ ] **Step 3: Decision gate**

- If PASS → proceed to Task 7 (cleanup).
- If FAIL → STOP. The model-A translation-offset assumption is insufficient for this session. Do NOT delete the old engines. Surface the report; the first lever is the spec's documented contingency (direct phase-correlation registration of the clicked frames to the reference instead of the chain-reduced translation), still model A. Re-evaluate before cleanup.

- [ ] **Step 4: Commit the CLI**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/validate_session.py
git commit -m "feat(pitch): validate_session CLI — held-out cross-end accuracy bar"
```

---

## Task 7: Cleanup — delete the dead per-frame engines (ONLY after Task 6 PASS)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py`
- Delete: `packages/soccer-vision/src/soccer_vision/pitch/calib_compare.py` and `tests/test_pitch_calib_compare.py`
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py` (remove unused K/outlier fields + window/seed/gate/gap params)
- Modify: `tests/test_pitch_calib_anchor.py` (remove tests for deleted engines)

- [ ] **Step 1: Delete the dead engines from `calib_anchor.py`**

Remove: `poses_by_pose_propagation` + `_rotation_from_chain` (Engine B), `poses_by_click_propagation`, `poses_by_gated_propagation`. Keep: `frame_homography`, `calibrate_clicked_frames`, `_reproj_rms_px`, `_fold_for_pose`, `flag_outlier_clicks`, `_robust_sqpnp` (spec §8 retains the outlier helpers).

- [ ] **Step 2: Delete `calib_compare.py` + its test**

```bash
git rm packages/soccer-vision/src/soccer_vision/pitch/calib_compare.py \
       packages/soccer-vision/tests/test_pitch_calib_compare.py
```

- [ ] **Step 3: Remove unused state.py fields/params**

Delete from `LabelerState.__init__`: `window`, `seed_size`, `gate_px`, `gap_dist`, `line_band` usages that referenced the gated engine; and the now-unused `self._K`, `self._outliers`, `self._calibrated`'s K dependency (keep `_calibrated` as the "≥1 segment solvable" flag). Remove `line_band` only if `_line_obs` is no longer used — NOTE line clicks are still stored and exported; keep `line_clicks`, `_line_obs`, `propagate_line_clicks` (lines may feed a future refine), but they no longer gate the point solve. Update `server.py`/`__main__.py` if they pass the removed kwargs.

- [ ] **Step 4: Remove tests for deleted engines**

In `tests/test_pitch_calib_anchor.py`, delete tests that call `poses_by_*propagation` / `compare_engines`. Keep tests for `flag_outlier_clicks`, `frame_homography`, `calibrate_clicked_frames`.

- [ ] **Step 5: Full suite + lint + types**

Run: `uv run pytest -q && uv run ruff check packages/soccer-vision/src && uv run mypy`
Expected: all green; coverage may shift. Fix any import errors from the deletions.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(pitch): delete dead per-frame calibration engines (global solve is live)"
```

---

## Self-Review

**Spec coverage:**
- §3 model (`H_f = H_global ∘ T_f`) → Task 1 (`solve_global`, `frame_homography`). ✓
- §4 module/seam → Task 1 (module) + Task 3 (seam). ✓
- §5 offsets (translation reduction of `M[f]`) → Task 1 (`_translation_of`). ✓
- §6 honest status (fold + cross-end) → Task 2 (`frame_status`) + Task 3 (`_status_of`); export honesty + block-until-idle → Task 4; frontend colour → Task 5. ✓
- §7 cross-end validation acceptance bar → Task 1 (`cross_end_holdout`) + Task 6 (run). ✓ (refined to leave-one-frame-out; flagged in header.)
- §8 cleanup → Task 7, gated on Task 6 PASS. ✓
- §9 testing (single-end "sky" regression; honest gate; offsets) → Task 1 Step 1, Task 2, Task 3 Step 1. ✓

**Placeholder scan:** No TBD/TODO. Every code step has complete code. The two human-run/decision steps (Task 6 Steps 2-3) are explicitly marked and necessary (require the real clip + cache).

**Type consistency:** `CalibFrame` gains `two_ended` (Task 3) and is used consistently in `_compute_poses`/`_apply_fits`/`_status_of`/`export`. `GlobalCalib.frame_homography` returns normalized image→pitch everywhere. `frame_status(h_norm, size, *, segment_two_ended)` signature matches its call in `_status_of`. `fold_of_norm` takes normalized H everywhere.

**Open items to confirm with the user (from header):** (1) leave-one-frame-out vs the spec's leave-one-end-out; (2) dropping `flag_outlier_clicks` from the live path (RANSAC subsumes it). Both noted at top.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-26-calibration-global-homography.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks. Task 6 is a hard gate (Patrick runs the validation on the real clip; cleanup waits on PASS).
2. **Inline Execution** — execute tasks in this session with checkpoints.

Note: Tasks 1–2 (the pure module) are pure and independent; Tasks 3–5 touch the live labeler; Task 6 is the human-run acceptance gate; Task 7 is gated on Task 6.
