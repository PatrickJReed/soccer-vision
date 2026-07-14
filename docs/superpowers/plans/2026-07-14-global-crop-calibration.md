# Global-Crop Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-anchor 6-DOF pose calibration with the exact virtual-PTZ crop model — one global homography per registration segment + a 2-DOF per-frame offset — plus honest status/confidence/export and two mechanical guards.

**Architecture:** New pure module `pitch/global_crop.py` (`solve_crop_session → CropCalib`) implementing the same duck-typed surface `LabelerState` already consumes from `PhysicalCalib` (`frame_homography/status/is_anchor/nearest_anchor_gap/anchor_h`). `H_f = H_g @ T(d_f)`; offsets at clicked frames come from clicks, unclicked frames use short-hop chain deltas relative to bracketing anchors. Spec: `docs/superpowers/specs/2026-07-14-global-crop-calibration-design.md`.

**Tech Stack:** numpy, OpenCV (`cv2.findHomography` via existing `fit_homography`), `scipy.optimize.least_squares`, `scipy.spatial.ConvexHull`. Python 3.11, mypy --strict, ruff.

**Conventions (read first):**
- All labeler-internal coords are NORMALIZED [0,1] image units. Export denormalizes: `H_px = H_norm @ diag(1/W, 1/H, 1)`.
- Chain `transforms[f] = M[f]` maps frame-f normalized coords → the segment's reference-frame coords (`manual_anchor.cumulative_transforms`). Under the crop model M ≈ translation, so a pixel `p` in frame f sits at `p + d_f` on the "canvas" (reference coords); `d_f init = M[f][:2,2] / M[f][2,2]`.
- All solver residuals are in METRES (`x·WIDTH_M(45.7), y·LENGTH_M(68.5)`); reports in feet (×3.28084).
- Run pytest from `packages/soccer-vision/`; run `uv run mypy` and `uv run ruff check` from the REPO ROOT (canonical config — src-only mypy masks test-file errors).
- Commit after every task. Never commit with failing tests.

**File map:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/global_crop.py` (Tasks 1–7)
- Create: `packages/soccer-vision/tests/test_global_crop.py` (Tasks 1–7)
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py`, `labeler/server.py`, `labeler/__main__.py` (Task 8)
- Create: `packages/soccer-vision/tests/test_labeler_state_crop.py` (Task 8)
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/validate_session.py` (Task 9)
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/mapper.py`, `pitch/manual_anchor.py`, `pitch/landmarks.py` (Task 10)

---

### Task 1: Module skeleton + synthetic crop world fixture

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/global_crop.py`
- Create: `packages/soccer-vision/tests/test_global_crop.py`

- [ ] **Step 1: Write the failing test** — the fixture builds a ground-truth crop world from a proven synthetic camera (same recipe as `test_physical_calib.py`), and the first test checks the geometry helpers round-trip.

```python
"""Tests for pitch/global_crop.py — the exact virtual-PTZ crop model."""
import numpy as np
import pytest
from numpy.typing import NDArray

from soccer_vision.pitch.calib_anchor import frame_homography
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click, LineClick
from soccer_vision.pitch import global_crop as gc

SIZE = (1920, 1080)
W, H = SIZE
K_TRUE = np.array([[1460.0, 0, W / 2], [0, 1460.0, H / 2], [0, 0, 1.0]])
# One fixed camera viewing midfield (proven pose family from test_physical_calib).
RVEC = np.array([[1.22], [0.0], [0.0]])
TVEC = np.array([[-22.0], [-3.0], [40.0]])


def _h_g_true() -> NDArray[np.float64]:
    """Ground-truth canvas(norm)->pitch homography from the physical camera."""
    h_px = frame_homography(K_TRUE, RVEC, TVEC)          # full-pixel image -> pitch
    return np.asarray(h_px @ np.diag([float(W), float(H), 1.0]), np.float64)


def _canvas_of_landmarks(h_g: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalized canvas position of each of the 21 landmarks (inverse map)."""
    inv = np.linalg.inv(h_g)
    pts = np.column_stack([PITCH_LANDMARKS, np.ones(len(PITCH_LANDMARKS))])
    q = (inv @ pts.T).T
    return np.asarray(q[:, :2] / q[:, 2:3], np.float64)


class CropWorld:
    """Ground truth: h_g + per-frame offsets + generated clicks + exact chain."""

    def __init__(self, n_frames: int = 240, drift: float = 0.0) -> None:
        self.n_frames = n_frames
        self.h_g = _h_g_true()
        canvas = _canvas_of_landmarks(self.h_g)
        span = canvas[:, 0].max() - canvas[:, 0].min()
        # Pan sweeps the canvas x-range so both field ends are seen by some frame.
        x0 = canvas[:, 0].min() - 0.1
        self.offsets = {
            f: np.array([x0 + span * 1.2 * f / (n_frames - 1) - 0.1, 0.0])
            for f in range(n_frames)
        }
        self.canvas = canvas
        # chain M[f] = T(d_f), optionally corrupted with linear drift in x
        self.transforms = {
            f: np.array([[1.0, 0.0, d[0] + drift * f], [0.0, 1.0, d[1]], [0.0, 0.0, 1.0]])
            for f, d in self.offsets.items()
        }

    def visible(self, frame: int) -> list[int]:
        d = self.offsets[frame]
        out = []
        for i, c in enumerate(self.canvas):
            x, y = c[0] - d[0], c[1] - d[1]
            if 0.02 <= x <= 0.98 and 0.02 <= y <= 0.98 and i != 5:
                out.append(i)
        return out

    def click(self, frame: int, kp_idx: int) -> Click:
        d = self.offsets[frame]
        c = self.canvas[kp_idx]
        return Click(frame=frame, kp_idx=kp_idx, x=float(c[0] - d[0]), y=float(c[1] - d[1]))

    def clicks_at(self, frame: int, ids: list[int] | None = None) -> list[Click]:
        ids = self.visible(frame) if ids is None else ids
        return [self.click(frame, i) for i in ids]


@pytest.fixture(scope="module")
def world() -> CropWorld:
    w = CropWorld()
    ends = {f: np.mean([PITCH_LANDMARKS[i][1] for i in w.visible(f)])
            for f in range(w.n_frames) if len(w.visible(f)) >= 4}
    # self-check: the pan really shows both ends (else the fixture constants are wrong)
    assert min(ends.values()) < 0.4 and max(ends.values()) > 0.6
    return w


def test_translation_and_compose_roundtrip(world: CropWorld) -> None:
    f = 30
    assert np.allclose(gc._translation(world.transforms[f]), world.offsets[f])
    h_f = world.h_g @ gc._t(world.offsets[f])
    c = world.clicks_at(f)[0]
    q = gc._apply(h_f, np.array([[c.x, c.y]]))[0]
    assert np.linalg.norm(q - PITCH_LANDMARKS[c.kp_idx]) < 1e-9


def test_residuals_are_metres(world: CropWorld) -> None:
    f = 30
    po = [(c.kp_idx, c.x, c.y) for c in world.clicks_at(f)]
    r = gc._point_residuals_m(world.h_g, world.offsets[f], po)
    assert r.shape == (2 * len(po),)
    assert np.abs(r).max() < 1e-6  # exact clicks -> zero metre residual
    lo = [("midline", *map(float, world.canvas[4] - world.offsets[f]))]
    rl = gc._line_residuals_m(world.h_g, world.offsets[f], lo)
    assert rl.shape == (1,) and abs(rl[0]) < 1e-6  # landmark 4 lies ON the midline
```

- [ ] **Step 2: Run to verify failure**

Run: `cd packages/soccer-vision && uv run pytest tests/test_global_crop.py -x -q`
Expected: FAIL — `module 'soccer_vision.pitch' has no attribute/module 'global_crop'`.

- [ ] **Step 3: Implement the skeleton**

```python
"""Global-crop calibration: ONE homography per registration segment + a 2-DOF
per-frame crop offset. Models the Trace virtual-PTZ exactly (each frame is a 2D
crop of one fixed view), so every click in a segment constrains the same global
homography and one click fully determines a clicked frame.

H_f = H_g @ T(d_f): pixel p in frame f sits at p + d_f on the segment's canvas
(the chain reference frame's coordinate system, normalized units). Offsets at
clicked frames come from clicks; unclicked frames use short-hop chain deltas
relative to their bracketing anchors — long chain compositions never enter.
Pure: no I/O. Spec: docs/superpowers/specs/2026-07-14-global-crop-calibration-design.md
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore[import-untyped]
from scipy.spatial import ConvexHull, QhullError  # type: ignore[import-untyped]

from soccer_vision.calib.field_model import LENGTH_M, METRES_TO_FEET, WIDTH_M
from soccer_vision.calib.validate import fold_count
from soccer_vision.pitch.homography import HomographyError, fit_homography
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click, LineClick

RANSAC_THRESH_PITCH = 0.012   # global-fit inlier gate, pitch units (~0.8 m)
HULL_AREA_MIN = 0.02          # min spread (pitch-units^2) of fit landmarks
Y_SPAN_ONE_END = 0.5          # below this y-span the session saw ~one end -> cap yellow
POINT_OK_FT = 6.0             # in-sample anchor tolerances (same convention as physical)
LINE_OK_FT = 4.0
FOLD_MIN, FOLD_MAX = 4, 15
GREEN_RADIUS = 100
DEFAULT_GAP_GUARD = 200
CONF_ANCHOR = 0.9
CONF_PROP_MAX, CONF_PROP_MIN = 0.8, 0.6
PRIOR_WEIGHT = 0.05           # weak chain prior for offset DOF the clicks don't constrain

_SCALE_M = np.array([WIDTH_M, LENGTH_M])
_FT = METRES_TO_FEET
# line_id -> (pitch axis index, constant): the five named lines are axis-aligned
_LINE_PITCH: dict[str, tuple[int, float]] = {
    "near_touchline": (0, 0.0),
    "far_touchline": (0, 1.0),
    "own_goal_line": (1, 0.0),
    "opp_goal_line": (1, 1.0),
    "midline": (1, 0.5),
}

PointObs = tuple[int, float, float]   # (kp_idx, x_norm, y_norm)
LineObs = tuple[str, float, float]    # (line_id, x_norm, y_norm)


def _translation(m: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Translation component of a (near-translation) chain transform."""
    a = np.asarray(m, dtype=np.float64)
    return np.asarray(a[:2, 2] / a[2, 2], dtype=np.float64)


def _t(d: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    return np.array([[1.0, 0.0, float(d[0])], [0.0, 1.0, float(d[1])], [0.0, 0.0, 1.0]])


def _apply(h: NDArray[np.floating[Any]], pts: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """pts (N,2) -> (N,2) under homography h (no w<=0 guard: solver-internal)."""
    p = np.column_stack([np.asarray(pts, np.float64), np.ones(len(pts))])
    q = (np.asarray(h, np.float64) @ p.T).T
    return np.asarray(q[:, :2] / q[:, 2:3], np.float64)


def _point_residuals_m(
    h_g: NDArray[np.floating[Any]], d: NDArray[np.floating[Any]], po: Sequence[PointObs]
) -> NDArray[np.float64]:
    """Per-point (2N,) metre residuals of clicks at offset d against their landmarks."""
    if not po:
        return np.zeros(0)
    pts = np.array([[x + d[0], y + d[1]] for _, x, y in po], np.float64)
    q = _apply(h_g, pts)
    lms = PITCH_LANDMARKS[[i for i, _, _ in po]]
    return np.asarray(((q - lms) * _SCALE_M).ravel(), np.float64)


def _line_residuals_m(
    h_g: NDArray[np.floating[Any]], d: NDArray[np.floating[Any]], lo: Sequence[LineObs]
) -> NDArray[np.float64]:
    """Per-line-click (N,) metre distances to the named (axis-aligned) pitch line."""
    if not lo:
        return np.zeros(0)
    pts = np.array([[x + d[0], y + d[1]] for _, x, y in lo], np.float64)
    q = _apply(h_g, pts)
    out = [
        (float(qi[ax]) - c) * float(_SCALE_M[ax])
        for (lid, _, _), qi in zip(lo, q, strict=True)
        for ax, c in (_LINE_PITCH[lid],)
    ]
    return np.array(out, np.float64)
```

Import hygiene: the block above lists only what Task 1's code uses plus `scipy`/`fold_count`/
`fit_homography` needed from Task 2 onward. If `uv run ruff check` flags an unused import at
THIS commit, remove it here and re-add it in the task that first uses it (scipy → Tasks 2–3,
`fold_count` → Task 5, `HomographyError`/`fit_homography` → Task 2, `LineClick` → Task 4).

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_global_crop.py -x -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/global_crop.py packages/soccer-vision/tests/test_global_crop.py
git commit -m "feat(pitch): global-crop core geometry + synthetic crop-world fixture"
```

---

### Task 2: Robust global fit `_fit_h_g`

**Files:** same two files.

- [ ] **Step 1: Write the failing tests**

```python
def test_fit_h_g_recovers_truth(world: CropWorld) -> None:
    canvas_pts, pitch_pts = [], []
    for f in range(0, world.n_frames, 20):
        for c in world.clicks_at(f):
            d = world.offsets[f]
            canvas_pts.append([c.x + d[0], c.y + d[1]])
            pitch_pts.append(PITCH_LANDMARKS[c.kp_idx])
    h = gc._fit_h_g(np.array(canvas_pts), np.array(pitch_pts))
    assert h is not None
    q = gc._apply(h, np.array(canvas_pts))
    assert np.abs((q - np.array(pitch_pts)) * gc._SCALE_M).max() < 0.05  # < 5 cm


def test_fit_h_g_rejects_outlier(world: CropWorld) -> None:
    canvas_pts, pitch_pts = [], []
    for f in range(0, world.n_frames, 20):
        for c in world.clicks_at(f):
            d = world.offsets[f]
            canvas_pts.append([c.x + d[0], c.y + d[1]])
            pitch_pts.append(PITCH_LANDMARKS[c.kp_idx])
    canvas_pts[0] = [canvas_pts[0][0] + 0.2, canvas_pts[0][1] + 0.2]  # gross outlier
    h = gc._fit_h_g(np.array(canvas_pts), np.array(pitch_pts))
    assert h is not None
    q = gc._apply(h, np.array(canvas_pts[1:]))
    assert np.abs((q - np.array(pitch_pts[1:])) * gc._SCALE_M).max() < 0.05


def test_fit_h_g_degenerate_inputs(world: CropWorld) -> None:
    three = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.1]])
    assert gc._fit_h_g(three, three) is None                       # < 4 points
    col_c = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
    col_p = np.array([[0.0, 0.0], [0.0, 0.3], [0.0, 0.6], [0.0, 0.9]])
    assert gc._fit_h_g(col_c, col_p) is None                       # collinear landmarks
    tiny = np.array([[0.5, 0.5], [0.51, 0.5], [0.5, 0.51], [0.51, 0.51]])
    assert gc._fit_h_g(tiny, tiny * 0.01 + 0.5) is None            # hull too small
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_global_crop.py -x -q`
Expected: FAIL — `module ... has no attribute '_fit_h_g'`.

- [ ] **Step 3: Implement**

```python
def _fit_h_g(
    canvas_pts: NDArray[np.floating[Any]], pitch_pts: NDArray[np.floating[Any]]
) -> NDArray[np.float64] | None:
    """RANSAC global fit canvas->pitch. None when the constraints are degenerate:
    < 4 distinct landmarks, or their pitch spread (hull area) is too small to pin a
    homography — the bootstrap-wait semantic (red, never a garbage fit)."""
    if len(canvas_pts) < 4:
        return None
    distinct = np.unique(np.round(np.asarray(pitch_pts, np.float64), 6), axis=0)
    if len(distinct) < 4:
        return None
    try:
        if ConvexHull(distinct).volume < HULL_AREA_MIN:  # 2-D: volume == area
            return None
    except QhullError:
        return None  # collinear
    try:
        h = fit_homography(np.asarray(canvas_pts, np.float64),
                           np.asarray(pitch_pts, np.float64),
                           ransac_thresh=RANSAC_THRESH_PITCH)
    except HomographyError:
        return None
    return np.asarray(h, np.float64)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_global_crop.py -x -q` → 5 passed.

- [ ] **Step 5: Commit** — `git commit -am "feat(pitch): global-crop robust H_g fit (RANSAC + hull degeneracy gate)"`

---

### Task 3: Per-frame offset solve `_solve_offset`

**Files:** same two files.

- [ ] **Step 1: Write the failing tests**

```python
def test_offset_from_single_click(world: CropWorld) -> None:
    f = 120
    c = world.clicks_at(f)[0]
    d = gc._solve_offset(world.h_g, [(c.kp_idx, c.x, c.y)], [], np.zeros(2))
    assert np.linalg.norm(d - world.offsets[f]) < 1e-4  # ONE click determines the frame


def test_offset_robust_to_one_bad_click(world: CropWorld) -> None:
    f = 120
    po = [(c.kp_idx, c.x, c.y) for c in world.clicks_at(f)]
    assert len(po) >= 4
    bad = (po[0][0], po[0][1] + 0.08, po[0][2] + 0.08)  # ~1.5 m wrong
    d = gc._solve_offset(world.h_g, [bad] + po[1:], [], world.offsets[f] + 0.05)
    assert np.linalg.norm(d - world.offsets[f]) < 0.005  # soft_l1 downweights it


def test_offset_two_axis_lines_full_solve(world: CropWorld) -> None:
    f = 120
    d_true = world.offsets[f]
    lo = [
        ("midline", float(world.canvas[4][0] - d_true[0]), float(world.canvas[4][1] - d_true[1])),
        ("far_touchline", float(world.canvas[1][0] - d_true[0]), float(world.canvas[1][1] - d_true[1])),
    ]
    assert gc._offset_axes(lo) == {0, 1}
    d = gc._solve_offset(world.h_g, [], lo, d_true + np.array([0.03, -0.03]))
    assert np.linalg.norm(d - d_true) < 1e-3


def test_offset_one_axis_line_keeps_prior_direction(world: CropWorld) -> None:
    f = 120
    d_true = world.offsets[f]
    lo = [("midline", float(world.canvas[4][0] - d_true[0]), float(world.canvas[4][1] - d_true[1]))]
    assert gc._offset_axes(lo) == {1}
    d0 = d_true + np.array([0.05, 0.05])  # drifted chain init
    d = gc._solve_offset(world.h_g, [], lo, d0, prior=True)
    r = gc._line_residuals_m(world.h_g, d, lo)
    assert abs(r[0]) < 0.2                       # the line got satisfied (metres)
    assert abs(d[0] - d0[0]) < 0.02              # unconstrained direction stayed near init
```

- [ ] **Step 2: Run to verify failure** — attribute error on `_solve_offset` / `_offset_axes`.

- [ ] **Step 3: Implement**

```python
def _offset_axes(lo: Sequence[LineObs]) -> set[int]:
    """Which pitch axes (0=x, 1=y) the frame's line clicks constrain."""
    return {_LINE_PITCH[lid][0] for lid, _, _ in lo}


def _solve_offset(
    h_g: NDArray[np.floating[Any]],
    po: Sequence[PointObs],
    lo: Sequence[LineObs],
    d0: NDArray[np.floating[Any]],
    *,
    prior: bool = False,
) -> NDArray[np.float64]:
    """2-DOF canvas offset for one frame by robust least squares (metre residuals,
    soft_l1). `prior=True` adds a weak pull toward d0 — used when the frame's clicks
    constrain fewer than 2 DOF (e.g. a single one-axis line), so the unconstrained
    direction stays at the chain init instead of wandering."""
    d_init = np.asarray(d0, np.float64)

    def fun(d: NDArray[np.float64]) -> NDArray[np.float64]:
        parts = [_point_residuals_m(h_g, d, po), _line_residuals_m(h_g, d, lo)]
        if prior:
            parts.append((d - d_init) * _SCALE_M * PRIOR_WEIGHT)
        return np.concatenate(parts)

    res = least_squares(fun, d_init, method="trf", loss="soft_l1", f_scale=0.5,
                        x_scale=1e-2)
    return np.asarray(res.x, np.float64)
```

- [ ] **Step 4: Run to verify pass** — 9 passed.
- [ ] **Step 5: Commit** — `git commit -am "feat(pitch): global-crop 2-DOF offset solve (points+lines, robust, chain prior)"`

---

### Task 4: `solve_crop_session` + `CropCalib` (alternation, brackets, segments)

**Files:** same two files.

- [ ] **Step 1: Write the failing tests**

```python
def _session(world: CropWorld, frames: list[int], *, drift: float = 0.0,
             lines: list[LineClick] | None = None,
             gap_guard: int = 200) -> gc.CropCalib:
    w = CropWorld(world.n_frames, drift=drift) if drift else world
    clicks = [c for f in frames for c in w.clicks_at(f)]
    return gc.solve_crop_session(clicks, lines or [], SIZE, w.transforms,
                                 gap_guard=gap_guard)


def _err_ft(calib: gc.CropCalib, world: CropWorld, frame: int, kp_idx: int) -> float:
    h = calib.frame_homography(frame)
    assert h is not None
    c = world.click(frame, kp_idx)
    q = gc._apply(h, np.array([[c.x, c.y]]))[0]
    return float(np.linalg.norm((q - PITCH_LANDMARKS[kp_idx]) * gc._SCALE_M) * gc._FT)


def test_solve_recovers_world(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    calib = _session(world, frames)
    for f in frames:
        assert calib.is_anchor(f)
        for i in world.visible(f):
            assert _err_ft(calib, world, f, i) < 1.0


def test_single_end_frame_places_far_end(world: CropWorld) -> None:
    """F-C1 regression — the point of this project. A frame clicked ONLY at one end
    must place the OTHER end's landmarks accurately, because H_g carries them."""
    frames = list(range(0, world.n_frames, 20))
    f_own = min(frames, key=lambda f: np.mean([PITCH_LANDMARKS[i][1] for i in world.visible(f)]))
    own_ids = [i for i in world.visible(f_own) if PITCH_LANDMARKS[i][1] < 0.5]
    assert len(own_ids) >= 1
    clicks = [c for f in frames if f != f_own for c in world.clicks_at(f)]
    clicks += world.clicks_at(f_own, own_ids[:1])   # ONE own-end click only
    calib = gc.solve_crop_session(clicks, [], SIZE, world.transforms)
    assert calib.is_anchor(f_own)
    # every landmark of the whole field projects to < 2 ft error under this frame's H
    for i in range(21):
        if i == 5:
            continue
        assert _err_ft(calib, world, f_own, i) < 2.0


def test_chain_drift_corrected_at_anchors(world: CropWorld) -> None:
    """Corrupt the chain with linear drift; anchor frames must still solve to truth
    (clicks win), and unclicked frames' error stays bounded by SHORT-hop drift."""
    drifted = CropWorld(world.n_frames, drift=2e-4)   # 0.048 norm ≈ 92 px over 240 frames
    frames = list(range(0, drifted.n_frames, 20))
    clicks = [c for f in frames for c in drifted.clicks_at(f)]
    calib = gc.solve_crop_session(clicks, [], SIZE, drifted.transforms)
    for f in frames:
        for i in drifted.visible(f)[:3]:
            assert _err_ft(calib, drifted, f, i) < 1.0        # anchors: click-solved
    mid = frames[3] + 10                                       # 10-frame hop
    i = drifted.visible(mid)[0]
    assert _err_ft(calib, drifted, mid, i) < 3.0               # short-hop bound, not 92 px


def test_segment_isolation(world: CropWorld) -> None:
    seg_of = {f: (0 if f < 120 else 1) for f in range(world.n_frames)}
    clicks = [c for f in (0, 40, 80) for c in world.clicks_at(f)]  # only segment 0 clicked
    calib = gc.solve_crop_session(clicks, [], SIZE, world.transforms, segment_of=seg_of)
    assert calib.frame_homography(60) is not None
    assert calib.frame_homography(150) is None                  # other segment: no H_g


def test_gap_guard(world: CropWorld) -> None:
    calib = _session(world, [0, 20], gap_guard=50)
    assert calib.frame_homography(30) is not None
    assert calib.frame_homography(200) is None                  # 180 > guard


def test_one_end_session_is_capped(world: CropWorld) -> None:
    own_frames = [f for f in range(world.n_frames)
                  if world.visible(f) and np.mean([PITCH_LANDMARKS[i][1]
                                                   for i in world.visible(f)]) < 0.35]
    clicks = [c for f in own_frames[:4] for c in world.clicks_at(f)]
    calib = gc.solve_crop_session(clicks, [], SIZE, world.transforms)
    seg = calib.segments.get(0)
    if seg is None:
        pytest.skip("one-end clicks degenerate for H_g on this geometry")
    assert seg.one_end_capped


def test_too_few_landmarks_no_anchor(world: CropWorld) -> None:
    calib = gc.solve_crop_session(world.clicks_at(120)[:3], [], SIZE, world.transforms)
    assert calib.anchor_h == {}
    assert calib.frame_homography(120) is None
```

- [ ] **Step 2: Run to verify failure** — attribute errors on `solve_crop_session` / `CropCalib`.

- [ ] **Step 3: Implement**

```python
@dataclass(frozen=True, eq=False)
class SegmentSolve:
    """One segment's solved global model."""

    h_g: NDArray[np.float64]                    # canvas(norm) -> pitch[0,1]
    offsets: dict[int, NDArray[np.float64]]     # click-solved anchor offsets
    anchor_status: dict[int, str]               # "green" | "yellow" per anchor
    one_end_capped: bool                        # clicked landmarks span < Y_SPAN_ONE_END


@dataclass(frozen=True, eq=False)
class CropCalib:
    """Solved session. Same duck-typed surface LabelerState uses from PhysicalCalib."""

    segments: dict[int, SegmentSolve]
    transforms: dict[int, NDArray[np.float64]]
    segment_of: dict[int, int]
    size: tuple[int, int]
    gap_guard: int = DEFAULT_GAP_GUARD

    def _segment(self, frame: int) -> int:
        return self.segment_of.get(frame, 0)

    @property
    def anchor_h(self) -> dict[int, NDArray[np.float64]]:
        """Anchor frame -> normalized H (bootstrap/duck-type parity with PhysicalCalib)."""
        out: dict[int, NDArray[np.float64]] = {}
        for seg_id, ss in self.segments.items():
            for f, d in ss.offsets.items():
                out[f] = np.asarray(ss.h_g @ _t(d), np.float64)
        return out

    def is_anchor(self, frame: int) -> bool:
        ss = self.segments.get(self._segment(frame))
        return ss is not None and frame in ss.offsets

    def _segment_anchors(self, frame: int) -> list[int]:
        ss = self.segments.get(self._segment(frame))
        if ss is None:
            return []
        return sorted(f for f in ss.offsets if f in self.transforms)

    def nearest_anchor_gap(self, frame: int) -> int | None:
        anchors = self._segment_anchors(frame)
        return min(abs(frame - a) for a in anchors) if anchors else None

    def used_anchors(self, frame: int) -> list[int]:
        """The bracket anchors an unclicked frame's offset actually interpolates from
        (F-C3: grading uses THESE, not the nearest green anywhere)."""
        anchors = self._segment_anchors(frame)
        if not anchors or frame not in self.transforms:
            return []
        if min(abs(frame - a) for a in anchors) > self.gap_guard:
            return []
        lo = [a for a in anchors if a < frame]
        hi = [a for a in anchors if a > frame]
        if lo and hi:
            return [lo[-1], hi[0]]
        return [lo[-1]] if lo else [hi[0]]

    def _offset_of(self, frame: int) -> NDArray[np.float64] | None:
        ss = self.segments.get(self._segment(frame))
        if ss is None:
            return None
        if frame in ss.offsets:
            return ss.offsets[frame]
        used = self.used_anchors(frame)
        if not used:
            return None
        t_f = _translation(self.transforms[frame])
        preds = [ss.offsets[a] + (t_f - _translation(self.transforms[a])) for a in used]
        if len(preds) == 1:
            return preds[0]
        a, b = used
        w = (frame - a) / (b - a)
        return np.asarray((1.0 - w) * preds[0] + w * preds[1], np.float64)

    def frame_homography(self, frame: int) -> NDArray[np.float64] | None:
        ss = self.segments.get(self._segment(frame))
        d = self._offset_of(frame)
        if ss is None or d is None:
            return None
        return np.asarray(ss.h_g @ _t(d), np.float64)


def _refine_h_g(
    h0: NDArray[np.float64],
    obs: Sequence[tuple[NDArray[np.float64], Sequence[PointObs], Sequence[LineObs]]],
) -> NDArray[np.float64]:
    """Refine the 8 free params of H_g against all well-constrained anchors' points
    AND line clicks (metre residuals, soft_l1). Falls back to h0 on non-finite output."""
    def fun(p: NDArray[np.float64]) -> NDArray[np.float64]:
        h = np.append(p, 1.0).reshape(3, 3)
        parts = [r for d, po, lo in obs
                 for r in (_point_residuals_m(h, d, po), _line_residuals_m(h, d, lo))]
        return np.concatenate(parts) if parts else np.zeros(1)

    p0 = (h0 / h0[2, 2]).ravel()[:8]
    res = least_squares(fun, p0, method="trf", loss="soft_l1", f_scale=0.5)
    h = np.append(res.x, 1.0).reshape(3, 3)
    return h if bool(np.isfinite(h).all()) else h0


def _grade_anchor(
    h_g: NDArray[np.float64], d: NDArray[np.float64],
    po: Sequence[PointObs], lo: Sequence[LineObs],
) -> str:
    """green iff the frame's own clicks fit in-sample within tolerance AND the clicks
    constrain both offset DOF (>=1 point, or lines spanning both axes)."""
    full_dof = bool(po) or _offset_axes(lo) == {0, 1}
    if not full_dof:
        return "yellow"
    pr = _point_residuals_m(h_g, d, po).reshape(-1, 2)
    pt_ft = [float(np.hypot(r[0], r[1]) * _FT) for r in pr]
    ln_ft = [abs(float(r)) * _FT for r in _line_residuals_m(h_g, d, lo)]
    if pt_ft and float(np.median(pt_ft)) > POINT_OK_FT:
        return "yellow"
    if ln_ft and float(np.median(ln_ft)) > LINE_OK_FT:
        return "yellow"
    return "green"


def solve_crop_session(
    points: Sequence[Click],
    lines: Sequence[LineClick],
    size: tuple[int, int],
    transforms: Mapping[int, NDArray[np.floating[Any]]],
    *,
    segment_of: Mapping[int, int] | None = None,
    gap_guard: int = DEFAULT_GAP_GUARD,
    rounds: int = 2,
) -> CropCalib:
    """Solve every segment: RANSAC H_g from chain-initialized canvas correspondences,
    then alternate (offset re-solve per clicked frame) <-> (H_g refine with points+lines).
    Segments without >=4 spread landmarks stay unsolved (bootstrap waits)."""
    tf = {f: np.asarray(m, np.float64) for f, m in transforms.items()}
    seg_of: dict[int, int] = dict(segment_of) if segment_of is not None else {}
    by_pt: dict[int, list[Click]] = {}
    for c in points:
        by_pt.setdefault(c.frame, []).append(c)
    by_ln: dict[int, list[LineClick]] = {}
    for lc in lines:
        by_ln.setdefault(lc.frame, []).append(lc)

    segments: dict[int, SegmentSolve] = {}
    clicked_frames = sorted(set(by_pt) | set(by_ln))
    seg_ids = sorted({seg_of.get(f, 0) for f in clicked_frames})
    for seg_id in seg_ids:
        frames = [f for f in clicked_frames if seg_of.get(f, 0) == seg_id and f in tf]
        if not frames:
            continue
        d: dict[int, NDArray[np.float64]] = {f: _translation(tf[f]) for f in frames}
        po_of: dict[int, list[PointObs]] = {
            f: [(int(c.kp_idx), float(c.x), float(c.y)) for c in by_pt.get(f, [])]
            for f in frames}
        lo_of: dict[int, list[LineObs]] = {
            f: [(str(lc.line_id), float(lc.x), float(lc.y)) for lc in by_ln.get(f, [])]
            for f in frames}
        canvas = [[x + d[f][0], y + d[f][1]] for f in frames for _, x, y in po_of[f]]
        pitch = [PITCH_LANDMARKS[i] for f in frames for i, _, _ in po_of[f]]
        h_g = _fit_h_g(np.array(canvas, np.float64), np.array(pitch, np.float64)) \
            if canvas else None
        if h_g is None:
            continue
        for _ in range(rounds):
            for f in frames:
                full_dof = bool(po_of[f]) or _offset_axes(lo_of[f]) == {0, 1}
                d[f] = _solve_offset(h_g, po_of[f], lo_of[f], d[f], prior=not full_dof)
            usable = [(d[f], po_of[f], lo_of[f]) for f in frames
                      if po_of[f] or _offset_axes(lo_of[f]) == {0, 1}]
            if usable:
                h_g = _refine_h_g(h_g, usable)
        grade = {f: _grade_anchor(h_g, d[f], po_of[f], lo_of[f]) for f in frames}
        ys = [PITCH_LANDMARKS[i][1] for f in frames for i, _, _ in po_of[f]]
        capped = bool(ys) and (max(ys) - min(ys)) < Y_SPAN_ONE_END
        segments[seg_id] = SegmentSolve(
            h_g=np.asarray(h_g, np.float64), offsets=d, anchor_status=grade,
            one_end_capped=capped)
    return CropCalib(segments, tf, seg_of, size, gap_guard)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_global_crop.py -x -q` → all pass. If `test_single_end_frame_places_far_end` fails on tolerance, print the errors and check the alternation converged (raise `rounds`) before touching tolerances — the 2 ft bound is the product requirement, not a suggestion.

- [ ] **Step 5: Commit** — `git commit -am "feat(pitch): solve_crop_session + CropCalib (alternating global fit, bracket offsets, F-C1 regression test)"`

---

### Task 5: Status + confidence (w-sign gate, fold band, used-anchor grading)

**Files:** same two files.

- [ ] **Step 1: Write the failing tests**

```python
def test_status_green_anchor_and_propagated(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    calib = _session(world, frames)
    assert calib.status(frames[2]) == "green"
    assert calib.status(frames[2] + 5) == "green"     # within GREEN_RADIUS of used anchors
    assert calib.status(999_999) == "red"


def test_status_uses_bracket_anchors_not_nearest_green(world: CropWorld) -> None:
    """F-C3: a frame bracketed by a YELLOW anchor must not be green just because a
    green anchor sits nearby on the other side of it."""
    frames = list(range(0, world.n_frames, 20))
    clicks = [c for f in frames for c in world.clicks_at(f)]
    calib = gc.solve_crop_session(clicks, [], SIZE, world.transforms)
    f_mid = frames[3]
    ss = calib.segments[0]
    forced = dict(ss.anchor_status)
    forced[f_mid] = "yellow"
    calib2 = gc.CropCalib(
        {0: gc.SegmentSolve(ss.h_g, ss.offsets, forced, ss.one_end_capped)},
        calib.transforms, calib.segment_of, calib.size, calib.gap_guard)
    probe = f_mid + 5   # bracketed by f_mid (yellow) and frames[4] (green)
    assert set(calib2.used_anchors(probe)) == {f_mid, frames[4]}
    assert calib2.status(probe) == "yellow"


def test_sky_pose_is_red(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    calib = _session(world, frames)
    ss = calib.segments[0]
    sky = ss.h_g.copy()
    sky[2, :] = np.array([0.9, -3.0, 1.0])   # corrupt: far half flips behind camera
    calib2 = gc.CropCalib(
        {0: gc.SegmentSolve(sky, ss.offsets, ss.anchor_status, ss.one_end_capped)},
        calib.transforms, calib.segment_of, calib.size, calib.gap_guard)
    f = frames[2]
    if gc._wsigns_ok(calib2.frame_homography(f), SIZE):
        pytest.skip("corruption did not flip w-signs on this geometry; strengthen row")
    assert calib2.status(f) == "red"


def test_green_implies_wsign_pass(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    calib = _session(world, frames)
    for f in range(0, world.n_frames, 7):
        if calib.status(f) == "green":
            h = calib.frame_homography(f)
            assert h is not None and gc._wsigns_ok(h, SIZE)


def test_confidence_mapping(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    calib = _session(world, frames)
    f_anchor = frames[2]
    assert gc.frame_confidence(calib, f_anchor) == pytest.approx(0.9)
    c_near = gc.frame_confidence(calib, f_anchor + 2)
    c_far = gc.frame_confidence(calib, f_anchor + 9)
    assert 0.6 <= c_far < c_near <= 0.8
    assert gc.frame_confidence(calib, 999_999) == 0.0
```

- [ ] **Step 2: Run to verify failure** — missing `status` / `_wsigns_ok` / `frame_confidence`.

- [ ] **Step 3: Implement** (add to `CropCalib` and module)

```python
def _wsigns_ok(h_norm: NDArray[np.floating[Any]], size: tuple[int, int]) -> bool:
    """True iff ALL 21 canonical landmarks project with w > 0 (in front of the camera)
    under the sign-normalized pitch->pixel map. A pose whose far end flips behind the
    camera plane ("lines in the sky") fails this even when its near field looks fine."""
    w, h = size
    h_px = np.asarray(h_norm, np.float64) @ np.diag([1.0 / w, 1.0 / h, 1.0])
    try:
        p = np.linalg.inv(h_px)
    except np.linalg.LinAlgError:
        return False
    if float((p @ np.array([0.5, 0.5, 1.0]))[2]) < 0:
        p = -p
    pts = np.column_stack([PITCH_LANDMARKS, np.ones(len(PITCH_LANDMARKS))])
    wz = (p @ pts.T).T[:, 2]
    return bool(np.all(wz > 1e-9))


def _fold_norm(h_norm: NDArray[np.floating[Any]], size: tuple[int, int]) -> int:
    """fold_count for a NORMALIZED image->pitch homography (sign-normalized)."""
    w, h = size
    h_px = np.asarray(h_norm, np.float64) @ np.diag([1.0 / w, 1.0 / h, 1.0])
    try:
        p = np.linalg.inv(h_px)
    except np.linalg.LinAlgError:
        return 0
    if float((p @ np.array([0.5, 0.5, 1.0]))[2]) < 0:
        p = -p
    return fold_count(p, size)
```

Add to `CropCalib`:

```python
    def status(self, frame: int) -> str:
        """green = click-solved anchor within in-sample tolerance, or a frame within
        GREEN_RADIUS whose USED bracket anchors are all green — in both cases the
        whole-field projection is physical (all-21 w>0, fold in band). yellow =
        propagated beyond radius / partially-constrained anchor / one-end-capped
        segment. red = no homography or an unphysical projection."""
        h = self.frame_homography(frame)
        if h is None:
            return "red"
        if not _wsigns_ok(h, self.size):
            return "red"
        if not FOLD_MIN <= _fold_norm(h, self.size) <= FOLD_MAX:
            return "red"
        ss = self.segments[self._segment(frame)]
        if ss.one_end_capped:
            return "yellow"
        if frame in ss.offsets:
            return ss.anchor_status.get(frame, "yellow")
        used = self.used_anchors(frame)
        near = min(abs(frame - a) for a in used)
        if near <= GREEN_RADIUS and all(ss.anchor_status.get(a) == "green" for a in used):
            return "green"
        return "yellow"


def frame_confidence(calib: Any, frame: int) -> float:
    """Honest export confidence for ANY engine exposing status/is_anchor/
    nearest_anchor_gap: 0.9 for a green anchor, 0.8 -> 0.6 ramp across GREEN_RADIUS
    for propagated green, 0.0 otherwise. Retires the constant-1.0 overclaim (F-C2)."""
    if calib.status(frame) != "green":
        return 0.0
    if calib.is_anchor(frame):
        return CONF_ANCHOR
    gap = calib.nearest_anchor_gap(frame) or 0
    return CONF_PROP_MAX - (CONF_PROP_MAX - CONF_PROP_MIN) * min(1.0, gap / GREEN_RADIUS)
```

- [ ] **Step 4: Run to verify pass** — all tests green.
- [ ] **Step 5: Commit** — `git commit -am "feat(pitch): crop status (w-sign no-sky gate, used-anchor grading) + honest frame_confidence"`

---

### Task 6: Diagnostics — crop-assumption report + implied camera

**Files:** same two files.

- [ ] **Step 1: Write the failing tests**

```python
def test_crop_assumption_report_pass_and_fail(world: CropWorld) -> None:
    pairs = {f: np.linalg.inv(world.transforms[f]) @ world.transforms[f + 1]
             for f in range(world.n_frames - 1)}
    rep = gc.crop_assumption_report(pairs, SIZE)
    assert rep["ok"] and rep["max_abs_rot_deg"] < 0.01 and abs(rep["max_scale_dev"]) < 1e-6
    # 0.05 rad in normalized coords ≈ 1.6° after the pixel-space aspect correction
    rot = np.array([[np.cos(0.05), -np.sin(0.05), 0], [np.sin(0.05), np.cos(0.05), 0], [0, 0, 1.0]])
    rep2 = gc.crop_assumption_report({0: rot}, SIZE)
    assert not rep2["ok"] and rep2["max_abs_rot_deg"] > 1.0


def test_implied_camera_recovers_focal(world: CropWorld) -> None:
    out = gc.implied_camera(world.h_g, SIZE, pp_canvas=np.array([0.5, 0.5]))
    assert out is not None
    f_px, c = out
    assert abs(f_px - 1460.0) / 1460.0 < 0.05      # within 5% of the true focal
    assert c[2] < 0 or c[2] > 0                     # finite center recovered
```

- [ ] **Step 2: Run to verify failure** — missing names.

- [ ] **Step 3: Implement**

```python
def crop_assumption_report(
    interframe: Mapping[int, NDArray[np.floating[Any]]], size: tuple[int, int]
) -> dict[str, Any]:
    """Decompose NORMALIZED inter-frame pair transforms in PIXEL space and report how
    translation-pure they are. ok=False means the crop model is questionable for this
    clip (rotation/zoom/perspective present) — a loud warning, not a hard failure."""
    w, h = size
    s = np.diag([1.0 / w, 1.0 / h, 1.0])
    s_inv = np.diag([float(w), float(h), 1.0])
    rots, scales, persp = [], [], []
    for m in interframe.values():
        g = s_inv @ (np.asarray(m, np.float64)) @ s     # back to pixel space
        g = g / g[2, 2]
        rots.append(math.degrees(math.atan2(g[1, 0], g[0, 0])))
        scales.append(math.sqrt(abs(float(np.linalg.det(g[:2, :2])))) - 1.0)
        persp.append(max(abs(g[2, 0]), abs(g[2, 1])))
    if not rots:
        return {"ok": True, "n": 0, "max_abs_rot_deg": 0.0, "max_scale_dev": 0.0,
                "max_perspective": 0.0}
    rep = {
        "n": len(rots),
        "max_abs_rot_deg": float(np.max(np.abs(rots))),
        "max_scale_dev": float(np.max(np.abs(scales))),
        "max_perspective": float(np.max(persp)),
    }
    rep["ok"] = bool(rep["max_abs_rot_deg"] <= 0.2 and rep["max_scale_dev"] <= 0.005
                     and rep["max_perspective"] <= 1e-5)
    return rep


def implied_camera(
    h_g: NDArray[np.floating[Any]], size: tuple[int, int],
    *, pp_canvas: NDArray[np.floating[Any]],
) -> tuple[float, NDArray[np.float64]] | None:
    """REPORT-ONLY physical decomposition of H_g: assuming square pixels and principal
    point pp_canvas (normalized canvas units — use the mean frame centre), recover the
    focal (px) from the plane-homography orthonormality constraints and the camera
    centre (metres). None if the constraints are inconsistent (non-physical H_g)."""
    w, h = size
    # pitch[0,1] -> canvas px, then pitch metres -> canvas px
    a = np.diag([float(w), float(h), 1.0]) @ np.linalg.inv(np.asarray(h_g, np.float64))
    a = a @ np.diag([1.0 / WIDTH_M, 1.0 / LENGTH_M, 1.0])
    cx, cy = float(pp_canvas[0]) * w, float(pp_canvas[1]) * h
    b = np.array([[1.0, 0, -cx], [0, 1.0, -cy], [0, 0, 1.0]]) @ a
    b1, b2 = b[:, 0], b[:, 1]
    # with D = diag(1/f^2, 1/f^2, 1): b1' D b2 = 0  and  |D^.5 b1| = |D^.5 b2|
    n1 = b1[0] * b2[0] + b1[1] * b2[1]
    d1 = -b1[2] * b2[2]
    n2 = (b1[0] ** 2 + b1[1] ** 2) - (b2[0] ** 2 + b2[1] ** 2)
    d2 = b2[2] ** 2 - b1[2] ** 2
    cands = [n / dd for n, dd in ((n1, d1), (n2, d2)) if abs(dd) > 1e-12 and n / dd > 0]
    if not cands:
        return None
    f2 = float(np.mean(cands))
    f_px = math.sqrt(f2)
    m = np.diag([1.0 / f_px, 1.0 / f_px, 1.0]) @ b
    lam = 1.0 / float(np.linalg.norm(m[:, 0]))
    r1, r2, t = lam * m[:, 0], lam * m[:, 1], lam * m[:, 2]
    r3 = np.cross(r1, r2)
    rmat = np.column_stack([r1, r2, r3])
    c = -rmat.T @ t
    return f_px, np.asarray(c, np.float64)
```

- [ ] **Step 4: Run to verify pass.** If `test_implied_camera_recovers_focal` misses 5%, print both candidate `f` values — they should agree; a large split means the pp assumption is off, in which case relax the tolerance to 10% (report-only tool) and note it in the docstring.
- [ ] **Step 5: Commit** — `git commit -am "feat(pitch): crop-assumption diagnostic + report-only implied-camera decomposition"`

---

### Task 7: Acceptance-gate functions for the crop engine

**Files:** same two files.

- [ ] **Step 1: Write the failing tests**

```python
def _ntl_line_clicks(world: CropWorld, frame: int, n: int = 3) -> list[LineClick]:
    d = world.offsets[frame]
    out = []
    for yv in np.linspace(0.15, 0.85, n):
        pitch = np.array([[0.0, float(yv)]])                 # ON the near touchline
        cpt = gc._apply(np.linalg.inv(world.h_g), pitch)[0]  # canvas position
        out.append(LineClick(frame=frame, line_id="near_touchline",
                             x=float(cpt[0] - d[0]), y=float(cpt[1] - d[1])))
    return out


def test_foreground_holdout_crop(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    clicks = [c for f in frames for c in world.clicks_at(f)]
    lines = [lc for f in frames[:3] for lc in _ntl_line_clicks(world, f)]
    errs = gc.foreground_holdout_crop(clicks, lines, SIZE, world.transforms)
    assert len(errs) == 9
    assert float(np.median(errs)) < 2.0                       # feet, exact synthetic data


def test_propagation_holdout_split_by_end(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    clicks = [c for f in frames for c in world.clicks_at(f)]
    errs, by_end = gc.propagation_holdout_crop(clicks, [], SIZE, world.transforms)
    assert errs and float(np.median(errs)) < 2.0
    assert set(by_end) <= {"own", "opp", "both"}
    assert sum(len(v) for v in by_end.values()) == len(errs)


def test_evaluate_crop_gate_passes_on_clean_world(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    clicks = [c for f in frames for c in world.clicks_at(f)]
    lines = [lc for f in frames[:3] for lc in _ntl_line_clicks(world, f)]
    rep = gc.evaluate_crop_gate(clicks, lines, SIZE, world.transforms)
    assert rep.passed_numeric
    assert rep.fg_n == 9 and rep.prop_n > 0
    assert rep.prop_by_end  # the F-C1 evidence table exists
```

- [ ] **Step 2: Run to verify failure** — missing names.

- [ ] **Step 3: Implement**

```python
_NEAR_TL_POINT_IDS = {0, 2}  # near-touchline endpoint landmarks (held out with its lines)


@dataclass(frozen=True)
class CropGateReport:
    """Held-out acceptance metrics (feet) for a session's crop calibration."""

    fg_median_ft: float
    fg_p90_ft: float
    fg_n: int
    prop_median_ft: float
    prop_p90_ft: float
    prop_n: int
    prop_by_end: dict[str, tuple[float, float, int]]  # end -> (median, p90, n)
    passed_numeric: bool


def foreground_holdout_crop(
    points: Sequence[Click], lines: Sequence[LineClick], size: tuple[int, int],
    transforms: Mapping[int, NDArray[np.floating[Any]]],
    *, segment_of: Mapping[int, int] | None = None,
    gap_guard: int = DEFAULT_GAP_GUARD,
) -> list[float]:
    """Remove ALL near-touchline evidence (its line clicks AND the point landmarks on
    it), re-solve, then measure the held line clicks' perpendicular error in feet."""
    held = [lc for lc in lines if lc.line_id == "near_touchline"]
    if not held:
        return []
    rest_l = [lc for lc in lines if lc.line_id != "near_touchline"]
    rest_p = [c for c in points if c.kp_idx not in _NEAR_TL_POINT_IDS]
    calib = solve_crop_session(rest_p, rest_l, size, transforms,
                               segment_of=segment_of, gap_guard=gap_guard)
    errs: list[float] = []
    for lc in held:
        h = calib.frame_homography(lc.frame)
        if h is None:
            continue
        q = _apply(h, np.array([[lc.x, lc.y]]))[0]
        errs.append(abs(float(q[0]) - 0.0) * float(_SCALE_M[0]) * _FT)
    return errs


def _end_of_frame(clicked: Sequence[Click]) -> str:
    ys = [float(PITCH_LANDMARKS[c.kp_idx][1]) for c in clicked]
    m = float(np.mean(ys))
    return "own" if m < 0.45 else "opp" if m > 0.55 else "both"


def propagation_holdout_crop(
    points: Sequence[Click], lines: Sequence[LineClick], size: tuple[int, int],
    transforms: Mapping[int, NDArray[np.floating[Any]]],
    *, segment_of: Mapping[int, int] | None = None,
    gap_guard: int = DEFAULT_GAP_GUARD,
) -> tuple[list[float], dict[str, list[float]]]:
    """Leave-one-anchor-out: drop each clicked frame entirely, re-solve, predict its
    point clicks. Returns (all errors ft, errors bucketed by which end the held frame
    saw) — the by-end buckets are the direct F-C1 measurement."""
    seg = dict(segment_of) if segment_of is not None else {}
    by_frame: dict[int, list[Click]] = {}
    for c in points:
        by_frame.setdefault(c.frame, []).append(c)
    errs: list[float] = []
    by_end: dict[str, list[float]] = {}
    for held, hcs in sorted(by_frame.items()):
        others = [f for f in by_frame if f != held and seg.get(f, 0) == seg.get(held, 0)]
        if not others or min(abs(held - f) for f in others) > gap_guard:
            continue
        calib = solve_crop_session(
            [c for c in points if c.frame != held],
            [lc for lc in lines if lc.frame != held],
            size, transforms, segment_of=segment_of, gap_guard=gap_guard)
        h = calib.frame_homography(held)
        if h is None:
            continue
        end = _end_of_frame(hcs)
        for c in hcs:
            q = _apply(h, np.array([[c.x, c.y]]))[0]
            e = float(np.linalg.norm((q - PITCH_LANDMARKS[c.kp_idx]) * _SCALE_M) * _FT)
            errs.append(e)
            by_end.setdefault(end, []).append(e)
    return errs, by_end


def evaluate_crop_gate(
    points: Sequence[Click], lines: Sequence[LineClick], size: tuple[int, int],
    transforms: Mapping[int, NDArray[np.floating[Any]]],
    *, segment_of: Mapping[int, int] | None = None,
    gap_guard: int = DEFAULT_GAP_GUARD,
) -> CropGateReport:
    """Numeric acceptance gate (same thresholds as the physical engine's):
    foreground med <= 5 ft & p90 <= 12 ft AND propagation med <= 5 ft."""
    fg = foreground_holdout_crop(points, lines, size, transforms,
                                 segment_of=segment_of, gap_guard=gap_guard)
    pr, by_end = propagation_holdout_crop(points, lines, size, transforms,
                                          segment_of=segment_of, gap_guard=gap_guard)
    def _stats(v: list[float]) -> tuple[float, float]:
        return ((float(np.median(v)), float(np.percentile(v, 90)))
                if v else (float("inf"), float("inf")))
    fg_med, fg_p90 = _stats(fg)
    pr_med, pr_p90 = _stats(pr)
    ends = {k: (*_stats(v), len(v)) for k, v in sorted(by_end.items())}
    passed = fg_med <= 5.0 and fg_p90 <= 12.0 and pr_med <= 5.0
    return CropGateReport(fg_med, fg_p90, len(fg), pr_med, pr_p90, len(pr), ends, passed)
```

- [ ] **Step 4: Run to verify pass**, then the whole module file: `uv run pytest tests/test_global_crop.py -q` → all pass.
- [ ] **Step 5: Commit** — `git commit -am "feat(pitch): crop acceptance gate (foreground holdout + LOO propagation split by field end)"`

---

### Task 8: LabelerState engine wiring + honest export + gate sidecar

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py`
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/server.py` (thread `engine` kwarg into `run(...)` and the `LabelerState(...)` construction at ~line 239)
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/__main__.py` (add `--engine` flag)
- Create: `packages/soccer-vision/tests/test_labeler_state_crop.py`

- [ ] **Step 1: Check how server.py serializes CalibFrame** — run `grep -n "residual\|n_points\|asdict\|status" packages/soccer-vision/src/soccer_vision/labeler/server.py | head -20`. If it accesses fields explicitly (expected), the new `confidence` field is invisible to the frontend and safe. If it uses `dataclasses.asdict`, the extra key is additive JSON — also safe. Note the finding in the commit message.

- [ ] **Step 2: Write the failing tests**

```python
"""Crop-engine wiring through LabelerState: bootstrap, dirty scoping, honest export."""
import json
import numpy as np
import pandas as pd

from soccer_vision.labeler.state import LabelerState
from tests.test_global_crop import SIZE, CropWorld


def _interframe(world: CropWorld) -> dict[int, np.ndarray]:
    # cumulative_transforms convention: interframe[f] maps frame f -> f+1, and
    # M[f] = M[f-1] @ inv(interframe[f-1]) with M[0] = I. With interframe[f] =
    # T(d_f) @ inv(T(d_{f+1})) the reconstructed canvas is world's shifted by the
    # constant d_0 — an origin choice H_g absorbs, so the solve is unaffected.
    return {f: world.transforms[f] @ np.linalg.inv(world.transforms[f + 1])
            for f in range(world.n_frames - 1)}


def _state(world: CropWorld, tmp_path) -> LabelerState:
    return LabelerState(interframe=_interframe(world), n_frames=world.n_frames, size=SIZE,
                        autosave_path=tmp_path / "clicks.json", engine="crop")


def test_crop_bootstrap_single_frame(tmp_path) -> None:
    """The crop engine bootstraps from ONE well-spread frame (physical needed 3)."""
    world = CropWorld()
    st = _state(world, tmp_path)
    f = 120
    ids = world.visible(f)
    assert len(ids) >= 4
    st.add_clicks(world.clicks_at(f, ids))
    st.wait_idle()
    cf = st.frame_homography(f)
    assert cf is not None and cf.is_anchor
    st.stop_worker()


def test_export_honest_confidence_and_gate_json(tmp_path) -> None:
    world = CropWorld()
    st = _state(world, tmp_path)
    for f in range(0, world.n_frames, 20):
        st.add_clicks(world.clicks_at(f))
    st.wait_idle()
    out = tmp_path / "out"
    st.export(out)
    st.stop_worker()
    df = pd.read_parquet(out / "homographies.parquet")
    assert not df.empty
    assert (df["confidence"] <= 0.9).all() and (df["confidence"] >= 0.6).all()
    assert not (df["confidence"] == 1.0).any()          # the F-C2 overclaim is retired
    gate = json.loads((out / "calib_gate.json").read_text())
    assert gate["engine"] == "crop"
    assert "prop_by_end" in gate and isinstance(gate["passed_numeric"], bool)


def test_point_click_dirty_scoped_to_segment(tmp_path) -> None:
    """LabelerState builds segment_of from interframe gaps: dropping key 119 breaks
    the chain at frame 120. In crop mode a point click re-solves only its own
    segment's H_g, so a click at 150 must dirty only frames >= 120."""
    world = CropWorld()
    interframe = {f: m for f, m in _interframe(world).items() if f != 119}
    st = LabelerState(interframe=interframe, n_frames=world.n_frames, size=SIZE,
                      autosave_path=tmp_path / "c.json", engine="crop")
    for f in (0, 40, 80, 140, 180, 220):
        st.add_clicks(world.clicks_at(f))
    st.wait_idle()
    marked: list[int] = []
    st._worker.mark_dirty = lambda frames: marked.extend(frames)  # type: ignore[method-assign]
    c = world.clicks_at(150)[0]
    st.add_click(150, c.kp_idx, c.x, c.y)
    assert marked and min(marked) >= 120        # only the second segment re-solves
    st.stop_worker()
```

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/test_labeler_state_crop.py -x -q` → TypeError: unexpected keyword `engine`.

- [ ] **Step 4: Implement the state changes.** In `state.py`:

Imports:

```python
from soccer_vision.pitch.global_crop import (
    CropCalib,
    evaluate_crop_gate,
    frame_confidence,
    solve_crop_session,
)
```

`CalibFrame` gains a field (after `n_points`):

```python
    confidence: float        # honest export confidence (0.9 anchor / 0.8->0.6 ramp / 0.0)
```

`__init__` signature gains `engine: str = "physical"` and stores `self._engine = engine`; annotation of `_last_calib` becomes `PhysicalCalib | CropCalib | None`.

`_solve` branches (docstring: crop solve is cheap and deterministic; no warm seed needed):

```python
        if self._engine == "crop":
            calib_c = solve_crop_session(
                clicks, lines, self.size, self._transforms,
                segment_of=self._segment_of, gap_guard=self._gap_guard)
            with self._lock:
                self._last_calib = calib_c
            return calib_c
```

(keep the existing physical path for `engine == "physical"`; adjust the return type of `_solve` and `_build_frame`'s `calib` parameter to `PhysicalCalib | CropCalib`).

`_build_frame` fills the new field:

```python
        return CalibFrame(
            H=h,
            status=calib.status(f),
            is_anchor=anchor,
            residual=None,
            n_points=counts.get(f, 0) if anchor else 0,
            confidence=frame_confidence(calib, f),
        )
```

`add_click` dirty scope (replace the `mark_dirty(range(self.n_frames))` line):

```python
        if self._calibrated or self._try_bootstrap():
            # crop mode: a point click only constrains its own segment's H_g; physical
            # mode: it feeds the shared focal K, so every frame is dirty.
            dirty = self._affected(frame) if self._engine == "crop" else range(self.n_frames)
            self._worker.mark_dirty(dirty)
```

Apply the same branch in `nudge_click` and in `remove_last`'s point path (`kind != "ln"`).

`export` — replace the constant-confidence entry and add the gate sidecar:

```python
            # Honest gate: only export GREEN frames; confidence comes from the engine's
            # grading (anchor 0.9, propagated 0.8->0.6), never a constant 1.0 (F-C2).
            entries[f] = HomographyEntry(
                denormalize_homography(cf.H, self.size), "manual", cf.confidence)
        homographies_to_parquet(entries, out / "homographies.parquet")
        if self._engine == "crop":
            with self._lock:
                pts = list(self._active_clicks())
                lns = list(self.line_clicks)
            rep = evaluate_crop_gate(pts, lns, self.size, self._transforms,
                                     segment_of=self._segment_of,
                                     gap_guard=self._gap_guard)
            (out / "calib_gate.json").write_text(json.dumps({
                "engine": "crop",
                "fg_median_ft": rep.fg_median_ft, "fg_p90_ft": rep.fg_p90_ft,
                "fg_n": rep.fg_n,
                "prop_median_ft": rep.prop_median_ft, "prop_p90_ft": rep.prop_p90_ft,
                "prop_n": rep.prop_n,
                "prop_by_end": {k: list(v) for k, v in rep.prop_by_end.items()},
                "passed_numeric": rep.passed_numeric,
            }, indent=2))
```

In `server.py`: `run(...)` gains `engine: str = "physical"` and passes `engine=engine` into the `LabelerState(...)` call (~line 239). In `__main__.py`: `ap.add_argument("--engine", choices=["physical", "crop"], default="physical", help="calibration engine (crop = global-crop model)")` and pass `engine=args.engine` to `run`.

- [ ] **Step 5: Run to verify pass** — new file passes AND the existing suite still passes:

Run: `uv run pytest tests/test_labeler_state_crop.py tests/test_labeler_state.py tests/test_labeler_refit_worker.py tests/test_labeler_server.py -q`
Expected: all pass. Existing tests construct `LabelerState` without `engine` → default `"physical"` keeps behavior identical, EXCEPT any test asserting exported `confidence == 1.0` — update such a test to assert `confidence == frame_confidence(...)` semantics (0.9/ramp), which is the intended behavior change for BOTH engines.

- [ ] **Step 6: Commit** — `git commit -am "feat(labeler): crop engine wiring, segment-scoped dirty, honest export confidence + calib_gate.json"`

---

### Task 9: `validate_session` — engine comparison + crop diagnostics

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/validate_session.py`
- Modify: `packages/soccer-vision/tests/test_validate_session.py` (extend)

- [ ] **Step 1: Write the failing test** (append to the existing test file, reusing its fixtures if it has chain/clicks builders; otherwise build from `CropWorld` as in Task 8):

```python
def test_run_crop_gate_and_report(tmp_path) -> None:
    from tests.test_global_crop import SIZE, CropWorld
    from soccer_vision.pitch.validate_session import crop_gate_from_session
    world = CropWorld()
    clicks = [c for f in range(0, world.n_frames, 20) for c in world.clicks_at(f)]
    rep = crop_gate_from_session(clicks, [], SIZE, world.transforms)
    assert rep.prop_n > 0
```

- [ ] **Step 2: Run to verify failure** — ImportError.

- [ ] **Step 3: Implement.** Add to `validate_session.py`:

```python
from soccer_vision.pitch.global_crop import (
    CropGateReport,
    crop_assumption_report,
    evaluate_crop_gate,
    implied_camera,
    solve_crop_session,
)


def crop_gate_from_session(points, lines, size, transforms, *, segment_of=None):
    # thin, testable seam mirroring run_gate's physical path
    return evaluate_crop_gate(points, lines, size, transforms, segment_of=segment_of)
```

(type annotations to match the module's style: `points: list[Click]` etc.)

Extend `main()`:
- `ap.add_argument("--engine", choices=["physical", "crop", "both"], default="both")`
- `ap.add_argument("--crop-check", action="store_true", help="print crop-assumption decomposition stats and exit")`
- For `--crop-check`: load chain, print `crop_assumption_report(interframe, size)` and exit.
- For crop/both: after loading (chain, clicks, lines, segments, transforms), print the crop report in the same format as the physical one, PLUS the by-end table and the implied camera:

```python
        rep_c = evaluate_crop_gate(points, lines, size, transforms, segment_of=segment_of)
        print("\n[crop] held-out acceptance gate (feet):")
        print(f"  foreground   median={rep_c.fg_median_ft:6.2f}  p90={rep_c.fg_p90_ft:6.2f}  n={rep_c.fg_n}")
        print(f"  propagation  median={rep_c.prop_median_ft:6.2f}  p90={rep_c.prop_p90_ft:6.2f}  n={rep_c.prop_n}")
        for end, (med, p90, n) in rep_c.prop_by_end.items():
            print(f"    held frame saw {end:>4}: median={med:6.2f}  p90={p90:6.2f}  n={n}")
        print(f"  NUMERIC: {'PASS' if rep_c.passed_numeric else 'FAIL'}")
        calib = solve_crop_session(points, lines, size, transforms, segment_of=segment_of)
        for seg_id, ss in sorted(calib.segments.items()):
            centers = np.array([[0.5 + d[0], 0.5 + d[1]] for d in ss.offsets.values()])
            cam = implied_camera(ss.h_g, size, pp_canvas=centers.mean(axis=0))
            if cam is not None:
                f_px, c = cam
                print(f"  segment {seg_id}: implied focal {f_px:.0f}px  centre "
                      f"({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})m  "
                      f"one_end_capped={ss.one_end_capped}")
```

- Give `render_spotcheck` an `engine: str = "physical"` parameter: when `"crop"`, replace the `solve_session(...)` call with `solve_crop_session(points, lines, size, transforms, segment_of=segment_of)` (the calib object is interface-compatible for `frame_homography/status/is_anchor`). Thread `--engine`'s value through `main()`'s spot-check call (for `both`, render with crop into `<out>/crop/` and physical into `<out>/physical/`).

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_validate_session.py -q`.
- [ ] **Step 5: Commit** — `git commit -am "feat(pitch): validate_session --engine both + --crop-check + crop spot-check renders"`

---

### Task 10: Mechanical riders — PitchMapper w-guard + real RANSAC thresholds

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/mapper.py:37-40`
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py` (the `fit_homography(image_pts, pitch_pts)` call inside `fit_frame_homographies`, ~line 272)
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/landmarks.py:136`
- Test: `packages/soccer-vision/tests/test_pitch_mapper.py`, `tests/test_pitch_landmarks.py` (extend)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pitch_mapper.py (append)
def test_behind_camera_points_map_to_nan() -> None:
    # H whose third row makes w <= 0 for the probe pixel: a "behind camera" mapping
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -0.01, 1.0]])
    det = pd.DataFrame({"frame": [0, 0], "x_px": [10.0, 10.0], "y_px": [50.0, 200.0]})
    out = PitchMapper().transform(det, {0: H})
    assert np.isfinite(out.loc[0, "x_pitch"])            # w = 0.5 > 0: normal mapping
    assert np.isnan(out.loc[1, "x_pitch"]) and np.isnan(out.loc[1, "y_pitch"])  # w = -1
```

```python
# tests/test_pitch_landmarks.py (append)
def test_build_frame_homographies_rejects_gross_outlier() -> None:
    ids = [0, 1, 2, 3, 9, 10, 13, 14]
    H_true = np.array([[6e-4, 1e-5, 0.05], [2e-5, 9e-4, 0.02], [1e-6, 2e-6, 1.0]])
    inv = np.linalg.inv(H_true)
    px = []
    for i in ids:
        v = inv @ np.array([*PITCH_LANDMARKS[i], 1.0])
        px.append(v[:2] / v[2])
    rows = [{"frame": 0, "kp_idx": i, "x_px": float(p[0]), "y_px": float(p[1]), "conf": 1.0}
            for i, p in zip(ids, px, strict=True)]
    rows[0]["x_px"] += 400.0                              # one gross mislabel
    hs = build_frame_homographies(pd.DataFrame(rows))
    assert 0 in hs
    good = np.array([[r["x_px"], r["y_px"]] for r in rows[1:]])
    q = (hs[0] @ np.column_stack([good, np.ones(len(good))]).T).T
    q = q[:, :2] / q[:, 2:3]
    tgt = PITCH_LANDMARKS[ids[1:]]
    assert np.abs(q - tgt).max() < 0.02                   # outlier did not bias the fit
```

(Both blocks append to existing test files — check their import headers and add any of
`numpy as np` / `pandas as pd` / `PitchMapper` / `PITCH_LANDMARKS` / `build_frame_homographies`
that are missing.)

- [ ] **Step 2: Run to verify failure** — mapper test fails (finite garbage instead of NaN); landmarks test fails (biased fit).

- [ ] **Step 3: Implement.** `mapper.py` — replace lines 37-40 with:

```python
            mapped = (H @ pts.T).T
            wcol = mapped[:, 2]
            bad = wcol <= 1e-9  # behind-camera / at-horizon: NaN, never a mirrored coord
            safe = np.where(bad, 1.0, wcol)
            mapped = mapped / safe[:, None]
            mapped[bad, :2] = np.nan
            x_pitch[group.index] = mapped[:, 0]
            y_pitch[group.index] = mapped[:, 1]
```

`manual_anchor.py` `fit_frame_homographies` — change the call to
`fit_homography(image_pts, pitch_pts, ransac_thresh=0.012)`.
`landmarks.py:136` — change to
`fit_homography(image_points, pitch_points, ransac_thresh=0.012)`
(0.012 pitch units ≈ 0.8 m in the destination space; the previous default did zero rejection).

- [ ] **Step 4: Run the full suite** — `uv run pytest -q` from `packages/soccer-vision/`. The thresholds touch existing fit paths: if any existing test now fails, inspect whether it relied on outliers being averaged in (fix the test's data, not the threshold) — report anything ambiguous rather than force-fitting.

- [ ] **Step 5: Commit** — `git commit -am "fix(pitch): PitchMapper behind-camera NaN guard + real RANSAC thresholds at live fit sites"`

---

### Task 11: Real-session validation (the §4 acceptance evidence) — REPORT, do not flip

**Files:** none modified — this is an evidence run.

- [ ] **Step 1: Locate the oceanside chain cache**

```bash
cd packages/soccer-vision && uv run python - <<'EOF'
from pathlib import Path
from soccer_vision.labeler.chain import _video_hash
for stem in ("training_clip", "oceanside_clip"):
    v = Path.home() / f"sv-labeler/{stem}.mp4"
    if v.exists():
        print(stem, Path.home() / f"sv-labeler/.sv_labeler_cache/{_video_hash(v)}.npz")
    else:
        print(stem, "VIDEO NOT FOUND — ask Patrick for the path")
EOF
```

(training_clip's chain is known-good at `~/sv-labeler/.sv_labeler_cache/ef2546eaddd5e6fc.npz` if the hash lookup misses due to a re-encode.)

- [ ] **Step 2: Run the crop-assumption check + both-engine gate on each session**

```bash
uv run python -m soccer_vision.pitch.validate_session \
  --chain ~/sv-labeler/.sv_labeler_cache/ef2546eaddd5e6fc.npz \
  --clicks ~/sv-labeler/.sv_labeler_cache/training_clip.clicks.json \
  --engine both --crop-check
uv run python -m soccer_vision.pitch.validate_session \
  --chain <oceanside chain from Step 1> \
  --clicks ~/sv-labeler/.sv_labeler_cache/oceanside_clip.clicks.json \
  --engine both
```

- [ ] **Step 3: Render spot-checks including the previously-sky frames**

```bash
uv run python -m soccer_vision.pitch.validate_session \
  --chain ~/sv-labeler/.sv_labeler_cache/ef2546eaddd5e6fc.npz \
  --clicks ~/sv-labeler/.sv_labeler_cache/training_clip.clicks.json \
  --engine crop \
  --video ~/sv-labeler/training_clip.mp4 --spot-out ~/sv-labeler/crop_spotcheck
```

Confirm frames 134 and 193 are among the renders (they are clicked frames, so `_spot_frames` includes them).

- [ ] **Step 4: Report** the full numbers table (both engines, both sessions, by-end split, implied focal vs the expected 1461–1471, crop-assumption stats) and the spot-check paths. **STOP: Patrick's visual verdict on the renders is part of the binding gate. Do not proceed to Task 12 without it.** Render and report only — do not interpret the images (standing preference).

---

### Task 12: Engine default flip + final verification (only after Task 11 passes)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/__main__.py` (default `--engine crop`)
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/server.py` (`run(..., engine: str = "crop")`)

- [ ] **Step 1: Preconditions** — Task 11 numeric gates PASS on both sessions AND Patrick approved the renders. If either failed, stop and report instead.

- [ ] **Step 2: Flip the defaults** — change both defaults from `"physical"` to `"crop"`. `LabelerState.__init__`'s own default stays `"physical"` so existing tests remain valid; the CLI/server defaults are what users hit.

- [ ] **Step 3: Full verification from the repo root**

```bash
cd packages/soccer-vision && uv run pytest -q            # expect: all pass
cd ../.. && uv run mypy && uv run ruff check src tests   # canonical invocations, repo root
```

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(labeler): global-crop engine is the default (gate passed on both real sessions)"
```

---

## Self-review checklist (run after writing, before execution)

1. Spec §1 model → Task 1/4. §2 Step 0 → Task 6; Steps 1–4 → Tasks 3–4; Step 5 → Task 5. §3 status/confidence/export → Tasks 5/8. §4 gate → Tasks 7/11. §5 integration → Tasks 8/9; riders → Task 10. §6 tests → Tasks 1–8. Engine flip discipline → Tasks 11/12. No gaps found.
2. All code blocks complete; no TBDs.
3. Names used across tasks: `_translation/_t/_apply/_point_residuals_m/_line_residuals_m/_offset_axes/_solve_offset/_fit_h_g/_refine_h_g/_grade_anchor/_wsigns_ok/_fold_norm/solve_crop_session/SegmentSolve/CropCalib/frame_confidence/crop_assumption_report/implied_camera/foreground_holdout_crop/propagation_holdout_crop/CropGateReport/evaluate_crop_gate/crop_gate_from_session` — consistent.
