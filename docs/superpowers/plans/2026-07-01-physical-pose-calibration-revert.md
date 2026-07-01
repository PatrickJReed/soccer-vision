# Physical-Pose Calibration Revert — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the free-homography bundle in the live labeler with a physical per-frame point+line calibration engine (bracket propagation for unclicked frames, per-frame coverage-graded status, foreground+propagation acceptance gate with a visual spot-check).

**Architecture:** New `pitch/physical_calib.py` (`solve_session` → `PhysicalCalib`) solves each clicked frame as a real camera pose (`K[r1|r2|t]`) against the rigid field, reusing `calib/calibrate.py` primitives. Unclicked frames are filled by chain-shifting the bracketing anchors' homographies and blending into one fitted homography. `LabelerState`, `validate_session`, and the overlay render are rewired onto it; `global_calib.py`'s bundle is deleted.

**Tech Stack:** Python, numpy, OpenCV (`solvePnP` SQPNP, `Rodrigues`, `findHomography`), scipy (`least_squares` inside the existing `refine_pose`), pandas/pyarrow (export), pytest.

**Spec:** `docs/superpowers/specs/2026-07-01-physical-pose-calibration-revert-design.md`

---

## File Structure

- **Create** `packages/soccer-vision/src/soccer_vision/pitch/physical_calib.py` — the engine: `PhysicalCalib`, `solve_session`, propagation, status, held-out metrics, `GateReport`.
- **Create** `packages/soccer-vision/tests/test_physical_calib.py` — unit tests.
- **Modify** `packages/soccer-vision/src/soccer_vision/labeler/state.py` — rewire `_solve`/`_build_frame`/`_status_of`/`export`, `CalibFrame`, imports.
- **Modify** `packages/soccer-vision/src/soccer_vision/pitch/validate_session.py` — `evaluate_gate` + spot-check renders.
- **Modify** `packages/soccer-vision/src/soccer_vision/viz/pitch_overlay.py` — clipped line-draw helper.
- **Delete** `packages/soccer-vision/src/soccer_vision/pitch/global_calib.py` and `packages/soccer-vision/tests/test_pitch_global_calib.py`; update `tests/test_labeler_state.py`, `tests/test_pitch_manual_anchor.py`, and the `calib_anchor.py` docstring reference.

Canonical checks after every task: `uv run pytest <files>`, `uv run mypy` (src+tests), `uv run ruff check`.

---

## Task 1: Physical engine core — `solve_session` + `PhysicalCalib` (anchors only)

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/physical_calib.py`
- Test: `packages/soccer-vision/tests/test_physical_calib.py`

- [ ] **Step 1: Write the failing test** (`tests/test_physical_calib.py`)

```python
import numpy as np
import cv2
from soccer_vision.pitch.physical_calib import solve_session, PhysicalCalib
from soccer_vision.pitch.manual_anchor import Click, LineClick
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.calib.calibrate import calibrate_camera

SIZE = (1920, 1080)


def _project(fp_ids, K, rvec, tvec, size):
    fp = field_points_3d()
    img = cv2.projectPoints(fp[fp_ids], rvec, tvec, K, None)[0].reshape(-1, 2)
    w, h = size
    return [(i, x / w, y / h) for i, (x, y) in zip(fp_ids, img)]


def _synthetic_clicks(size):
    # a plausible sideline camera: build clicks by projecting known landmarks
    K = np.array([[1460.0, 0, size[0] / 2], [0, 1460.0, size[1] / 2], [0, 0, 1]])
    rvec = np.array([[1.2], [0.0], [0.0]])          # tilt down
    tvec = np.array([[-22.0], [-3.0], [40.0]])
    ids = [0, 1, 4, 9, 10, 13, 14]                  # own+opp spread
    proj = _project(ids, K, rvec, tvec, size)
    return [Click(frame=100, kp_idx=i, x=x, y=y) for i, x, y in proj], K, rvec, tvec


def test_solve_session_recovers_anchor():
    clicks, K, rvec, tvec = _synthetic_clicks(SIZE)
    calib = solve_session(clicks, [], SIZE, {100: np.eye(3)})
    assert calib.is_anchor(100)
    H = calib.frame_homography(100)
    assert H is not None
    # clicked points map back to their canonical landmarks within tolerance
    from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
    for c in clicks:
        q = H @ np.array([c.x, c.y, 1.0]); q = q[:2] / q[2]
        assert np.linalg.norm(q - PITCH_LANDMARKS[c.kp_idx]) < 0.02  # <~1.4 ft
    assert calib.frame_homography(999) is None  # no such anchor, no transform -> None
```

- [ ] **Step 2: Run it — expect ImportError / fail.** `uv run pytest tests/test_physical_calib.py -x`

- [ ] **Step 3: Implement `physical_calib.py` (core only).**

```python
"""Physical per-frame calibration for a fixed camera: each clicked (anchor) frame is a
real camera pose H = K[r1|r2|t] against the rigid 9v9 field; unclicked frames are filled by
bracket-propagating the neighbouring anchors through the inter-frame chain. Replaces the
free-homography bundle (pitch.global_calib). Pure: no I/O."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.calibrate import CalibError, calibrate_camera, refine_pose
from soccer_vision.calib.field_model import (
    LENGTH_M, METRES_TO_FEET, WIDTH_M, field_line_3d, field_points_3d,
)
from soccer_vision.calib.validate import fold_count
from soccer_vision.pitch.calib_anchor import flag_outlier_clicks, frame_homography
from soccer_vision.pitch.homography import HomographyError, fit_homography
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click, LineClick

FOLD_MIN, FOLD_MAX = 4, 15
DEFAULT_GAP_GUARD = 200
FOREGROUND_OK_FT = 8.0
GRID_N = 9
_FT = METRES_TO_FEET
_SCALE = np.array([WIDTH_M, LENGTH_M])


def _group(items: Sequence) -> dict[int, list]:
    d: dict[int, list] = {}
    for it in items:
        d.setdefault(it.frame, []).append(it)
    return d


def _apply(h: NDArray, pts: NDArray) -> NDArray:
    """pts (N,2 or N,3) -> (N,2) under homography h."""
    p = np.asarray(pts, dtype=np.float64)
    if p.shape[1] == 2:
        p = np.column_stack([p, np.ones(len(p))])
    q = (np.asarray(h, dtype=np.float64) @ p.T).T
    return q[:, :2] / q[:, 2:3]


def _line_perp_feet(qpitch: NDArray, line_id: str) -> float:
    p1, p2 = field_line_3d(line_id)
    a, b = p1[:2], p2[:2]
    pm = np.asarray(qpitch) * _SCALE
    ab = b - a
    L = math.hypot(ab[0], ab[1])
    cross = ab[0] * (a[1] - pm[1]) - ab[1] * (a[0] - pm[0])
    return float(abs(cross) / L * _FT) if L > 1e-9 else float("nan")


def _fold(h_norm: NDArray, size: tuple[int, int]) -> int:
    """fold_count for a NORMALIZED image->pitch homography (sign-normalized)."""
    w, h = size
    h_px = np.asarray(h_norm, dtype=np.float64) @ np.diag([1.0 / w, 1.0 / h, 1.0])
    try:
        h_p2px = np.linalg.inv(h_px)
    except np.linalg.LinAlgError:
        return 0
    if float((h_p2px @ np.array([0.5, 0.5, 1.0]))[2]) < 0:
        h_p2px = -h_p2px
    return fold_count(h_p2px, size)


def _anchor_pose(k, po, lo, seed_pose):
    ids = [i for i, _, _ in po]
    img = np.array([[x, y] for _, x, y in po], dtype=np.float64)
    fp = field_points_3d()
    ok, rv, tv = cv2.solvePnP(fp[ids], img, np.asarray(k), None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    if seed_pose is not None:
        rv, tv = np.asarray(seed_pose[0]), np.asarray(seed_pose[1])
    try:
        rv, tv = refine_pose(k, rv, tv, po, lo)
    except CalibError:
        pass
    return np.asarray(rv, dtype=np.float64), np.asarray(tv, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class PhysicalCalib:
    K: NDArray
    poses: dict[int, tuple[NDArray, NDArray]]
    anchor_h: dict[int, NDArray]              # normalized image -> pitch[0,1]
    coverage_grade: dict[int, str]           # anchor -> "green" | "yellow"
    transforms: dict[int, NDArray]
    size: tuple[int, int]
    gap_guard: int = DEFAULT_GAP_GUARD

    def is_anchor(self, frame: int) -> bool:
        return frame in self.anchor_h

    def nearest_anchor_gap(self, frame: int) -> int | None:
        if not self.anchor_h:
            return None
        return min(abs(frame - a) for a in self.anchor_h)

    def frame_homography(self, frame: int) -> NDArray | None:
        return None  # extended in Task 2 (anchor lookup added here in Task 1)


def solve_session(
    points: Sequence[Click],
    lines: Sequence[LineClick],
    size: tuple[int, int],
    transforms: Mapping[int, NDArray],
    *,
    min_points: int = 4,
    gap_guard: int = DEFAULT_GAP_GUARD,
    seed: "PhysicalCalib | None" = None,
) -> PhysicalCalib:
    w, h = size
    tf = {f: np.asarray(m, dtype=np.float64) for f, m in transforms.items()}
    by_pt = _group(points)
    by_ln = _group(lines)
    obs = {f: [(c.kp_idx, c.x * w, c.y * h) for c in cs] for f, cs in by_pt.items()}
    try:
        K = calibrate_camera(obs, size, min_points=6).K
    except CalibError:
        return PhysicalCalib(np.eye(3), {}, {}, {}, tf, size, gap_guard)
    clean, _flagged = flag_outlier_clicks(points, K, size)
    by_clean = _group(clean)
    diag = np.diag([float(w), float(h), 1.0])
    poses: dict[int, tuple[NDArray, NDArray]] = {}
    anchor_h: dict[int, NDArray] = {}
    grade: dict[int, str] = {}
    for f in sorted(by_clean):
        pcs = by_clean[f]
        lcs = by_ln.get(f, [])
        if len({c.kp_idx for c in pcs}) < min_points:
            continue
        po = [(c.kp_idx, c.x * w, c.y * h) for c in pcs]
        lo = [(l.line_id, l.x * w, l.y * h) for l in lcs]
        pose = _anchor_pose(K, po, lo, seed.poses.get(f) if seed else None)
        if pose is None:
            continue
        rv, tv = pose
        poses[f] = (rv, tv)
        anchor_h[f] = frame_homography(K, rv, tv) @ diag
        grade[f] = "yellow"  # real grade in Task 3
    return PhysicalCalib(K, poses, anchor_h, grade, tf, size, gap_guard)
```

Add the anchor branch to `frame_homography` now (propagation lands in Task 2):

```python
    def frame_homography(self, frame: int) -> NDArray | None:
        if frame in self.anchor_h:
            return self.anchor_h[frame]
        return None
```

- [ ] **Step 4: Run tests — expect PASS.** `uv run pytest tests/test_physical_calib.py -x`
- [ ] **Step 5: mypy + ruff, then commit.**

```bash
uv run mypy && uv run ruff check
git add packages/soccer-vision/src/soccer_vision/pitch/physical_calib.py packages/soccer-vision/tests/test_physical_calib.py
git commit -m "feat(calib): physical per-frame engine core (solve_session + PhysicalCalib anchors)"
```

---

## Task 2: Chain-shift + bracket propagation

**Files:**
- Modify: `pitch/physical_calib.py`
- Test: `tests/test_physical_calib.py`

- [ ] **Step 1: Write the failing test.**

```python
def test_bracket_propagation_on_translating_chain():
    # two anchors that are pure 2D translations of one base view; interior frame must be
    # recovered by chain-shift+blend to sub-pixel (drift-free synthetic chain).
    clicks, K, rvec, tvec = _synthetic_clicks(SIZE)
    a0 = [Click(frame=0, kp_idx=c.kp_idx, x=c.x, y=c.y) for c in clicks]
    # anchor at frame 20 = same scene shifted by dx in NORMALIZED coords
    dx = 0.05
    a20 = [Click(frame=20, kp_idx=c.kp_idx, x=c.x + dx, y=c.y) for c in clicks]
    T = {0: np.eye(3), 20: np.array([[1, 0, -dx], [0, 1, 0], [0, 0, 1.0]]),
         10: np.array([[1, 0, -dx / 2], [0, 1, 0], [0, 0, 1.0]])}
    calib = solve_session(a0 + a20, [], SIZE, T)
    H10 = calib.frame_homography(10)  # unclicked interior frame
    assert H10 is not None
    from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
    # a landmark seen at (x+dx/2) in frame 10 should map to its canonical position
    for c in clicks:
        q = _apply_pt(H10, c.x + dx / 2, c.y)
        assert np.linalg.norm(q - PITCH_LANDMARKS[c.kp_idx]) < 0.03


def test_gap_guard_returns_none():
    clicks, *_ = _synthetic_clicks(SIZE)
    T = {i: np.eye(3) for i in range(0, 500)}
    calib = solve_session([Click(frame=100, kp_idx=c.kp_idx, x=c.x, y=c.y) for c in clicks],
                          [], SIZE, T, gap_guard=200)
    assert calib.frame_homography(150) is not None   # within gap (one-sided shift)
    assert calib.frame_homography(400) is None        # >200 from the only anchor


def _apply_pt(H, x, y):
    q = H @ np.array([x, y, 1.0]); return q[:2] / q[2]
```

- [ ] **Step 2: Run — expect fail** (propagation returns None). `uv run pytest tests/test_physical_calib.py -x`

- [ ] **Step 3: Implement propagation.** Add module functions and replace `frame_homography`:

```python
def _grid(n: int) -> NDArray:
    xs = np.linspace(0.0, 1.0, n)
    gx, gy = np.meshgrid(xs, xs)
    return np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])


def _shift_h(anchor_h: NDArray, m_a: NDArray, m_t: NDArray) -> NDArray:
    return np.asarray(anchor_h) @ np.linalg.inv(np.asarray(m_a)) @ np.asarray(m_t)


def _bracket_h(h_lo: NDArray, h_hi: NDArray, w: float) -> NDArray:
    g = _grid(GRID_N)
    blended = (1.0 - w) * _apply(h_lo, g) + w * _apply(h_hi, g)
    try:
        return np.asarray(fit_homography(g[:, :2], blended), dtype=np.float64)
    except HomographyError:
        return h_lo if w < 0.5 else h_hi
```

Replace `PhysicalCalib.frame_homography`:

```python
    def frame_homography(self, frame: int) -> NDArray | None:
        if frame in self.anchor_h:
            return self.anchor_h[frame]
        if not self.anchor_h or frame not in self.transforms:
            return None
        if (gap := self.nearest_anchor_gap(frame)) is None or gap > self.gap_guard:
            return None
        anchors = sorted(self.anchor_h)
        lo = [a for a in anchors if a < frame]
        hi = [a for a in anchors if a > frame]
        if lo and hi and lo[-1] in self.transforms and hi[0] in self.transforms:
            a, b = lo[-1], hi[0]
            h_lo = _shift_h(self.anchor_h[a], self.transforms[a], self.transforms[frame])
            h_hi = _shift_h(self.anchor_h[b], self.transforms[b], self.transforms[frame])
            return _bracket_h(h_lo, h_hi, (frame - a) / (b - a))
        a = lo[-1] if lo else hi[0]
        if a not in self.transforms:
            return None
        return _shift_h(self.anchor_h[a], self.transforms[a], self.transforms[frame])
```

- [ ] **Step 4: Run tests — expect PASS.**
- [ ] **Step 5: mypy + ruff, commit.** `git commit -m "feat(calib): bracket propagation (chain-shift + fitted blend) with gap guard"`

---

## Task 3: Coverage self-check + per-frame status

**Files:** Modify `pitch/physical_calib.py`; Test `tests/test_physical_calib.py`

- [ ] **Step 1: Write the failing test.**

```python
def _near_tl_clicks(frame, K, rvec, tvec, size, n=3):
    # project points on the near touchline (x=0) to pixels -> LineClicks
    from soccer_vision.calib.field_model import LENGTH_M
    w, h = size
    obj = np.array([[0.0, y, 0.0] for y in np.linspace(5, LENGTH_M - 5, n)])
    img = cv2.projectPoints(obj, rvec, tvec, K, None)[0].reshape(-1, 2)
    return [LineClick(frame=frame, line_id="near_touchline", x=x / w, y=y / h) for x, y in img]


def test_status_grades():
    clicks, K, rvec, tvec = _synthetic_clicks(SIZE)
    T = {i: np.eye(3) for i in range(0, 400)}
    # anchor with a correct near-TL -> GREEN
    green_lines = _near_tl_clicks(100, K, rvec, tvec, SIZE)
    calib = solve_session([Click(100, c.kp_idx, c.x, c.y) for c in clicks], green_lines, SIZE, T)
    assert calib.status(100) == "green"
    # same anchor without any near-TL -> YELLOW (foreground unverified)
    calib2 = solve_session([Click(100, c.kp_idx, c.x, c.y) for c in clicks], [], SIZE, T)
    assert calib2.status(100) == "yellow"
    # propagated within gap -> yellow; beyond gap -> red
    assert calib2.status(150) == "yellow"
    assert calib2.status(390) == "red"
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement.** Add the errors helper + self-check, populate grade in `solve_session`, add `status`:

```python
def _foreground_errors(k, po, line_clicks, size):
    """Held-out near-TL feet errors for one frame (refit WITHOUT its near-TL), or None if
    the frame has no near-touchline click."""
    if not any(l.line_id == "near_touchline" for l in line_clicks):
        return None
    w, h = size
    lo_fit = [(l.line_id, l.x * w, l.y * h) for l in line_clicks if l.line_id != "near_touchline"]
    pose = _anchor_pose(k, po, lo_fit, None)
    if pose is None:
        return None
    rv, tv = pose
    h_norm = frame_homography(k, rv, tv) @ np.diag([float(w), float(h), 1.0])
    errs = [_line_perp_feet(_apply(h_norm, np.array([[l.x, l.y]]))[0], "near_touchline")
            for l in line_clicks if l.line_id == "near_touchline"]
    return errs or None


def _grade(k, po, line_clicks, size):
    errs = _foreground_errors(k, po, line_clicks, size)
    if errs is None:
        return "yellow"
    return "green" if float(np.median(errs)) <= FOREGROUND_OK_FT else "yellow"
```

In `solve_session`, replace `grade[f] = "yellow"` with `grade[f] = _grade(K, po, lcs, size)`.

Add to `PhysicalCalib`:

```python
    def status(self, frame: int) -> str:
        h = self.frame_homography(frame)
        if h is None:
            return "red"
        if not FOLD_MIN <= _fold(h, self.size) <= FOLD_MAX:
            return "red"
        if frame in self.anchor_h:
            return self.coverage_grade.get(frame, "yellow")
        return "yellow"
```

- [ ] **Step 4: Run — PASS.**  **Step 5: mypy + ruff, commit.** `git commit -m "feat(calib): coverage-graded per-frame status (anchor self-check)"`

---

## Task 4: Acceptance metrics — `evaluate_gate`

**Files:** Modify `pitch/physical_calib.py`; Test `tests/test_physical_calib.py`

- [ ] **Step 1: Write the failing test.**

```python
from soccer_vision.pitch.physical_calib import evaluate_gate, GateReport


def test_gate_passes_on_clean_multi_anchor():
    clicks, K, rvec, tvec = _synthetic_clicks(SIZE)
    T = {i: np.eye(3) for i in range(0, 200)}
    pts, lns = [], []
    for f in (20, 60, 100, 140):                     # several anchors, all with near-TL
        pts += [Click(f, c.kp_idx, c.x, c.y) for c in clicks]
        lns += _near_tl_clicks(f, K, rvec, tvec, SIZE)
    rep = evaluate_gate(pts, lns, SIZE, T)
    assert isinstance(rep, GateReport)
    assert rep.fg_median_ft <= 5.0 and rep.prop_median_ft <= 5.0
    assert rep.passed_numeric
```

- [ ] **Step 2: Run — expect ImportError.**

- [ ] **Step 3: Implement.**

```python
@dataclass(frozen=True)
class GateReport:
    fg_median_ft: float
    fg_p90_ft: float
    fg_n: int
    prop_median_ft: float
    prop_p90_ft: float
    prop_n: int
    passed_numeric: bool


def foreground_holdout(points, lines, size, *, min_points=4):
    w, h = size
    by_pt, by_ln = _group(points), _group(lines)
    K = calibrate_camera(
        {f: [(c.kp_idx, c.x * w, c.y * h) for c in cs] for f, cs in by_pt.items()},
        size, min_points=6).K
    errs: list[float] = []
    for f, pcs in by_pt.items():
        if len({c.kp_idx for c in pcs}) < min_points:
            continue
        po = [(c.kp_idx, c.x * w, c.y * h) for c in pcs]
        fe = _foreground_errors(K, po, by_ln.get(f, []), size)
        if fe:
            errs.extend(fe)
    return errs


def propagation_holdout(points, lines, size, transforms, *, gap_guard=DEFAULT_GAP_GUARD):
    by_pt = _group(points)
    anchors = sorted(f for f in by_pt if len({c.kp_idx for c in by_pt[f]}) >= 4)
    errs: list[float] = []
    for held in anchors:
        others = [a for a in anchors if a != held]
        if not others or min(abs(held - a) for a in others) > gap_guard:
            continue
        rest_p = [c for c in points if c.frame != held]
        rest_l = [l for l in lines if l.frame != held]
        calib = solve_session(rest_p, rest_l, size, transforms, gap_guard=gap_guard)
        H = calib.frame_homography(held)
        if H is None:
            continue
        for c in by_pt[held]:
            q = _apply(H, np.array([[c.x, c.y]]))[0]
            errs.append(float(np.linalg.norm((q - PITCH_LANDMARKS[c.kp_idx]) * _SCALE) * _FT))
    return errs


def evaluate_gate(points, lines, size, transforms, *, gap_guard=DEFAULT_GAP_GUARD):
    fg = foreground_holdout(points, lines, size)
    pr = propagation_holdout(points, lines, size, transforms, gap_guard=gap_guard)
    fg_med = float(np.median(fg)) if fg else float("inf")
    fg_p90 = float(np.percentile(fg, 90)) if fg else float("inf")
    pr_med = float(np.median(pr)) if pr else float("inf")
    pr_p90 = float(np.percentile(pr, 90)) if pr else float("inf")
    passed = fg_med <= 5.0 and fg_p90 <= 12.0 and pr_med <= 5.0
    return GateReport(fg_med, fg_p90, len(fg), pr_med, pr_p90, len(pr), passed)
```

- [ ] **Step 4: Run — PASS.**  **Step 5: mypy + ruff, commit.** `git commit -m "feat(calib): evaluate_gate (foreground + propagation held-out)"`

---

## Task 5: Clipped overlay render helper

**Files:** Modify `viz/pitch_overlay.py`; Test `tests/test_pitch_overlay.py` (add case)

- [ ] **Step 1: Write the failing test.**

```python
import numpy as np
from soccer_vision.viz.pitch_overlay import clipped_polyline

def test_clipped_polyline_drops_behind_and_offscreen():
    # H maps pitch[0,1]->px; craft points where one endpoint is behind camera (w<=0)
    # and one off-frame; only the valid in-frame point should survive.
    H = np.array([[1000.0, 0, 200], [0, 1000.0, 200], [0, 0, 1.0]])
    pts = np.array([[0.1, 0.1], [0.15, 0.15]])       # both in front, near frame
    out = clipped_polyline(H, pts, size=(1920, 1080), margin=80)
    assert all(0 - 80 <= x <= 1920 + 80 and 0 - 80 <= y <= 1080 + 80 for x, y in out)
    behind = np.array([[0.1, 0.1]])
    Hneg = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])  # forces w<0
    assert clipped_polyline(Hneg, behind, size=(1920, 1080), margin=80) == []
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement `clipped_polyline` in `viz/pitch_overlay.py`.**

```python
def clipped_polyline(h_pitch_to_px, pts, *, size, margin=80):
    """Project pitch points through a pitch->px homography, keeping only those in front of
    the camera (w>0) and within the frame + margin. Returns a list of (int,int) pixels."""
    import numpy as _np
    w, h = size
    out = []
    for px, py in _np.asarray(pts, dtype=float):
        v = _np.asarray(h_pitch_to_px, dtype=float) @ _np.array([px, py, 1.0])
        if v[2] <= 1e-9:
            continue
        x, y = v[0] / v[2], v[1] / v[2]
        if -margin <= x <= w + margin and -margin <= y <= h + margin:
            out.append((int(x), int(y)))
    return out
```

Note: the live labeler overlay is drawn in the frontend JS; this Python helper backs the `validate_session` spot-check renders (Task 7) and QA. Frontend clipping is a separate follow-up (out of scope).

- [ ] **Step 4: Run — PASS.**  **Step 5: mypy + ruff, commit.** `git commit -m "feat(viz): clipped_polyline for in-front/in-frame overlay rendering"`

---

## Task 6: Rewire `LabelerState` onto the physical engine

**Files:** Modify `labeler/state.py`; Test `tests/test_labeler_state.py`

- [ ] **Step 1: Update the test** to reflect the physical status/export (edit existing bundle-based assertions).

```python
def test_state_physical_anchor_green_and_export(tmp_path):
    from soccer_vision.labeler.state import LabelerState
    from soccer_vision.pitch.manual_anchor import Click, LineClick
    # ... build a small interframe (identity chain) + a synthetic anchor's clicks+near-TL ...
    st = LabelerState(interframe, n_frames, size=SIZE)
    st.add_clicks(anchor_clicks)
    st.add_line_clicks(near_tl_lines)
    st.wait_idle()
    assert st._status_of(anchor_frame) == "green"
    st.export(tmp_path)
    import pandas as pd
    hom = pd.read_parquet(tmp_path / "homographies.parquet")
    assert anchor_frame in set(hom["frame"].tolist())     # green frame exported
```

- [ ] **Step 2: Run — expect fail** (still bundle-wired).

- [ ] **Step 3: Rewire `state.py`.**
  - Imports: delete the `global_calib` import block; add
    `from soccer_vision.pitch.physical_calib import PhysicalCalib, solve_session`.
  - Replace `CalibFrame`:

```python
@dataclass(frozen=True, eq=False)
class CalibFrame:
    H: NDArray[np.float64]   # NORMALIZED image -> pitch[0,1]
    status: str              # "green" | "yellow" | "red" (from PhysicalCalib.status)
    is_anchor: bool
```

  - Replace `self._last_bundle: BundleCalib | None` with `self._last_calib: PhysicalCalib | None = None`.
  - `_solve` → return a `PhysicalCalib` (snapshot points **and** line_clicks under the lock; pass both to `solve_session`; warm-start with `seed=self._last_calib`; store it):

```python
    def _solve(self) -> PhysicalCalib:
        with self._lock:
            pts = list(self._active_clicks())
            lns = list(self.line_clicks)
            seed = self._last_calib
        calib = solve_session(pts, lns, self.size, self._transforms,
                              gap_guard=self._gap_guard, seed=seed)
        with self._lock:
            self._last_calib = calib
        return calib
```

  - `_build_frame(calib, f)`:

```python
    def _build_frame(self, calib: PhysicalCalib, f: int) -> CalibFrame | None:
        h = calib.frame_homography(f)
        if h is None:
            return None
        return CalibFrame(H=h, status=calib.status(f), is_anchor=calib.is_anchor(f))
```

  - `_compute_dirty`: drop the `two_ended` tuple — `calib = self._solve()`; build with `self._build_frame(calib, f)`.
  - `_try_bootstrap`: calibrated once `calib.anchor_h` is non-empty:
    `calib = self._solve(); if not calib.anchor_h: return False; ... self._calibrated = True`.
  - `_status_of`: `cf = self._fits.get(f); return cf.status if cf else "red"`.
  - `export`: gate on `cf.status == "green"`; confidence `1.0` for green (drop the residual-based
    penalty and `residual_px_threshold`); everything else (block-until-idle, keypoints, line_clicks
    parquet) unchanged.
  - Add `self._gap_guard = DEFAULT_GAP_GUARD` (import from physical_calib) in `__init__`; drop
    `residual_px_threshold`.

- [ ] **Step 4: Run — PASS.**  **Step 5: full labeler suite + mypy + ruff, commit.**
`git commit -m "feat(labeler): LabelerState runs on the physical per-frame calib"`

---

## Task 7: Rewrite `validate_session` (gate + spot-check)

**Files:** Modify `pitch/validate_session.py`; Test `tests/test_validate_session.py` (new/updated)

- [ ] **Step 1: Write the failing test** — call the new `run_gate(chain, clicks)` helper on a tiny fixture and assert a `GateReport` comes back.

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement.** Replace the `cross_end_holdout` body:
  - Load points **and** lines (`clicks_from_sidecar` + `line_clicks_from_sidecar`).
  - `report = evaluate_gate(points, lines, size, transforms)`; print fg/prop median+p90 and
    `PASS/FAIL` from `report.passed_numeric`.
  - Add optional `--video` + `--spot-out`: when given, render N spread frames (incl. the sparsest
    anchor and one no-line frame) via `clipped_polyline` over the decoded frame, write PNGs, and
    print `REQUIRED: review <dir> and confirm foreground/overlay before trusting green`.
  - Keep it a thin CLI wrapping a testable `run_gate(...) -> GateReport`.

- [ ] **Step 4: Run — PASS.**  **Step 5: mypy + ruff, commit.**
`git commit -m "feat(calib): validate_session gate = foreground+propagation + visual spot-check"`

---

## Task 8: Delete the bundle + clean up

**Files:** Delete `pitch/global_calib.py`, `tests/test_pitch_global_calib.py`; edit `tests/test_pitch_manual_anchor.py`, `tests/test_labeler_state.py`, `pitch/calib_anchor.py` docstring.

- [ ] **Step 1:** `git rm packages/soccer-vision/src/soccer_vision/pitch/global_calib.py packages/soccer-vision/tests/test_pitch_global_calib.py`
- [ ] **Step 2:** Remove any `global_calib`/bundle-symbol imports from remaining tests; delete tests that only exercised deleted symbols; update `calib_anchor.py`'s docstring reference to `global_calib.solve_bundle`.
- [ ] **Step 3:** `grep -rn "global_calib\|solve_bundle\|BundleCalib\|cross_end_holdout\|fold_of_norm\|two_ended_segments\|solve_global\|GlobalCalib" packages/soccer-vision/src packages/soccer-vision/tests` → expect **no** matches.
- [ ] **Step 4:** Full suite + `uv run mypy` + `uv run ruff check` — all green.
- [ ] **Step 5: Commit.** `git commit -m "refactor(calib): delete the free-homography bundle (replaced by physical engine)"`

---

## Task 9: Real-session validation

**Files:** none (verification only)

- [ ] **Step 1:** Run the gate on the training session:

```bash
cd packages/soccer-vision && uv run python -m soccer_vision.pitch.validate_session \
  --chain ~/sv-labeler/.sv_labeler_cache/ef2546eaddd5e6fc.npz \
  --clicks ~/sv-labeler/.sv_labeler_cache/training_clip.clicks.json \
  --video ~/sv-labeler/training_clip.mp4 --spot-out ~/sv-labeler/gate_spotcheck
```

- [ ] **Step 2:** Confirm foreground median ≈ 3.6 ft / propagation median ≈ 3.8 ft and `PASS`
  (matches this session's experiment). Investigate any regression before proceeding.
- [ ] **Step 3:** Open the spot-check PNGs for Patrick to review (render only; he interprets), incl.
  a sparse/no-line frame (expected yellow, rougher). Do not claim visual pass — that's his call.

---

## Self-review

- **Spec coverage:** solve_session/PhysicalCalib (T1), bracket propagation + gap guard (T2), coverage-graded status (T3), gate (T4), clipped overlay (T5), LabelerState rewire + honest export (T6), validate_session gate + spot-check (T7), deletion (T8), real-session acceptance (T9) — every spec section maps to a task.
- **Type consistency:** `frame_homography` returns `NDArray | None` everywhere; `PhysicalCalib` field names match usage across T1–T7; `CalibFrame` new fields (`H`, `status`, `is_anchor`) used consistently in `state.py`.
- **Placeholders:** none — each code step carries real implementation; T6/T7 give exact edits against the current file contents.
- **Note:** frontend JS overlay clipping is explicitly out of scope (T5 note); the Python `clipped_polyline` backs the spot-check.
