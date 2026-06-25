# Camera Calibration Phase 2 — Line Constraints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add line-constraint pose refinement to the camera-calibration core — a clicked pixel asserted to lie on a known field line tightens an otherwise under-constrained pose (the near touchline and full midline have no clickable point landmarks).

**Architecture:** Three pure additions to the existing `soccer_vision.calib` package. (1) A field-lines registry in `field_model.py` mapping named pitch lines to their two landmark-index endpoints. (2) A `line_residual` helper in `calibrate.py` = perpendicular pixel distance from a clicked point to the image line through a field line's projected endpoints. (3) `refine_pose` in `calibrate.py` = a `scipy.optimize.least_squares` solve over the 6 pose DOF whose residual vector concatenates point reprojection errors and line distances, focal held fixed, seeded from a Phase-1 pose. Validated entirely on synthetic cameras — no labeler UI, no real data (Phase 3).

**Tech Stack:** Python, NumPy, OpenCV (`cv2.projectPoints`, `cv2.Rodrigues`), SciPy (`scipy.optimize.least_squares`), pytest, mypy (strict), ruff.

**Spec:** `docs/superpowers/specs/2026-06-25-camera-calibration-phase2-design.md`

**Conventions to honor (from Phase 1):**
- The world-metres→pixel homography is `homography_from_pose(K, rvec, tvec)` (already exists in `calibrate.py`): `H = K [r1 | r2 | t]`, maps `(X, Y, 1)` field-metres on Z=0 to pixels.
- Field-metre landmark positions are `field_points_3d()` → `(21, 3)`, Z=0. Landmark index map (from `pitch/landmarks.py`): 0 corner_own_left `(0,0)`, 1 corner_own_right `(W,0)`, 2 corner_opp_left `(0,L)`, 3 corner_opp_right `(W,L)`, 4 halfway_far `(W, L/2)`, 5 halfway_near `(0, L/2)` (hidden under camera). `W = WIDTH_M = 45.7`, `L = LENGTH_M = 68.5`.
- `CalibError` (already in `calibrate.py`) is the package's calibration exception.
- `cv2` rvec/tvec are `(3, 1)` float64 arrays; keep that shape.
- Lint gate: **`uv run mypy` from the repo root** (`/Users/patrickreed/Sandbox/soccer-vision`), and `uv run ruff check` from `packages/soccer-vision`. Tests: `uv run pytest` from `packages/soccer-vision`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `packages/soccer-vision/src/soccer_vision/calib/field_model.py` | Field geometry (points + now lines) | Add `FIELD_LINES`, `field_line_3d` |
| `packages/soccer-vision/src/soccer_vision/calib/calibrate.py` | Camera solve (pose↔homography, global calib, now line refine) | Add `line_residual`, `refine_pose` |
| `packages/soccer-vision/tests/test_calib_lines.py` | All Phase 2 tests (registry + residual + refine + gate) | Create |

---

## Task 1: Field-lines registry

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/calib/field_model.py`
- Test: `packages/soccer-vision/tests/test_calib_lines.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `packages/soccer-vision/tests/test_calib_lines.py` with:

```python
"""Phase 2 tests: field-lines registry, line residual, line-constrained pose refine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from soccer_vision.calib.field_model import (
    LENGTH_M,
    WIDTH_M,
    FIELD_LINES,
    field_line_3d,
    field_points_3d,
)


def test_field_line_3d_near_touchline_endpoints() -> None:
    p1, p2 = field_line_3d("near_touchline")
    fp = field_points_3d()
    assert np.allclose(p1, fp[0])  # corner_own_left
    assert np.allclose(p2, fp[2])  # corner_opp_left
    assert p1[2] == 0.0 and p2[2] == 0.0  # on the Z=0 plane


def test_field_line_3d_midline_spans_full_width() -> None:
    # midline = (5, 4): halfway_near (x=0) -> halfway_far (x=WIDTH), both at y=L/2
    p1, p2 = field_line_3d("midline")
    assert np.allclose(p1, [0.0, LENGTH_M / 2, 0.0])
    assert np.allclose(p2, [WIDTH_M, LENGTH_M / 2, 0.0])


def test_field_lines_registry_has_five_named_lines() -> None:
    assert set(FIELD_LINES) == {
        "near_touchline", "far_touchline", "own_goal_line", "opp_goal_line", "midline",
    }


def test_field_line_3d_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        field_line_3d("not_a_line")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_lines.py -v`
Expected: FAIL with `ImportError: cannot import name 'FIELD_LINES'` (and `field_line_3d`).

- [ ] **Step 3: Implement the registry**

In `packages/soccer-vision/src/soccer_vision/calib/field_model.py`, after `field_points_3d`, add:

```python
# Named pitch lines as pairs of landmark indices whose 3D positions are the line's
# two endpoints. These cover the regions with no clickable point landmark: the near
# touchline (x=0, under the camera) and the FULL midline (halfway_near/idx 5 is
# hidden, so only halfway_far/idx 4 is a clickable point — but the line is known).
FIELD_LINES: dict[str, tuple[int, int]] = {
    "near_touchline": (0, 2),  # corner_own_left -> corner_opp_left (x=0)
    "far_touchline": (1, 3),   # corner_own_right -> corner_opp_right (x=WIDTH_M)
    "own_goal_line": (0, 1),   # corner_own_left -> corner_own_right (y=0)
    "opp_goal_line": (2, 3),   # corner_opp_left -> corner_opp_right (y=LENGTH_M)
    "midline": (5, 4),         # halfway_near -> halfway_far (y=LENGTH_M/2)
}


def field_line_3d(line_id: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The two 3D metre endpoints of a named field line (on the Z=0 plane)."""
    if line_id not in FIELD_LINES:
        raise KeyError(f"unknown line {line_id!r}; valid: {sorted(FIELD_LINES)}")
    i, j = FIELD_LINES[line_id]
    fp = field_points_3d()
    return fp[i], fp[j]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_lines.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/calib/field_model.py tests/test_calib_lines.py`
Expected: no errors.
Run (from repo root `/Users/patrickreed/Sandbox/soccer-vision`): `uv run mypy`
Expected: no NEW errors in `field_model.py` / `test_calib_lines.py` (pre-existing errors elsewhere are unchanged — see `reference_operational` note).

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/calib/field_model.py packages/soccer-vision/tests/test_calib_lines.py
git commit -m "feat(calib): field-lines registry (FIELD_LINES, field_line_3d)"
```

---

## Task 2: Line residual (point-to-projected-line distance)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/calib/calibrate.py`
- Test: `packages/soccer-vision/tests/test_calib_lines.py`

- [ ] **Step 1: Write the failing tests**

Append to `packages/soccer-vision/tests/test_calib_lines.py`. First add `line_residual` to the calibrate import and a shared synthetic-camera helper, then the tests:

```python
from soccer_vision.calib.calibrate import (
    homography_from_pose,
    line_residual,
)

_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)


def _look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV (world->camera) rvec, tvec for a camera at `eye` looking at `target`."""
    eye_a, target_a, up_a = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    fwd = target_a - eye_a
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_a)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rmat = np.vstack([right, down, fwd])
    rvec, _ = cv2.Rodrigues(rmat)
    return rvec, (-rmat @ eye_a).reshape(3, 1)


def test_line_residual_zero_on_the_line() -> None:
    # A pixel that is the projection of a point ON near_touchline (x=0) must have ~0
    # distance to the projected near_touchline.
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    h = homography_from_pose(_K, rvec, tvec)
    p1, p2 = field_line_3d("near_touchline")
    on_line_world = np.array([0.0, 30.0, 0.0])  # x=0 -> on near_touchline
    px = cv2.projectPoints(on_line_world.reshape(1, 3), rvec, tvec, _K, np.zeros(5))[0].reshape(2)
    d = line_residual(h, p1, p2, (float(px[0]), float(px[1])))
    assert d < 0.01


def test_line_residual_equals_perpendicular_offset() -> None:
    # Take an on-line pixel, then offset it perpendicular to the projected line by a
    # known number of pixels; the residual should equal that offset.
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    h = homography_from_pose(_K, rvec, tvec)
    p1, p2 = field_line_3d("near_touchline")
    a = h @ np.array([p1[0], p1[1], 1.0]); a = a[:2] / a[2]
    b = h @ np.array([p2[0], p2[1], 1.0]); b = b[:2] / b[2]
    ell = np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])
    n = np.array([ell[0], ell[1]]) / np.hypot(ell[0], ell[1])  # unit normal
    on_line = (a + b) / 2.0
    offset_px = 7.5
    clicked = on_line + offset_px * n
    d = line_residual(h, p1, p2, (float(clicked[0]), float(clicked[1])))
    assert abs(d - offset_px) < 1e-6


def test_line_residual_finite_when_endpoints_offscreen() -> None:
    # near_touchline endpoints (corners) project far off-screen in a midfield view,
    # but the residual must still be a correct finite distance.
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    h = homography_from_pose(_K, rvec, tvec)
    p1, p2 = field_line_3d("near_touchline")
    d = line_residual(h, p1, p2, (500.0, 500.0))
    assert np.isfinite(d) and d >= 0.0


def test_line_residual_degenerate_endpoints_returns_zero() -> None:
    # A homography that collapses both endpoints to the same pixel -> degenerate line.
    p1, p2 = field_line_3d("near_touchline")
    h = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 200.0], [0.0, 0.0, 1.0]])  # maps everything to (100,200)
    d = line_residual(h, p1, p2, (300.0, 400.0))
    assert d == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_lines.py -k line_residual -v`
Expected: FAIL with `ImportError: cannot import name 'line_residual'`.

- [ ] **Step 3: Implement `line_residual`**

In `packages/soccer-vision/src/soccer_vision/calib/calibrate.py`, add `import math` near the top (after `from dataclasses import dataclass`), and add this function after `pitch_homography`:

```python
def line_residual(
    h_world: NDArray[np.floating],
    p1_3d: NDArray[np.floating],
    p2_3d: NDArray[np.floating],
    clicked_px: tuple[float, float],
) -> float:
    """Perpendicular pixel distance from clicked_px to the image line through the
    projections of the two world-metre line endpoints under h_world.

    Returns 0.0 if the projected line is degenerate (endpoints coincide or project
    at infinity) so a least-squares residual vector keeps a constant length.
    """
    h = np.asarray(h_world, dtype=np.float64)

    def _project(p: NDArray[np.floating]) -> NDArray[np.float64] | None:
        v = h @ np.array([float(p[0]), float(p[1]), 1.0])
        if abs(v[2]) < 1e-9:
            return None
        return np.asarray(v[:2] / v[2], dtype=np.float64)

    a = _project(p1_3d)
    b = _project(p2_3d)
    if a is None or b is None:
        return 0.0
    ell = np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])
    norm = math.hypot(float(ell[0]), float(ell[1]))
    if norm < 1e-9:
        return 0.0
    u, v = clicked_px
    return float(abs(ell[0] * u + ell[1] * v + ell[2]) / norm)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_lines.py -k line_residual -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/calib/calibrate.py tests/test_calib_lines.py`
Expected: no errors.
Run (from repo root): `uv run mypy`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/calib/calibrate.py packages/soccer-vision/tests/test_calib_lines.py
git commit -m "feat(calib): line_residual (point-to-projected-line distance)"
```

---

## Task 3: Line-constrained pose refinement (`refine_pose`) + synthetic gate

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/calib/calibrate.py`
- Test: `packages/soccer-vision/tests/test_calib_lines.py`

- [ ] **Step 1: Write the failing tests**

Append to `packages/soccer-vision/tests/test_calib_lines.py`. Extend the calibrate import to add `CalibError` and `refine_pose`:

```python
from soccer_vision.calib.calibrate import CalibError, refine_pose  # noqa: E402  (add to existing import)
```

(If preferred, merge these names into the single `from soccer_vision.calib.calibrate import (...)` block at the top of the file rather than a second import — the engineer should keep one import block.)

Then add:

```python
def _project_ids(
    rvec: np.ndarray, tvec: np.ndarray, ids: list[int], k: np.ndarray = _K
) -> dict[int, tuple[float, float]]:
    fp = field_points_3d()
    px = cv2.projectPoints(fp[ids], rvec, tvec, k, np.zeros(5))[0].reshape(-1, 2)
    return {i: (float(px[n, 0]), float(px[n, 1])) for n, i in enumerate(ids)}


def test_refine_pose_too_few_constraints_raises() -> None:
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    # 2 points (4 residuals) + 1 line (1 residual) = 5 < 6
    pts = _project_ids(rvec, tvec, [1, 3])
    point_obs = [(i, x, y) for i, (x, y) in pts.items()]
    line_obs = [("near_touchline", 500.0, 500.0)]
    with pytest.raises(CalibError):
        refine_pose(_K, rvec, tvec, point_obs, line_obs)


def _inframe(px: np.ndarray, w: int = 1920, h: int = 1080) -> bool:
    return bool(0 < px[0] < w and 0 < px[1] < h)


def test_refine_pose_roundtrip_recovers_true_pose() -> None:
    # Noiseless full obs from a known pose; seed a perturbed pose; refine must return
    # to the truth (point reprojection residual ~0).
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    ids = [1, 3, 4, 6, 7, 8, 10, 12, 14, 16]
    pts = _project_ids(rvec, tvec, ids)
    point_obs = [(i, x, y) for i, (x, y) in pts.items()]
    # one midline line click: a pixel on the midline (y=L/2), in frame
    on_line = cv2.projectPoints(
        np.array([[22.0, 34.25, 0.0]]), rvec, tvec, _K, np.zeros(5))[0].reshape(2)
    line_obs = [("midline", float(on_line[0]), float(on_line[1]))]
    rvec0 = rvec + np.array([[0.03], [-0.02], [0.01]])
    tvec0 = tvec + np.array([[1.5], [-1.0], [0.8]])
    rr, tt = refine_pose(_K, rvec0, tvec0, point_obs, line_obs)
    # reprojection of the clicked points under the recovered pose is ~exact
    proj = cv2.projectPoints(field_points_3d()[ids], rr, tt, _K, np.zeros(5))[0].reshape(-1, 2)
    truth = np.array([pts[i] for i in ids])
    assert np.max(np.linalg.norm(proj - truth, axis=1)) < 0.5


def test_line_constraint_tightens_underconstrained_midline() -> None:
    # THE GATE (Patrick's complaint #1: the midline's NEAR part is never labeled,
    # only the far end). A Trace-like midfield sideline view. Every clickable point
    # landmark in view sits at x >= ~22 m (centre line and the far half) -- the NEAR
    # half of the visible midline (x ~ 3..20 m) has NO point support, so points-only
    # must EXTRAPOLATE it. Under click noise that extrapolation swings; a single
    # midline line click in the near region pins it. Averaged over fixed noise draws,
    # points+line beats points-only on a held-out near-midline point. All clicks are
    # IN FRAME (a realistic labeler click), unlike the near touchline which projects
    # below the frame bottom in this geometry.
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 40.0, 0.0))
    fp = field_points_3d()
    allpx = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    # realistic: click EVERY in-frame landmark (idx 5 is hidden under the camera)
    ids = [i for i in range(21) if i != 5 and _inframe(allpx[i])]
    clean = {i: (float(allpx[i, 0]), float(allpx[i, 1])) for i in ids}
    # visible span of the midline (y = L/2); held-out + line-click both in its NEAR,
    # point-free region, at DIFFERENT points (so the line must generalise, not memo)
    y_mid = 34.25
    xs = np.linspace(0.0, 45.7, 80)
    mlpx = cv2.projectPoints(
        np.column_stack([xs, np.full(80, y_mid), np.zeros(80)]), rvec, tvec, _K, np.zeros(5)
    )[0].reshape(-1, 2)
    vis_x = sorted(float(x) for x, px in zip(xs, mlpx, strict=True) if _inframe(px))
    x_held, x_click = vis_x[0] + 3.0, vis_x[0] + 8.0
    held_world = np.array([[x_held, y_mid, 0.0]])
    held_px = cv2.projectPoints(held_world, rvec, tvec, _K, np.zeros(5))[0].reshape(2)
    line_px = cv2.projectPoints(
        np.array([[x_click, y_mid, 0.0]]), rvec, tvec, _K, np.zeros(5))[0].reshape(2)
    assert _inframe(held_px) and _inframe(line_px)  # realistic in-frame clicks

    rng = np.random.default_rng(2)
    err_points_only: list[float] = []
    err_points_line: list[float] = []
    for _ in range(10):
        noisy = {i: (x + rng.normal(0, 2.0), y + rng.normal(0, 2.0)) for i, (x, y) in clean.items()}
        point_obs = [(i, x, y) for i, (x, y) in noisy.items()]
        line_obs = [("midline", float(line_px[0]), float(line_px[1]))]
        ra, ta = refine_pose(_K, rvec, tvec, point_obs, [])
        rb, tb = refine_pose(_K, rvec, tvec, point_obs, line_obs)
        pa = cv2.projectPoints(held_world, ra, ta, _K, np.zeros(5))[0].reshape(2)
        pb = cv2.projectPoints(held_world, rb, tb, _K, np.zeros(5))[0].reshape(2)
        err_points_only.append(float(np.linalg.norm(pa - held_px)))
        err_points_line.append(float(np.linalg.norm(pb - held_px)))

    mean_only = float(np.mean(err_points_only))
    mean_line = float(np.mean(err_points_line))
    # the line constraint must MATERIALLY tighten the under-constrained near midline.
    # Prototyped value with this seed: ratio ~0.77 (≈23% lower held-out error).
    assert mean_line < mean_only, f"line did not help: only={mean_only:.1f} line={mean_line:.1f}"
    assert mean_line < 0.85 * mean_only, f"line gain too small: only={mean_only:.1f} line={mean_line:.1f}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_lines.py -k refine_pose -v`
Expected: FAIL with `ImportError: cannot import name 'refine_pose'`.

- [ ] **Step 3: Implement `refine_pose`**

In `packages/soccer-vision/src/soccer_vision/calib/calibrate.py`, add the scipy import at the top with the other imports:

```python
from scipy.optimize import least_squares
```

and `field_line_3d` to the existing field_model import line:

```python
from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M, field_line_3d, field_points_3d
```

Then add at the end of the file:

```python
def refine_pose(
    k: NDArray[np.floating],
    rvec0: NDArray[np.floating],
    tvec0: NDArray[np.floating],
    point_obs: list[tuple[int, float, float]],
    line_obs: list[tuple[str, float, float]],
    *,
    min_constraints: int = 6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Refine a camera pose (rvec, tvec) against point + line observations.

    Minimizes, via scipy least-squares over the 6 pose DOF (focal `k` held fixed):
    per point, the 2 reprojection-error components; per line click, the 1
    point-to-projected-line distance. Seeded at (rvec0, tvec0) — e.g. a Phase-1
    `calibrate_camera` pose. Raises CalibError if the constraint count
    (`2*len(point_obs) + len(line_obs)`) is below `min_constraints` (a 6-DOF pose
    needs >= 6 residuals) or the solve does not converge to a finite pose.
    """
    n_constraints = 2 * len(point_obs) + len(line_obs)
    if n_constraints < min_constraints:
        raise CalibError(
            f"need >= {min_constraints} constraints to refine a 6-DOF pose; "
            f"got {n_constraints} ({len(point_obs)} points + {len(line_obs)} lines)")

    k_arr = np.asarray(k, dtype=np.float64)
    dist0 = np.zeros(5, dtype=np.float64)
    fp = field_points_3d()
    point_ids = [int(kp) for kp, _, _ in point_obs]
    obj = fp[point_ids].astype(np.float64) if point_ids else np.zeros((0, 3))
    img = np.array([[x, y] for _, x, y in point_obs], dtype=np.float64).reshape(-1, 2)
    lines = [(field_line_3d(lid), (float(x), float(y))) for lid, x, y in line_obs]

    def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:].reshape(3, 1)
        parts: list[NDArray[np.float64]] = []
        if len(point_ids):
            proj = cv2.projectPoints(obj, rvec, tvec, k_arr, dist0)[0].reshape(-1, 2)
            parts.append((proj - img).ravel())
        if lines:
            h = homography_from_pose(k_arr, rvec, tvec)
            parts.append(np.array([line_residual(h, p1, p2, px) for (p1, p2), px in lines]))
        return np.concatenate(parts) if parts else np.zeros(0)

    x0 = np.concatenate(
        [np.asarray(rvec0, dtype=np.float64).ravel(), np.asarray(tvec0, dtype=np.float64).ravel()])
    sol = least_squares(residuals, x0, method="trf")
    if sol.status < 1 or not np.all(np.isfinite(sol.x)):
        raise CalibError(
            f"pose refinement did not converge (status {sol.status}); seed pose kept by caller")
    return sol.x[:3].reshape(3, 1).copy(), sol.x[3:].reshape(3, 1).copy()
```

- [ ] **Step 4: Run the Phase 2 tests to verify they pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_lines.py -v`
Expected: PASS (all 12: 4 registry + 4 residual + 4 refine/gate).

If the gate test (`test_line_constraint_tightens_underconstrained_midline`) does NOT show the gap, do NOT weaken the assertion to force a pass. This construction was prototyped to give ratio ~0.77 with `default_rng(2)`; if it regresses, the likely cause is a construction bug (e.g., `refine_pose` not actually consuming `line_obs`, or the residual sign/projection wrong). Re-check that `refine_pose` includes the line residual and that `line_px`/`held_px` are the near-midline points. Report findings rather than masking them.

- [ ] **Step 5: Run the FULL calib test suite (no regressions)**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_field_model.py tests/test_calib_calibrate.py tests/test_calib_validate.py tests/test_calib_lines.py -v`
Expected: all PASS (Phase 1 tests unchanged, Phase 2 added).

- [ ] **Step 6: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/calib/calibrate.py tests/test_calib_lines.py`
Expected: no errors.
Run (from repo root): `uv run mypy`
Expected: no new errors in `calibrate.py` / `test_calib_lines.py`.

- [ ] **Step 7: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/calib/calibrate.py packages/soccer-vision/tests/test_calib_lines.py
git commit -m "feat(calib): refine_pose — line-constrained pose refinement (point+line residuals)"
```

---

## Done criteria

- `FIELD_LINES` / `field_line_3d` expose the 5 named pitch lines as 3D metre endpoints.
- `line_residual` returns the perpendicular pixel distance to a projected field line, degenerate-safe (0.0).
- `refine_pose` recovers a true pose from point+line obs, raises `CalibError` below 6 constraints, and a synthetic gate proves a line constraint materially tightens an under-constrained near-midline view (mean held-out error with the line < 0.85× without; prototyped ~0.77×).
- Full calib suite green; `ruff` and root `mypy` clean for the touched files.
- No labeler UI, no real-data run (Phase 3).
