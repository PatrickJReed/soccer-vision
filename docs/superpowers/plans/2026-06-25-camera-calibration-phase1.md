# Camera-Calibration Field Registration — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A drift-free, fold-free camera-calibration core — shared focal + per-frame pose against the known 9v9 field (OpenCV `calibrateCamera`) — plus a validation that proves it on existing clip+training clicks.

**Architecture:** New `soccer_vision/calib/` subpackage: `field_model.py` (3D field points in metres), `calibrate.py` (pose→homography helpers + `calibrate_camera` returning a `CalibResult`), `validate.py` (fold/accuracy metrics). A notebook runs the validation on the real clicks. No labeler changes (Phase 3), no lines (Phase 2).

**Tech Stack:** Python 3.11, numpy, OpenCV (`cv2.calibrateCamera`/`solvePnP`/`projectPoints`/`Rodrigues`). pytest, mypy strict (bare `uv run mypy` from REPO ROOT), ruff.

---

## CRITICAL conventions for every task

- **GIT (ABSOLUTE):** commit only on the current branch (`feat/camera-calib-phase1`) via `git add <paths> && git commit`. NEVER checkout/switch/reset/stash/rebase. Read-only git is fine.
- **mypy:** bare `uv run mypy` from the REPO ROOT only (inside `packages/soccer-vision` gives a bogus "Duplicate __main__"). Zero new errors; annotate test helpers and `tmp_path: Path`.
- **ruff:** imports at top (sorted), no `;`-joined statements; lint changed src AND tests.
- Existing APIs (read the files):
  - `soccer_vision.pitch.landmarks.PITCH_LANDMARKS` — `(21, 2)` float64; col 0 = fraction of WIDTH, col 1 = fraction of LENGTH; both in [0,1]. `NEAR_HALFWAY_IDX == 5` (the hidden landmark).
  - `cv2.calibrateCamera(objectPoints, imagePoints, imageSize, K, dist, flags=...) -> (rms, K, dist, rvecs, tvecs)`.
  - `cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP) -> (ok, rvec, tvec)`.

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/soccer_vision/calib/__init__.py` | subpackage marker | Create |
| `src/soccer_vision/calib/field_model.py` | 9v9 field points in metres + dims | Create |
| `src/soccer_vision/calib/calibrate.py` | pose→homography helpers, `calibrate_camera`, `CalibResult` | Create |
| `src/soccer_vision/calib/validate.py` | fold count, feet error, leave-one-out accuracy | Create |
| `tests/test_calib_field_model.py` | field-model tests | Create |
| `tests/test_calib_calibrate.py` | synthetic-camera round-trip + outlier | Create |
| `tests/test_calib_validate.py` | metric tests | Create |
| `examples/calib_validate.ipynb` | run the validation on real clicks | Create |

---

## Task 1: 9v9 field model (metres)

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/calib/__init__.py`
- Create: `packages/soccer-vision/src/soccer_vision/calib/field_model.py`
- Test: `packages/soccer-vision/tests/test_calib_field_model.py`

- [ ] **Step 1: Write the failing test**

Create `packages/soccer-vision/tests/test_calib_field_model.py`:

```python
"""Tests for the 9v9 field model in metres."""

from __future__ import annotations

import numpy as np
from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M, field_points_3d


def test_field_points_shape_and_plane() -> None:
    pts = field_points_3d()
    assert pts.shape == (21, 3)
    assert np.allclose(pts[:, 2], 0.0)  # planar, Z=0


def test_field_points_corners_in_metres() -> None:
    pts = field_points_3d()
    assert np.allclose(pts[0], [0.0, 0.0, 0.0])                 # corner_own_left  (0,0)
    assert np.allclose(pts[3], [WIDTH_M, LENGTH_M, 0.0])        # corner_opp_right (1,1)
    assert np.allclose(pts[1], [WIDTH_M, 0.0, 0.0])             # corner_own_right (1,0)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_field_model.py -q`
Expected: FAIL — `ModuleNotFoundError: soccer_vision.calib`

- [ ] **Step 3: Implement**

Create `packages/soccer-vision/src/soccer_vision/calib/__init__.py` (empty file).

Create `packages/soccer-vision/src/soccer_vision/calib/field_model.py`:

```python
"""The 9v9 pitch as a known rigid 3D model (planar, Z=0) in metres.

Camera calibration solves the physical camera against THIS fixed structure, so a
real camera pose can't fold the far field into view the way a free homography can.
Dimensions are the nominal US Soccer 9v9 mid-range from PitchSpec's docstring.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

LENGTH_M: float = 68.5  # goal-to-goal (the canonical y axis)
WIDTH_M: float = 45.7   # touchline-to-touchline (the canonical x axis)
METRES_TO_FEET: float = 3.28084


def field_points_3d() -> NDArray[np.float64]:
    """The 21 canonical landmarks as real-world metres on the Z=0 plane.

    PITCH_LANDMARKS col 0 is a fraction of WIDTH, col 1 a fraction of LENGTH.
    """
    pts = np.zeros((len(PITCH_LANDMARKS), 3), dtype=np.float64)
    pts[:, 0] = PITCH_LANDMARKS[:, 0] * WIDTH_M
    pts[:, 1] = PITCH_LANDMARKS[:, 1] * LENGTH_M
    return pts
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_field_model.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Gate + commit**

REPO ROOT `uv run mypy 2>&1 | tail -1` → Success; ruff both files → clean.

```bash
git add packages/soccer-vision/src/soccer_vision/calib/__init__.py packages/soccer-vision/src/soccer_vision/calib/field_model.py packages/soccer-vision/tests/test_calib_field_model.py
git commit -m "feat(calib): 9v9 field model in metres (planar Z=0)"
```

---

## Task 2: pose → homography helpers

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/calib/calibrate.py`
- Test: `packages/soccer-vision/tests/test_calib_calibrate.py`

- [ ] **Step 1: Write the failing test**

Create `packages/soccer-vision/tests/test_calib_calibrate.py`:

```python
"""Tests for camera calibration: homography helpers + calibrate_camera."""

from __future__ import annotations

import cv2
import numpy as np
from soccer_vision.calib.calibrate import homography_from_pose, pitch_homography
from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M, field_points_3d


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV (world->camera) rvec, tvec for a camera at `eye` looking at `target`."""
    fwd = target - eye; fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up); right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    rmat = np.vstack([right, down, fwd])
    rvec, _ = cv2.Rodrigues(rmat)
    return rvec, (-rmat @ eye).reshape(3, 1)


_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]])


def test_homography_from_pose_matches_projectpoints() -> None:
    rvec, tvec = _look_at(np.array([20.0, -10, 12]), np.array([22.0, 34, 0]), np.array([0.0, 0, 1]))
    fp = field_points_3d()
    H = homography_from_pose(_K, rvec, tvec)
    proj = (H @ np.column_stack([fp[:, 0], fp[:, 1], np.ones(len(fp))]).T).T
    proj = proj[:, :2] / proj[:, 2:3]
    truth, _ = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))
    assert np.allclose(proj, truth.reshape(-1, 2), atol=1e-6)


def test_pitch_homography_scales_canonical() -> None:
    from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
    rvec, tvec = _look_at(np.array([20.0, -10, 12]), np.array([22.0, 34, 0]), np.array([0.0, 0, 1]))
    Hw = homography_from_pose(_K, rvec, tvec)
    Hp = pitch_homography(Hw)
    canon = np.column_stack([PITCH_LANDMARKS[:, 0], PITCH_LANDMARKS[:, 1], np.ones(21)])
    a = (Hp @ canon.T).T; a = a[:, :2] / a[:, 2:3]
    fp = field_points_3d()
    b = (Hw @ np.column_stack([fp[:, 0], fp[:, 1], np.ones(21)]).T).T; b = b[:, :2] / b[:, 2:3]
    assert np.allclose(a, b, atol=1e-6)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_calibrate.py -k "homography or pitch" -q`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement**

Create `packages/soccer-vision/src/soccer_vision/calib/calibrate.py`:

```python
"""Camera calibration against the known 9v9 field: shared focal + per-frame pose.

A per-frame homography H = K [r1 | r2 | t] comes from a PHYSICAL camera pose, so it
cannot fold the far field into view (the failure of the free per-frame homography);
and each frame is solved directly against the field, so there is no chained-
registration drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M, field_points_3d


def homography_from_pose(
    k: NDArray[np.floating], rvec: NDArray[np.floating], tvec: NDArray[np.floating]
) -> NDArray[np.float64]:
    """World-metres (X, Y, Z=0) -> pixel homography for a camera (K, rvec, tvec)."""
    rmat, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    cols = np.column_stack([rmat[:, 0], rmat[:, 1], np.asarray(tvec, dtype=np.float64).ravel()])
    return np.asarray(np.asarray(k, dtype=np.float64) @ cols, dtype=np.float64)


def pitch_homography(h_world: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert a world-metres->pixel homography to canonical-[0,1]^2 -> pixel."""
    return np.asarray(np.asarray(h_world, dtype=np.float64) @ np.diag([WIDTH_M, LENGTH_M, 1.0]),
                      dtype=np.float64)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_calibrate.py -k "homography or pitch" -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Gate + commit**

REPO ROOT mypy → Success; ruff → clean.

```bash
git add packages/soccer-vision/src/soccer_vision/calib/calibrate.py packages/soccer-vision/tests/test_calib_calibrate.py
git commit -m "feat(calib): pose->homography helpers (world & canonical)"
```

---

## Task 3: calibrate_camera + CalibResult

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/calib/calibrate.py`
- Test: `packages/soccer-vision/tests/test_calib_calibrate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_calib_calibrate.py` (add to the imports: `from soccer_vision.calib.calibrate import CalibError, CalibResult, calibrate_camera`):

```python
def _observations(eyes: list[tuple[float, float, float]], target=(22.85, 34.25, 0.0),
                  k=_K, w: int = 1920, h: int = 1080) -> dict[int, list[tuple[int, float, float]]]:
    fp = field_points_3d()
    obs: dict[int, list[tuple[int, float, float]]] = {}
    for i, e in enumerate(eyes):
        rvec, tvec = _look_at(np.array(e, float), np.array(target, float), np.array([0.0, 0, 1]))
        px, _ = cv2.projectPoints(fp, rvec, tvec, k, np.zeros(5))
        px = px.reshape(-1, 2)
        ids = [j for j in range(21) if j != 5 and 0 < px[j, 0] < w and 0 < px[j, 1] < h]
        if len(ids) >= 6:
            obs[i] = [(j, float(px[j, 0]), float(px[j, 1])) for j in ids]
    return obs


# 8 elevated, varied viewpoints -> the whole field projects in-frame; well-conditioned focal
_EYES = [(8.0, 4, 70), (33, 14, 80), (18, 59, 75), (40, 44, 85),
         (23, -1, 90), (3, 34, 78), (31, 64, 82), (38, 19, 88)]


def test_calibrate_recovers_known_focal() -> None:
    obs = _observations(_EYES)
    res = calibrate_camera(obs, (1920, 1080), min_points=6)
    assert isinstance(res, CalibResult)
    assert abs(res.K[0, 0] - 1400.0) < 30.0          # focal recovered (~2%)
    assert max(res.rms_px.values()) < 1.0            # near-perfect reprojection (noiseless)
    # the recovered homography reprojects the field to the same pixels as the pose
    fp = field_points_3d(); f0 = res.frames[0]
    Hpix = (res.homography(f0) @ np.column_stack([fp[:, 0], fp[:, 1], np.ones(21)]).T).T
    Hpix = Hpix[:, :2] / Hpix[:, 2:3]
    rv, tv = res.poses[f0]
    truth, _ = cv2.projectPoints(fp, rv, tv, res.K, np.zeros(5))
    assert np.allclose(Hpix, truth.reshape(-1, 2), atol=1.0)


def test_calibrate_rejects_outlier_view() -> None:
    obs = _observations(_EYES)
    # add a garbage view (random pixels) that should be rejected
    obs[99] = [(j, 5.0 * j, 7.0 * j) for j in range(0, 12)]
    res = calibrate_camera(obs, (1920, 1080), min_points=6, rms_reject_px=20.0)
    assert res.n_excluded >= 1
    assert 99 not in res.frames
    assert abs(res.K[0, 0] - 1400.0) < 40.0


def test_calibrate_too_few_views_raises() -> None:
    obs = _observations(_EYES[:2])
    try:
        calibrate_camera(obs, (1920, 1080), min_points=6)
    except CalibError as e:
        assert "view" in str(e).lower()
    else:
        raise AssertionError("expected CalibError")
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_calibrate.py -k "calibrate" -q`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement**

Append to `calibrate.py`:

```python
class CalibError(Exception):
    """Calibration could not be solved (too few/degenerate views, implausible focal)."""


@dataclass
class CalibResult:
    K: NDArray[np.float64]                                  # shared 3x3 intrinsics
    poses: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]  # frame -> (rvec, tvec)
    rms_px: dict[int, float]                                # frame -> reprojection RMS (px)
    frames: list[int]
    n_excluded: int

    def homography(self, frame: int) -> NDArray[np.float64]:
        rvec, tvec = self.poses[frame]
        return homography_from_pose(self.K, rvec, tvec)

    def pitch_homography(self, frame: int) -> NDArray[np.float64]:
        return pitch_homography(self.homography(frame))


def _build_views(
    observations: dict[int, list[tuple[int, float, float]]], min_points: int
) -> tuple[list[int], list[NDArray[np.float32]], list[NDArray[np.float32]]]:
    fp = field_points_3d()
    frames: list[int] = []
    objp: list[NDArray[np.float32]] = []
    imgp: list[NDArray[np.float32]] = []
    for f in sorted(observations):
        seen: dict[int, tuple[float, float]] = {}
        for kp, x, y in observations[f]:
            seen[int(kp)] = (float(x), float(y))  # last wins on duplicates
        ids = sorted(seen)
        if len(ids) < min_points:
            continue
        frames.append(f)
        objp.append(fp[ids].astype(np.float32))
        imgp.append(np.array([seen[i] for i in ids], dtype=np.float32))
    return frames, objp, imgp


def _per_view_rms(
    objp: list[NDArray[np.float32]], imgp: list[NDArray[np.float32]],
    k: NDArray[np.float64], dist: NDArray[np.float64],
    rvecs: list[NDArray[np.float64]], tvecs: list[NDArray[np.float64]],
) -> list[float]:
    out: list[float] = []
    for o, im, rv, tv in zip(objp, imgp, rvecs, tvecs, strict=True):
        proj, _ = cv2.projectPoints(o, rv, tv, k, dist)
        d = proj.reshape(-1, 2) - im
        out.append(float(np.sqrt(np.mean(np.sum(d * d, axis=1)))))
    return out


_FLAGS = (
    cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
    | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
    | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
)


def calibrate_camera(
    observations: dict[int, list[tuple[int, float, float]]],
    frame_size: tuple[int, int],
    *,
    min_points: int = 6,
    focal_init: float | None = None,
    rms_reject_px: float = 50.0,
) -> CalibResult:
    """Shared-focal + per-frame-pose calibration against the 9v9 field.

    observations: {frame: [(kp_idx, x_px, y_px), ...]}. Estimates ONE focal across
    all frames (principal point fixed at centre, no distortion) + a per-frame pose;
    one outlier-view rejection pass on reprojection RMS.
    """
    w, h = frame_size
    frames, objp, imgp = _build_views(observations, min_points)
    if len(frames) < 3:
        raise CalibError(
            f"need >= 3 calibratable views (>= {min_points} landmarks each); got {len(frames)}")

    f0 = float(focal_init if focal_init is not None else w)

    def _solve(op: list[NDArray[np.float32]], ip: list[NDArray[np.float32]]) -> tuple[
        NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        k0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]], dtype=np.float64)
        d0 = np.zeros(5, dtype=np.float64)
        _, k, dist, rvecs, tvecs = cv2.calibrateCamera(op, ip, (w, h), k0, d0, flags=_FLAGS)
        return k, dist, list(rvecs), list(tvecs)

    k, dist, rvecs, tvecs = _solve(objp, imgp)
    rms = _per_view_rms(objp, imgp, k, dist, rvecs, tvecs)

    keep = [i for i, r in enumerate(rms) if r <= rms_reject_px]
    n_excluded = len(frames) - len(keep)
    if 0 < n_excluded <= len(frames) - 3:
        frames = [frames[i] for i in keep]
        objp = [objp[i] for i in keep]
        imgp = [imgp[i] for i in keep]
        k, dist, rvecs, tvecs = _solve(objp, imgp)
        rms = _per_view_rms(objp, imgp, k, dist, rvecs, tvecs)

    focal = float(k[0, 0])
    if not 0.1 * w < focal < 50 * w:
        raise CalibError(
            f"implausible focal {focal:.0f}px (frame width {w}); too few views or pose diversity")

    return CalibResult(
        K=np.asarray(k, dtype=np.float64),
        poses={f: (np.asarray(rvecs[i], np.float64), np.asarray(tvecs[i], np.float64))
               for i, f in enumerate(frames)},
        rms_px={f: rms[i] for i, f in enumerate(frames)},
        frames=frames,
        n_excluded=n_excluded,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_calibrate.py -q`
Expected: PASS (5 tests). If `test_calibrate_recovers_known_focal` fails on the focal tolerance, the synthetic views may be under-conditioned — widen the spread of `_EYES` (more varied positions/heights) until cv2 recovers the focal; do NOT loosen the assertion beyond ~5%.

- [ ] **Step 5: Gate + commit**

REPO ROOT mypy → Success; ruff → clean; full suite `uv run pytest -q | tail -1`.

```bash
git add packages/soccer-vision/src/soccer_vision/calib/calibrate.py packages/soccer-vision/tests/test_calib_calibrate.py
git commit -m "feat(calib): calibrate_camera (shared focal + per-frame pose) with outlier rejection"
```

---

## Task 4: validation metrics

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/calib/validate.py`
- Test: `packages/soccer-vision/tests/test_calib_validate.py`

- [ ] **Step 1: Write the failing test**

Create `packages/soccer-vision/tests/test_calib_validate.py`:

```python
"""Tests for the calibration validation metrics."""

from __future__ import annotations

import cv2
import numpy as np
from soccer_vision.calib.calibrate import homography_from_pose, pitch_homography
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.calib.validate import fold_count, leave_one_out_feet, reproj_error_feet


def _look_at(eye, target, up=(0.0, 0, 1)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rmat = np.vstack([right, down, fwd]); rvec, _ = cv2.Rodrigues(rmat)
    return rvec, (-rmat @ eye).reshape(3, 1)


_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]])


def test_fold_count_full_field_view() -> None:
    # high overhead view -> most of the 21 landmarks in-frame
    rvec, tvec = _look_at((23.0, 4, 80), (22.85, 34.25, 0))
    Hp = pitch_homography(homography_from_pose(_K, rvec, tvec))
    assert fold_count(Hp, (1920, 1080)) >= 15


def test_reproj_error_feet_zero_for_exact() -> None:
    rvec, tvec = _look_at((20.0, -8, 12), (22.85, 34.25, 0))
    Hw = homography_from_pose(_K, rvec, tvec)
    fp = field_points_3d()
    px, _ = cv2.projectPoints(fp[3:4], rvec, tvec, _K, np.zeros(5))
    err = reproj_error_feet(Hw, fp[3], tuple(px.reshape(2)))
    assert err < 0.1  # exact projection -> ~0 ft


def test_leave_one_out_feet_perfect_projection() -> None:
    fp = field_points_3d()
    rvec, tvec = _look_at((20.0, -8, 12), (22.85, 34.25, 0))
    px, _ = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5)); px = px.reshape(-1, 2)
    ids = [0, 1, 3, 9, 13, 14, 16, 4]
    obs = {0: [(i, float(px[i, 0]), float(px[i, 1])) for i in ids]}
    errs = leave_one_out_feet(obs, _K, (1920, 1080), min_other=4)
    allv = [v for vals in errs.values() for v in vals]
    assert allv and max(allv) < 1.0  # perfect data -> sub-foot held-out error
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_validate.py -q`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement**

Create `packages/soccer-vision/src/soccer_vision/calib/validate.py`:

```python
"""Validation metrics for the camera-calibration core: fold-free and accuracy.

These are pure (given a homography / camera), so the gate logic is unit-tested —
the lesson of the anchor_cov coverage gate that shipped untested.
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.calibrate import homography_from_pose
from soccer_vision.calib.field_model import METRES_TO_FEET, field_points_3d
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS


def fold_count(h_pitch: NDArray[np.floating], frame_size: tuple[int, int]) -> int:
    """How many of the 21 canonical landmarks project IN-FRAME through h_pitch.

    A folding homography pulls the far field in -> count near 21 on a shallow view;
    a physical camera keeps only the visible slice -> count ~6-12.
    """
    w, h = frame_size
    pts = np.column_stack([PITCH_LANDMARKS[:, 0], PITCH_LANDMARKS[:, 1], np.ones(len(PITCH_LANDMARKS))])
    pr = (np.asarray(h_pitch, dtype=np.float64) @ pts.T).T
    wz = pr[:, 2]
    denom = np.where(np.abs(pr[:, 2:3]) < 1e-9, 1e-9, pr[:, 2:3])
    uv = pr[:, :2] / denom
    return int(np.sum((wz > 0) & (uv[:, 0] > 0) & (uv[:, 0] < w) & (uv[:, 1] > 0) & (uv[:, 1] < h)))


def reproj_error_feet(
    h_world: NDArray[np.floating], field_pt_3d: NDArray[np.floating],
    clicked_px: tuple[float, float],
) -> float:
    """Map a clicked pixel back to field metres (via inv H) and compare to the true
    field position; error in feet."""
    h_inv = np.linalg.inv(np.asarray(h_world, dtype=np.float64))
    v = h_inv @ np.array([clicked_px[0], clicked_px[1], 1.0])
    fld = v[:2] / v[2]
    return float(np.hypot(fld[0] - field_pt_3d[0], fld[1] - field_pt_3d[1]) * METRES_TO_FEET)


def leave_one_out_feet(
    observations: dict[int, list[tuple[int, float, float]]],
    k: NDArray[np.floating],
    frame_size: tuple[int, int],
    *,
    min_other: int = 4,
) -> dict[int, list[float]]:
    """Held-out accuracy: per frame, fit the pose from all-but-one landmark and
    measure the held-out landmark's reprojection error in feet. Returns
    {kp_idx: [feet errors across frames]}."""
    fp = field_points_3d()
    out: dict[int, list[float]] = defaultdict(list)
    for obs in observations.values():
        seen: dict[int, tuple[float, float]] = {int(kp): (float(x), float(y)) for kp, x, y in obs}
        ids = sorted(seen)
        if len(ids) < min_other + 1:
            continue
        for held in ids:
            others = [i for i in ids if i != held]
            op = fp[others].astype(np.float64)
            ip = np.array([seen[i] for i in others], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(op, ip, np.asarray(k, np.float64), None,
                                          flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                continue
            h_world = homography_from_pose(k, rvec, tvec)
            out[held].append(reproj_error_feet(h_world, fp[held], seen[held]))
    return dict(out)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_calib_validate.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Gate + commit**

REPO ROOT mypy → Success; ruff → clean.

```bash
git add packages/soccer-vision/src/soccer_vision/calib/validate.py packages/soccer-vision/tests/test_calib_validate.py
git commit -m "feat(calib): validation metrics (fold count, held-out feet accuracy)"
```

---

## Task 5: validation notebook + real-data acceptance (controller)

**Files:**
- Create: `examples/calib_validate.ipynb`

- [ ] **Step 1: Build the notebook**

Create the notebook via a throwaway python script (DELETE after; do not commit it). `json.dump(nb, indent=1)`, `nbformat=4`, `nbformat_minor=0`. Cells:

Cell 0 (markdown):
```
# Camera-calibration validation (Phase 1)
Runs the drift-free calibration on existing labeler clicks (clip + training) and
checks: FOLD-FREE (shallow frames don't project the whole field in), DRIFT-FREE
(held-out error flat across the full game), ACCURACY (held-out feet vs the free
homography). Local; no GPU.
```

Cell 1 (code):
```python
import json, numpy as np, cv2
from pathlib import Path
from collections import defaultdict
from soccer_vision.calib.calibrate import calibrate_camera
from soccer_vision.calib.validate import fold_count, leave_one_out_feet
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

CACHE = Path.home()/"sv-labeler/.sv_labeler_cache"
SESSIONS = {"clip": "clip.clicks.json", "training": "training.clicks.json"}
FRAME = (1920, 1080)

def load_obs(name):
    raw = json.loads((CACHE/SESSIONS[name]).read_text())
    clicks = raw if isinstance(raw, list) else raw["clicks"]
    obs = defaultdict(list)
    for c in clicks:
        obs[c["frame"]].append((c["kp_idx"], c["x"]*FRAME[0], c["y"]*FRAME[1]))
    return dict(obs)
```

Cell 2 (code) — calibrate + report per session:
```python
for name in SESSIONS:
    obs = load_obs(name)
    res = calibrate_camera(obs, FRAME, min_points=6)
    print(f"=== {name}: {len(res.frames)} calibrated frames, {res.n_excluded} excluded ===")
    print(f"  shared focal: {res.K[0,0]:.0f}px  (frame width {FRAME[0]})  reproj RMS median {np.median(list(res.rms_px.values())):.1f}px")
    # FOLD-FREE: landmarks in-frame per calibrated frame (folding=~21, slice=~6-12)
    folds = [fold_count(res.pitch_homography(f), FRAME) for f in res.frames]
    print(f"  fold check: in-frame landmarks/frame min {min(folds)} median {int(np.median(folds))} max {max(folds)}  (folding would be ~21)")
    # ACCURACY: held-out feet
    errs = leave_one_out_feet({f: obs[f] for f in res.frames}, res.K, FRAME, min_other=4)
    allv = [v for vals in errs.values() for v in vals]
    print(f"  held-out accuracy: median {np.median(allv):.1f}ft  90th {np.percentile(allv,90):.1f}ft")
    # DRIFT-FREE: per-frame fit RMS early-vs-late. The camera calib solves each
    # frame independently against the field (no chaining), so RMS should be FLAT
    # across the whole game (contrast the chained homography's 4->95 ft growth).
    fr = np.array(sorted(res.frames)); rms_arr = np.array([res.rms_px[f] for f in fr])
    third = max(1, len(fr) // 3)
    print(f"  drift check (per-frame fit RMS): early-third {np.median(rms_arr[:third]):.1f}px "
          f"vs late-third {np.median(rms_arr[-third:]):.1f}px  (flat = drift-free)")
```

(The drift cell's per-frame aggregation is approximate; the controller refines it when running — the key output is the early-vs-late comparison.)

Cell 3 (markdown):
```
## Overlays (Patrick assesses)
The next cell renders the calibrated pitch on sampled frames for visual review.
Claude renders; Patrick interprets.
```

Cell 4 (code) — render overlay sheet (requires the session video at ~/sv-labeler/<name>.mp4):
```python
EDGES=[(0,1),(1,3),(3,2),(2,0),(4,6),(9,10),(11,12),(9,11),(10,12),(13,14),(15,16),(13,15),(14,16),(17,18),(19,20)]
name="training"; obs=load_obs(name); res=calibrate_camera(obs,FRAME,min_points=6)
cap=cv2.VideoCapture(str(Path.home()/f"sv-labeler/{name}.mp4")); cells=[]
import random; random.seed(0)
for f in sorted(random.sample(res.frames, min(8,len(res.frames)))):
    cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,img=cap.read()
    if not ok: continue
    Hp=res.pitch_homography(f); canon=np.column_stack([PITCH_LANDMARKS[:,0],PITCH_LANDMARKS[:,1],np.ones(21)])
    pr=(Hp@canon.T).T; uv=pr[:,:2]/pr[:,2:3]
    for a,b in EDGES:
        cv2.line(img,(int(uv[a,0]),int(uv[a,1])),(int(uv[b,0]),int(uv[b,1])),(40,200,255),3)
    cells.append(img)
cap.release()
cw=640; tiles=[cv2.resize(c,(cw,int(c.shape[0]*cw/c.shape[1]))) for c in cells]
rows=[np.hstack(tiles[i:i+4]+[np.zeros_like(tiles[0])]*(4-len(tiles[i:i+4]))) for i in range(0,len(tiles),4)]
cv2.imwrite("/tmp/calib_overlay.jpg",np.vstack(rows)); print("wrote /tmp/calib_overlay.jpg")
```

- [ ] **Step 2: Validate + commit**

Run: `python3 -c "import json; nb=json.load(open('examples/calib_validate.ipynb')); assert nb['nbformat']==4; assert len(nb['cells'])==5; s=''.join(x for c in nb['cells'] for x in c['source']); assert 'calibrate_camera' in s and 'fold_count' in s; print('OK')"`
Expected: `OK`. Confirm only the notebook is staged (throwaway script deleted).

```bash
git add examples/calib_validate.ipynb
git commit -m "docs(examples): calib_validate notebook — fold/drift/accuracy on real clicks"
```

- [ ] **Step 3 (controller, NOT a subagent): run the validation**

The controller runs `calibrate_camera` + the metrics on the real `clip` and `training` clicks (loaded from the autosave sidecars), reports the **go/no-go**:
- **Fold-free:** in-frame landmarks/frame ~6-12 (NOT ~21) on shallow frames.
- **Drift-free:** held-out error early-third ≈ late-third on the full `training` game (flat), vs the chained homography's 4→95 ft growth.
- **Accuracy:** held-out median feet competitive with / better than the per-frame free homography; no catastrophic frames.
- Render the overlay sheet for Patrick to assess.

This is the decisive Phase-1 result. If the shared-focal estimate is unstable on the real fixed-camera data (a known risk — a fixed camera that mostly pans is weakly observable for focal), the finding is to FIX the focal (measure it once from a well-distributed session / known FOV) rather than estimate it — fold that into the Phase 1 report, not a code change here.

---

## Final verification

- [ ] `cd packages/soccer-vision && uv run pytest -q` — all pass.
- [ ] REPO ROOT `uv run mypy 2>&1 | tail -1` — Success.
- [ ] `cd packages/soccer-vision && uv run ruff check src/soccer_vision/calib/ tests/test_calib_field_model.py tests/test_calib_calibrate.py tests/test_calib_validate.py` — clean.
