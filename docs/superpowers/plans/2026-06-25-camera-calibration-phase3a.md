# Camera Calibration Phase 3a — Registration Engine + A/B Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two calibration-based per-frame homography engines (A: propagate clicks → `refine_pose`; B: calibrate clicked frames → propagate the pose via the chain's recovered rotation) plus a head-to-head comparison harness vs the current free fit, on the full training-game session.

**Architecture:** A new pure module `pitch/calib_anchor.py` reuses the registration chain primitives (`build_segments`, `cumulative_transforms`, and a newly-extracted `propagate_clicks`) and the Phase 1/2 calib core (`calibrate_camera`, `refine_pose`, `homography_from_pose`, `pitch_homography`, `fold_count`). It works internally in full-pixel image space against 3D field metres and emits the labeler's full-pixel image→pitch export homography. No `labeler/` code changes (that's Phase 3b).

**Tech Stack:** Python, NumPy, OpenCV (`cv2.solvePnP/SQPNP`, `projectPoints`, `Rodrigues`, SVD via numpy), scipy (inside `refine_pose`), pytest, mypy (strict), ruff.

**Spec:** `docs/superpowers/specs/2026-06-25-camera-calibration-phase3a-design.md`

**Verified before writing (against real cv2/numpy):** Engine B rotation-from-chain round-trips to 2.6e-11 px; Engine A SQPNP-seed + `refine_pose` recovers a pose to 8e-13 px; `frame_homography` round-trips to 1e-16. The synthetic constructions below are taken from those prototypes.

**Conventions (from Phases 1–2 and the labeler):**
- `homography_from_pose(K, rvec, tvec)` → world-metres (X,Y,1 on Z=0) → pixel. `pitch_homography(h_world)` → canonical-[0,1]² → pixel. rvec/tvec are `(3,1)` float64; `(R,t)` is world→camera (`x_cam = R X + t`), camera centre `C = -Rᵀt`.
- `calibrate_camera(observations: dict[frame, list[(kp_idx, x_px, y_px)]], frame_size, *, min_points=6) -> CalibResult` with `.K`, `.poses: dict[frame, (rvec, tvec)]`.
- `refine_pose(K, rvec0, tvec0, point_obs: list[(kp_idx,x_px,y_px)], line_obs: list[(line_id,x_px,y_px)], *, min_constraints=6) -> (rvec, tvec)`; raises `CalibError`.
- `fold_count(h_pitch, frame_size) -> int`; `field_points_3d() -> (21,3)`.
- The labeler stores **normalized [0,1]** clicks; the registration chain (`cumulative_transforms` output `M[f]`) is **normalized**; export homographies are **full-pixel image→pitch**. `normalize_homography`/`denormalize_homography` live in `labeler/chain.py`. `Click(frame, kp_idx, x, y)` (normalized x,y) and `build_segments`/`cumulative_transforms` live in `pitch/manual_anchor.py`.
- Lint gate: **`uv run mypy` from the REPO ROOT** (`/Users/patrickreed/Sandbox/soccer-vision`); `uv run ruff check` + `uv run pytest` from `packages/soccer-vision`. The repo has pre-existing mypy errors in OTHER files — only keep YOUR touched files clean.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py` | Registration primitives + free fit | Extract `propagate_clicks`; `fit_frame_homographies` calls it |
| `packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py` | Calibration-based per-frame engines + back-end | Create |
| `packages/soccer-vision/src/soccer_vision/pitch/calib_compare.py` | `compare_engines` head-to-head metrics | Create |
| `packages/soccer-vision/tests/test_pitch_calib_anchor.py` | Engine + back-end tests | Create |
| `packages/soccer-vision/tests/test_pitch_calib_compare.py` | Comparison-harness smoke test | Create |
| `examples/calib_anchor_compare.ipynb` | The run the user does on real data | Create (thin) |

---

## Task 1: Extract `propagate_clicks` (shared, DRY)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_calib_anchor.py` (create)

- [ ] **Step 1: Write the failing test**

Create `packages/soccer-vision/tests/test_pitch_calib_anchor.py`:

```python
"""Phase 3a tests: shared propagation + calibration registration engines."""

from __future__ import annotations

import cv2
import numpy as np
from soccer_vision.pitch.manual_anchor import Click, build_segments, cumulative_transforms, propagate_clicks


def test_propagate_clicks_carries_a_click_along_an_identity_chain() -> None:
    # Identity inter-frame transforms -> a click at frame 0 propagates UNCHANGED to
    # every frame within the window, for its own landmark.
    interframe = {i: np.eye(3) for i in range(5)}  # frames 0..5 linked
    seg = build_segments(interframe, 6)
    transforms = cumulative_transforms(interframe, seg)
    clicks = [Click(frame=0, kp_idx=3, x=0.4, y=0.6)]
    prop = propagate_clicks(clicks, transforms, seg, window=10)
    assert prop[2][3] == (0.4, 0.6)  # frame 2, landmark 3
    assert prop[5][3] == (0.4, 0.6)
    # outside the window -> not present
    prop_small = propagate_clicks(clicks, transforms, seg, window=1)
    assert 3 not in prop_small.get(5, {})


def test_propagate_clicks_respects_segments() -> None:
    # A gap (missing link at 2) splits segments; a click in segment 0 does not reach
    # segment 1.
    interframe = {0: np.eye(3), 1: np.eye(3), 3: np.eye(3)}  # link missing at 2
    seg = build_segments(interframe, 5)
    transforms = cumulative_transforms(interframe, seg)
    clicks = [Click(frame=0, kp_idx=1, x=0.5, y=0.5)]
    prop = propagate_clicks(clicks, transforms, seg, window=10)
    assert 1 in prop[1]            # same segment
    assert 4 not in prop or 1 not in prop.get(4, {})  # frame 4 is a different segment
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -v`
Expected: FAIL with `ImportError: cannot import name 'propagate_clicks'`.

- [ ] **Step 3: Extract `propagate_clicks` and refit `fit_frame_homographies`**

In `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py`, add this function ABOVE `fit_frame_homographies`:

```python
def propagate_clicks(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    *,
    window: int,
    frames: Sequence[int] | None = None,
) -> dict[int, dict[int, tuple[float, float]]]:
    """For each target frame, the propagated pixel of each landmark.

    A click maps into a target frame g iff they share a segment and
    |click.frame - g| <= window; for each landmark the nearest such click wins
    (first in click order on ties). Returns {frame: {kp_idx: (x, y)}} in the same
    (normalized) image space as the clicks/transforms.
    """
    out: dict[int, dict[int, tuple[float, float]]] = {}
    if not clicks or not transforms:
        return out
    if frames is None:
        target = sorted(transforms)
    else:
        target = sorted(set(transforms) & {int(f) for f in frames})
    if not target:
        return out
    frame_arr = np.array(target, dtype=np.int64)
    n = len(frame_arr)
    m_stack = np.stack([np.asarray(transforms[int(f)], dtype=np.float64) for f in frame_arr])
    m_inv = np.linalg.inv(m_stack)
    frame_seg = np.array([segment_of[int(f)] for f in frame_arr], dtype=np.int64)

    k = len(clicks)
    click_frame = np.array([c.frame for c in clicks], dtype=np.int64)
    click_seg = np.array([segment_of.get(c.frame, -1) for c in clicks], dtype=np.int64)
    click_kp = np.array([c.kp_idx for c in clicks], dtype=np.int64)

    pos = np.full((k, n, 2), np.nan)
    for j, c in enumerate(clicks):
        src_m = transforms.get(c.frame)
        if src_m is None:
            continue
        ref = np.asarray(src_m, dtype=np.float64) @ np.array([c.x, c.y, 1.0])
        dst = m_inv @ ref
        pos[j] = dst[:, :2] / dst[:, 2:3]

    dist = np.abs(click_frame[:, None] - frame_arr[None, :])
    usable = (click_seg[:, None] == frame_seg[None, :]) & (dist <= window)
    usable &= ~np.isnan(pos[:, :, 0])

    big = np.iinfo(np.int64).max
    for kp in sorted({int(v) for v in click_kp}):
        rows = np.where(click_kp == kp)[0]
        d = np.where(usable[rows], dist[rows], big)
        choice = d.argmin(axis=0)  # first minimal row wins ties (click order)
        ok = d[choice, np.arange(n)] != big
        chosen = pos[rows[choice], np.arange(n)]
        for gi in np.where(ok)[0]:
            out.setdefault(int(frame_arr[gi]), {})[kp] = (
                float(chosen[gi, 0]), float(chosen[gi, 1]))
    return out
```

Then REPLACE the body of `fit_frame_homographies` (everything after its docstring) with a version that calls the helper:

```python
    fits: dict[int, FrameFit] = {}
    propagated = propagate_clicks(
        clicks, transforms, segment_of, window=window, frames=frames)
    lm = np.asarray(landmarks, dtype=np.float64)
    for f, kpmap in propagated.items():
        if len(kpmap) < min_points:
            continue
        idxs = sorted(kpmap)
        image_pts = np.array([kpmap[i] for i in idxs], dtype=np.float64)
        pitch_pts = lm[idxs]
        try:
            H = fit_homography(image_pts, pitch_pts)
        except HomographyError:
            continue
        errs = np.linalg.norm(_apply(H, image_pts) - pitch_pts, axis=1)
        fits[f] = FrameFit(H=H, residual=float(np.median(errs)), n_points=len(idxs))
    return fits
```

(Keep the `fit_frame_homographies` signature and docstring unchanged.)

- [ ] **Step 4: Run the new tests + the FULL manual_anchor suite (no regression)**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -v && uv run pytest -k "manual_anchor or labeler" -v`
Expected: new tests PASS; ALL existing manual_anchor/labeler tests still PASS (the refactor is behavior-preserving — same propagation, same fit).

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/pitch/manual_anchor.py tests/test_pitch_calib_anchor.py`
Then from REPO ROOT: `uv run mypy`
Expected: clean for the touched files; no new errors.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py packages/soccer-vision/tests/test_pitch_calib_anchor.py
git commit -m "refactor(pitch): extract propagate_clicks shared by free fit + calib engine"
```

---

## Task 2: `calib_anchor` front-end + back-end primitives

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_calib_anchor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pitch_calib_anchor.py` (add the imports to the existing import block at the top, plus the `_look_at` helper if not present):

```python
from soccer_vision.calib.calibrate import homography_from_pose, pitch_homography
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.pitch.calib_anchor import (
    FramePose,
    calibrate_clicked_frames,
    frame_homography,
)

_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)


def _look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    e, t, u = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    fwd = t - e
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, u)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rmat = np.vstack([right, down, fwd])
    rvec, _ = cv2.Rodrigues(rmat)
    return rvec, (-rmat @ e).reshape(3, 1)


def test_frame_homography_round_trips_pixel_to_pitch() -> None:
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    h_pitch = pitch_homography(homography_from_pose(_K, rvec, tvec))  # canon[0,1]^2 -> px
    h_img2pitch = frame_homography(_K, rvec, tvec)                    # px -> canon[0,1]^2
    canon = np.array([0.3, 0.6, 1.0])
    px = h_pitch @ canon; px = px / px[2]
    back = h_img2pitch @ np.array([px[0], px[1], 1.0]); back = back / back[2]
    assert np.hypot(back[0] - 0.3, back[1] - 0.6) < 1e-9


def test_calibrate_clicked_frames_recovers_focal() -> None:
    # synthetic clicks (NORMALIZED) on several elevated views -> shared focal ~1400
    from soccer_vision.pitch.manual_anchor import Click
    eyes = [(8.0, 4, 70), (33, 14, 80), (18, 59, 75), (40, 44, 85), (23, -1, 90), (3, 34, 78)]
    fp = field_points_3d()
    clicks: list[Click] = []
    for fidx, e in enumerate(eyes):
        rvec, tvec = _look_at(e, (22.85, 34.25, 0.0))
        px = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080:
                clicks.append(Click(fidx, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080))
    k, poses = calibrate_clicked_frames(clicks, (1920, 1080), min_points=6)
    assert abs(k[0, 0] - 1400.0) < 40.0
    assert len(poses) >= 3
    f0 = sorted(poses)[0]
    assert poses[f0][0].shape == (3, 1) and poses[f0][1].shape == (3, 1)


def test_frame_pose_dataclass_fields() -> None:
    fp_ = FramePose(rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)),
                    residual_px=1.5, n_points=7, fold_count=9)
    assert fp_.residual_px == 1.5 and fp_.n_points == 7 and fp_.fold_count == 9
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -k "frame_homography or calibrate_clicked or frame_pose" -v`
Expected: FAIL with `ModuleNotFoundError: ...calib_anchor`.

- [ ] **Step 3: Create the module**

Create `packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py`:

```python
"""Calibration-based per-frame homography engines for a fixed-camera session.

Two engines turn a session's clicks + the registration chain into a calibrated
homography for every frame: A (propagate clicks, then refine_pose per frame) and B
(calibrate clicked frames, then propagate the pose via the chain's recovered camera
rotation). Both reuse the Phase 1/2 calib core, so each frame is a real camera pose
(no fold) solved against the field directly (no homography-chaining drift).

Internally full-pixel image space; emits the labeler's full-pixel image->pitch[0,1]
homography (the export format). Pure: no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.calibrate import (
    CalibError,
    calibrate_camera,
    homography_from_pose,
    pitch_homography,
    refine_pose,
)
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.calib.validate import fold_count
from soccer_vision.pitch.manual_anchor import Click, propagate_clicks


@dataclass(frozen=True, eq=False)
class FramePose:
    """A per-frame calibrated camera pose + quality."""

    rvec: NDArray[np.float64]
    tvec: NDArray[np.float64]
    residual_px: float   # reprojection RMS over the frame's obs (nan if none)
    n_points: int        # point landmarks used (0 if pose-propagated)
    fold_count: int      # landmarks projecting in-frame (slice size, ~6-12)


def frame_homography(
    k: NDArray[np.floating], rvec: NDArray[np.floating], tvec: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Full-pixel image -> pitch-[0,1]^2 homography (the labeler export format)."""
    h_pitch = pitch_homography(homography_from_pose(k, rvec, tvec))  # canon[0,1]^2 -> px
    return np.asarray(np.linalg.inv(h_pitch), dtype=np.float64)


def calibrate_clicked_frames(
    clicks: Sequence[Click],
    size: tuple[int, int],
    *,
    min_points: int = 6,
) -> tuple[NDArray[np.float64], dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]]:
    """Shared focal + per-clicked-frame pose from the DIRECTLY-clicked frames.

    Clicks are normalized [0,1]; converted to full pixel for the calib core. Raises
    CalibError if too few/degenerate clicked views.
    """
    w, h = size
    obs: dict[int, list[tuple[int, float, float]]] = {}
    for c in clicks:
        obs.setdefault(c.frame, []).append((c.kp_idx, c.x * w, c.y * h))
    result = calibrate_camera(obs, size, min_points=min_points)
    return result.K, result.poses


def _reproj_rms_px(
    k: NDArray[np.floating],
    rvec: NDArray[np.floating],
    tvec: NDArray[np.floating],
    point_obs: Sequence[tuple[int, float, float]],
) -> float:
    """Reprojection RMS (px) of point_obs under the pose; nan if no points."""
    if not point_obs:
        return float("nan")
    obj = field_points_3d()[[int(kp) for kp, _, _ in point_obs]].astype(np.float64)
    img = np.array([[x, y] for _, x, y in point_obs], dtype=np.float64)
    proj = cv2.projectPoints(
        obj, rvec, tvec, np.asarray(k, dtype=np.float64), np.zeros(5))[0].reshape(-1, 2)
    d = proj - img
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _fold_for_pose(
    k: NDArray[np.floating],
    rvec: NDArray[np.floating],
    tvec: NDArray[np.floating],
    size: tuple[int, int],
) -> int:
    """fold_count for a pose's pitch homography."""
    return fold_count(pitch_homography(homography_from_pose(k, rvec, tvec)), size)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -v`
Expected: PASS (Task 1 tests + the 3 new ones).

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/pitch/calib_anchor.py tests/test_pitch_calib_anchor.py`
Then from REPO ROOT: `uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py packages/soccer-vision/tests/test_pitch_calib_anchor.py
git commit -m "feat(pitch): calib_anchor scaffolding (FramePose, frame_homography, calibrate_clicked_frames)"
```

---

## Task 3: Engine A — `poses_by_click_propagation`

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_calib_anchor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pitch_calib_anchor.py`:

```python
from soccer_vision.pitch.calib_anchor import poses_by_click_propagation
from soccer_vision.pitch.manual_anchor import build_segments, cumulative_transforms


def _clicks_on_frames(
    frame_eyes: dict[int, tuple[float, float, float]], size: tuple[int, int] = (1920, 1080)
):
    """Synthetic normalized clicks: project the field into each frame's camera."""
    from soccer_vision.pitch.manual_anchor import Click
    w, h = size
    fp = field_points_3d()
    clicks: list[Click] = []
    poses = {}
    for f, e in frame_eyes.items():
        rvec, tvec = _look_at(e, (22.85, 34.25, 0.0))
        poses[f] = (rvec, tvec)
        px = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            if j != 5 and 0 < px[j, 0] < w and 0 < px[j, 1] < h:
                clicks.append(Click(f, j, float(px[j, 0]) / w, float(px[j, 1]) / h))
    return clicks, poses


def test_engine_a_recovers_poses_on_clicked_frames() -> None:
    # 4 elevated, diverse views directly clicked; identity-ish chain links them all.
    eyes = {0: (8.0, 4, 70), 3: (33, 14, 80), 6: (18, 59, 75), 9: (40, 44, 85)}
    clicks, true_poses = _clicks_on_frames(eyes)
    # a registration chain that links frames 0..9 (use identity transforms: clicks at
    # a frame stay put; only same-frame clicks feed that frame -> tests the per-frame
    # calibrate+refine path, not propagation accuracy).
    interframe = {i: np.eye(3) for i in range(9)}
    seg = build_segments(interframe, 10)
    transforms = cumulative_transforms(interframe, seg)
    k, _poses = calibrate_clicked_frames(clicks, (1920, 1080), min_points=6)
    out = poses_by_click_propagation(clicks, transforms, seg, k, (1920, 1080),
                                     window=360, min_points=4)
    fp = field_points_3d()
    for f in eyes:
        assert f in out
        fpose = out[f]
        proj = cv2.projectPoints(fp, true_poses[f][0], true_poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        rec = cv2.projectPoints(fp, fpose.rvec, fpose.tvec, _K, np.zeros(5))[0].reshape(-1, 2)
        # recovered pose reprojects the field to ~the same pixels as truth
        assert np.median(np.linalg.norm(proj - rec, axis=1)) < 2.0
        assert fpose.fold_count < 21  # not a fold


def test_engine_a_line_obs_path_runs() -> None:
    # a frame given an extra line observation still solves (the refine_pose line path
    # is exercised; real line propagation is Phase 3b).
    eyes = {0: (8.0, 4, 70), 3: (33, 14, 80), 6: (18, 59, 75)}
    clicks, _ = _clicks_on_frames(eyes)
    interframe = {i: np.eye(3) for i in range(6)}
    seg = build_segments(interframe, 7)
    transforms = cumulative_transforms(interframe, seg)
    k, _poses = calibrate_clicked_frames(clicks, (1920, 1080), min_points=6)
    line_obs = {0: [("midline", 960.0, 540.0)]}
    out = poses_by_click_propagation(clicks, transforms, seg, k, (1920, 1080),
                                     window=360, min_points=4, line_obs=line_obs)
    assert 0 in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -k engine_a -v`
Expected: FAIL with `ImportError: cannot import name 'poses_by_click_propagation'`.

- [ ] **Step 3: Implement Engine A**

Append to `packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py`:

```python
def poses_by_click_propagation(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    k: NDArray[np.floating],
    size: tuple[int, int],
    *,
    window: int,
    min_points: int = 4,
    line_obs: Mapping[int, Sequence[tuple[str, float, float]]] | None = None,
) -> dict[int, FramePose]:
    """Engine A: propagate clicks into each frame, then refine_pose (focal fixed).

    Per target frame: gather window-propagated point landmarks (and any supplied
    pixel-space line_obs for that frame), seed a pose with SQPNP on the propagated
    points, and refine with refine_pose. A frame with < min_points propagated points
    (SQPNP needs them) or a non-converging solve is left uncovered. line_obs are
    pre-propagated, pixel-space lines per frame (3a real run is point-only; line
    propagation is Phase 3b). Returns {frame: FramePose}.
    """
    w, h = size
    k_arr = np.asarray(k, dtype=np.float64)
    fp = field_points_3d()
    propagated = propagate_clicks(clicks, transforms, segment_of, window=window)
    out: dict[int, FramePose] = {}
    for f, kpmap in propagated.items():
        if len(kpmap) < min_points:
            continue
        idxs = sorted(kpmap)
        # propagated positions are normalized -> convert to pixel for the calib core
        point_obs = [(i, kpmap[i][0] * w, kpmap[i][1] * h) for i in idxs]
        lobs = list(line_obs.get(f, [])) if line_obs else []
        obj = fp[idxs].astype(np.float64)
        img = np.array([[x, y] for _, x, y in point_obs], dtype=np.float64)
        ok, rvec0, tvec0 = cv2.solvePnP(obj, img, k_arr, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        try:
            rvec, tvec = refine_pose(k_arr, rvec0, tvec0, point_obs, lobs)
        except CalibError:
            continue
        out[f] = FramePose(
            rvec=rvec,
            tvec=tvec,
            residual_px=_reproj_rms_px(k_arr, rvec, tvec, point_obs),
            n_points=len(idxs),
            fold_count=_fold_for_pose(k_arr, rvec, tvec, size),
        )
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -k engine_a -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/pitch/calib_anchor.py tests/test_pitch_calib_anchor.py`
Then from REPO ROOT: `uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py packages/soccer-vision/tests/test_pitch_calib_anchor.py
git commit -m "feat(pitch): Engine A — poses_by_click_propagation (SQPNP seed + refine_pose)"
```

---

## Task 4: Engine B — `_rotation_from_chain` + `poses_by_pose_propagation`

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_calib_anchor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pitch_calib_anchor.py`:

```python
from soccer_vision.labeler.chain import normalize_homography
from soccer_vision.pitch.calib_anchor import poses_by_pose_propagation


def _pan_sequence(n: int = 7):
    """A fixed-centre camera panning across n frames; returns (poses, interframe_norm)."""
    center = np.array([-8.0, 34.0, 8.0])
    poses = {}
    for f, dy in enumerate(np.linspace(-6, 6, n)):
        rvec, tvec = _look_at(tuple(center), (22.85, 34.0 + dy, 0.0))
        poses[f] = (rvec, tvec)
    interframe = {}
    for i in range(n - 1):
        ri, _ = cv2.Rodrigues(poses[i][0])
        rj, _ = cv2.Rodrigues(poses[i + 1][0])
        g_px = _K @ rj @ np.linalg.inv(ri) @ np.linalg.inv(_K)
        interframe[i] = normalize_homography(g_px, (1920, 1080))
    return poses, interframe


def test_engine_b_recovers_panned_poses_from_one_clicked_frame() -> None:
    poses, interframe = _pan_sequence(7)
    seg = build_segments(interframe, 7)
    transforms = cumulative_transforms(interframe, seg)
    # only frame 0 is "clicked" (calibrated); propagate its pose to all others
    clicked_poses = {0: poses[0]}
    out = poses_by_pose_propagation(transforms, seg, _K, clicked_poses, (1920, 1080))
    fp = field_points_3d()
    for f in range(7):
        assert f in out
        truth = cv2.projectPoints(fp, poses[f][0], poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        rec = cv2.projectPoints(fp, out[f].rvec, out[f].tvec, _K, np.zeros(5))[0].reshape(-1, 2)
        assert np.max(np.linalg.norm(truth - rec, axis=1)) < 0.5  # prototype: ~1e-11
        assert out[f].n_points == 0


def test_engine_b_uncovered_without_clicked_neighbor() -> None:
    poses, interframe = _pan_sequence(5)
    seg = build_segments(interframe, 5)
    transforms = cumulative_transforms(interframe, seg)
    # a clicked frame in a DIFFERENT (nonexistent) segment -> no frame covered
    out = poses_by_pose_propagation(transforms, seg, _K, {}, (1920, 1080))
    assert out == {}


def test_engine_b_nonrotation_chain_does_not_crash() -> None:
    # a degenerate inter-frame transform -> SVD nearest-rotation, still returns a pose
    poses, interframe = _pan_sequence(3)
    interframe[1] = normalize_homography(
        np.array([[1.0, 0.3, 5.0], [0.0, 1.2, 2.0], [1e-4, 0.0, 1.0]]), (1920, 1080))
    seg = build_segments(interframe, 3)
    transforms = cumulative_transforms(interframe, seg)
    out = poses_by_pose_propagation(transforms, seg, _K, {0: poses[0]}, (1920, 1080))
    assert all(np.all(np.isfinite(p.rvec)) and np.all(np.isfinite(p.tvec)) for p in out.values())
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -k engine_b -v`
Expected: FAIL with `ImportError: cannot import name 'poses_by_pose_propagation'`.

- [ ] **Step 3: Implement Engine B**

Append to `packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py`:

```python
def _rotation_from_chain(
    g_px: NDArray[np.floating], k: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Relative camera rotation from an inter-frame PIXEL homography G = K R_rel K^-1.

    M = K^-1 G K equals R_rel up to scale; the nearest rotation (SVD U V^T, with the
    sign fix for a proper rotation) is returned, so a noisy / non-rotation transform
    degrades gracefully to the closest valid rotation.
    """
    m = np.linalg.inv(np.asarray(k, dtype=np.float64)) @ np.asarray(g_px, dtype=np.float64) \
        @ np.asarray(k, dtype=np.float64)
    u, _s, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u = u.copy()
        u[:, -1] *= -1
        r = u @ vt
    return np.asarray(r, dtype=np.float64)


def poses_by_pose_propagation(
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    k: NDArray[np.floating],
    clicked_poses: Mapping[int, tuple[NDArray[np.floating], NDArray[np.floating]]],
    size: tuple[int, int],
    *,
    frames: Sequence[int] | None = None,
) -> dict[int, FramePose]:
    """Engine B: propagate each clicked frame's pose to neighbours via the chain.

    For a target frame f, take the nearest clicked frame c in the same segment, the
    chain transform G_{c->f} = inv(M[f]) @ M[c] (converted normalized->pixel), recover
    the relative camera rotation, and compose it onto c's pose, keeping the (fixed)
    optical centre. A frame with no clicked frame in its segment is left uncovered.
    Returns {frame: FramePose} (n_points=0; residual_px=nan — no obs at f).
    """
    w, h = size
    d_mat = np.diag([float(w), float(h), 1.0])
    d_inv = np.diag([1.0 / w, 1.0 / h, 1.0])
    k_arr = np.asarray(k, dtype=np.float64)
    clicked = sorted(clicked_poses)
    clicked_seg = {c: segment_of.get(c) for c in clicked}
    targets = sorted(transforms) if frames is None else [
        int(f) for f in frames if int(f) in transforms]
    out: dict[int, FramePose] = {}
    for f in targets:
        seg = segment_of.get(f)
        cands = [c for c in clicked if clicked_seg[c] == seg]
        if not cands:
            continue
        c = min(cands, key=lambda cc: abs(cc - f))
        rc, _ = cv2.Rodrigues(np.asarray(clicked_poses[c][0], dtype=np.float64))
        tc = np.asarray(clicked_poses[c][1], dtype=np.float64).reshape(3)
        if c == f:
            r_f, t_f = rc, tc.reshape(3, 1)
        else:
            g_norm = np.linalg.inv(np.asarray(transforms[f], dtype=np.float64)) \
                @ np.asarray(transforms[c], dtype=np.float64)
            g_px = d_mat @ g_norm @ d_inv
            r_rel = _rotation_from_chain(g_px, k_arr)
            r_f = r_rel @ rc
            center = -rc.T @ tc                 # fixed optical centre
            t_f = (-r_f @ center).reshape(3, 1)
        rvec_f, _ = cv2.Rodrigues(r_f)
        out[f] = FramePose(
            rvec=np.asarray(rvec_f, dtype=np.float64),
            tvec=np.asarray(t_f, dtype=np.float64),
            residual_px=float("nan"),
            n_points=0,
            fold_count=_fold_for_pose(k_arr, rvec_f, t_f, size),
        )
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -k engine_b -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full file + lint + typecheck**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -v` (all green)
Run: `uv run ruff check src/soccer_vision/pitch/calib_anchor.py tests/test_pitch_calib_anchor.py`
Then from REPO ROOT: `uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py packages/soccer-vision/tests/test_pitch_calib_anchor.py
git commit -m "feat(pitch): Engine B — poses_by_pose_propagation (rotation-from-chain)"
```

---

## Task 5: Comparison harness — `compare_engines` + notebook + smoke test

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/calib_compare.py`
- Create test: `packages/soccer-vision/tests/test_pitch_calib_compare.py`
- Create: `examples/calib_anchor_compare.ipynb`

- [ ] **Step 1: Write the failing test**

Create `packages/soccer-vision/tests/test_pitch_calib_compare.py`:

```python
"""Smoke test for the three-way engine comparison harness."""

from __future__ import annotations

import cv2
import numpy as np
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.pitch.calib_compare import EngineMetrics, compare_engines
from soccer_vision.pitch.manual_anchor import Click


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    e, t, u = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    fwd = t - e; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, u); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rvec, _ = cv2.Rodrigues(np.vstack([right, down, fwd]))
    return rvec, (-np.vstack([right, down, fwd]) @ e).reshape(3, 1)


def test_compare_engines_returns_three_engine_metrics() -> None:
    _K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)
    fp = field_points_3d()
    eyes = {0: (8.0, 4, 70), 2: (33, 14, 80), 4: (18, 59, 75), 6: (40, 44, 85)}
    clicks: list[Click] = []
    for f, e in eyes.items():
        rvec, tvec = _look_at(e, (22.85, 34.25, 0.0))
        px = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080:
                clicks.append(Click(f, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080))
    interframe = {i: np.eye(3) for i in range(6)}
    result = compare_engines(clicks, interframe, 7, (1920, 1080), window=360)
    assert set(result) == {"free_fit", "engine_a", "engine_b"}
    for m in result.values():
        assert isinstance(m, EngineMetrics)
        assert m.n_covered >= 0 and m.n_folded >= 0
        assert 0.0 <= m.coverage_fraction <= 1.0
        assert isinstance(m.per_frame_median_ft, dict)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: ...calib_compare`.

- [ ] **Step 3: Implement `compare_engines`**

Create `packages/soccer-vision/src/soccer_vision/pitch/calib_compare.py`:

```python
"""Three-way head-to-head: free fit vs Engine A vs Engine B on one session.

Pure metrics (coverage, folds, held-out accuracy in feet) so the comparison logic
is unit-tested; the heavy real-data run lives in examples/calib_anchor_compare.ipynb
(the user runs it and assesses the rendered overlays).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.field_model import METRES_TO_FEET, field_points_3d
from soccer_vision.calib.validate import reproj_error_feet
from soccer_vision.pitch.calib_anchor import (
    FramePose,
    calibrate_clicked_frames,
    frame_homography,
    poses_by_click_propagation,
    poses_by_pose_propagation,
)
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import (
    Click,
    FrameFit,
    build_segments,
    cumulative_transforms,
    fit_frame_homographies,
)


@dataclass(frozen=True)
class EngineMetrics:
    """One engine's full-game numbers.

    accuracy here is the REPROJECTION error (feet) of the clicked landmarks under
    the engine's homography at the directly-clicked frames (NOT leave-one-out — a
    uniform, per-engine measure; the held-out calib-model accuracy is Phase-1's
    `leave_one_out_feet`, run alongside in the notebook). `per_frame_median_ft` is
    the per-clicked-frame median, indexed by frame, so the notebook can plot drift
    (error vs frame number).
    """

    n_covered: int            # frames with a homography
    coverage_fraction: float  # n_covered / n_frames
    n_folded: int             # covered frames whose fold_count >= fold_threshold
    median_accuracy_ft: float # reprojection error (feet) over all clicked landmarks, nan if none
    p90_accuracy_ft: float
    per_frame_median_ft: dict[int, float]  # {clicked_frame: median feet} for the drift plot


def _free_fit_homographies(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    size: tuple[int, int],
    *,
    window: int,
) -> dict[int, NDArray[np.float64]]:
    """The current free fit -> full-pixel image->pitch homographies (export form)."""
    w, h = size
    fits: dict[int, FrameFit] = fit_frame_homographies(
        clicks, transforms, segment_of, PITCH_LANDMARKS, window=window)
    # FrameFit.H is normalized image->pitch; denormalize to full-pixel image->pitch
    # (H_px = H_norm @ diag(1/W,1/H,1); same as labeler.chain.denormalize_homography,
    # inlined to keep pitch/ independent of labeler/).
    s = np.diag([1.0 / w, 1.0 / h, 1.0])
    return {f: np.asarray(fit.H, dtype=np.float64) @ s for f, fit in fits.items()}


def _calib_homographies(poses: Mapping[int, FramePose], k: NDArray[np.floating]) \
        -> dict[int, NDArray[np.float64]]:
    return {f: frame_homography(k, p.rvec, p.tvec) for f, p in poses.items()}


def _accuracy_feet(
    homographies: Mapping[int, NDArray[np.floating]],
    clicks_by_frame: Mapping[int, list[tuple[int, float, float]]],
) -> dict[int, list[float]]:
    """Per CLICKED frame with a homography, the feet error of each clicked landmark.

    A homography is image(px)->pitch[0,1]; invert and scale pitch[0,1]->metres to map
    a clicked pixel to field metres, compare to the landmark's true metres. Returns
    {frame: [feet errors]} so the caller can aggregate AND index by frame (drift).
    """
    fp = field_points_3d()
    width_m = float(fp[:, 0].max())
    length_m = float(fp[:, 1].max())
    out: dict[int, list[float]] = {}
    for f, obs in clicks_by_frame.items():
        h = homographies.get(f)
        if h is None:
            continue
        hinv = np.linalg.inv(np.asarray(h, dtype=np.float64))
        errs: list[float] = []
        for kp, x_px, y_px in obs:
            v = hinv @ np.array([x_px, y_px, 1.0])
            pitch = v[:2] / v[2]
            mx, my = pitch[0] * width_m, pitch[1] * length_m
            errs.append(float(np.hypot(mx - fp[kp, 0], my - fp[kp, 1]) * METRES_TO_FEET))
        if errs:
            out[f] = errs
    return out


def compare_engines(
    clicks: Sequence[Click],
    interframe: Mapping[int, NDArray[np.floating]],
    n_frames: int,
    size: tuple[int, int],
    *,
    window: int = 360,
    fold_threshold: int = 16,
    min_points: int = 6,
) -> dict[str, EngineMetrics]:
    """Run all three engines on one session; return per-engine metrics.

    fold_threshold: a covered frame with >= this many in-frame landmarks is counted
    as a fold (a fixed Trace crop never shows the whole field). Accuracy is the
    reprojection error (feet) of the clicked landmarks at the directly-clicked frames
    (uniform per-engine; NOT leave-one-out).
    """
    w, h = size
    segment_of = build_segments(interframe, n_frames)
    transforms = cumulative_transforms(interframe, segment_of)
    clicks_by_frame: dict[int, list[tuple[int, float, float]]] = {}
    for c in clicks:
        clicks_by_frame.setdefault(c.frame, []).append((c.kp_idx, c.x * w, c.y * h))

    k, clicked_poses = calibrate_clicked_frames(clicks, size, min_points=min_points)

    free_h = _free_fit_homographies(clicks, transforms, segment_of, size, window=window)
    a_poses = poses_by_click_propagation(
        clicks, transforms, segment_of, k, size, window=window)
    b_poses = poses_by_pose_propagation(transforms, segment_of, k, clicked_poses, size)
    a_h = _calib_homographies(a_poses, k)
    b_h = _calib_homographies(b_poses, k)

    def _metrics(
        homs: Mapping[int, NDArray[np.floating]],
        folds: Mapping[int, int] | None,
    ) -> EngineMetrics:
        n_cov = len(homs)
        if folds is not None:
            n_fold = sum(1 for fc in folds.values() if fc >= fold_threshold)
        else:
            # free fit: fold via its pitch homography's in-frame landmark count
            from soccer_vision.calib.validate import fold_count

            def _ff(hp: NDArray[np.floating]) -> int:
                # homs are image->pitch; the pitch-of-image is its inverse
                return fold_count(np.linalg.inv(np.asarray(hp, dtype=np.float64)), size)
            n_fold = sum(1 for hp in homs.values() if _ff(hp) >= fold_threshold)
        per_frame = _accuracy_feet(homs, clicks_by_frame)
        flat = [e for errs in per_frame.values() for e in errs]
        med = float(np.median(flat)) if flat else float("nan")
        p90 = float(np.percentile(flat, 90)) if flat else float("nan")
        per_frame_median = {f: float(np.median(errs)) for f, errs in per_frame.items()}
        return EngineMetrics(
            n_covered=n_cov,
            coverage_fraction=n_cov / n_frames if n_frames else 0.0,
            n_folded=n_fold,
            median_accuracy_ft=med,
            p90_accuracy_ft=p90,
            per_frame_median_ft=per_frame_median,
        )

    return {
        "free_fit": _metrics(free_h, None),
        "engine_a": _metrics(a_h, {f: p.fold_count for f, p in a_poses.items()}),
        "engine_b": _metrics(b_h, {f: p.fold_count for f, p in b_poses.items()}),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_compare.py -v`
Expected: PASS.

- [ ] **Step 5: Create the thin notebook**

Create `examples/calib_anchor_compare.ipynb` — a minimal notebook (use the existing `examples/calib_validate.ipynb` as the structural reference) with cells covering all five spec metrics:
1. Markdown: title + "loads the full-game chain `.npz` + the session's exported `keypoints.parquet`, runs `compare_engines`, prints the three-way table, plots drift, times each engine, and renders overlays at reference frames for visual assessment."
2. Load: `from soccer_vision.labeler.chain import load_chain`; `from soccer_vision.labeler.state import clicks_from_keypoints_parquet`; load the chain `(interframe, n_frames, size)` and the clicks (`clicks_from_keypoints_parquet` returns normalized `Click`s).
3. **Coverage / folds / accuracy:** `from soccer_vision.pitch.calib_compare import compare_engines`; `result = compare_engines(clicks, interframe, n_frames, size)`; print each `EngineMetrics` (n_covered, coverage_fraction, n_folded, median/p90 accuracy).
4. **Drift:** for each engine, scatter `per_frame_median_ft` (y = feet error) vs frame (x) — a flat band = no drift; growth/spikes = drift. (Use matplotlib; one subplot per engine.)
5. **Runtime:** time each engine call directly (`import time; t=time.perf_counter(); poses_by_click_propagation(...); dt=time.perf_counter()-t`) — A vs B vs free-fit wall-clock at 58k frames (the interactive-recompute cost for 3b). Build `segment_of`/`transforms` once via `build_segments`/`cumulative_transforms`, then call each engine.
6. **Held-out calib accuracy (optional reference):** run Phase-1 `validate.leave_one_out_feet` on the clicked frames for the held-out (not reprojection) number.
7. **Overlays:** for a few reference frames, draw the free-fit / A / B reprojected pitch (reuse `viz/pitch_overlay.py`) onto the frame for the USER to assess (render only — do NOT self-assess images).

Keep it thin — the engines and metrics are tested in code; the notebook is glue the user runs on real data. Validate it parses:
Run: `cd packages/soccer-vision && uv run python -c "import json; json.load(open('../../examples/calib_anchor_compare.ipynb'))"`
Expected: no error (valid notebook JSON).

- [ ] **Step 6: Lint + typecheck + commit**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/pitch/calib_compare.py tests/test_pitch_calib_compare.py`
Then from REPO ROOT: `uv run mypy`
Expected: clean.

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/calib_compare.py packages/soccer-vision/tests/test_pitch_calib_compare.py examples/calib_anchor_compare.ipynb
git commit -m "feat(pitch): compare_engines harness (free-fit vs A vs B) + notebook"
```

---

## Done criteria
- `propagate_clicks` extracted; `fit_frame_homographies` unchanged in behavior (existing suite green).
- `calib_anchor`: `FramePose`, `frame_homography` (px→pitch export form), `calibrate_clicked_frames`, Engine A (`poses_by_click_propagation`, SQPNP+refine, line_obs path wired), Engine B (`poses_by_pose_propagation` + `_rotation_from_chain`, recovers panned poses, degenerate-safe).
- `compare_engines` returns the three-way `EngineMetrics` (coverage / folds / reprojection feet + per-frame drift).
- A thin notebook for the user's real full-game run.
- Full suite + ruff + root mypy green for touched files. No `labeler/` integration (Phase 3b).

## Out of scope (Phase 3b)
Wiring an engine into `LabelerState`/server/export; the line-click UI + line propagation; incremental recompute; real line-click validation; the free-fit→calibration swap decision.
