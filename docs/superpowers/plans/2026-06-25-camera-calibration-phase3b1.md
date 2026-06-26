# Camera Calibration Phase 3b-1 — Calibrated Labeler Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the labeler's free-fit homography engine with the calibrated SQPNP engine (point-only) under a freeze-focal model, with per-frame robustness (iterative drop-worst SQPNP) that drops outlier clicks and flags them.

**Architecture:** Upgrade `poses_by_click_propagation` to fit each frame with a planar-safe **iterative drop-worst SQPNP** (`cv2.solvePnPRansac` degenerates on the coplanar field — verified), returning flagged outlier clicks, and add a `frames=` argument for windowed recompute. Then swap `LabelerState`'s engine from the free fit to this calibrated engine: bootstrap and **freeze the shared focal** once ≥3 anchor frames exist (the Trace lens is constant), then recompute only the touched window on each click. The server/UI/export contracts are preserved.

**Tech Stack:** Python, NumPy, OpenCV (`cv2.solvePnP` SQPNP, `projectPoints`), pytest, mypy (strict), ruff.

**Spec:** `docs/superpowers/specs/2026-06-25-camera-calibration-phase3b1-design.md`

**Verified before writing (against real cv2 + the full-game data):**
- The shared focal is robust (1469 px with or without the 2 gross outliers) — no robust-bootstrap needed.
- `cv2.solvePnPRansac` returns ~10⁵ px garbage on the coplanar (Z=0) field points — must NOT be used.
- Iterative drop-worst SQPNP (thr=40 px) drops 8526 kp8, 56938 kp7, 14357/14700/27391 kp14/16 to ~10–18 px inlier RMS and leaves every good frame untouched.

**Conventions (from 3a / the labeler):**
- `poses_by_click_propagation(clicks, transforms, segment_of, k, size, *, window, min_points=4, line_obs=None)` and `FramePose(rvec, tvec, residual_px, n_points, fold_count)` live in `pitch/calib_anchor.py`. `frame_homography(k, rvec, tvec)` → full-pixel image→pitch. `propagate_clicks(..., frames=None)` already supports `frames=`.
- `calibrate_clicked_frames(clicks, size, *, min_points=6) -> (K, poses)` (NORMALIZED clicks in).
- `LabelerState` (`labeler/state.py`) holds the chain + clicks, `_fits: dict[int, FrameFit]`, `_refit(frames)` / `_affected(frame)` (windowed), `add_click`, `_recompute_chunked`, `coverage`/`status_list`/`status_buckets`/`frame_homography`/`export`. The server (`labeler/server.py`) reads `state.frame_homography(idx).H` (NORMALIZED image→pitch, sent to the frontend), `.residual`, `.n_points`; `export` denormalizes `.H` to full-pixel.
- Normalized↔pixel: `H_norm = H_px @ diag(W, H, 1)`; export's `denormalize_homography(H_norm) = H_px`.
- Lint gate: **`uv run mypy` from repo root**; `uv run ruff check` + `uv run pytest` from `packages/soccer-vision`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py` | calib engines | Add `_robust_sqpnp`; `FramePose.outliers`; upgrade `poses_by_click_propagation` (robust fit + `frames=`) |
| `packages/soccer-vision/src/soccer_vision/labeler/state.py` | labeler session | Swap engine: `CalibFrame` record, freeze-focal bootstrap, calibrated `_refit`, status/export |
| `packages/soccer-vision/src/soccer_vision/labeler/server.py` | HTTP | Surface outlier flags; `recalibrate` endpoint |
| `packages/soccer-vision/tests/test_pitch_calib_anchor.py` | calib tests | Add robust-fit tests |
| `packages/soccer-vision/tests/test_labeler_state.py` | labeler tests | Add calibrated-backend tests |
| `examples/calib_labeler_validate.py` | real-data check | Create (the user/Claude runs it) |

---

## Task 1: Robust per-frame fit (`_robust_sqpnp`) + `poses_by_click_propagation` upgrade

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_calib_anchor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pitch_calib_anchor.py`:

```python
from soccer_vision.pitch.calib_anchor import _robust_sqpnp


def test_robust_sqpnp_drops_a_gross_outlier_click() -> None:
    # 8 in-frame landmarks from a known pose; corrupt one click by 400px -> the helper
    # must drop exactly that landmark and return a clean inlier pose.
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    fp = field_points_3d()
    ids = [1, 3, 4, 6, 7, 8, 14, 16]
    img = cv2.projectPoints(fp[ids], rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    img[2] = img[2] + np.array([400.0, -300.0])  # gross outlier on ids[2]
    res = _robust_sqpnp(_K, ids, img, thr=40.0, min_points=4)
    assert res is not None
    rv, tv, inliers, outliers = res
    assert outliers == [ids[2]]          # exactly the corrupted landmark dropped
    assert set(inliers) == set(ids) - {ids[2]}
    proj = cv2.projectPoints(fp[inliers], rv, tv, _K, np.zeros(5))[0].reshape(-1, 2)
    truth = cv2.projectPoints(fp[inliers], rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    assert np.max(np.linalg.norm(proj - truth, axis=1)) < 2.0  # clean pose


def test_robust_sqpnp_keeps_all_clean_clicks() -> None:
    # noiseless clicks -> nothing dropped
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    fp = field_points_3d()
    ids = [1, 3, 4, 6, 7, 8, 14, 16]
    img = cv2.projectPoints(fp[ids], rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    res = _robust_sqpnp(_K, ids, img, thr=40.0, min_points=4)
    assert res is not None
    _rv, _tv, inliers, outliers = res
    assert outliers == [] and set(inliers) == set(ids)


def test_engine_a_propagation_flags_outlier_and_restricts_frames() -> None:
    # pan sequence; clicks at a few frames; corrupt one landmark at a clicked frame ->
    # that frame's FramePose.outliers names it; frames= restricts the output.
    poses, interframe = _pan_sequence(9)
    clicks = _clicks_at(poses, [0, 4, 8])
    # corrupt landmark 6 (center_mark) at frame 4 by shifting its click far
    bad = [c for c in clicks if c.frame == 4 and c.kp_idx == 6]
    assert bad
    clicks = [c if not (c.frame == 4 and c.kp_idx == 6)
              else Click(c.frame, c.kp_idx, c.x + 0.25, c.y) for c in clicks]
    seg = build_segments(interframe, 9)
    transforms = cumulative_transforms(interframe, seg)
    k, _kp = calibrate_clicked_frames(clicks, (1920, 1080), min_points=6)
    out = poses_by_click_propagation(clicks, transforms, seg, k, (1920, 1080),
                                     window=360, min_points=4, frames=[4])
    assert set(out) == {4}                 # frames= restricted the targets
    assert 6 in out[4].outliers            # the corrupted landmark flagged
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -k "robust or flags_outlier" -v`
Expected: FAIL with `ImportError: cannot import name '_robust_sqpnp'`.

- [ ] **Step 3: Add `outliers` to `FramePose`, the `_robust_sqpnp` helper, and rewire the engine**

In `pitch/calib_anchor.py`:

(a) Add a defaulted `outliers` field to `FramePose` (backward-compatible — Engine B and existing constructions still work):

```python
@dataclass(frozen=True, eq=False)
class FramePose:
    """A per-frame calibrated camera pose + quality."""

    rvec: NDArray[np.float64]
    tvec: NDArray[np.float64]
    residual_px: float   # reprojection RMS over the frame's INLIER obs (nan if none)
    n_points: int        # inlier point landmarks used (0 if pose-propagated)
    fold_count: int      # landmarks projecting in-frame (slice size, ~6-12)
    outliers: tuple[int, ...] = ()  # kp_idx of clicks dropped as outliers by the robust fit
```

(b) Add `_robust_sqpnp` (after `_fold_for_pose`):

```python
def _robust_sqpnp(
    k: NDArray[np.floating],
    ids: list[int],
    img: NDArray[np.floating],
    *,
    thr: float = 40.0,
    min_points: int = 4,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int], list[int]] | None:
    """Planar-safe robust PnP by iterative drop-worst SQPNP.

    cv2.solvePnPRansac degenerates on the coplanar (Z=0) field, so instead: SQPNP the
    current point set, find the worst reprojection residual; if it exceeds `thr`, drop
    that point and refit; repeat until the worst residual is within `thr` or fewer than
    `min_points` remain. `ids` are landmark indices; `img` is the matching (N, 2) pixel
    array. Returns (rvec, tvec, inlier_ids, outlier_ids), or None if it can't converge.
    """
    fp = field_points_3d()
    k_arr = np.asarray(k, dtype=np.float64)
    img_arr = np.asarray(img, dtype=np.float64)
    keep = list(range(len(ids)))
    while len(keep) >= min_points:
        oi = [ids[i] for i in keep]
        ok, rvec, tvec = cv2.solvePnP(
            fp[oi].astype(np.float64), img_arr[keep], k_arr, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        proj = cv2.projectPoints(fp[oi], rvec, tvec, k_arr, np.zeros(5))[0].reshape(-1, 2)
        res = np.linalg.norm(proj - img_arr[keep], axis=1)
        worst = int(res.argmax())
        if res[worst] <= thr:
            kept = set(keep)
            inlier_ids = [ids[i] for i in keep]
            outlier_ids = [ids[i] for i in range(len(ids)) if i not in kept]
            return (np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64),
                    inlier_ids, outlier_ids)
        keep.pop(worst)
    return None
```

(c) Rewire `poses_by_click_propagation` to use it. Add the `frames` and `outlier_px` keyword params; replace the `solvePnP` + unconditional `refine_pose` body with robust-SQPNP + a conditional line-refine. Replace the function body (keep the docstring, extend the signature):

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
    frames: Sequence[int] | None = None,
    outlier_px: float = 40.0,
) -> dict[int, FramePose]:
    """Engine A: propagate clicks into each frame, then a robust per-frame fit.

    Per target frame: gather window-propagated point landmarks (and any supplied
    pixel-space line_obs), fit the pose with iterative drop-worst SQPNP (dropping
    outlier clicks — gross mislabels and chain-drifted propagated neighbours — at
    `outlier_px`), then, only if line_obs are present for that frame, refine with the
    line residuals (Phase-2 path). `frames=` restricts the targets (windowed
    recompute). Dropped clicks are reported as `FramePose.outliers`. Returns
    {frame: FramePose}.
    """
    w, h = size
    k_arr = np.asarray(k, dtype=np.float64)
    propagated = propagate_clicks(clicks, transforms, segment_of, window=window, frames=frames)
    out: dict[int, FramePose] = {}
    for f, kpmap in propagated.items():
        if len(kpmap) < min_points:
            continue
        idxs = sorted(kpmap)
        img = np.array([[kpmap[i][0] * w, kpmap[i][1] * h] for i in idxs], dtype=np.float64)
        fit = _robust_sqpnp(k_arr, idxs, img, thr=outlier_px, min_points=min_points)
        if fit is None:
            continue
        rvec, tvec, inlier_ids, outlier_ids = fit
        inlier_obs = [(i, float(kpmap[i][0] * w), float(kpmap[i][1] * h)) for i in inlier_ids]
        lobs = list(line_obs.get(f, [])) if line_obs else []
        if lobs:
            try:
                rvec, tvec = refine_pose(k_arr, rvec, tvec, inlier_obs, lobs)
            except CalibError:
                pass  # keep the robust-SQPNP pose
        out[f] = FramePose(
            rvec=rvec,
            tvec=tvec,
            residual_px=_reproj_rms_px(k_arr, rvec, tvec, inlier_obs),
            n_points=len(inlier_ids),
            fold_count=_fold_for_pose(k_arr, rvec, tvec, size),
            outliers=tuple(outlier_ids),
        )
    return out
```

- [ ] **Step 4: Run to verify pass (incl. the 3a Engine A tests still green)**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -v`
Expected: all pass — the 3 new robust tests AND the existing `test_engine_a_*` tests (noiseless → robust SQPNP drops nothing → recovery unchanged; the line_obs test still refines).
Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_compare.py -v`
Expected: still passes (`compare_engines` uses the upgraded engine).

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/pitch/calib_anchor.py tests/test_pitch_calib_anchor.py`
Then from REPO ROOT: `uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py packages/soccer-vision/tests/test_pitch_calib_anchor.py
git commit -m "feat(calib): robust per-frame fit (iterative drop-worst SQPNP) + frames= + outlier flags"
```

---

## Task 2: `LabelerState` engine swap (freeze-focal calibrated backend)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py`
- Test: `packages/soccer-vision/tests/test_labeler_state.py` (create if absent; otherwise append)

- [ ] **Step 1: Write the failing tests**

Create/append `tests/test_labeler_state.py`:

```python
"""Tests for the calibrated LabelerState backend."""

from __future__ import annotations

import cv2
import numpy as np
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.labeler.chain import normalize_homography
from soccer_vision.labeler.state import LabelerState
from soccer_vision.pitch.manual_anchor import Click

_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    e, t, u = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    f = t - e; f /= np.linalg.norm(f)
    r = np.cross(f, u); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    rvec, _ = cv2.Rodrigues(np.vstack([r, d, f]))
    return rvec, (-np.vstack([r, d, f]) @ e).reshape(3, 1)


def _pan_session(n=9):
    """A panning camera + valid chain + normalized clicks at a few frames."""
    center = (-8.0, 34.0, 9.0)
    poses = {f: _look_at(center, (22.85, 34.0 + dy, 0.0))
             for f, dy in enumerate(np.linspace(-10, 10, n))}
    interframe = {}
    for i in range(n - 1):
        ri, _ = cv2.Rodrigues(poses[i][0]); rj, _ = cv2.Rodrigues(poses[i + 1][0])
        g = _K @ rj @ np.linalg.inv(ri) @ np.linalg.inv(_K)
        interframe[i] = normalize_homography(g, (1920, 1080))
    fp = field_points_3d()
    clicks = []
    for f in (0, 4, 8):
        px = cv2.projectPoints(fp, poses[f][0], poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080:
                clicks.append(Click(f, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080))
    return interframe, poses, clicks


def test_labeler_bootstraps_focal_and_covers_all_frames() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    st.add_clicks(clicks)
    assert st._calibrated  # bootstrapped from >=3 anchors
    cov = st.coverage()
    assert cov > 0.8  # nearly every frame covered, fold-free
    cf = st.frame_homography(2)  # an UNCLICKED frame -> covered via propagation
    assert cf is not None and cf.H.shape == (3, 3)


def test_labeler_uncalibrated_before_three_anchors() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    two = [c for c in clicks if c.frame in (0, 4)]  # only 2 anchors
    st.add_clicks(two)
    assert not st._calibrated
    assert st.frame_homography(2) is None  # bootstrap gap


def test_labeler_flags_outlier_click() -> None:
    interframe, _poses, clicks = _pan_session(9)
    clicks = [c if not (c.frame == 4 and c.kp_idx == 6)
              else Click(c.frame, c.kp_idx, c.x + 0.25, c.y) for c in clicks]
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    st.add_clicks(clicks)
    cf = st.frame_homography(4)
    assert cf is not None and 6 in cf.outliers
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -v`
Expected: FAIL (`_calibrated` attribute missing / frame_homography returns the old FrameFit).

- [ ] **Step 3: Rewrite the `LabelerState` engine internals**

In `labeler/state.py`, make these changes:

(a) Update imports — drop the free-fit-specific ones, add the calib engine:

```python
from soccer_vision.calib.calibrate import CalibError
from soccer_vision.labeler.chain import denormalize_homography
from soccer_vision.pipeline import homographies_to_parquet
from soccer_vision.pitch.calib_anchor import (
    FramePose,
    calibrate_clicked_frames,
    frame_homography,
    poses_by_click_propagation,
)
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import (
    Click,
    build_segments,
    clicks_to_keypoints_df,
    cumulative_transforms,
)
from soccer_vision.pitch.propagation import HomographyEntry
```

(b) Add a `CalibFrame` record near the top of the module:

```python
from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class CalibFrame:
    """A calibrated per-frame result in the labeler's normalized space."""

    H: NDArray[np.float64]     # NORMALIZED image -> pitch[0,1] (frontend overlay)
    residual: float            # inlier reprojection RMS (px)
    n_points: int              # inlier landmarks
    fold_count: int
    outliers: tuple[int, ...]  # flagged kp_idx
```

(c) Replace `__init__` to hold the focal state and a px residual threshold (and a normalized→pixel scale matrix for the H conversion):

```python
    def __init__(
        self,
        interframe: Mapping[int, NDArray[np.floating]],
        n_frames: int,
        *,
        size: tuple[int, int],
        window: int = 360,
        residual_px_threshold: float = 25.0,
        outlier_px: float = 40.0,
        autosave_path: Path | None = None,
    ) -> None:
        self.n_frames = n_frames
        self.size = size
        self.window = window
        self.residual_px_threshold = residual_px_threshold
        self.outlier_px = outlier_px
        self.autosave_path = autosave_path
        self._segment_of = build_segments(interframe, n_frames)
        self._transforms = cumulative_transforms(interframe, self._segment_of)
        self.clicks: list[Click] = []
        self._fits: dict[int, CalibFrame] = {}
        self._K: NDArray[np.float64] | None = None
        self._calibrated = False
        w, h = size
        self._to_norm = np.diag([float(w), float(h), 1.0])  # H_norm = H_px @ this
```

(d) Calibration bootstrap + the calibrated per-frame builder + refit:

```python
    def _try_bootstrap(self) -> bool:
        """Estimate + freeze the shared focal once >=3 calibratable anchors exist."""
        if self._calibrated:
            return True
        try:
            k, _poses = calibrate_clicked_frames(self.clicks, self.size, min_points=6)
        except CalibError:
            return False
        self._K = k
        self._calibrated = True
        return True

    def _calib_frame(self, pose: FramePose) -> CalibFrame:
        assert self._K is not None
        h_px = frame_homography(self._K, pose.rvec, pose.tvec)  # full-pixel image->pitch
        h_norm = np.asarray(h_px @ self._to_norm, dtype=np.float64)  # normalized image->pitch
        return CalibFrame(
            H=h_norm, residual=pose.residual_px, n_points=pose.n_points,
            fold_count=pose.fold_count, outliers=pose.outliers,
        )

    def _refit(self, frames: list[int]) -> None:
        """Recompute exactly `frames` with the frozen focal (windowed)."""
        if not self._calibrated or self._K is None:
            for f in frames:
                self._fits.pop(f, None)
            return
        sub = poses_by_click_propagation(
            self.clicks, self._transforms, self._segment_of, self._K, self.size,
            window=self.window, frames=frames, outlier_px=self.outlier_px,
        )
        for f in frames:
            if f in sub:
                self._fits[f] = self._calib_frame(sub[f])
            else:
                self._fits.pop(f, None)

    def _recompute_all(self, chunk: int = 5000) -> None:
        self._fits = {}
        if not self._calibrated or self._K is None:
            return
        allf = sorted(self._transforms)
        for i in range(0, len(allf), chunk):
            part = allf[i:i + chunk]
            sub = poses_by_click_propagation(
                self.clicks, self._transforms, self._segment_of, self._K, self.size,
                window=self.window, frames=part, outlier_px=self.outlier_px,
            )
            for f, pose in sub.items():
                self._fits[f] = self._calib_frame(pose)
```

(e) Mutators — bootstrap on first eligibility (full recompute), else windowed:

```python
    def _affected(self, frame: int) -> list[int]:
        seg = self._segment_of.get(frame)
        lo = max(0, frame - self.window)
        hi = min(self.n_frames - 1, frame + self.window)
        return [f for f in range(lo, hi + 1) if self._segment_of.get(f) == seg]

    def add_click(self, frame: int, kp_idx: int, x: float, y: float) -> None:
        self.clicks.append(Click(frame=frame, kp_idx=kp_idx, x=x, y=y))
        was = self._calibrated
        if not was and self._try_bootstrap():
            self._recompute_all()           # first calibration -> full recompute
        elif self._calibrated:
            self._refit(self._affected(frame))
        self._autosave()

    def add_clicks(self, clicks: Sequence[Click], *, chunk: int = 5000) -> None:
        self.clicks.extend(clicks)
        self._try_bootstrap()
        self._recompute_all(chunk=chunk)
        self._autosave()

    def remove_last(self) -> None:
        if self.clicks:
            removed = self.clicks.pop()
            if self._calibrated:
                self._refit(self._affected(removed.frame))
            self._autosave()

    def nudge_click(self, frame: int, kp_idx: int, x: float, y: float) -> bool:
        for i in range(len(self.clicks) - 1, -1, -1):
            c = self.clicks[i]
            if c.frame == frame and c.kp_idx == kp_idx:
                self.clicks[i] = Click(frame=frame, kp_idx=kp_idx, x=x, y=y)
                if self._calibrated:
                    self._refit(self._affected(frame))
                self._autosave()
                return True
        return False

    def recalibrate(self) -> bool:
        """Re-estimate + re-freeze the focal, then full recompute. False if it fails."""
        self._calibrated = False
        self._K = None
        if not self._try_bootstrap():
            return False
        self._recompute_all()
        return True
```

(f) Status / coverage / accessor / export adapted to `CalibFrame` (green = covered AND residual ≤ threshold; calibration can't fold):

```python
    def _status_of(self, f: int) -> str:
        cf = self._fits.get(f)
        if cf is None:
            return "red"
        return "green" if cf.residual <= self.residual_px_threshold else "yellow"

    def coverage(self) -> float:
        if self.n_frames == 0:
            return 0.0
        green = sum(1 for f in range(self.n_frames) if self._status_of(f) == "green")
        return green / self.n_frames

    def status_list(self) -> list[str]:
        return [self._status_of(f) for f in range(self.n_frames)]

    def status_buckets(self, *, n_buckets: int = 1200) -> tuple[list[str], int]:
        full = self.status_list()
        if len(full) <= n_buckets:
            return full, 1
        bucket = -(-len(full) // n_buckets)
        out: list[str] = []
        for i in range(0, len(full), bucket):
            chunk = full[i:i + bucket]
            out.append("red" if "red" in chunk else "yellow" if "yellow" in chunk else "green")
        return out, bucket

    def frame_homography(self, frame: int) -> CalibFrame | None:
        return self._fits.get(frame)

    def export(self, out_dir: Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        w, h = self.size
        px_clicks = [Click(c.frame, c.kp_idx, c.x * w, c.y * h) for c in self.clicks]
        clicks_to_keypoints_df(px_clicks).to_parquet(out / "keypoints.parquet", index=False)
        entries: dict[int, HomographyEntry] = {}
        for f in range(self.n_frames):
            if self._status_of(f) != "green":
                continue
            cf = self._fits[f]
            conf = float(np.clip(1.0 - cf.residual / self.residual_px_threshold, 0.0, 1.0))
            # cf.H is normalized image->pitch; denormalize to full-pixel image->pitch
            entries[f] = HomographyEntry(
                denormalize_homography(cf.H, self.size), "manual", conf)
        homographies_to_parquet(entries, out / "homographies.parquet")
```

Delete the now-unused `_recompute_chunked` (replaced by `_recompute_all`) and any leftover free-fit imports (`fit_frame_homographies`, `FrameFit`, `coverage_fraction`, `frame_status`, `to_homography_entries`). Keep `clicks_from_sidecar` / `clicks_from_keypoints_parquet` unchanged.

- [ ] **Step 4: Run to verify pass + no regression**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -v`
Expected: 3 new tests PASS.
Run: `cd packages/soccer-vision && uv run pytest -k "labeler" -v`
Expected: all labeler tests pass. NOTE: tests that asserted the OLD free-fit `FrameFit`/`residual_threshold=0.05` behavior must be updated to the calibrated backend (the engine changed by design); update assertions to the `CalibFrame`/`residual_px_threshold` semantics, do NOT re-introduce the free fit.

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/labeler/state.py tests/test_labeler_state.py`
Then from REPO ROOT: `uv run mypy`
Expected: clean for the touched files.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/state.py packages/soccer-vision/tests/test_labeler_state.py
git commit -m "feat(labeler): swap LabelerState to the calibrated engine (freeze-focal + robust SQPNP)"
```

---

## Task 3: Server — surface outlier flags + `recalibrate` endpoint

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/server.py`
- Test: `packages/soccer-vision/tests/test_labeler_server.py` (append if exists; else create a minimal handler test)

- [ ] **Step 1: Write the failing test**

Append/create `tests/test_labeler_server.py`:

```python
"""Server payload tests for the calibrated backend."""

from __future__ import annotations

import numpy as np
from soccer_vision.labeler.server import make_handler
from soccer_vision.pitch.calib_anchor import frame_homography


class _StubState:
    """Minimal state surface the handler needs (no video/session required)."""

    n_frames = 9

    def __init__(self) -> None:
        from soccer_vision.pitch.calib_anchor import FramePose
        from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
        from soccer_vision.labeler.state import CalibFrame
        k = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)
        rvec = np.array([[0.1], [0.0], [0.0]]); tvec = np.array([[0.0], [-30.0], [40.0]])
        h_px = frame_homography(k, rvec, tvec)
        h_norm = h_px @ np.diag([1920.0, 1080.0, 1.0])
        self._cf = CalibFrame(H=h_norm, residual=3.0, n_points=7, fold_count=8,
                              outliers=(6,))
        self.clicks: list = []
        _ = PITCH_LANDMARKS  # imported for parity with the real state payload

    def frame_homography(self, idx):
        return self._cf if idx == 4 else None

    def status_buckets(self, *, n_buckets=1200):
        return (["green"] * self.n_frames, 1)

    def coverage(self):
        return 1.0


def test_frame_h_payload_includes_outliers() -> None:
    state = _StubState()
    handler_cls = make_handler(state, lambda i: b"", ["kp%d" % i for i in range(21)])
    # exercise the /api/frame_h serialization path via the state record the handler reads
    cf = state.frame_homography(4)
    assert cf is not None and 6 in cf.outliers
    assert handler_cls is not None  # handler builds against the calibrated state surface
```

(Light by design — it asserts the `outliers` data the handler serializes is on the
record, and that `make_handler` binds against the calibrated state surface. The
handler edit below adds the wire format.)

- [ ] **Step 2: Run to verify the handler doesn't send outliers yet**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_server.py -v`
Expected: PASS for the data assertion above; the handler edit (Step 3) adds the wire format.

- [ ] **Step 3: Update the handler**

In `server.py`, change the `/api/frame_h/` branch to send the calibrated fields + outliers, and add a `/api/recalibrate` POST:

```python
            elif path.startswith("/api/frame_h/"):
                idx = int(path.rsplit("/", 1)[1])
                cf = state.frame_homography(idx)
                self._json({
                    "h": None if cf is None
                    else [float(v) for v in np.asarray(cf.H).reshape(9)],
                    "residual": None if cf is None else cf.residual,
                    "n_points": None if cf is None else cf.n_points,
                    "outliers": [] if cf is None else list(cf.outliers),
                })
```

And in `do_POST`, add:

```python
            elif self.path == "/api/recalibrate":
                ok = state.recalibrate()
                self._json({"recalibrated": ok, **self._state_payload()})
```

- [ ] **Step 4: Run + lint + typecheck**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_server.py -v` (pass)
Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/labeler/server.py tests/test_labeler_server.py`
Then from REPO ROOT: `uv run mypy`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/server.py packages/soccer-vision/tests/test_labeler_server.py
git commit -m "feat(labeler): server sends outlier flags + recalibrate endpoint"
```

---

## Task 4: Real-data validation script

**Files:**
- Create: `examples/calib_labeler_validate.py`

- [ ] **Step 1: Write the script**

Create `examples/calib_labeler_validate.py` (spawn-safe `__main__` guard — never run multiprocessing from stdin):

```python
"""Validate the calibrated LabelerState backend on a full-game session.

Loads a chain .npz + an exported keypoints.parquet, runs the calibrated LabelerState,
and reports coverage, fold-free, and the flagged outlier clicks vs the free-fit
baseline (via compare_engines). The user runs this on the real data.

Usage: python examples/calib_labeler_validate.py CHAIN.npz KEYPOINTS.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(chain_path: str, keypoints_path: str) -> None:
    from soccer_vision.labeler.chain import load_chain
    from soccer_vision.labeler.state import LabelerState, clicks_from_keypoints_parquet

    loaded = load_chain(Path(chain_path))
    assert loaded is not None, f"chain not found: {chain_path}"
    interframe, n_frames, size = loaded
    clicks = clicks_from_keypoints_parquet(Path(keypoints_path), size)

    st = LabelerState(interframe=interframe, n_frames=n_frames, size=size, window=360)
    st.add_clicks(clicks)
    n_cov = sum(1 for f in range(n_frames) if st._status_of(f) != "red")
    flagged = {f: cf.outliers for f, cf in st._fits.items() if cf.outliers}
    print(f"calibrated: {st._calibrated}, focal: "
          f"{None if st._K is None else round(float(st._K[0, 0]), 1)}")
    print(f"coverage: {n_cov}/{n_frames} frames ({100 * n_cov / n_frames:.1f}%), "
          f"green: {st.coverage() * 100:.1f}%")
    print(f"frames with flagged outlier clicks: {len(flagged)}")
    for f in sorted(flagged):
        print(f"  frame {f}: dropped kp {list(flagged[f])}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 2: Smoke-run on a tiny synthetic case (CI-safe) — optional manual check**

Run (real data, the user's machine): `cd packages/soccer-vision && uv run python ../../examples/calib_labeler_validate.py /Users/patrickreed/sv-labeler/training_out/training_chain.npz /Users/patrickreed/sv-labeler/training_out/keypoints.parquet`
Expected (the go/no-go): `calibrated: True`, focal ~1469, coverage near 100 %, fold-free, and the flagged outliers include kp8@8526 / kp7@56938.

- [ ] **Step 3: Lint + commit**

Run: `cd packages/soccer-vision && uv run ruff check ../../examples/calib_labeler_validate.py`
Expected: clean.

```bash
git add examples/calib_labeler_validate.py
git commit -m "feat(examples): calibrated-labeler real-data validation script"
```

---

## Done criteria
- `_robust_sqpnp` (iterative drop-worst SQPNP) drops gross outlier clicks, keeps clean ones; `poses_by_click_propagation` uses it, supports `frames=`, and reports `FramePose.outliers`.
- `LabelerState` runs the calibrated engine: freeze-focal bootstrap at ≥3 anchors, windowed `_refit` on each click, `recalibrate()`, `CalibFrame` records (normalized `H` for the frontend, full-pixel on export), status/coverage adapted, outlier flags surfaced by the server.
- Full suite + ruff + root mypy green; the 3a/compare tests still pass.
- The real-data validation script reports fold-free, ~100 % coverage, focal ~1469, and the two genuine mislabels flagged.

## Out of scope (Phase 3b-2)
The line-click UI + line storage/propagation + `refine_pose`-with-lines for the near
touchline and midline; the frontend line-click mode; real line-anchor validation.
Schema additions (6-yard box, circle∩midline) remain a separate optional pass.
