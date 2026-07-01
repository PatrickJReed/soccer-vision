# Gated Click Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the labeler's "aggregate every in-window click and fit once" pose engine with a proximity-seeded, consistency-gated fit so a far click drifted across the chain can never corrupt a frame that has good local clicks, and true gaps stay honestly red.

**Architecture:** A new `poses_by_gated_propagation` engine: per frame, propagate landmarks (tracking each one's source frame-distance), bail to red if the nearest support is past `gap_dist`, seed an SQPNP pose from the nearest `seed_size` landmarks, accept farther landmarks only if they reproject within `gate_px` of that seed, then do the final SQPNP (+ existing line refine). `LabelerState` swaps its per-frame fit to this engine. The old `poses_by_click_propagation` stays for the Phase-3a `compare_engines` tooling.

**Tech Stack:** Python 3, numpy, OpenCV (`cv2.solvePnP` SQPNP, `projectPoints`), scipy (`refine_pose`); pytest; mypy strict (run from repo root); ruff (E,F,I,B,UP,RUF).

---

## Conventions for every task

- Run pytest/ruff from `packages/soccer-vision/`; run **mypy from the repo root** `/Users/patrickreed/Sandbox/soccer-vision` (running it from inside the package reports bogus errors — a known double-cd artifact).
- Branch is `feat/labeler-gated-propagation` (already created). Do NOT create a new branch.
- The clicks/transforms live in **normalized** [0,1] image space; SQPNP works in **full pixels** (×W, ×H). `field_points_3d()` returns the 21 landmarks as 3D metres (Z=0).

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/soccer_vision/pitch/manual_anchor.py` | Add `propagate_clicks_with_distance` (the existing numpy propagation, now also returning each landmark's source frame-distance); `propagate_clicks` becomes a thin wrapper that drops the distance. | 1 |
| `tests/test_pitch_manual_anchor.py` | Distance-reporting + nearest-wins + delegation tests. | 1 |
| `src/soccer_vision/pitch/calib_anchor.py` | Add `poses_by_gated_propagation` (seed + gate + gap_dist). `poses_by_click_propagation` unchanged. | 2 |
| `tests/test_pitch_calib_anchor.py` | Synthetic gate tests: corrupted-far rejected, stay-red gap, consistent-far accepted. | 2 |
| `src/soccer_vision/labeler/state.py` | `__init__` gains `seed_size`/`gate_px`/`gap_dist`; `_compute_poses` calls the gated engine (mapping `window`→`max_reach`). | 3 |
| `tests/test_labeler_state.py` | Params-wired + clean-session no-regression test. | 3 |

---

## Task 1: `propagate_clicks_with_distance`

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_manual_anchor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pitch_manual_anchor.py` (it already imports `Click`, `build_segments`, `cumulative_transforms`, `propagate_clicks`; add `propagate_clicks_with_distance` to the `manual_anchor` import):

```python
def test_propagate_clicks_with_distance_reports_source_distance() -> None:
    interframe = {i: np.eye(3) for i in range(6)}
    seg = build_segments(interframe, 6)
    transforms = cumulative_transforms(interframe, seg)
    prop = propagate_clicks_with_distance(
        [Click(frame=2, kp_idx=3, x=0.4, y=0.6)], transforms, seg, window=10)
    assert prop[2][3] == (0.4, 0.6, 0.0)   # at source -> distance 0
    assert prop[5][3] == (0.4, 0.6, 3.0)   # 3 frames away -> distance 3


def test_propagate_clicks_with_distance_nearest_source_wins() -> None:
    interframe = {i: np.eye(3) for i in range(6)}
    seg = build_segments(interframe, 6)
    transforms = cumulative_transforms(interframe, seg)
    clicks = [Click(0, 3, 0.1, 0.1), Click(4, 3, 0.9, 0.9)]
    prop = propagate_clicks_with_distance(clicks, transforms, seg, window=10)
    assert prop[3][3] == (0.9, 0.9, 1.0)   # frame 3 nearer to 4 (d=1) than to 0 (d=3)


def test_propagate_clicks_delegates_and_drops_distance() -> None:
    interframe = {i: np.eye(3) for i in range(4)}
    seg = build_segments(interframe, 4)
    transforms = cumulative_transforms(interframe, seg)
    prop = propagate_clicks([Click(0, 3, 0.4, 0.6)], transforms, seg, window=10)
    assert prop[2][3] == (0.4, 0.6)   # still a 2-tuple, unchanged
```

(If `np` is not imported in that test file, add `import numpy as np`.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -k "with_distance or delegates" -v`
Expected: FAIL (`propagate_clicks_with_distance` not defined).

- [ ] **Step 3: Implement**

In `manual_anchor.py`, rename the body of the current `propagate_clicks` into a new
`propagate_clicks_with_distance` that emits `(x, y, dist)`, and make `propagate_clicks`
delegate. Replace the entire existing `def propagate_clicks(...) -> dict[int, dict[int, tuple[float, float]]]:`
function with these two:

```python
def propagate_clicks_with_distance(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    *,
    window: int,
    frames: Sequence[int] | None = None,
) -> dict[int, dict[int, tuple[float, float, float]]]:
    """Like propagate_clicks, but each landmark also carries the frame-distance to its
    winning source click (`|source.frame - target|`). Returns {frame: {kp: (x, y, dist)}}.
    """
    out: dict[int, dict[int, tuple[float, float, float]]] = {}
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
        choice = d.argmin(axis=0)            # first minimal row wins ties (click order)
        chosen_dist = d[choice, np.arange(n)]
        ok = chosen_dist != big
        chosen = pos[rows[choice], np.arange(n)]
        for gi in np.where(ok)[0]:
            out.setdefault(int(frame_arr[gi]), {})[kp] = (
                float(chosen[gi, 0]), float(chosen[gi, 1]), float(chosen_dist[gi]))
    return out


def propagate_clicks(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    *,
    window: int,
    frames: Sequence[int] | None = None,
) -> dict[int, dict[int, tuple[float, float]]]:
    """For each target frame, the propagated pixel of each landmark (nearest source
    click wins). Returns {frame: {kp_idx: (x, y)}} in normalized image space. Thin
    wrapper over propagate_clicks_with_distance that drops the source distance.
    """
    return {
        g: {kp: (xyd[0], xyd[1]) for kp, xyd in kps.items()}
        for g, kps in propagate_clicks_with_distance(
            clicks, transforms, segment_of, window=window, frames=frames).items()
    }
```

- [ ] **Step 4: Run to verify pass + no regression**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -v` (new + existing pass)
Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -q` (propagate_clicks consumers unaffected)
Expected: all pass.

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/pitch/manual_anchor.py tests/test_pitch_manual_anchor.py`
Then REPO ROOT: `cd /Users/patrickreed/Sandbox/soccer-vision && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py packages/soccer-vision/tests/test_pitch_manual_anchor.py
git commit -m "feat(pitch): propagate_clicks_with_distance (source frame-distance per landmark)"
```

---

## Task 2: `poses_by_gated_propagation`

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_calib_anchor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pitch_calib_anchor.py`. Add `poses_by_gated_propagation` to the
`calib_anchor` import block (which already imports `poses_by_click_propagation`,
`FramePose`, etc.). The helpers `_K`, `_look_at`, `field_points_3d`, `Click`,
`build_segments`, `cumulative_transforms` are already imported in that file.

```python
def _static_chain(n: int) -> tuple[dict[int, np.ndarray], dict[int, int], dict[int, np.ndarray]]:
    interframe = {i: np.eye(3) for i in range(n)}
    seg = build_segments(interframe, n)
    transforms = cumulative_transforms(interframe, seg)
    return interframe, seg, transforms


def test_gated_propagation_rejects_corrupted_far_click() -> None:
    # Identity chain, static camera: a frame with good NEAR clicks plus a CORRUPTED
    # far click (simulating chain drift) -> the far click is gated out, pose stays true,
    # and the old aggregate engine is dragged off by the same bad click.
    _interframe, seg, transforms = _static_chain(400)
    rvec, tvec = _look_at((-8.0, 34.0, 20.0), (22.85, 34.0, 0.0))
    fp = field_points_3d()
    px = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    in_ids = [j for j in range(21) if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080]
    assert len(in_ids) >= 8
    near, far_id = in_ids[:7], in_ids[7]
    clicks = [Click(0, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080) for j in near]
    clicks.append(Click(200, far_id, float(px[far_id, 0] + 300.0) / 1920,
                        float(px[far_id, 1]) / 1080))  # corrupted, 300px off, 200 frames away
    out = poses_by_gated_propagation(
        clicks, transforms, seg, _K, (1920, 1080),
        max_reach=360, seed_size=6, gate_px=60.0, gap_dist=180, min_points=4, frames=[0])
    assert 0 in out
    assert out[0].n_points == 7  # 7 near kept; corrupted far rejected
    rec = cv2.projectPoints(fp, out[0].rvec, out[0].tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    gated_err = float(np.median(np.linalg.norm(px[in_ids] - rec[in_ids], axis=1)))
    assert gated_err < 1.0  # recovered ~= truth
    old = poses_by_click_propagation(
        clicks, transforms, seg, _K, (1920, 1080), window=360, min_points=4, frames=[0])
    rec_old = cv2.projectPoints(fp, old[0].rvec, old[0].tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    old_err = float(np.median(np.linalg.norm(px[in_ids] - rec_old[in_ids], axis=1)))
    assert old_err > gated_err  # the aggregate engine is corrupted; gating is better


def test_gated_propagation_stays_red_beyond_gap_dist() -> None:
    _interframe, seg, transforms = _static_chain(400)
    rvec, tvec = _look_at((-8.0, 34.0, 20.0), (22.85, 34.0, 0.0))
    fp = field_points_3d()
    px = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    in_ids = [j for j in range(21) if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080]
    clicks = [Click(300, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080) for j in in_ids[:7]]
    out = poses_by_gated_propagation(
        clicks, transforms, seg, _K, (1920, 1080),
        max_reach=360, seed_size=6, gate_px=60.0, gap_dist=180, min_points=4, frames=[0])
    assert 0 not in out  # nearest click 300 frames away (> gap_dist 180) -> red
    out2 = poses_by_gated_propagation(
        clicks, transforms, seg, _K, (1920, 1080),
        max_reach=360, seed_size=6, gate_px=60.0, gap_dist=320, min_points=4, frames=[0])
    assert 0 in out2  # a larger gap_dist recovers it -> gap_dist is the gate


def test_gated_propagation_accepts_consistent_far_clicks() -> None:
    _interframe, seg, transforms = _static_chain(200)
    rvec, tvec = _look_at((-8.0, 34.0, 20.0), (22.85, 34.0, 0.0))
    fp = field_points_3d()
    px = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    in_ids = [j for j in range(21) if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080]
    assert len(in_ids) >= 10
    near, far = in_ids[:6], in_ids[6:10]
    clicks = [Click(0, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080) for j in near]
    clicks += [Click(100, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080) for j in far]
    out = poses_by_gated_propagation(
        clicks, transforms, seg, _K, (1920, 1080),
        max_reach=360, seed_size=6, gate_px=60.0, gap_dist=180, min_points=4, frames=[0])
    assert 0 in out
    assert out[0].n_points == 10  # consistent (clean-chain) far clicks accepted -> deeper fit
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -k gated -v`
Expected: FAIL (`poses_by_gated_propagation` not defined).

- [ ] **Step 3: Implement**

In `calib_anchor.py`: add `propagate_clicks_with_distance` to the `manual_anchor` import (the
file already imports `propagate_clicks` there — add the new name alongside). Then add the
function (place it right after `poses_by_click_propagation`):

```python
def poses_by_gated_propagation(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    k: NDArray[np.floating],
    size: tuple[int, int],
    *,
    max_reach: int,
    seed_size: int = 6,
    gate_px: float = 60.0,
    gap_dist: int = 180,
    min_points: int = 4,
    line_obs: Mapping[int, Sequence[tuple[str, float, float]]] | None = None,
    frames: Sequence[int] | None = None,
) -> dict[int, FramePose]:
    """Reliability-aware Engine A. Per target frame: propagate landmarks (tracking each
    one's source frame-distance), red if the nearest is past `gap_dist`, seed an SQPNP pose
    from the nearest `seed_size` landmarks, accept farther landmarks only if they reproject
    within `gate_px` of the seed, then final SQPNP (+ refine_pose when line_obs present). A
    far click drifted across the chain can't corrupt a frame with good local clicks; true
    gaps stay red. Returns {frame: FramePose}.
    """
    w, h = size
    k_arr = np.asarray(k, dtype=np.float64)
    fp = field_points_3d()
    prop = propagate_clicks_with_distance(
        clicks, transforms, segment_of, window=max_reach, frames=frames)
    out: dict[int, FramePose] = {}
    for f, kpmap in prop.items():
        if len(kpmap) < min_points:
            continue
        if min(xyd[2] for xyd in kpmap.values()) > gap_dist:  # no reliable nearby anchor
            continue
        items = sorted(kpmap.items(), key=lambda kv: kv[1][2])  # ascending source distance
        kps = [kp for kp, _ in items]
        img = np.array([[v[0] * w, v[1] * h] for _, v in items], dtype=np.float64)
        n_seed = min(seed_size, len(kps))
        if n_seed < min_points:
            continue
        ok, rvec0, tvec0 = cv2.solvePnP(
            fp[kps[:n_seed]].astype(np.float64), img[:n_seed], k_arr, None,
            flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        proj = cv2.projectPoints(
            fp[kps].astype(np.float64), rvec0, tvec0, k_arr, np.zeros(5))[0].reshape(-1, 2)
        err = np.sqrt(np.sum((proj - img) ** 2, axis=1))
        keep = [i for i in range(len(kps)) if i < n_seed or err[i] <= gate_px]
        if len(keep) < min_points:
            continue
        keep_kps = [kps[i] for i in keep]
        keep_img = img[keep]
        point_obs = [(keep_kps[i], float(keep_img[i, 0]), float(keep_img[i, 1]))
                     for i in range(len(keep_kps))]
        ok2, rvec, tvec = cv2.solvePnP(
            fp[keep_kps].astype(np.float64), keep_img, k_arr, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok2:
            continue
        rvec = np.asarray(rvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        lobs = list(line_obs.get(f, [])) if line_obs else []
        if lobs:
            try:
                rvec, tvec = refine_pose(k_arr, rvec, tvec, point_obs, lobs)
            except CalibError:
                pass  # keep the SQPNP pose
        out[f] = FramePose(
            rvec=rvec,
            tvec=tvec,
            residual_px=_reproj_rms_px(k_arr, rvec, tvec, point_obs),
            n_points=len(keep_kps),
            fold_count=_fold_for_pose(k_arr, rvec, tvec, size),
        )
    return out
```

(Confirm `CalibError`, `refine_pose`, `field_points_3d`, `_reproj_rms_px`, `_fold_for_pose`,
`FramePose`, `Click` are already imported in `calib_anchor.py` — they are, used by
`poses_by_click_propagation`. Only `propagate_clicks_with_distance` is a new import.)

- [ ] **Step 4: Run to verify pass + no regression**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_calib_anchor.py -v`
Expected: the 3 new gated tests PASS; existing calib_anchor tests still pass.

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/pitch/calib_anchor.py tests/test_pitch_calib_anchor.py`
Then REPO ROOT: `cd /Users/patrickreed/Sandbox/soccer-vision && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py packages/soccer-vision/tests/test_pitch_calib_anchor.py
git commit -m "feat(pitch): poses_by_gated_propagation — proximity seed + consistency gate + gap_dist"
```

---

## Task 3: `LabelerState` uses the gated engine

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py`
- Test: `packages/soccer-vision/tests/test_labeler_state.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_labeler_state.py` (reuse the existing `_pan_session` helper):

```python
def test_state_gated_params_default_and_clean_session_fits() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080), window=360)
    try:
        assert (st.seed_size, st.gate_px, st.gap_dist) == (6, 60.0, 180)  # defaults wired
        st.add_clicks(clicks)
        st.wait_idle(timeout=10)
        # clean pan, gated engine: clicked frames are covered and accurate (no regression)
        for c in sorted({cl.frame for cl in clicks})[:3]:
            cf = st.frame_homography(c)
            assert cf is not None and cf.residual < 60.0
    finally:
        st.stop_worker()


def test_state_gap_dist_reds_far_frames() -> None:
    # A tighter gap_dist must cover a STRICT SUBSET of frames (far ones go red) -- proves
    # gap_dist is wired through to the engine, without depending on exact click positions.
    interframe, _poses, clicks = _pan_session(40)
    wide = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080),
                        window=360, gap_dist=180)
    tight = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080),
                         window=360, gap_dist=3)
    try:
        wide.add_clicks(clicks); wide.wait_idle(timeout=10)
        tight.add_clicks(clicks); tight.wait_idle(timeout=10)
        covered_wide = {f for f in range(40) if wide.frame_homography(f) is not None}
        covered_tight = {f for f in range(40) if tight.frame_homography(f) is not None}
        assert covered_tight < covered_wide  # strict subset: tight gap_dist reds far frames
        assert covered_tight  # but clicked frames are still covered
    finally:
        wide.stop_worker(); tight.stop_worker()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -k "gated_params or gap_dist_reds" -v`
Expected: FAIL (`LabelerState` has no `seed_size`/`gate_px`/`gap_dist`).

- [ ] **Step 3: Implement**

In `state.py`:

(a) Add `poses_by_gated_propagation` to the `calib_anchor` import block (which already imports
`poses_by_click_propagation`).

(b) In `__init__`, add the three params after `line_band: int = 60,`:
```python
        seed_size: int = 6,
        gate_px: float = 60.0,
        gap_dist: int = 180,
```
and store them next to `self.line_band = line_band`:
```python
        self.seed_size = seed_size
        self.gate_px = gate_px
        self.gap_dist = gap_dist
```

(c) In `_compute_poses`, replace the `poses_by_click_propagation(...)` call with the gated
engine (the `window` becomes `max_reach`; `line_obs` is passed through as before):
```python
        return poses_by_gated_propagation(
            clicks, self._transforms, self._segment_of, k, self.size,
            max_reach=self.window, seed_size=self.seed_size, gate_px=self.gate_px,
            gap_dist=self.gap_dist, frames=frames, line_obs=line_obs)
```

- [ ] **Step 4: Run to verify pass + no regression**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -v` (new + existing pass)
Run: `cd packages/soccer-vision && uv run pytest -k labeler -q` (all pass)
Expected: all pass. (The async equivalence test still holds — both async and synchronous paths
now route through the same gated engine.)

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/labeler/state.py tests/test_labeler_state.py`
Then REPO ROOT: `cd /Users/patrickreed/Sandbox/soccer-vision && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/state.py packages/soccer-vision/tests/test_labeler_state.py
git commit -m "feat(labeler): LabelerState uses gated propagation engine"
```

---

## Done criteria
- `propagate_clicks_with_distance` returns each landmark's source frame-distance; `propagate_clicks` delegates (unchanged output).
- `poses_by_gated_propagation`: gap_dist red, nearest-N seed, consistency gate, final fit (+ line refine); tested (corrupted-far rejected, stay-red gap, consistent-far accepted).
- `LabelerState` routes `_compute_poses` through the gated engine; `poses_by_click_propagation` untouched for `compare_engines`.
- Full suite + ruff + root mypy green.

## Real validation (Patrick, visual)
Relaunch the labeler on the saved training-clip session (gated engine):
```bash
cd /Users/patrickreed/Sandbox/soccer-vision/packages/soccer-vision
uv run python -m soccer_vision.labeler --video /Users/patrickreed/sv-labeler/training_clip.mp4 \
  --export-dir /Users/patrickreed/sv-labeler/training_clip_out --port 8000 --workers 1
```
The prototype predicts the early densely-clicked frames drop from 315–3370 px to 8–20 px and
~94% trustworthy green at `gap_dist=180`. Claude renders the per-frame residuals / coverage;
Patrick assesses whether the overlay now projects correctly.

## Deferred
Chain-reliability model from repeated clicks; coverage-weighting; a `--gap-dist`/`--gate-px`
CLI flag; UI distinguishing "red: no clicks" from "red: unreliable".
