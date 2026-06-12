# Labeler Full-Game Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make full-game labeling sessions practical: parallel chain precompute (~4h → <1h), incremental window-scoped recompute (sub-second clicks at 58k frames), crash-safe autosave, bucketed timeline, numeric residual readout, and drag-to-nudge.

**Architecture:** `fit_frame_homographies` gains a `frames` subset parameter (the projection/selection math is unchanged, just column-scoped). `LabelerState` exploits the invariant that a click mutation only affects fits within ±window of the mutated frame: incremental refits per mutation, chunked full recompute for bulk loads, atomic JSON autosave per mutation. `compute_chain` parallelizes via multiprocessing over contiguous pair chunks (same cache format). Server payloads scale via status buckets.

**Tech Stack:** Python 3.11, numpy, pandas, OpenCV, stdlib multiprocessing/http.server, vanilla JS. pytest, mypy strict (bare `uv run mypy` from REPO ROOT — checks tests too), ruff.

---

## CRITICAL conventions for every task

- **GIT (ABSOLUTE):** commit only on the current branch via `git add <paths> && git commit`. NEVER checkout/switch/reset/stash/rebase — prior agents corrupted branches doing this. Read-only git is fine.
- **mypy:** run bare `uv run mypy` from the REPO ROOT (running inside `packages/soccer-vision` gives bogus duplicate-module errors). New files/changes must add ZERO errors; annotate test helpers; `tmp_path: Path`.
- **ruff:** imports at top (sorted), no `;`-joined statements; lint changed src AND test files.
- Existing signatures you will touch (read the files before editing):
  - `pitch/manual_anchor.py::fit_frame_homographies(clicks, transforms, segment_of, landmarks, *, window, min_points=4)` — vectorized: builds `frames = np.array(sorted(transforms))`, projects every click into every frame, nearest-per-landmark, fits per frame.
  - `labeler/state.py::LabelerState(interframe, n_frames, *, size, window=360, residual_threshold=0.05)` with `_recompute/add_click/add_clicks/remove_last/coverage/status_list/frame_homography/export`; module function `clicks_from_keypoints_parquet(path, size)`.
  - `labeler/chain.py::compute_chain(video_path, *, cache_dir=None, downscale=1.0, player_boxes=None)`; `save_chain/load_chain/normalize_homography`; `_video_hash`.
  - `labeler/server.py::make_handler(state, frame_jpeg, landmark_names, *, landmark_xy=None, export_dir=None)` (GET `/`, `/app.js`, `/api/state`, `/api/frame_h/<i>`, `/api/frame/<i>`; POST `/api/click`, `/api/undo`, `/api/export`) and `run(video_path, *, port=8000, downscale_display=0.5, export_dir=None, window=360, resume=None)`.
  - `labeler/static/app.js` — canvas UI (status list timeline, click/undo/export, grid overlay).

## File Structure

| File | Change |
|---|---|
| `src/soccer_vision/pitch/manual_anchor.py` | `frames` subset param on fit |
| `src/soccer_vision/labeler/state.py` | incremental refit, chunked bulk, nudge, autosave, buckets |
| `src/soccer_vision/labeler/chain.py` | `workers` parallel precompute |
| `src/soccer_vision/labeler/server.py` | bucket payload, frame_h residual, nudge endpoint, autosave/workers wiring |
| `src/soccer_vision/labeler/__main__.py` | `--workers` flag |
| `src/soccer_vision/labeler/static/app.js`, `index.html` | buckets, residual, drag-nudge |
| `tests/test_pitch_manual_anchor.py` | subset tests |
| `tests/test_labeler_state.py` | incremental/property/nudge/autosave/bucket tests |
| `tests/test_labeler_chain.py` | parallel test |
| `tests/test_labeler_server.py` | payload/nudge tests |

---

## Task 1: `frames` subset parameter on the fit

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_manual_anchor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pitch_manual_anchor.py` (imports already cover what's needed):

```python
def test_fit_frames_subset_matches_full() -> None:
    n = 6
    interframe = _identity_chain(n)
    seg = build_segments(interframe, n)
    transforms = cumulative_transforms(interframe, seg)
    clicks = _clicks_one_per_frame()
    full = fit_frame_homographies(clicks, transforms, seg, PITCH_LANDMARKS, window=10)
    subset = fit_frame_homographies(
        clicks, transforms, seg, PITCH_LANDMARKS, window=10, frames=[2, 3]
    )
    assert set(subset) == {2, 3}
    for f in (2, 3):
        assert np.allclose(subset[f].H, full[f].H)
        assert subset[f].n_points == full[f].n_points
        assert np.isclose(subset[f].residual, full[f].residual)


def test_fit_frames_subset_ignores_unknown_frames() -> None:
    n = 3
    interframe = _identity_chain(n)
    seg = build_segments(interframe, n)
    transforms = cumulative_transforms(interframe, seg)
    out = fit_frame_homographies(
        _clicks_one_per_frame()[:4], transforms, seg, PITCH_LANDMARKS,
        window=10, frames=[1, 99],
    )
    assert set(out) <= {1}
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -k subset -v`
Expected: FAIL — unexpected keyword `frames`.

- [ ] **Step 3: Implement**

In `fit_frame_homographies`:
- Signature: add `frames: Sequence[int] | None = None` after `min_points`.
- Docstring: add "frames: restrict fitting to these target frames (candidate
  clicks still come from anywhere within the window); None = all frames."
- Change the frames-array line from `frames = np.array(sorted(transforms), dtype=np.int64)` to:

```python
    if frames is None:
        target = sorted(transforms)
    else:
        target = sorted(set(transforms) & {int(f) for f in frames})
    if not target:
        return fits
    frame_arr = np.array(target, dtype=np.int64)
```

and rename every subsequent use of the local `frames` array to `frame_arr`
(`n = len(frame_arr)`, `m_stack`/`frame_seg`/`index_of` comprehensions,
`dist = np.abs(click_frame[:, None] - frame_arr[None, :])`,
`fits[int(frame_arr[gi])] = ...`). The click projection loop's
`index_of.get(c.frame)` lookup now misses clicks whose OWN frame is outside the
subset — but their projections into subset frames are still needed. Fix: build a
SEPARATE `src_transform` lookup from the full `transforms` mapping:

```python
    pos = np.full((k, n, 2), np.nan)
    for j, c in enumerate(clicks):
        src_m = transforms.get(c.frame)
        if src_m is None:
            continue
        ref = np.asarray(src_m, dtype=np.float64) @ np.array([c.x, c.y, 1.0])
        dst = m_inv @ ref
        pos[j] = dst[:, :2] / dst[:, 2:3]
```

(This replaces the old `index_of`-based source lookup; `index_of` may then be
unused — remove it if so.)

- [ ] **Step 4: Run all manual_anchor tests**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -v`
Expected: PASS (30 tests — 28 existing + 2 new). The existing tests pin that
`frames=None` behavior is unchanged.

- [ ] **Step 5: Gate + commit**

REPO ROOT `uv run mypy 2>&1 | tail -1` → Success; package `uv run ruff check src/soccer_vision/pitch/manual_anchor.py tests/test_pitch_manual_anchor.py` → clean.

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py packages/soccer-vision/tests/test_pitch_manual_anchor.py
git commit -m "feat(manual_anchor): frames-subset parameter on fit_frame_homographies"
```

---

## Task 2: Incremental + chunked recompute in LabelerState

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py`
- Test: `packages/soccer-vision/tests/test_labeler_state.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_labeler_state.py` (add `random` to top imports):

```python
def _full_recompute_reference(st: LabelerState) -> LabelerState:
    """Fresh state replaying st's clicks via bulk load (full recompute)."""
    ref = LabelerState(
        interframe={i: np.eye(3) for i in range(st.n_frames - 1)},
        n_frames=st.n_frames, size=st.size, window=st.window,
    )
    ref.add_clicks(list(st.clicks))
    return ref


def _assert_fits_equal(a: LabelerState, b: LabelerState) -> None:
    fa = {f: a.frame_homography(f) for f in range(a.n_frames)}
    fb = {f: b.frame_homography(f) for f in range(b.n_frames)}
    keys_a = {f for f, v in fa.items() if v is not None}
    keys_b = {f for f, v in fb.items() if v is not None}
    assert keys_a == keys_b
    for f in keys_a:
        va, vb = fa[f], fb[f]
        assert va is not None and vb is not None
        assert np.allclose(va.H, vb.H)
        assert np.isclose(va.residual, vb.residual)


def test_incremental_equals_full_after_mutation_sequence() -> None:
    rng = random.Random(0)
    st = _state(n=40)
    for step in range(30):
        action = rng.random()
        if action < 0.6 or not st.clicks:
            idx = rng.choice(_IDXS)
            px, py = PITCH_LANDMARKS[idx] * _SCALE
            st.add_click(frame=rng.randrange(40), kp_idx=idx,
                         x=float(px) + rng.random(), y=float(py) + rng.random())
        elif action < 0.8:
            st.remove_last()
        else:
            c = st.clicks[rng.randrange(len(st.clicks))]
            st.nudge_click(c.frame, c.kp_idx, c.x + 0.5, c.y + 0.5)
    _assert_fits_equal(st, _full_recompute_reference(st))


def test_remove_last_clears_lost_fits() -> None:
    st = _state()
    for f, idx in enumerate(_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        st.add_click(frame=f, kp_idx=idx, x=float(px), y=float(py))
    assert st.frame_homography(3) is not None
    for _ in range(3):
        st.remove_last()
    # only 3 landmarks remain -> no frame can fit
    assert all(st.frame_homography(f) is None for f in range(st.n_frames))


def test_bulk_add_chunked_matches_unchunked() -> None:
    st_small_chunk = _state(n=20)
    st_default = _state(n=20)
    clicks = []
    for f, idx in enumerate(_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        clicks.append(Click(frame=f * 3, kp_idx=idx, x=float(px), y=float(py)))
    st_small_chunk.add_clicks(clicks, chunk=4)
    st_default.add_clicks(clicks)
    _assert_fits_equal(st_small_chunk, st_default)
```

(`nudge_click` is implemented in Task 3 — for THIS task, add a temporary
minimal `nudge_click` only if you implement Tasks 2+3 in one commit; otherwise
adjust: implement `nudge_click` in this task as part of the mutation API since
the property test needs it. DECISION: implement `nudge_click` here in Task 2;
Task 3 covers autosave/sidecar/buckets only.)

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -k "incremental or clears_lost or chunked" -v`
Expected: FAIL (no `nudge_click`, no `chunk` param).

- [ ] **Step 3: Implement in `state.py`**

Replace `_recompute`/mutation methods with:

```python
    def _refit(self, frames: list[int]) -> None:
        """Refit exactly `frames`, replacing/removing their cached fits."""
        sub = fit_frame_homographies(
            self.clicks, self._transforms, self._segment_of,
            PITCH_LANDMARKS, window=self.window, frames=frames,
        )
        for f in frames:
            if f in sub:
                self._fits[f] = sub[f]
            else:
                self._fits.pop(f, None)

    def _affected(self, frame: int) -> list[int]:
        """Frames whose fit can change when a click at `frame` mutates."""
        seg = self._segment_of.get(frame)
        lo = max(0, frame - self.window)
        hi = min(self.n_frames - 1, frame + self.window)
        return [f for f in range(lo, hi + 1) if self._segment_of.get(f) == seg]

    def _recompute_chunked(self, chunk: int = 5000) -> None:
        self._fits = {}
        all_frames = sorted(self._transforms)
        for i in range(0, len(all_frames), chunk):
            part = all_frames[i:i + chunk]
            self._fits.update(fit_frame_homographies(
                self.clicks, self._transforms, self._segment_of,
                PITCH_LANDMARKS, window=self.window, frames=part,
            ))

    def add_click(self, frame: int, kp_idx: int, x: float, y: float) -> None:
        self.clicks.append(Click(frame=frame, kp_idx=kp_idx, x=x, y=y))
        self._refit(self._affected(frame))
        self._autosave()

    def add_clicks(self, clicks: Sequence[Click], *, chunk: int = 5000) -> None:
        """Bulk-add (resume/sidecar load) with one chunked full recompute."""
        self.clicks.extend(clicks)
        self._recompute_chunked(chunk=chunk)
        self._autosave()

    def remove_last(self) -> None:
        if self.clicks:
            removed = self.clicks.pop()
            self._refit(self._affected(removed.frame))
            self._autosave()

    def nudge_click(self, frame: int, kp_idx: int, x: float, y: float) -> bool:
        """Move the MOST RECENT click matching (frame, kp_idx). False if none."""
        for i in range(len(self.clicks) - 1, -1, -1):
            c = self.clicks[i]
            if c.frame == frame and c.kp_idx == kp_idx:
                self.clicks[i] = Click(frame=frame, kp_idx=kp_idx, x=x, y=y)
                self._refit(self._affected(frame))
                self._autosave()
                return True
        return False
```

For THIS task, add a no-op `_autosave` stub (Task 3 fills it):

```python
    def _autosave(self) -> None:
        """Persist clicks to the sidecar; wired in by autosave_path (Task 3)."""
```

(Keep the constructor unchanged in this task. `Sequence` already imported.)

- [ ] **Step 4: Run all state tests**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -v`
Expected: PASS (all prior + 3 new = 11).

- [ ] **Step 5: Gate + commit**

REPO ROOT mypy → Success; ruff on both files → clean.

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/state.py packages/soccer-vision/tests/test_labeler_state.py
git commit -m "feat(labeler): incremental window-scoped refit + chunked bulk recompute + nudge"
```

---

## Task 3: Autosave sidecar + status buckets

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py`
- Test: `packages/soccer-vision/tests/test_labeler_state.py`

- [ ] **Step 1: Write the failing tests**

Append (add `import json` to top imports if absent):

```python
def test_autosave_writes_and_loads_round_trip(tmp_path: Path) -> None:
    side = tmp_path / "v.clicks.json"
    st = LabelerState(
        interframe={i: np.eye(3) for i in range(5)},
        n_frames=6, size=(1920, 1080), window=10, autosave_path=side,
    )
    st.add_click(0, 0, 0.25, 0.5)
    st.add_click(1, 3, 0.75, 0.5)
    assert side.exists()
    loaded = clicks_from_sidecar(side)
    assert [(c.frame, c.kp_idx) for c in loaded] == [(0, 0), (1, 3)]
    assert np.isclose(loaded[0].x, 0.25)


def test_autosave_updates_on_undo_and_nudge(tmp_path: Path) -> None:
    side = tmp_path / "v.clicks.json"
    st = LabelerState(
        interframe={i: np.eye(3) for i in range(5)},
        n_frames=6, size=(1920, 1080), window=10, autosave_path=side,
    )
    st.add_click(0, 0, 0.25, 0.5)
    st.add_click(1, 3, 0.75, 0.5)
    st.nudge_click(1, 3, 0.8, 0.6)
    assert np.isclose(clicks_from_sidecar(side)[1].x, 0.8)
    st.remove_last()
    assert len(clicks_from_sidecar(side)) == 1


def test_status_buckets_worst_status_rule() -> None:
    st = _state(n=12)
    for f, idx in enumerate(_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        st.add_click(frame=f, kp_idx=idx, x=float(px), y=float(py))
    buckets, bucket_size = st.status_buckets(n_buckets=4)
    assert len(buckets) == 4
    assert bucket_size == 3
    full = st.status_list()
    for b, states in enumerate(
        [full[i:i + bucket_size] for i in range(0, 12, bucket_size)]
    ):
        if "red" in states:
            assert buckets[b] == "red"
        elif "yellow" in states:
            assert buckets[b] == "yellow"
        else:
            assert buckets[b] == "green"
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -k "autosave or buckets" -v`
Expected: FAIL (no `autosave_path` kwarg / `clicks_from_sidecar` / `status_buckets`).

- [ ] **Step 3: Implement in `state.py`**

Constructor: add keyword param `autosave_path: Path | None = None` (after
`residual_threshold`); store `self.autosave_path = autosave_path`. Add `import
json` and `import os` to top imports. Replace the `_autosave` stub:

```python
    def _autosave(self) -> None:
        """Atomically persist normalized clicks to the sidecar (if configured)."""
        if self.autosave_path is None:
            return
        payload = [
            {"frame": c.frame, "kp_idx": c.kp_idx, "x": c.x, "y": c.y}
            for c in self.clicks
        ]
        tmp = self.autosave_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.autosave_path)

    def status_buckets(self, *, n_buckets: int = 1200) -> tuple[list[str], int]:
        """Downsampled timeline: worst status per bucket (red > yellow > green)."""
        full = self.status_list()
        if len(full) <= n_buckets:
            return full, 1
        bucket_size = -(-len(full) // n_buckets)  # ceil division
        out: list[str] = []
        for i in range(0, len(full), bucket_size):
            chunk = full[i:i + bucket_size]
            if "red" in chunk:
                out.append("red")
            elif "yellow" in chunk:
                out.append("yellow")
            else:
                out.append("green")
        return out, bucket_size
```

Module-level loader (next to `clicks_from_keypoints_parquet`):

```python
def clicks_from_sidecar(path: Path) -> list[Click]:
    """Load the autosave sidecar (normalized coords) back into Clicks."""
    data = json.loads(Path(path).read_text())
    return [
        Click(frame=int(d["frame"]), kp_idx=int(d["kp_idx"]),
              x=float(d["x"]), y=float(d["y"]))
        for d in data
    ]
```

NOTE on the bucket test math: n=12, n_buckets=4 → bucket_size=3 ✓. When
`len(full) <= n_buckets` the full list is returned with bucket_size=1 (the
bake-off-scale regression path).

- [ ] **Step 4: Run all state tests**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -v`
Expected: PASS (14).

- [ ] **Step 5: Gate + commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/state.py packages/soccer-vision/tests/test_labeler_state.py
git commit -m "feat(labeler): autosave sidecar + status buckets"
```

---

## Task 4: Parallel chain precompute

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/chain.py`
- Test: `packages/soccer-vision/tests/test_labeler_chain.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_labeler_chain.py` (add imports `cv2`, and
`from soccer_vision.labeler.chain import compute_chain` to the top block):

```python
def _write_pan_video(path: Path, n: int = 40) -> None:
    """Synthetic textured video with a slow pan so registration succeeds."""
    rng = np.random.default_rng(0)
    big = (rng.random((300, 500, 3)) * 255).astype(np.uint8)
    big = cv2.GaussianBlur(big, (5, 5), 0)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))  # type: ignore[attr-defined]
    for i in range(n):
        x = 2 * i  # 2px pan per frame
        vw.write(big[20:260, x:x + 320])
    vw.release()


def test_parallel_chain_equals_serial(tmp_path: Path) -> None:
    video = tmp_path / "pan.mp4"
    _write_pan_video(video)
    serial_if, n1, size1 = compute_chain(video, cache_dir=tmp_path / "c1", workers=1)
    par_if, n2, size2 = compute_chain(video, cache_dir=tmp_path / "c2", workers=3)
    assert (n1, size1) == (n2, size2)
    assert set(serial_if) == set(par_if)
    for k in serial_if:
        assert np.allclose(serial_if[k], par_if[k], atol=1e-9)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_chain.py -k parallel -v`
Expected: FAIL — unexpected keyword `workers`.

- [ ] **Step 3: Implement in `chain.py`**

Add to imports: `import os` and `from multiprocessing import Pool`. Add a
TOP-LEVEL worker (spawn-safe — macOS uses spawn):

```python
def _chain_worker(
    args: tuple[str, int, int, float],
) -> dict[int, NDArray[np.float64]]:
    """Register pairs [start, end) of one video chunk (runs in a subprocess)."""
    video_path, start, end, downscale = args
    cap = cv2.VideoCapture(video_path)
    pos = 0

    def read_frame(idx: int) -> NDArray[np.uint8] | None:
        nonlocal pos
        if idx < pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            pos = idx
        while pos < idx:
            if not cap.grab():
                return None
            pos += 1
        ok, frame = cap.read()
        pos += 1
        return frame if ok else None

    boxes = pd.DataFrame(
        columns=["frame", "class", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    )
    try:
        return compute_interframe_homographies(
            read_frame, set(range(start, end)), boxes, downscale=downscale
        )
    finally:
        cap.release()
```

In `compute_chain`: add `workers: int | None = None` keyword param (docstring:
"parallel chunked registration; None = cpu_count()-1; forced to 1 when
player_boxes is given — masking is per-frame data the workers don't carry").
Replace the single-pass registration block (the `cap = ...` through
`interframe_px = compute_interframe_homographies(...)` section) with:

```python
    n_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    if player_boxes is not None:
        n_workers = 1
    if n_workers <= 1:
        # (existing serial path, unchanged: open cap, read_frame, ORB pass)
        ...existing code...
    else:
        n_pairs = n_frames - 1
        bounds = np.linspace(0, n_pairs, n_workers + 1).astype(int)
        jobs = [
            (str(video_path), int(bounds[i]), int(bounds[i + 1]), downscale)
            for i in range(n_workers)
            if bounds[i] < bounds[i + 1]
        ]
        interframe_px = {}
        with Pool(processes=len(jobs)) as pool:
            for part in pool.imap_unordered(_chain_worker, jobs):
                interframe_px.update(part)
                print(f"chain: {len(interframe_px)}/{n_pairs} pairs registered")
```

(The serial branch keeps the EXACT existing code; only the parallel branch is
new. `n_frames`, `width`, `height` are read from a probe `VideoCapture` before
branching, as the existing code already does.)

- [ ] **Step 4: Run chain tests**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_chain.py -v`
Expected: PASS (5: 4 existing + 1 new). The parallel test spawns subprocesses —
runtime a few seconds.

- [ ] **Step 5: Gate + commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/chain.py packages/soccer-vision/tests/test_labeler_chain.py
git commit -m "feat(labeler): parallel chunked chain precompute (workers param)"
```

---

## Task 5: Server payloads, nudge endpoint, wiring; CLI --workers

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/server.py`
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/__main__.py`
- Test: `packages/soccer-vision/tests/test_labeler_server.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_labeler_server.py`, the `_serve()` helper's `LabelerState(...)`
stays unchanged. Update the existing `test_click_then_state_reports_coverage`
assertion `assert len(state["status"]) == 6` to:

```python
        assert len(state["status_buckets"]) == 6
        assert state["bucket_size"] == 1
```

Append:

```python
def test_frame_h_includes_residual_and_n_points() -> None:
    httpd, _ = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        for f, idx in enumerate([0, 3, 6, 11, 16, 19]):
            px, py = PITCH_LANDMARKS[idx] * 1000.0
            _post(f"{base}/api/click",
                  {"frame": f, "kp_idx": int(idx), "x": float(px), "y": float(py)})
        fh = _get(f"{base}/api/frame_h/3")
        assert fh["h"] is not None
        assert fh["residual"] is not None and fh["residual"] < 0.05
        assert fh["n_points"] == 6
        assert _get(f"{base}/api/frame_h/0")["h"] is not None
    finally:
        httpd.shutdown()


def test_nudge_endpoint_moves_click() -> None:
    httpd, state = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _post(f"{base}/api/click", {"frame": 0, "kp_idx": 0, "x": 0.2, "y": 0.3})
        out = _post(f"{base}/api/nudge", {"frame": 0, "kp_idx": 0, "x": 0.4, "y": 0.5})
        assert out["n_clicks"] == 1
        assert np.isclose(state.clicks[0].x, 0.4)
    finally:
        httpd.shutdown()
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_server.py -v`
Expected: FAIL (payload has `status`, no nudge route).

- [ ] **Step 3: Implement**

In `server.py` `_state_payload`: replace the `"status": state.status_list(),`
entry with:

```python
                **dict(zip(("status_buckets", "bucket_size"),
                           state.status_buckets(), strict=True)),
```

…or more plainly:

```python
            buckets, bucket_size = state.status_buckets()
            return {
                "n_frames": state.n_frames,
                "coverage": state.coverage(),
                "status_buckets": buckets,
                "bucket_size": bucket_size,
                "n_clicks": len(state.clicks),
                "landmark_names": landmark_names,
                "landmark_xy": xy,
            }
```

`/api/frame_h/<i>` handler: include residual/n_points:

```python
            elif path.startswith("/api/frame_h/"):
                idx = int(path.rsplit("/", 1)[1])
                fit = state.frame_homography(idx)
                self._json({
                    "h": None if fit is None
                    else [float(v) for v in np.asarray(fit.H).reshape(9)],
                    "residual": None if fit is None else fit.residual,
                    "n_points": None if fit is None else fit.n_points,
                })
```

`do_POST`: add before the 404 fallback:

```python
            elif self.path == "/api/nudge":
                found = state.nudge_click(
                    int(payload["frame"]), int(payload["kp_idx"]),
                    float(payload["x"]), float(payload["y"]),
                )
                if found:
                    self._json(self._state_payload())
                else:
                    self._json({"error": "no click at frame/kp_idx"}, code=404)
```

`run()`: add params `workers: int | None = None` (pass to `compute_chain`) and
wire autosave: after `compute_chain` returns, compute the sidecar path and
construct the state with it; precedence resume > sidecar:

```python
    interframe, n_frames, size = compute_chain(video_path, workers=workers)
    cache_dir = Path(video_path).parent / ".sv_labeler_cache"
    sidecar = cache_dir / f"{Path(video_path).stem}.clicks.json"
    state = LabelerState(
        interframe=interframe, n_frames=n_frames, size=size, window=window,
        autosave_path=sidecar,
    )
    if resume is not None:
        state.add_clicks(clicks_from_keypoints_parquet(resume, size))
        print(f"resumed {len(state.clicks)} clicks from {resume}")
    elif sidecar.exists():
        state.add_clicks(clicks_from_sidecar(sidecar))
        print(f"restored {len(state.clicks)} clicks from autosave {sidecar}")
```

(Import `clicks_from_sidecar` next to the existing state imports inside `run`.
NOTE: `compute_chain`'s default `cache_dir` is exactly
`Path(video_path).parent / ".sv_labeler_cache"` — keep the sidecar path
consistent with it.)

In `__main__.py`: add `--workers` (`type=int, default=None`, help "parallel
chain precompute workers (default: cores-1)") and pass `workers=args.workers`
to `run(...)`.

- [ ] **Step 4: Run server tests**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_server.py -v`
Expected: PASS (5: 3 existing—1 updated—+ 2 new).

- [ ] **Step 5: Gate + commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/server.py packages/soccer-vision/src/soccer_vision/labeler/__main__.py packages/soccer-vision/tests/test_labeler_server.py
git commit -m "feat(labeler): bucketed state payload, residual in frame_h, nudge endpoint, autosave+workers wiring"
```

---

## Task 6: Frontend — buckets, residual, drag-nudge

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/static/app.js`
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/static/index.html`

No unit tests (canvas UI; verified in Task 7). Make EXACTLY these changes:

- [ ] **Step 1: index.html**

Change the stats span `<span>frame <b id="res">—</b></span>` to
`<span>residual <b id="res">—</b></span>`.

- [ ] **Step 2: app.js — buckets**

- Globals: replace `let status = [];` usage semantics with buckets: add
  `let bucketSize = 1;` (keep variable name `status` for the bucket array to
  minimize churn).
- `applyState(st)`: set `status = st.status_buckets; bucketSize = st.bucket_size;`
  (replacing `status = st.status;`).
- `jumpRed(dir)`: operate on bucket indices; current bucket =
  `Math.floor(cur / bucketSize)`; scan buckets in `dir`; on finding a red
  bucket b, `loadFrame(Math.min(st_n_frames - 1, b * bucketSize))` — store
  `nFrames` from state (`let nFrames = 0;` set in applyState) and clamp:

```javascript
function jumpRed(dir){
  let b = Math.floor(cur / bucketSize) + dir;
  while(b >= 0 && b < status.length){
    if(status[b] === "red"){ loadFrame(Math.min(nFrames - 1, b * bucketSize)); return; }
    b += dir;
  }
}
```

- The timeline renderer already iterates `status` — unchanged (now buckets).

- [ ] **Step 3: app.js — numeric residual**

In `loadFrame(i)`, the `/api/frame_h/` fetch now returns `{h, residual, n_points}`:

```javascript
  const fh = await api(`/api/frame_h/${i}`);
  curH = fh.h;
  const resEl = document.getElementById("res");
  if (fh.residual == null) { resEl.textContent = "—"; resEl.style.color = ""; }
  else {
    resEl.textContent = `${fh.residual.toFixed(3)} (${fh.n_points} pts)`;
    resEl.style.color = fh.residual <= 0.05 ? "#39d98a" : "#ffb454";
  }
```

(Replace the previous `res` update that wrote the status word.)

- [ ] **Step 4: app.js — drag-nudge**

Replace `canvas.onclick` with mousedown/mousemove/mouseup handlers:

```javascript
let dragging = null;  // {kp_idx} while dragging an existing dot
let didDrag = false;

function canvasNorm(e){
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
}

canvas.onmousedown = (e) => {
  const [x, y] = canvasNorm(e);
  for (const c of clicks) {
    if (c.frame !== cur) continue;
    const dx = (c.x - x) * canvas.width;
    const dy = (c.y - y) * canvas.height;
    if (Math.hypot(dx, dy) < 10) { dragging = { kp_idx: c.kp_idx, c }; return; }
  }
  dragging = null;
};

canvas.onmousemove = (e) => {
  if (!dragging) return;
  didDrag = true;
  const [x, y] = canvasNorm(e);
  dragging.c.x = x; dragging.c.y = y;   // live local preview
  drawFrame();
};

canvas.onmouseup = async (e) => {
  if (dragging && didDrag) {
    const [x, y] = canvasNorm(e);
    applyState(await postJSON("/api/nudge",
      { frame: cur, kp_idx: dragging.kp_idx, x, y }));
    const fh = await api(`/api/frame_h/${cur}`); curH = fh.h; drawFrame();
  }
  dragging = null;
};

canvas.onclick = async (e) => {
  if (didDrag) { didDrag = false; return; }   // suppress click after a drag
  const [x, y] = canvasNorm(e);
  clicks.push({ frame: cur, kp_idx: armed, x, y }); placed.add(armed);
  applyState(await postJSON("/api/click", { frame: cur, kp_idx: armed, x, y }));
  const fh = await api(`/api/frame_h/${cur}`); curH = fh.h; drawFrame();
};
```

(Keep the rest — undo/grid/export/scrub — unchanged, but make sure `nFrames`
is declared and set in `applyState`: `nFrames = st.n_frames;`.)

- [ ] **Step 5: Sanity + commit**

`node --check` is unavailable; sanity via the server test suite (static files
just need to exist) + Task 7 manual run.
Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_server.py -q` → pass.

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/static/
git commit -m "feat(labeler): bucketed timeline, numeric residual, drag-to-nudge"
```

---

## Task 7: Acceptance (controller-run + Patrick)

Not a subagent task.

- [ ] **Bake-off regression (controller):** relaunch the labeler on
  `~/sv-labeler/clip.mp4` with `--resume ~/sv-labeler/out/keypoints.parquet`;
  confirm via `/api/state` that coverage equals the pre-feature value (~82%
  green at frame level — compare `coverage`, which is bucket-independent), a
  click POST round-trips in < 1 s, and the sidecar file appears after the first
  mutation. Kill the server, relaunch WITHOUT `--resume`, confirm clicks
  restored from the sidecar.
- [ ] **Full-game precompute benchmark (controller):** run the chain precompute
  on `~/sv-labeler/training.mp4`... (the 32-min video — copy from wherever
  Patrick stored it; if absent, ask). Time it with default workers; target
  well under an hour. Report the measured wall-clock.
- [ ] **Full-game labeling session (Patrick):** label `training.mp4` passages;
  verify click latency, autosave behavior, nudge UX, bucketed timeline.

---

## Final verification

- [ ] `cd packages/soccer-vision && uv run pytest -q` — all pass.
- [ ] REPO ROOT `uv run mypy 2>&1 | tail -1` — Success.
- [ ] `uv run ruff check src/ tests/` — clean.
