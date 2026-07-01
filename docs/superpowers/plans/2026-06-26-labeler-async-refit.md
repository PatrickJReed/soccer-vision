# Labeler Responsive Refit (async worker + scoped line-refine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every labeler click return in <~50ms by moving the windowed pose refit off the request thread onto a background worker, with the viewed frame fit synchronously and line-refine scoped to a band.

**Architecture:** A generic `RefitWorker` daemon thread owns a dirty frame-set + revision counter and runs a compute/apply callback pair, cancelling a stale pass when a newer edit arrives. `LabelerState` fits the clicked frame synchronously (instant overlay) then `mark_dirty`s the affected window for the worker; reads/writes of `_fits` are guarded by one `RLock`. Line clicks propagate only within `±line_band` so `refine_pose` runs on a bounded set of frames. The server stays serial `HTTPServer`; the frontend polls `/api/state` while a refit is pending.

**Tech Stack:** Python 3 (`threading`, stdlib `http.server`), numpy, OpenCV (SQPNP), scipy (`least_squares` via `refine_pose`); vanilla-JS canvas frontend; pytest; mypy (strict, run from repo root); ruff.

---

## Conventions for every task

- All commands run from `packages/soccer-vision/` **except mypy**, which runs from the repo root `/Users/patrickreed/Sandbox/soccer-vision` (a known double-cd artifact makes mypy report bogus errors when run from inside the package).
- Branch is `feat/labeler-async-refit` (already created). Do NOT create a new branch.
- Test-data helper: `tests/test_labeler_state.py` already has `_pan_session(n) -> (interframe, _poses, clicks)` and `_K`/`_look_at`. Reuse them.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/soccer_vision/labeler/refit_worker.py` (new) | Generic background "recompute a dirty set" thread: dirty-set, revision/cancellation, `wait_idle`, `pending`. Calibration-agnostic. | 1 |
| `tests/test_labeler_refit_worker.py` (new) | `RefitWorker` unit tests with trivial compute/apply (no calibration, no sleeps). | 1 |
| `src/soccer_vision/labeler/state.py` | Refactor refit into `_compute_poses`/`_compute_dirty`/`_apply_fits`; add `RLock` + `line_band` scoping (Task 2); construct/start the worker, `_refit_one`, rewire interactive mutators, lock-guard reads, `pending`/`wait_idle`/`stop_worker`, `export` waits idle (Task 3). | 2, 3 |
| `tests/test_labeler_state.py` | line-band scoping test (Task 2); instant-current-frame + async-equivalence + pending tests (Task 3). | 2, 3 |
| `src/soccer_vision/labeler/server.py` | `pending` in `/api/state`; `run()` starts/stops the worker. | 4 |
| `tests/test_labeler_server.py` | `/api/state` exposes `pending`. | 4 |
| `src/soccer_vision/labeler/static/app.js` | Poll `/api/state` while `pending > 0`; stop at 0. | 5 |

---

## Task 1: `RefitWorker` — generic background dirty-set recompute

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/labeler/refit_worker.py`
- Test: `packages/soccer-vision/tests/test_labeler_refit_worker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_labeler_refit_worker.py`:

```python
"""RefitWorker unit tests: deterministic via wait_idle(), no sleeps, no calibration."""

from __future__ import annotations

import threading

from soccer_vision.labeler.refit_worker import RefitWorker


def test_marks_get_computed_and_applied() -> None:
    applied: dict[int, int] = {}

    def compute(frames, is_cancelled):
        return {f: f * 10 for f in frames}

    def apply(results):
        applied.update(results)

    w = RefitWorker(compute, apply)
    w.start()
    try:
        w.mark_dirty([1, 2, 3])
        w.wait_idle(timeout=5)
        assert applied == {1: 10, 2: 20, 3: 30}
        assert w.pending() == 0
    finally:
        w.stop()


def test_overlapping_marks_union() -> None:
    seen_batches: list[list[int]] = []

    def compute(frames, is_cancelled):
        seen_batches.append(sorted(frames))
        return {f: f for f in frames}

    w = RefitWorker(compute, lambda results: None)
    w.start()
    try:
        w.mark_dirty([1, 2])
        w.mark_dirty([2, 3])
        w.wait_idle(timeout=5)
        # every requested frame was computed at least once (union, nothing dropped)
        computed = {f for batch in seen_batches for f in batch}
        assert computed == {1, 2, 3}
    finally:
        w.stop()


def test_cancellation_requeues_and_eventually_applies() -> None:
    # First compute blocks until a second mark_dirty bumps the revision, so the
    # first pass is cancelled; the worker must re-run and eventually apply.
    bumped = threading.Event()
    applied: dict[int, int] = {}
    calls = {"n": 0}

    def compute(frames, is_cancelled):
        calls["n"] += 1
        if calls["n"] == 1:
            bumped.wait(timeout=5)      # let the test add a newer mark first
            if is_cancelled():
                return None             # superseded -> discard
        return {f: f for f in frames}

    def apply(results):
        applied.update(results)

    w = RefitWorker(compute, apply)
    w.start()
    try:
        w.mark_dirty([1])
        w.mark_dirty([2])              # bumps revision while pass 1 is inside compute
        bumped.set()
        w.wait_idle(timeout=5)
        assert applied == {1: 1, 2: 2}  # nothing lost despite the cancel
    finally:
        w.stop()


def test_stop_is_idempotent_and_joins() -> None:
    w = RefitWorker(lambda frames, c: {}, lambda r: None)
    w.start()
    w.stop()
    w.stop()  # no error on double stop
    assert w.pending() == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_refit_worker.py -v`
Expected: FAIL (`No module named 'soccer_vision.labeler.refit_worker'`).

- [ ] **Step 3: Implement `RefitWorker`**

Create `src/soccer_vision/labeler/refit_worker.py`:

```python
"""A generic background "recompute a dirty set" worker.

Calibration-agnostic: owns one daemon thread, a dirty frame-set, and a monotonic
revision counter. On each edit the owner calls mark_dirty(frames); the worker
snapshots the set, runs the compute callback OFF the lock (so callers/readers are
not blocked), and merges via the apply callback. A newer mark_dirty bumps the
revision, so an in-flight pass can cancel itself (is_cancelled()) and be re-run with
the unioned set — rapid edits coalesce instead of piling up. wait_idle() blocks until
the set is drained (used by tests and by export).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T")
# Compute returns {frame: result} for the requested frames, or None if it detected
# cancellation mid-pass and the partial result must be discarded.


class RefitWorker(Generic[T]):
    def __init__(
        self,
        compute: Callable[[list[int], Callable[[], bool]], dict[int, T] | None],
        apply: Callable[[dict[int, T]], None],
    ) -> None:
        self._compute = compute
        self._apply = apply
        self._cv = threading.Condition()
        self._dirty: set[int] = set()
        self._revision = 0
        self._inflight = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="refit-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            if not self._running:
                return
            self._running = False
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def mark_dirty(self, frames: Iterable[int]) -> None:
        batch = list(frames)
        if not batch:
            return
        with self._cv:
            self._dirty.update(batch)
            self._revision += 1
            self._idle.clear()
            self._cv.notify_all()

    def pending(self) -> int:
        with self._cv:
            return len(self._dirty) + self._inflight

    def wait_idle(self, timeout: float | None = None) -> None:
        self._idle.wait(timeout)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._running and not self._dirty:
                    self._idle.set()
                    self._cv.wait()
                if not self._running:
                    return
                frames = sorted(self._dirty)
                self._dirty.clear()
                rev = self._revision
                self._inflight = len(frames)

            def is_cancelled(_r: int = rev) -> bool:
                with self._cv:
                    return self._revision != _r

            results = self._compute(frames, is_cancelled)

            if results is None:
                # Superseded: put the frames back so they are recomputed with the
                # newer edit's set; do not apply the discarded partial.
                with self._cv:
                    self._dirty.update(frames)
                    self._inflight = 0
                continue

            self._apply(results)
            with self._cv:
                self._inflight = 0
                if not self._dirty:
                    self._idle.set()
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_refit_worker.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/labeler/refit_worker.py tests/test_labeler_refit_worker.py`
Then from REPO ROOT: `cd /Users/patrickreed/Sandbox/soccer-vision && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/refit_worker.py packages/soccer-vision/tests/test_labeler_refit_worker.py
git commit -m "feat(labeler): RefitWorker — background dirty-set recompute with cancellation"
```

---

## Task 2: `LabelerState` refit refactor + `RLock` + scoped line-refine (still synchronous)

This task introduces NO threading. It extracts the refit body into reusable
`_compute_poses` / `_compute_dirty` / `_apply_fits`, adds the `RLock` and the
`line_band` param, and scopes line propagation to `line_band`. All existing behavior
stays synchronous, so the existing suite must remain green.

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py`
- Test: `packages/soccer-vision/tests/test_labeler_state.py`

- [ ] **Step 1: Write the failing test (line-band scoping)**

Append to `tests/test_labeler_state.py`:

```python
def test_line_obs_scoped_to_line_band() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080),
                      window=360, line_band=1)   # band of +/-1 frame
    st.add_clicks(clicks)
    st.line_clicks.append(LineClick(frame=4, line_id="midline", x=0.5, y=0.5))
    # _line_obs over all frames: only frames within +/-1 of frame 4 carry the line obs
    obs = st._line_obs(list(range(9)))
    carrying = sorted(f for f, lst in obs.items() if lst)
    assert carrying == [3, 4, 5]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -k line_obs_scoped -v`
Expected: FAIL (`LabelerState.__init__` has no `line_band`; or scoping uses `window`, so far frames also carry obs).

- [ ] **Step 3: Implement the refactor**

In `state.py`:

(a) Add `import threading` at the top (after `import os`).

(b) Add the `line_band` param to `__init__` (after `window: int = 360,`):
```python
        line_band: int = 60,
```
and store it + the lock + a chunk size in `__init__` (after `self.window = window`):
```python
        self.line_band = line_band
        self._lock = threading.RLock()
        self._refit_chunk = 256
```

(c) Change `_line_obs` to propagate lines with `line_band` instead of `window`:
```python
    def _line_obs(self, frames: Sequence[int] | None) -> dict[int, list[tuple[str, float, float]]]:
        w, h = self.size
        prop = propagate_line_clicks(
            self.line_clicks, self._transforms, self._segment_of,
            window=self.line_band, frames=frames)
        return {f: [(lid, x * w, y * h) for (lid, x, y) in lst] for f, lst in prop.items()}
```

(d) Add the three reusable methods (place them right before the current `_refit`):
```python
    def _compute_poses(self, frames: Sequence[int]) -> dict[int, FramePose]:
        """SQPNP (+ selective refine) for `frames`. Snapshots inputs under the lock,
        then solves OFF the lock (numpy/scipy release the GIL) so readers aren't blocked.
        Returns only the frames that solved (>= min_points)."""
        if not self._calibrated or self._K is None:
            return {}
        with self._lock:
            clicks = self._active_clicks()
            k = self._K
            line_obs = self._line_obs(frames)
        return poses_by_click_propagation(
            clicks, self._transforms, self._segment_of, k, self.size,
            window=self.window, frames=frames, line_obs=line_obs)

    def _compute_dirty(
        self, frames: Sequence[int], is_cancelled: Callable[[], bool]
    ) -> dict[int, FramePose | None] | None:
        """Compute `frames` in chunks (checking cancellation between chunks). Returns a
        map over EVERY requested frame -> pose-or-None (None = no longer solvable, so the
        applier pops any stale fit). None return = the whole pass was cancelled."""
        out: dict[int, FramePose | None] = {}
        ordered = list(frames)
        for i in range(0, len(ordered), self._refit_chunk):
            if is_cancelled():
                return None
            chunk = ordered[i:i + self._refit_chunk]
            solved = self._compute_poses(chunk)
            for f in chunk:
                out[f] = solved.get(f)
        return out

    def _apply_fits(self, results: dict[int, FramePose | None]) -> None:
        """Merge computed poses into _fits under the lock: set solved frames, pop the rest."""
        with self._lock:
            for f, pose in results.items():
                if pose is None:
                    self._fits.pop(f, None)
                else:
                    self._fits[f] = self._calib_frame(pose)
```
Add `Callable` to the `collections.abc` import line (it currently imports `Mapping, Sequence`):
```python
from collections.abc import Callable, Mapping, Sequence
```

(e) Reimplement `_refit` and `_recompute_all` on top of the new helpers (keep them
synchronous for now; callers unchanged this task):
```python
    def _refit(self, frames: list[int]) -> None:
        if not self._calibrated or self._K is None:
            with self._lock:
                for f in frames:
                    self._fits.pop(f, None)
            return
        result = self._compute_dirty(frames, lambda: False)
        assert result is not None  # never cancelled with a constant-False predicate
        self._apply_fits(result)

    def _recompute_all(self, chunk: int = 5000) -> None:
        with self._lock:
            self._fits = {}
        if not self._calibrated or self._K is None:
            return
        result = self._compute_dirty(sorted(self._transforms), lambda: False)
        assert result is not None
        self._apply_fits(result)
```
(The `chunk` param is retained for signature compatibility with `add_clicks` but is no
longer used — chunking is now `self._refit_chunk`. Leave the `add_clicks` call site as is.)

(f) Guard the status reads with the lock so they are consistent once the worker writes
concurrently (Task 3). Replace `_status_of` and `frame_homography`:
```python
    def _status_of(self, f: int) -> str:
        with self._lock:
            cf = self._fits.get(f)
        if cf is None:
            return "red"
        return "green" if cf.residual <= self.residual_px_threshold else "yellow"

    def frame_homography(self, frame: int) -> CalibFrame | None:
        with self._lock:
            return self._fits.get(frame)
```
(`coverage`, `status_list`, `status_buckets` call `_status_of`, so they inherit the guard.)
In `export`, the green-check + `self._fits[f]` read should also be consistent; wrap the
loop's fit access — replace the export loop body's `cf = self._fits[f]` with a locked read:
```python
        for f in range(self.n_frames):
            if self._status_of(f) != "green":
                continue
            with self._lock:
                cf = self._fits[f]
            conf = float(np.clip(1.0 - cf.residual / self.residual_px_threshold, 0.0, 1.0))
            entries[f] = HomographyEntry(
                denormalize_homography(cf.H, self.size), "manual", conf)
```

- [ ] **Step 4: Run to verify pass + no regression**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -v`
Expected: the new `test_line_obs_scoped_to_line_band` PASSES; all existing state tests still pass.
Run: `cd packages/soccer-vision && uv run pytest -k labeler -q`
Expected: all pass.

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/labeler/state.py tests/test_labeler_state.py`
Then from REPO ROOT: `cd /Users/patrickreed/Sandbox/soccer-vision && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/state.py packages/soccer-vision/tests/test_labeler_state.py
git commit -m "refactor(labeler): extract _compute_poses/_compute_dirty/_apply_fits + RLock + line_band scoping"
```

---

## Task 3: `LabelerState` async wiring (worker + instant current frame)

Now make the interactive mutators non-blocking: fit the clicked frame synchronously,
enqueue the affected window to the worker.

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py`
- Test: `packages/soccer-vision/tests/test_labeler_state.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_labeler_state.py`:

```python
def test_add_click_fits_current_frame_synchronously() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080), window=360)
    try:
        # bootstrap on all but the last click, synchronously (bulk)
        st.add_clicks(clicks[:-1])
        last = clicks[-1]
        st.add_click(last.frame, last.kp_idx, last.x, last.y)
        # the clicked frame is fit on return, BEFORE the background worker drains
        assert st.frame_homography(last.frame) is not None
    finally:
        st.stop_worker()


def test_async_refit_matches_synchronous_full_recompute() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080),
                      window=360, line_band=60)
    try:
        for c in clicks:
            st.add_click(c.frame, c.kp_idx, c.x, c.y)
        st.wait_idle(timeout=10)
        fits_async = {f: cf.H.copy() for f, cf in st._fits.items()}
        # synchronous full recompute on the SAME calibrated state (same K)
        ref = st._compute_dirty(sorted(st._transforms), lambda: False)
        assert ref is not None
        expected = {f: st._calib_frame(p).H for f, p in ref.items() if p is not None}
        assert set(fits_async) == set(expected)
        for f in expected:  # np is imported at module top (used by _pan_session/_K)
            np.testing.assert_allclose(fits_async[f], expected[f], atol=1e-6)
    finally:
        st.stop_worker()


def test_pending_drains_to_zero() -> None:
    interframe, _poses, clicks = _pan_session(40)
    st = LabelerState(interframe=interframe, n_frames=40, size=(1920, 1080), window=360)
    try:
        for c in clicks:
            st.add_click(c.frame, c.kp_idx, c.x, c.y)
        st.wait_idle(timeout=10)
        assert st.pending() == 0
    finally:
        st.stop_worker()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -k "synchronously or async_refit or pending_drains" -v`
Expected: FAIL (`LabelerState` has no `stop_worker`/`wait_idle`/`pending`).

- [ ] **Step 3: Implement the worker wiring**

In `state.py`:

(a) Import the worker (after the `manual_anchor` import block):
```python
from soccer_vision.labeler.refit_worker import RefitWorker
```

(b) In `__init__`, after `self._to_norm = ...`, construct and start the worker (annotate the
attribute so the generic parameter is fixed to `FramePose | None`):
```python
        self._worker: RefitWorker[FramePose | None] = RefitWorker(
            self._compute_dirty, self._apply_fits)
        self._worker.start()
```

(c) Add the lifecycle/passthrough methods and the synchronous one-frame fit (place near
`_apply_fits`):
```python
    def _refit_one(self, frame: int) -> None:
        """Synchronously fit just `frame` (instant overlay for the clicked frame)."""
        result = self._compute_dirty([frame], lambda: False)
        assert result is not None
        self._apply_fits(result)

    def wait_idle(self, timeout: float | None = None) -> None:
        self._worker.wait_idle(timeout)

    def pending(self) -> int:
        return self._worker.pending()

    def stop_worker(self) -> None:
        self._worker.stop()
```

(d) Rewire the interactive mutators to "instant current frame + enqueue window". Replace
`add_click`:
```python
    def add_click(self, frame: int, kp_idx: int, x: float, y: float) -> None:
        with self._lock:
            self.clicks.append(Click(frame=frame, kp_idx=kp_idx, x=x, y=y))
            self._seq.append("pt")
        if not self._calibrated and self._try_bootstrap():
            self._refit_one(frame)                       # clicked frame instant
            self._worker.mark_dirty(range(self.n_frames))  # everything else in background
        elif self._calibrated:
            self._refit_one(frame)
            self._worker.mark_dirty(self._affected(frame))
        self._autosave()
```
Replace `add_line_click`:
```python
    def add_line_click(self, frame: int, line_id: str, x: float, y: float) -> None:
        with self._lock:
            self.line_clicks.append(LineClick(frame=frame, line_id=line_id, x=x, y=y))
            self._seq.append("ln")
        if self._calibrated:
            self._refit_one(frame)
            self._worker.mark_dirty(self._affected(frame))
        self._autosave()
```
Replace the refit section of `remove_last` (the `if self._calibrated: self._refit(...)` part):
```python
        if self._calibrated:
            self._refit_one(removed_frame)
            self._worker.mark_dirty(self._affected(removed_frame))
        self._autosave()
```
Replace the refit section of `nudge_click` (the `if self._calibrated: self._refit(...)`):
```python
                if self._calibrated:
                    self._refit_one(frame)
                    self._worker.mark_dirty(self._affected(frame))
                self._autosave()
                return True
```
Replace `recalibrate` so the full recompute is enqueued (K + outliers stay synchronous):
```python
    def recalibrate(self) -> bool:
        self._calibrated = False
        self._K = None
        self._outliers = {}
        if not self._try_bootstrap():
            with self._lock:
                self._fits = {}
            return False
        with self._lock:
            self._fits = {}
        self._worker.mark_dirty(range(self.n_frames))
        return True
```
(`add_clicks` is the bulk/boot path and STAYS synchronous — it still calls
`self._recompute_all(...)`. Leave it unchanged.)

(e) Make `export` wait for a settled fit — at the very top of `export`, before any writes:
```python
    def export(self, out_dir: Path) -> None:
        self.wait_idle(timeout=30)
        out = Path(out_dir)
        ...
```

(f) Mark the now-unused synchronous `_refit` for removal: it is no longer called by any
mutator (all interactive callers use `_refit_one` + `mark_dirty`; bulk uses
`_recompute_all`). Delete the `_refit` method added in Task 2 step 3(e). Verify with:
```bash
cd packages/soccer-vision && grep -rn "self._refit(" src/ tests/
```
Expected: no matches (only `_refit_one`, `_recompute_all`, `_compute_dirty`). If any match
remains, convert it to `_refit_one(frame)` + `mark_dirty(self._affected(frame))`.

- [ ] **Step 4: Run to verify pass + no regression**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -v`
Expected: new async tests PASS; existing tests still pass.
Run: `cd packages/soccer-vision && uv run pytest -k labeler -q`
Expected: all pass. (If any existing test that does an interactive `add_click`/`add_line_click`
then asserts a FAR in-window frame's coverage now fails, add `st.wait_idle()` before that
assertion — the clicked frame is synchronous but far frames settle in the background.)

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/labeler/state.py tests/test_labeler_state.py`
Then from REPO ROOT: `cd /Users/patrickreed/Sandbox/soccer-vision && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/state.py packages/soccer-vision/tests/test_labeler_state.py
git commit -m "feat(labeler): async windowed refit — instant current frame, background worker"
```

---

## Task 4: Server — expose `pending`, start/stop the worker

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/server.py`
- Test: `packages/soccer-vision/tests/test_labeler_server.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_labeler_server.py`:

```python
def test_state_payload_includes_pending() -> None:
    httpd, _ = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        out = _get(f"{base}/api/state")
        assert "pending" in out
        assert isinstance(out["pending"], int)
    finally:
        httpd.shutdown()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_server.py -k pending -v`
Expected: FAIL (`KeyError: 'pending'`).

- [ ] **Step 3: Implement**

In `server.py`:

(a) Add `pending` to `_state_payload` (inside the returned dict, after `"line_names": lines,`):
```python
                "pending": state.pending(),
```

(b) In `run()`, stop the worker on shutdown so the process exits cleanly. The current
`run()` ends with:
```python
    try:
        httpd.serve_forever()
    finally:
        cap.release()
```
Change the `finally` to also stop the worker:
```python
    try:
        httpd.serve_forever()
    finally:
        cap.release()
        state.stop_worker()
```
(The worker auto-starts in `LabelerState.__init__`, so no explicit start is needed in `run()`.)

- [ ] **Step 4: Run + lint + typecheck**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_server.py -v`
Expected: all pass (incl. the new `pending` test).
Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/labeler/server.py tests/test_labeler_server.py`
Then from REPO ROOT: `cd /Users/patrickreed/Sandbox/soccer-vision && uv run mypy`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/server.py packages/soccer-vision/tests/test_labeler_server.py
git commit -m "feat(labeler): expose refit pending count + stop worker on shutdown"
```

---

## Task 5: Frontend — poll `/api/state` while a refit is pending

No JS unit harness exists; this task is verified by a brace/identifier audit (node is
unavailable) and Patrick's interactive check. The change is small: after a click, if the
returned `pending` is > 0, poll `/api/state` (~750 ms) to fill the coverage timeline, and
stop polling when `pending` reaches 0.

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/static/app.js`

- [ ] **Step 1: Add the poll loop + a `pending` field on applyState**

Read `app.js` first to confirm identifiers (`applyState`, `api`, `cur`, `curH`, `drawFrame`).
Near the other top-level `let` globals (after `let ... lineClicks = [];`), add:
```javascript
let pendingPoll = null;   // setInterval handle while a background refit is in flight
```
In `applyState(st)`, after the existing `nFrames = st.n_frames;` line, capture pending and
(re)start/stop the poll loop:
```javascript
  maybePoll(st.pending || 0);
```
Add the `maybePoll` function just below `applyState`:
```javascript
function maybePoll(pending){
  if(pending > 0 && pendingPoll === null){
    pendingPoll = setInterval(async () => {
      const st = await api("/api/state");
      applyState(st);                                   // refresh timeline/coverage
      const fh = await api(`/api/frame_h/${cur}`); curH = fh.h; drawFrame();
      if((st.pending || 0) === 0){ clearInterval(pendingPoll); pendingPoll = null; }
    }, 750);
  } else if(pending === 0 && pendingPoll !== null){
    clearInterval(pendingPoll); pendingPoll = null;
  }
}
```
(Note: `applyState` calls `maybePoll`, and the poll calls `applyState` — `maybePoll` is
guarded by the `pendingPoll === null` check so it will not stack intervals.)

- [ ] **Step 2: Syntax sanity check (node unavailable)**

Run a brace/paren balance check and confirm referenced identifiers exist:
```bash
cd /Users/patrickreed/Sandbox/soccer-vision
python3 - <<'PY'
import re
s=open("packages/soccer-vision/src/soccer_vision/labeler/static/app.js").read()
s2=re.sub(r'//[^\n]*','',s)
s2=re.sub(r'`(?:\\.|[^`\\])*`','""',s2); s2=re.sub(r'"(?:\\.|[^"\\])*"','""',s2); s2=re.sub(r"'(?:\\.|[^'\\])*'","''",s2)
st=[]; pairs={')':'(',']':'[','}':'{'}; ok=True
for ch in s2:
    if ch in '([{': st.append(ch)
    elif ch in pairs:
        if not st or st[-1]!=pairs[ch]: ok=False; break
        st.pop()
print("balanced" if ok and not st else "UNBALANCED")
for name in ["function maybePoll","pendingPoll","applyState","maybePoll(st.pending"]:
    print(name, "->", "ok" if name in s else "MISSING")
PY
```
Expected: `balanced` and every identifier `ok`.

- [ ] **Step 3: Manual verification (Patrick)**

The labeler can be relaunched on the training clip:
```bash
cd /Users/patrickreed/Sandbox/soccer-vision/packages/soccer-vision
uv run python -m soccer_vision.labeler --video /Users/patrickreed/sv-labeler/training_clip.mp4 \
  --export-dir /Users/patrickreed/sv-labeler/training_clip_out --port 8000 --workers 1
```
Verify: place ~25 point clicks — each click returns instantly (no freeze); the coverage
timeline visibly fills in over the following second; a line click is also instant and the
overlay near it updates. Report observations (do not assess overlay accuracy — that is
Patrick's call).

- [ ] **Step 4: Commit**

```bash
cd /Users/patrickreed/Sandbox/soccer-vision
git add packages/soccer-vision/src/soccer_vision/labeler/static/app.js
git commit -m "feat(labeler): poll /api/state while a background refit is pending (frontend)"
```

---

## Done criteria
- `RefitWorker` (new file): dirty-set + revision/cancellation + `wait_idle`/`pending`, unit-tested without calibration.
- `LabelerState`: `_compute_poses`/`_compute_dirty`/`_apply_fits` under one `RLock`; interactive mutators fit the clicked frame synchronously then `mark_dirty` the affected window; bulk `add_clicks` stays synchronous; `recalibrate` enqueues; `export` waits idle; line propagation scoped to `line_band`. Tested: instant current frame, async-equals-synchronous equivalence, line-band scoping, pending drains.
- Server: `/api/state` exposes `pending`; worker stopped on shutdown.
- Frontend: polls while `pending > 0`, stops at 0; manually verified responsive.
- Full suite + ruff + root mypy green.

## Real validation (Patrick, visual)
Relaunch on the training clip, place 25+ clicks plus near-touchline/midline line clicks, and
confirm clicking stays responsive throughout and the overlay still snaps. Claude renders;
Patrick assesses.

## Deferred
`ThreadingHTTPServer`/multiprocessing; model-assisted auto-init; any calibration-math change.
