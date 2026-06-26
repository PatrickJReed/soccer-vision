# Labeler responsive refit — async background worker + scoped line-refine (design)

**Date:** 2026-06-26
**Status:** Design approved, pending implementation plan
**Depends on:** the calibrated labeler (`labeler/state.py` `LabelerState`, `labeler/server.py`,
`labeler/static/{index.html,app.js}`), `pitch/calib_anchor.poses_by_click_propagation`
(the per-frame SQPNP + selective `refine_pose` engine), `pitch/manual_anchor.propagate_line_clicks`.

## Problem

After calibration bootstraps (~3 anchor frames × ≥6 landmarks, ≈20 clicks in), **every
click runs a synchronous pose re-fit over the whole ±`window` of frames inside the HTTP
request handler**, on a single-threaded `HTTPServer`. The handler blocks until the refit
finishes, so the UI freezes:

- **Point click:** ~2×`window` SQPNP solves (≈720 frames at `window=360`) → seconds on a
  fanless MacBook Air.
- **Line click:** far worse — `_line_obs` propagates one line click across the entire
  ±`window`, and `poses_by_click_propagation` runs `refine_pose` (scipy `least_squares`,
  ~10–50 ms/frame) on **every** frame in that window → tens of seconds, "almost unresponsive."

The freeze first appears at ~20 clicks because the first ~18 clicks pre-date the focal
bootstrap and only *store* (no refit). The bootstrap click then runs a full 2700-frame
`_recompute_all`, and every click after refits the window.

### Why this design (coherence with the field)

Published sports-field calibration repos (TVCalib, PnLCalib, SoccerNet `sn-calibration`,
Sportlight, `sportsfield_release`) are all **batch/offline ML-automatic** — they never face
an interactive per-click recompute, so they offer no precedent to copy. We are interactive
because our regime is different: ML pitch-keypoint fine-tuning failed on our small, OOD
(youth 9v9, faint markings, fixed wide-angle Trace camera) data (Phase 3.5b), and the
fixed no-parallax camera makes "click once, propagate" cheap. The labeler is the per-game
workhorse and the data-generation bridge toward an eventual cross-game model.

The recompute-on-edit loop **is** a solved problem — in interactive-annotation tooling
(CVAT/Roboflow: "embed once, decode many" + caching) and SfM GUIs (COLMAP `ControllerThread`,
Meshroom `TaskThread`: heavy solve on a background thread, UI updates via a timer, never
blocking the UI thread). Even SoccerNet's "automatic" pipeline depended on humans annotating
ground truth first — it just **decoupled annotation from the solve**. Our freeze exists
because we coupled them in real time; this design decouples them, bringing us into coherence
with how the field already separates these concerns. Canonical UI law agrees (Nielsen
0.1 s/1 s/10 s; web.dev "don't block the main thread"; React optimistic UI). Python's default
`HTTPServer` is documented serial ("each request must be completed before the next") — our
exact freeze — and NumPy/SciPy release the GIL in native kernels, so a background worker
running `solvePnP`/`least_squares` genuinely progresses while the server serves polls.

## Goal

Every click (point or line) returns in **<~50 ms** regardless of click count or line
constraints. The wide-`window` fit — and the cross-pan accuracy it buys on shallow
goal-mouth views — still happens, just in the background; the coverage timeline fills in a
beat later. No change to exported homographies' correctness.

## Non-goals

- `ThreadingHTTPServer` / multiprocessing (single user, one browser → requests already
  serialize; one worker thread + one lock is sufficient and avoids handler-vs-handler races).
- Changing the calibration math, the focal bootstrap, outlier preprocessing, or the export
  format. This is a *scheduling* change (when refits run), plus one *scoping* change
  (`_line_obs` band) — not a change to *what* a fully-settled fit produces.
- A model-assisted (auto-init + click-correct) path — deferred to the ML cross-game phase.
- New field lines / landmark schema.

## Design

### Component 1 — `RefitWorker` (`labeler/refit_worker.py`)

A generic, calibration-agnostic "background recompute of a dirty set" unit. It owns one
daemon thread, a dirty frame-set, a monotonic revision counter, a `threading.Lock`, and a
wake `threading.Condition`/`Event`. Constructed with two callbacks supplied by `LabelerState`:

- `compute(frames: list[int], is_cancelled: Callable[[], bool]) -> dict[int, FramePose]`
- `apply(results: dict[int, FramePose]) -> None`

Public API:
- `mark_dirty(frames: Iterable[int]) -> None` — union into the dirty set, bump revision, wake.
- `start() -> None` / `stop() -> None` — lifecycle (daemon; `stop` joins).
- `wait_idle(timeout: float | None = None) -> None` — block until the dirty set is drained
  (test/determinism barrier; also used by `export`).
- `pending() -> int` — count of frames still dirty/in-flight (for the `/api/state` indicator).

Worker loop: under the lock, snapshot the dirty set + the current revision, clear the dirty
set; release the lock; call `compute(snapshot, is_cancelled)` **lock-free**, where
`is_cancelled()` returns True once the revision has advanced (a newer edit arrived); on
return, if not cancelled, re-acquire the lock and `apply(results)`; if cancelled, drop the
partial results and loop (the new dirty set is already queued). Compute in chunks so
cancellation is checked between chunks (bounded wasted work). The worker knows nothing about
poses, clicks, or `K`.

### Component 2 — `LabelerState` integration

`LabelerState` constructs and owns a `RefitWorker`, passing:
- `compute = lambda frames, is_cancelled: self._compute_poses(frames, is_cancelled)` —
  a refactor of the current `poses_by_click_propagation(...)` call (snapshots
  `_active_clicks()` + `_line_obs(frames)` under the lock at the start of the call), with a
  per-chunk `is_cancelled()` check.
- `apply = self._apply_fits` — merges `{frame: pose}` into `self._fits` under the lock
  (popping frames that failed to solve, mirroring today's `_refit`).

A single `self._lock` guards `self._fits` (and the `pending`/status reads). Mutators change to
**instant-return + enqueue**:

- `add_click(frame, …)`: append + `_seq` + autosave; if not yet calibrated, attempt
  `_try_bootstrap()` (cheap); **synchronously fit only the clicked frame**
  (`self._refit_one(frame)` → `_compute_poses([frame], …)` → merge) so its overlay is correct
  on return; then `worker.mark_dirty(self._affected(frame))` (on bootstrap: mark **all**
  frames dirty). Return.
- `add_line_click(frame, …)`: same shape — synchronous `_refit_one(frame)`, then
  `mark_dirty(self._affected(frame))`.
- `remove_last()` / `recalibrate()`: synchronous current-frame (or removed-frame) fit where
  applicable, then `mark_dirty` the affected range (recalibrate marks all). `recalibrate`'s
  K + outlier re-estimate stays synchronous (cheap, ~170 clicked frames); only the 2700-frame
  recompute is enqueued.
- `export(...)`: `worker.wait_idle()` first, so the written homographies reflect a fully
  settled fit.

`_refit_one(frame)` is the small synchronous path (one-frame `_compute_poses`). The old
`_refit(frames)` / `_recompute_all()` are removed in favor of `mark_dirty` + the worker
(their compute body becomes `_compute_poses`).

### Component 3 — scoped line-refine (the C piece)

`_line_obs(frames)` today emits a propagated line observation for **every** in-`window` frame,
so `refine_pose` runs across the whole window. Change: a line click at frame `F` emits
observations only for frames within **±`line_band`** of `F` (default `line_band=60` ≈ ±2 s),
intersected with the requested `frames`. Outside the band, frames stay SQPNP-only (no line
constraint) — the constraint stays where it is geometrically valuable (the line is visible
near where it was clicked) and total `refine_pose` work is bounded. `line_band` is a
`LabelerState` constructor param (default 60), independent of the point-propagation `window`;
optional `--line-band` CLI flag. **Implementation:** `_line_obs` calls the existing
`propagate_line_clicks(..., window=self.line_band, frames=frames)` — i.e. pass `line_band` as
the propagation window for *lines* (point clicks still use the wider `self.window`). This
reuses `propagate_line_clicks` unchanged: a line click only reaches frames within
`±line_band` of its own frame, so `refine_pose` runs only there. No new code in
`propagate_line_clicks`.

### Component 4 — server

Server stays serial `HTTPServer`. Handlers become fast (no in-handler heavy compute). Reads
of `_fits` (`/api/state` status buckets, `/api/frame_h`) acquire `state._lock` briefly.
`/api/state` payload gains `pending: int` (= `worker.pending()`). `run()` calls
`state.start_worker()` after construction and `state.stop_worker()` in the `finally`. No new
endpoints.

### Component 5 — frontend (`static/app.js`)

The click handler already fetches `/api/frame_h/cur` after posting — that returns the
synchronously-fit current frame, so the overlay is immediate. Add a **poll loop**: when a
mutating action returns `pending > 0`, `setInterval` poll `/api/state` (~750 ms); each poll
`applyState(...)` refreshes the coverage timeline, and re-fetches `/api/frame_h/cur` if the
current frame was in the refit; stop polling when `pending === 0`. No `index.html` change.

## Error handling

- A frame that fails to solve in `_compute_poses` is caught per-frame and popped from
  `_fits` (today's `_refit` behavior); the worker never dies on one bad frame.
- Cancellation is cooperative (revision check between chunks) — no thread killing, bounded
  wasted compute.
- Lock discipline: **compute lock-free; snapshot inputs and merge outputs under the lock**;
  critical sections are O(frames) dict updates only.
- Worker is a daemon; `stop()` joins on shutdown. A second click while the worker is mid-pass
  bumps the revision → current pass cancels and restarts with the unioned dirty set
  (coalescing).
- `export` and tests use `wait_idle()` to avoid racing a partial fit.

## Testing

Determinism comes from `RefitWorker.wait_idle()` (no sleeps).

- **Equivalence (the gate):** for a synthetic `_pan_session`, after a sequence of
  `add_click`/`add_line_click` + `wait_idle()`, `_fits` is identical (within tolerance) to a
  direct synchronous `_compute_poses` over all frames **using the same `line_band`** — proving
  the *backgrounding* changed timing, not output. (The line-band scoping is a separate,
  intentional output change vs. the old full-`window` line behavior, covered by its own test
  below — the equivalence test holds `line_band` fixed on both sides.)
- **Instant current frame:** after `add_click(F, …)` (before `wait_idle`), `frame_homography(F)`
  is already present (synchronous one-frame fit), while a far in-window frame may still be
  pending.
- **Coalescing + cancellation:** `RefitWorker` unit tests with a trivial `compute_fn` —
  overlapping `mark_dirty` ranges union; a revision bump mid-`compute` makes `is_cancelled()`
  return True and the pass restarts; `pending()` reaches 0 after `wait_idle()`.
- **Scoped line-refine:** `_line_obs` (or `propagate_line_clicks` with the band) emits
  observations only within `±line_band` of each line click's frame (pure-function test).
- **Server:** a click endpoint returns promptly with `pending > 0`; after `wait_idle`,
  `/api/state` reports the settled coverage and `pending == 0`; bad inputs unchanged.
- **`RefitWorker` in isolation:** start/mark_dirty/wait_idle/stop lifecycle with a counting
  `compute_fn`, no calibration involved.

## Deferred

Model-assisted auto-init (ML cross-game phase); `ThreadingHTTPServer`/multiprocessing;
any change to the calibration math or export format.
