# Labeler Full-Game Readiness (design)

**Date:** 2026-06-11
**Status:** Design approved, pending implementation plan
**Depends on:** Interactive anchor labeler (shipped + field-tested 2026-06-11).

## Problem

The labeler is proven on the 2-minute bake-off clip (82% coverage from one
session) but four measured properties block its real use case — full games
(e.g. `training.mp4`: 32 min, 58,179 frames):

1. **Chain precompute is serial-eager:** ~15 CPU-min per 2 min of video → ~4 h
   for a 32-min game before the first click.
2. **Per-click recompute is O(clicks × frames):** the vectorized fit projects
   every click into every frame (~1 GB at 600 clicks × 100k frames) and refits
   all frames — tens of seconds per click at full-game scale.
3. **Clicks live only in memory until Export:** a crash after 45 min of
   labeling loses the session.
4. **The UI ships one status string per frame per response** (~1 MB at
   full-game length) and the timeline renders one div per frame.

## Goal

Make a full-game labeling session practical: precompute under ~an hour
(one-time, cached), sub-second clicks at 58k frames, crash-safe sessions, a
responsive UI — plus two session-tested UX riders (numeric residual readout,
point nudge).

## Non-goals

- Lazy/on-demand registration (considered; parallel-eager chosen for
  simplicity and identical math).
- Keyboard arming for landmarks 10–20 (offered, declined).
- Any change to registration math, window/residual defaults, or cache format.

## Decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Precompute scaling | Parallel eager (multiprocessing chunks; same math, same cache) |
| Recompute scaling | Incremental: refit only ±window of the mutated click |
| Crash safety | Autosave sidecar JSON per mutation; auto-load on startup |
| UI scaling | Server-side bucketed timeline (~1,200 buckets) |
| UX riders | Numeric residual + n_points readout; drag-to-nudge placed clicks |

## Design

### 1. Parallel chain precompute (`labeler/chain.py`)

`compute_chain` gains `workers: int | None = None` (default `max(1,
cpu_count() - 1)`). The pair range `[0, n-1)` splits into `workers` contiguous
chunks; each worker is a top-level function (spawn-safe) that opens its OWN
`VideoCapture`, registers pairs in its chunk (reading frames chunk-start to
chunk-end + 1, i.e. 1-frame overlap), and returns its `{i: G}` dict. The parent
merges, normalizes (existing `normalize_homography`), and writes the SAME
`.npz` cache — existing caches remain valid and are still checked first.
Per-chunk completion progress is printed. `workers=1` must reproduce the serial
path exactly (test: chunked == serial on a synthetic video).

### 2. Incremental recompute (`pitch/manual_anchor.py` + `labeler/state.py`)

`fit_frame_homographies` gains `frames: Sequence[int] | None = None`: when
given, only those frames are fitted (candidate clicks still come from anywhere
within the window — the projection/selection math is unchanged, just
row-scoped). Correctness invariant: a click mutation can only change fits
within ±window of the mutated click's frame (nearest-per-landmark selection is
window-bounded), so:

- `add_click` / `remove_last` / `nudge_click`: refit only
  `[frame - window, frame + window]` (clamped, same segment) and merge into the
  cached fits dict (replacing entries; frames that lose their fit are removed).
- `add_clicks` (bulk/resume): one full recompute, **chunked** over frames
  (~5,000 per chunk) so the (clicks × frames × 2) projection array stays ~50 MB
  instead of ~1 GB.

Property test (load-bearing): after ANY sequence of add/remove/nudge, the
incrementally-maintained fits equal a fresh full recompute.

### 3. Autosave (`labeler/state.py` + `labeler/server.py`)

Every mutation atomically (tmp + rename) writes the click list to a sidecar
`<video stem>.clicks.json` in the same directory as the chain cache
(`.sv_labeler_cache/`). Format: `[{"frame": int, "kp_idx": int, "x": float,
"y": float}]` in normalized coords. On `run()` startup: explicit `--resume
<keypoints.parquet>` takes precedence; otherwise an existing sidecar is loaded
automatically (count printed). Export behavior unchanged.

### 4. Bucketed timeline (`labeler/server.py` + `static/app.js`)

`/api/state` replaces the per-frame `status` list with `status_buckets`: a
fixed-size list (~1,200) where each bucket reports the worst status inside it
(red if any red, else yellow if any yellow, else green) plus `bucket_size`.
The frontend timeline renders buckets; red-jump buttons jump to the start
frame of the next/previous red bucket. Scrubbing remains frame-accurate.

### 5. UX riders

- **Numeric residual:** `/api/frame_h/<i>` adds `residual: float | null` and
  `n_points: int | null`; the stats bar shows `residual 0.018 (6 pts)` styled
  by the status color.
- **Point nudge:** mousedown within ~10 px (canvas space) of a dot placed ON
  the current frame arms a drag; mouseup POSTs `/api/nudge {frame, kp_idx, x,
  y}` (normalized coords); the server updates that click in place (the click
  matching frame + kp_idx), autosaves, incrementally refits ±window, and
  returns the state payload. Propagated (dashed/other-frame) dots are not
  draggable.

## Acceptance

- `training.mp4` (58k frames) chain precompute completes in well under an hour
  wall-clock on Patrick's laptop (8 perf cores); cache hit on relaunch.
- Click latency at 58k frames with hundreds of clicks: < 1 s.
- Kill the server mid-session, relaunch → clicks restored from sidecar.
- Bake-off clip workflow (2-min clip) behaves identically to today (regression:
  same coverage from the same resumed clicks).

## Testing

- **Pure:** subset-fit equivalence (fits(frames=S) == full fits restricted to
  S); the incremental-vs-full property test across mutation sequences;
  chunked bulk recompute equals unchunked; bucket downsampling (worst-status
  rule, edge buckets).
- **Chain:** parallel (workers=2/3) merge equals serial on a synthetic video;
  cache round-trip unchanged.
- **Server:** nudge endpoint (moves the right click, refits); autosave file
  written on click/undo/nudge and loaded on a fresh `LabelerState` via the
  startup path; `/api/state` bucket payload shape.
- **Manual acceptance:** the full `training.mp4` session (Patrick).
