# Interactive Anchor Labeler (design)

**Date:** 2026-06-05
**Status:** Design approved, pending implementation plan
**Depends on:** Phase 3 (pitch homography + pipeline), Phase 3.5a (homography propagation)

## Problem

Pitch-homography coverage is the binding constraint on the analytics layer. The
Phase 3.5b ML keypoint model is the path to *cross-game, hands-off* coverage, but
a 66-frame bootstrap failed to learn (21 visually-similar landmarks on faint OOD
markings need far more data), and training a good model is a large up-front
investment.

For a **single fixed-camera video**, that ML detour is unnecessary. The Trace
camera has no parallax, so every frame relates to every other by a homography
recoverable from background texture alone (the basis of 3.5a). Therefore each
landmark only needs to be located **once**; frame-to-frame registration carries
it everywhere. A person willing to spend a few minutes per video can anchor the
field by hand and get a complete per-frame mapping — no model, more reliable.

## Goal

A **local browser app** that lets a user interactively click field landmarks on a
video (each landmark once), spreads those clicks across frames via registration,
computes a per-frame field→pitch homography with a **live confidence/coverage
readout**, and exports `homographies.parquet` + `keypoints.parquet` that feed the
existing analytics pipeline unchanged.

## Non-goals

- Replacing the Phase 3.5b ML model. The model remains the lever for onboarding
  many games without per-game clicking; this tool is the per-video manual path.
- Player detection / masking as a hard dependency (CPU-only by default).
- Multi-camera or parallax footage (the approach relies on the no-parallax
  property).
- Any change to the downstream analytics (`assemble_from_homographies` and below
  are reused as-is).

## Decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Labeling UI | Custom "click-once" tool (not Roboflow reuse) |
| Live feedback | Real-time coverage + per-frame confidence; click until green |
| Environment | Local browser app, CPU-only |
| Anchoring model | Point-level: each click propagates across frames via registration |
| Confidence metric | The per-frame homography fit's own reprojection residual |
| Player masking | Off by default; optional via a trajectories parquet |

## Architecture & components

Four pieces; the heavy math reuses existing package code.

1. **Pure point-anchor core** — `pitch/manual_anchor.py` (new). Given the
   inter-frame registration chain + the set of clicks, returns per-frame
   homographies + per-frame confidence. No I/O; fully unit-testable. Reuses
   `compute_interframe_homographies`, `fit_homography`, `PITCH_LANDMARKS`.

2. **Local server** — `labeler/server.py` (new). Stdlib `http.server` (zero new
   deps). Loads the video, runs the one-time registration precompute (cached),
   holds click state, exposes JSON endpoints. Per-click recompute is pure/fast.

3. **Browser frontend** — `labeler/static/index.html` + `app.js` (new). Canvas +
   `fetch`, no framework. Scrub, arm a landmark, click to place, see overlays,
   watch the live coverage timeline.

4. **Output** — `keypoints.parquet` (the clicks: `frame, kp_idx, x_px, y_px,
   conf=1.0`) + `homographies.parquet` (computed per-frame), feeding the existing
   `assemble_from_homographies` → analytics.

**Data flow:** video → (precompute) inter-frame chain → clicks accumulate →
point-anchor core fits per-frame homographies + confidence → browser shows
coverage → user clicks until green → export → existing pipeline.

## The point-anchor algorithm (the new core)

**Inter-frame chain (reused, precomputed once).**
`compute_interframe_homographies` gives `G[i]`: frame *i* → frame *i+1* pixels.
Build a **cumulative transform** `M[f]` = frame *f* → reference-frame (frame 0)
pixels: `M[0] = I`, `M[f] = M[f-1] @ inv(G[f-1])`. Any clicked point *p* in frame
*c* maps into frame *g* as `inv(M[g]) @ M[c] @ p`. Computed once per video,
cached to disk.

Gaps in the chain (a frame pair that failed registration) split the video into
registration-connected segments; cumulative transforms are defined within a
segment. Clicks only propagate within their segment. Segment boundaries are
reported (they appear as forced red zones until anchored on both sides).

**Per-frame homography from propagated clicks.** For a target frame *g*:
1. Gather every click within a drift-bounded window `|c − g| ≤ W` (same segment),
   map each into *g*'s pixels via the cumulative transforms.
2. Collect `(pixel_in_g ↔ PITCH_LANDMARKS[kp_idx])` correspondences — one per
   distinct landmark; if the same landmark was clicked in several frames, the
   nearest-chain-distance one wins.
3. If ≥4 distinct landmarks → `fit_homography` (RANSAC) → store H + residual.

**Confidence = the fit's reprojection residual.** With >4 landmarks the fit is
over-determined, so its mean reprojection error (pitch units) is a direct,
honest accuracy estimate. A frame is **covered** when it has ≥4 landmarks **and**
residual ≤ a threshold (default 0.05, matching the existing gate). This drives
the live timeline: green = covered & low residual, yellow = few landmarks /
borderline, red = none / failed.

**Self-correcting:** drift over long chains shows up *as* high residual → that
frame goes yellow/red → the user adds a click nearby → residual drops → green.
Re-clicking the same landmark at a few times through the video is how drift is
bounded. `W` is tunable; the real registration reach is discovered empirically
through the tool (the 3.5a probe saw no accuracy knee out to gap 60, so reach may
be generous).

**v1 simplification:** registration runs **without player-masking** (masking
needs the GPU detector). Background features dominate and RANSAC rejects
moving-player outliers — a reasonable CPU-only default, with a slight accuracy
trade-off. Optional `--player-boxes trajectories_px.parquet` enables masking when
detection has been run.

## Interface

A single-page canvas app (see mockup `labeler-mockup.html`):

- **Main frame** — the current video frame. **Arm** a landmark from the
  right-side palette (numbered 0–20, named, color-coded by tier), then **click**
  its location on the frame. Placed landmarks show a ✓. #5 is hidden (never
  visible).
- **Green pitch-grid overlay** — the current frame's fitted homography reprojected
  onto the field. If the green lines sit on the real markings, the homography is
  right (primary correctness check).
- **Dots** — solid numbered = clicked in this frame; dashed hollow = propagated in
  from clicks in other frames.
- **Coverage timeline** (whole video) — green/yellow/red segments, with a
  playhead. Live stats: overall coverage %, this-frame residual, landmarks in
  frame, total clicks.
- **Controls** — prev/next red zone (jump the playhead to uncovered stretches),
  undo last click, nudge a placed point, toggle grid overlay, export. Keyboard
  shortcuts: number keys arm a landmark; arrows scrub.

**Loop:** arm → click → watch coverage grow → jump to a red zone → click → repeat
until the bar is green and the grid overlay sits on the lines everywhere → export.

## Output & integration

- `keypoints.parquet` — the raw clicks (`frame, kp_idx, x_px, y_px, conf=1.0`),
  for provenance and cheap re-tuning.
- `homographies.parquet` — per-frame `HomographyEntry` (written via the existing
  `homographies_to_parquet`), consumed by the existing
  `assemble_from_homographies(trajectories_px_path, homographies_path, out_dir)`
  → `trajectories` + `phases`. No downstream change.

The user still runs detection (`analyze_video` Stage 1, GPU) separately to get
`trajectories_px` for the players; this tool replaces only the homography stage
for videos where the ML pitch model is weak.

## Operational details

- **Precompute caching** — the registration pass (full-res ORB; minutes on a
  laptop, longer for a full game) writes the inter-frame chain + cumulative
  transforms to a sidecar keyed by video hash. Reopening is instant.
- **Display proxy** — registration uses full-res frames (ORB recall, per 3.5a);
  the browser is served downscaled JPEGs; clicks scale back to full-res coords.
- **Session persistence** — clicks autosave to a sidecar JSON keyed by video;
  close and resume. Export is the final parquet write.

## Testing & acceptance

- **Pure core** (`manual_anchor.py`): unit tests on synthetic inter-frame chains
  + clicks → known homographies and residuals; window/segment behavior; <4-landmark
  frames left uncovered; residual rises with injected drift.
- **Server**: endpoint-level integration tests (serve frame, add/remove click,
  coverage JSON, export writes valid parquet).
- **Frontend**: manual verification (canvas app).
- **End-to-end acceptance**: launch on the bake-off clip, place a handful of
  anchors, confirm the green grid overlay sits on the markings and coverage
  climbs, then `assemble_from_homographies` runs on the export and produces a
  phases parquet (the existing pipeline smoke path).

## File structure

| File | Responsibility | Action |
|---|---|---|
| `pitch/manual_anchor.py` | Point-propagation core: cumulative transforms, per-frame fit, residual confidence | Create |
| `labeler/__init__.py` | Package marker | Create |
| `labeler/__main__.py` | `python -m soccer_vision.labeler --video … [--player-boxes …]` launcher | Create |
| `labeler/server.py` | Stdlib HTTP server: frame/click/coverage/export endpoints + precompute cache | Create |
| `labeler/static/index.html` | Canvas UI shell | Create |
| `labeler/static/app.js` | Scrub/arm/click/overlay/timeline logic | Create |
| `tests/test_pitch_manual_anchor.py` | Pure-core unit tests | Create |
| `tests/test_labeler_server.py` | Endpoint integration tests | Create |

Reused unchanged: `pitch/propagation.py` (`compute_interframe_homographies`),
`pitch/homography.py` (`fit_homography`), `pitch/landmarks.py` (`PITCH_LANDMARKS`),
`pipeline.py` (`homographies_to_parquet`, `assemble_from_homographies`).
