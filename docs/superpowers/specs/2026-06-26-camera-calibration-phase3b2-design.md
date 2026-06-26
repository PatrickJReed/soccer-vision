# Camera Calibration — Phase 3b-2: Line-Click UI (near touchline + midline) (design)

**Date:** 2026-06-26
**Status:** Design approved, pending implementation plan
**Depends on:** Phase 2 (`calib.calibrate.refine_pose`, `line_residual`,
`field_model.FIELD_LINES` / `field_line_3d`), Phase 3b-1 (the calibrated
`LabelerState`; `poses_by_click_propagation` already takes a per-frame `line_obs`
arg and runs `refine_pose` on frames that have line constraints), the labeler
(`labeler/state.py`, `labeler/server.py`, `labeler/static/{index.html,app.js}`,
`pitch/manual_anchor.propagate_clicks`).

## Problem

The calibrated labeler (3b-1) is point-only. Two pitch regions have no clickable
*point* and stay under-constrained (Patrick's overlay critique, confirmed): the
**near touchline** (under the camera, often below the frame) and the **near part of
the midline** (only the far halfway landmark is a point). Phase 2 built the math —
`refine_pose` with **line constraints** (a clicked pixel asserted to lie on a known
field line) — and 3b-1's `poses_by_click_propagation` already consumes a per-frame
`line_obs`. What's missing is the **line-click data path**: a UI to place line
clicks, storage, chain-propagation, and wiring them into the per-frame refine.

## Goal (Phase 3b-2)

Add line-clicking end-to-end: arm a line in the palette → click along it on the
canvas → the constraint is stored, propagated through the chain, and fed to
`refine_pose` on the affected frames — tightening the near touchline / midline (and
the other three lines) where points can't reach. Validated visually on a real
session (no clean numeric gate exists without ground truth there).

## Non-goals

- Landmark-schema additions (6-yard goal box, circle∩midline) — a separate deferred
  pass (the synthetic test showed only ~5-9 % gain and they don't fix the rejections).
- A numeric line-accuracy gate (validation is visual — Patrick assesses the overlay).
- Changing the point-click flow, the focal bootstrap, or the outlier preprocessing
  (lines don't touch the focal).
- New field lines beyond the 5 in `FIELD_LINES`.

## Design

### Data model — `LineClick`
- `LineClick(frame: int, line_id: str, x: float, y: float)` (normalized [0,1] x,y) —
  a pixel asserted to lie on `line_id` (one of the 5 `FIELD_LINES`). Lives in
  `pitch/manual_anchor.py` next to `Click`.

### Line-click propagation — `pitch/manual_anchor.py`
- `propagate_line_clicks(line_clicks, transforms, segment_of, *, window, frames=None)
  -> dict[int, list[tuple[str, float, float]]]`: for each target frame, chain-map
  every line click in the same segment within `window` to that frame's pixels (a
  pixel on line L maps to a pixel still on L, since the line is rigid and the chain
  is the camera's no-parallax motion) and emit `(line_id, x, y)`. Unlike
  `propagate_clicks`, **all** in-window line clicks contribute (each is an
  independent constraint), not nearest-wins. Returns NORMALIZED pixels; the caller
  scales ×(w,h). Pure, tested. Factor out the shared chain-projection from
  `propagate_clicks` if it keeps both DRY; otherwise mirror it.

### `LabelerState` integration — `labeler/state.py`
- `self.line_clicks: list[LineClick]` alongside `self.clicks`.
- `add_line_click(frame, line_id, x, y)`: append, then `_refit(self._affected(frame))`
  (windowed), autosave. (No bootstrap/recalibrate — lines don't affect the focal.)
- `_refit` / `_recompute_all` build per-frame `line_obs` via `propagate_line_clicks`
  (scaled to pixel) and pass it to `poses_by_click_propagation(..., line_obs=...)`,
  which already runs `refine_pose` on frames that have line constraints (SQPNP seed
  for points, refine for lines — the Phase-2 path). **Decision (approved): line
  clicks propagate and `refine_pose` runs on the affected frames** — bounded cost
  (refine only where line clicks exist; point-only frames stay ~135 ms; a few
  seconds when line clicks are in the window; a slower one-time full recompute).
- `remove_last`: pop the most-recent action whether point or line (track insertion
  order across both lists — e.g. a small `_history` of `("pt"|"ln", index)` or
  compare the two lists' last-append; simplest: a single `_order` list of refs).
- Persistence: the **autosave sidecar JSON** gains a `line_clicks` array (so a crash/
  reopen restores them); `export` writes a **`line_clicks.parquet`** (frame, line_id,
  x_px, y_px) next to `keypoints.parquet`, and the resume path loads it if present.
  The point `keypoints.parquet` and `homographies.parquet` are unchanged — the
  exported homographies already bake in the line refine, so downstream consumers need
  no change.

### Server — `labeler/server.py`
- `POST /api/line_click {frame, line_id, x, y}` → `state.add_line_click(...)` →
  state payload.
- `GET /api/clicks` returns `{clicks: [...], line_clicks: [...]}`.
- `POST /api/undo` → `state.remove_last()` (points or lines).
- `GET /api/lines` (or fold into `/api/state`) → the 5 `FIELD_LINES` names for the
  palette. `export` POST unchanged.

### Frontend — `labeler/static/{index.html, app.js}`
- **LINES palette section** under the landmark list: the 5 line names. Arming state
  becomes one-of: `armedKind ∈ {"point", "line"}` + the armed id; arming a line
  clears the armed point and vice versa. A canvas click places a point (armed point)
  or a `LineClick` (armed line) at the current frame via the matching endpoint.
- **Drawing:** line clicks as small diamonds in a per-line colour (distinct from the
  point dots), labelled with a short line tag; when the overlay is on, also draw the
  model's projected lines so the fit is visible. Point dots / drag-nudge unchanged.
- **Undo** button hits `/api/undo` (now point-or-line). Keyboard 0–9 still arms
  points; lines are armed by palette click. Load (`/api/clicks`) restores both.

## Error handling
- A line click on a frame with too few total constraints to refine → `refine_pose`
  raises `CalibError`, caught in `poses_by_click_propagation` (keeps the SQPNP pose);
  the frame is still covered.
- A `line_id` not in `FIELD_LINES` → `field_line_3d` raises `KeyError`; the server
  validates and 400s a bad `line_id`.
- Line clicks before the focal is bootstrapped (< 3 anchors) → stored but not yet
  applied (no calibrated homographies until bootstrap; lines join on the next refit).
- Undo with no clicks of either kind → no-op.

## Testing
- **`propagate_line_clicks` (synthetic):** a line click at one frame chain-maps to a
  pixel that still lies on the same field line in a neighbour frame (within tolerance,
  using a known pan chain); all in-window line clicks are emitted (not nearest-wins);
  `frames=` restricts targets.
- **End-to-end refine (the gate, mirrors Phase 2):** a `LabelerState` on a synthetic
  shallow/under-constrained view — adding a midline (or near-touchline) line click
  measurably tightens that frame's pose toward truth vs point-only (held-out error
  down), proving the line constraint flows UI→store→propagate→refine.
- **`LabelerState` behaviour:** `add_line_click` recomputes `_affected`; `remove_last`
  pops the right kind; export/resume round-trip line clicks.
- **Server:** `/api/line_click` adds a line click and refits; `/api/clicks` returns
  both lists; bad `line_id` → 400.
- **Real validation (visual):** Patrick clicks the near touchline + midline on a
  session; the rendered overlay snaps onto those edges. Claude renders, Patrick
  assesses (no self-interpretation of images).

## Deferred to later
Schema additions (goal box, circle∩midline); the model-path (keypoints→pose);
any cross-game onboarding.
