# Labeler: Delete Specific Labels — Design (2026-07-29)

**Goal.** Targeted deletion of individual labels (point clicks and line clicks) on any
frame, from the labeler UI. Today the only removal tools are global Undo
(`remove_last` — strictly chronological) and `nudge` (move, not remove) — fixing a
specific bad label on an old frame is impossible without undoing everything after it.
Immediate driver: cleaning the audited bad clicks on home_g4_oceanside frames
128/166/220 (Patrick, 2026-07-29).

## UX

Right-click (two-finger click) within ~14 canvas px of a rendered marker on the
CURRENT frame deletes that label — the nearest point or line click, whichever is
closer. The browser context menu is suppressed on the canvas only. A hover cue marks
the would-be-deleted marker (ring highlight) when the cursor is within deletion
range. No other interaction changes (left-click add, drag-nudge, Undo, Export).
Hit-testing is done in canvas pixel space (dx·width, dy·height — NOT raw normalized
units, which would mix the 16:9 axes).

## Backend (`labeler/state.py`)

- `delete_click(frame: int, kp_idx: int) -> int` — removes **all** point clicks for
  that landmark on that frame (duplicates die together; `add_click` appends, so
  stacked mislabels are possible and must clear in one gesture). Returns the count.
- `delete_line_click(frame: int, line_id: str, x: float, y: float, *, eps: float = 1e-6) -> bool`
  — removes the nearest line click of `line_id` on `frame` within `eps` (normalized
  Euclidean). The UI sends the exact stored coordinates of the marker it hit-tested
  (float64 round-trips exactly through JSON), so `eps` is only float-paranoia
  tolerance; callers with approximate coords may pass a larger eps.
- **`_seq` lockstep invariant is the crux:** the i-th `"pt"` entry of `_seq`
  corresponds to `self.clicks[i]` and the i-th `"ln"` to `self.line_clicks[i]`.
  Deletion removes the matching `_seq` positions in the same `self._lock` block that
  mutates the list, so Undo keeps working correctly on the surviving clicks.
- Dirty semantics mirror `remove_last`: line deletion is segment-scoped
  (`_affected(frame)`); point deletion is segment-scoped in crop mode and
  whole-session in physical mode. `_autosave()` after every deletion; no
  re-bootstrap logic (same policy as undo: the calibration refreshes on the next
  worker pass).

## Server (`labeler/server.py`)

Two POSTs mirroring the `nudge` endpoint's shape and error convention:
- `/api/delete_click` `{frame, kp_idx}` → state payload, or `{"error": ...}` 404 if
  nothing matched.
- `/api/delete_line_click` `{frame, line_id, x, y}` → state payload or 404.

## Frontend (`labeler/static/app.js`)

`canvas.oncontextmenu`: preventDefault → `canvasNorm(e)` → nearest-label hit-test
(current frame only, both click arrays, canvas-pixel distance ≤ 14) → POST the
matching endpoint → on success refresh exactly like the Undo handler (applyState +
re-fetch `/api/clicks` + `placed` + `frame_h` + `drawFrame`). Hover cue piggybacks
on the existing mousemove/drawFrame path.

## Testing

State-level (mirror existing test_labeler_state patterns): deletes remove exactly
the targeted clicks; duplicates all removed; `_seq` invariant holds (undo after a
targeted delete pops the correct surviving click); return values (0/None-found
cases); dirty-marking scope per engine; autosave sidecar reflects deletion.
Server-level (mirror test_labeler_server): round-trip both endpoints incl. the 404
path. Frontend is untested in this repo (no JS harness) — verified by hand.

## Out of scope

Multi-select/rubber-band deletion; deleting from non-current frames; undo of a
deletion (Undo continues to pop the chronological last click; a deleted click is
gone); any engine change.
