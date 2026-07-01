# Physical-Pose Calibration Revert — Design

**Date:** 2026-07-01
**Status:** Approved (brainstorm complete) — ready for implementation plan
**Supersedes:** `2026-06-26-calibration-global-homography-design.md` and
`2026-06-26-global-reference-calibration-design.md` (the free-homography bundle and the
mosaic/global-reference directions). Those approaches are retired; see Evidence below.

## Goal

Replace the free-homography bundle (`pitch/global_calib.py::solve_bundle` / `BundleCalib`)
in the live labeler with a **physical per-frame point+line calibration engine**: each
clicked (anchor) frame is solved directly against the rigid 9v9 field as a real camera
pose; unclicked frames are filled by **bracket propagation** through the inter-frame chain;
frames carry a **per-frame, coverage-graded trust status**; and a session is accepted only
after a **foreground + propagation numeric gate plus a visual spot-check**.

## Background / motivation

A free 8-DOF homography is under-constrained by our data (clicks sit in a thin far-end
band, median y-spread ~183 px of 1080) and is non-physical (it can put the horizon inside
the field → "lines in the sky"). A physical camera pose `H = K[r1|r2|t]` cannot do that.
`refine_pose` was fixed (commit 73f87da) to minimize in **pitch space** (uniform units) so
near-horizon line clicks no longer destabilise the solve.

### Evidence gathered this session (why these decisions)

Head-to-head on the real training session (`/tmp/panning_experiment.py`,
`/tmp/prop_experiment.py`, `/tmp/diag_shared.py`), metre-based feet, two held-out protocols:

- **Model choice.** Independent physical (point+line) vs shipped free bundle vs a
  shared-(K,C) panning bundle. Foreground held-out (remove ALL near-touchline, predict it):
  independent 3.6 / p90 8.2 ft; free bundle 3.0 / 33; shared-panning 37.5 / 40. The
  shared-center idea was **refuted** (it only ties when C is pinned, never beats, and is
  fragile — its free-C fit slides C to a wrong (11,30,−1.6) m). Independent physical has the
  tightest foreground **tail** (p90 8.2 vs 33), is physical, and uses the line clicks.

- **Propagation choice.** Leave-one-anchor-out, predict the held frame's points:
  copy-nearest (no shift) 8.5 / 86; nearest + chain-shift 4.2 / 28.8; **bracket (shift both
  bracketing anchors, distance-weight) 3.8 / 22.1**; rotation-SLERP 7.0 / 70; shipped bundle
  4.6 / 25.9. Bracket wins median AND tail. Error vs gap-to-nearest-anchor (nearest+shift):
  gap 0–10 → 3.5 ft, 11–30 → 4.5, 31–80 → 5.1, 81–200 → 5.3, **201+ → 19.8**. So propagation
  is trustworthy to ~200-frame gaps (~7 s @ 30 fps) and must be flagged beyond.

## Decisions (locked)

1. **Scope:** Full replacement — rip the bundle out of the live path; `LabelerState`,
   export, status, and `validate_session` all run on the physical engine.
2. **Propagation:** Bracket-interpolation with a ~200-frame gap guard (evidence above).
3. **Status:** Per-frame + coverage grade (anchors self-checked against their own held-out
   foreground).
4. **Gate:** Numeric (foreground held-out + leave-one-anchor-out propagation) **plus** a
   required visual spot-check.

## Architecture

One engine, every frame anchored to the same rigid field. New module
`pitch/physical_calib.py` replaces the bundle in `global_calib.py`, reusing the physical
primitives already in place:

- `calib/calibrate.py`: `calibrate_camera` (shared focal), `refine_pose` (fixed,
  point+line), `homography_from_pose`, `pitch_homography`. **Kept as-is.**
- `calib/validate.py`: `fold_count`. **Kept.**
- `pitch/calib_anchor.py`: `frame_homography` (pose → normalized image→pitch),
  `flag_outlier_clicks` / `_robust_sqpnp` (per-frame robust preprocessing). **Kept/reused.**

### Coordinate conventions (unchanged from the codebase)

- Clicks are **normalized** image coords `x,y ∈ [0,1]`. Point clicks carry `kp_idx`; line
  clicks carry `line_id ∈ {near_touchline, far_touchline, own_goal_line, opp_goal_line,
  midline}`.
- Pitch is canonical `[0,1]²` (x = width fraction, y = length fraction), metres via
  `WIDTH_M=45.7`, `LENGTH_M=68.5`.
- A **frame homography** is normalized-image → pitch `[0,1]²` (the export format). Physical
  pose → this via `H_norm = inv(pitch_homography(homography_from_pose(K,rvec,tvec))) @ diag(W_px,H_px,1)`.
- Chain transforms `M[f]` (from `manual_anchor.cumulative_transforms`) map **normalized
  frame f → normalized reference**. Chain transfer of anchor `a`'s homography to frame `t`:
  `H_t = H_a @ inv(M[a]) @ M[t]`.

## Components / interfaces

### `pitch/physical_calib.py`

```python
FOLD_MIN, FOLD_MAX = 4, 15          # reuse the plausibility window from global_calib
DEFAULT_GAP_GUARD = 200             # frames; propagation beyond this is untrusted (evidence)
FOREGROUND_OK_FT = 8.0              # an anchor's own held-out near-TL error to grade green
GRID_N = 9                          # grid resolution for bracket-blend homography fitting

@dataclass(frozen=True, eq=False)
class PhysicalCalib:
    K: NDArray                              # shared 3x3 intrinsics
    poses: dict[int, tuple[NDArray, NDArray]]   # anchor frame -> (rvec, tvec)
    anchor_h: dict[int, NDArray]            # anchor frame -> normalized image->pitch H (cached)
    coverage_grade: dict[int, str]          # anchor frame -> "green" | "yellow"
    transforms: dict[int, NDArray]          # frame -> M[f] (normalized frame->reference)
    size: tuple[int, int]
    gap_guard: int = DEFAULT_GAP_GUARD

    def is_anchor(self, frame: int) -> bool
    def nearest_anchor_gap(self, frame: int) -> int | None   # min |frame - anchor|, None if no anchors
    def frame_homography(self, frame: int) -> NDArray | None  # anchor | bracket | None (beyond gap)
    def status(self, frame: int) -> str                       # "green" | "yellow" | "red"

def solve_session(
    points: Sequence[Click],
    lines: Sequence[LineClick],
    size: tuple[int, int],
    transforms: Mapping[int, NDArray],
    *,
    min_points: int = 4,
    gap_guard: int = DEFAULT_GAP_GUARD,
    seed: PhysicalCalib | None = None,       # warm-start refine_pose from prior poses
) -> PhysicalCalib
```

**`solve_session`:**
1. Shared `K = calibrate_camera({frame: point_obs}, size, min_points=6).K`.
2. `flag_outlier_clicks(points, K, size)` → drop per-frame outlier point clicks.
3. For each frame with ≥ `min_points` distinct point landmarks: SQPNP seed
   (`cv2.solvePnP(..., SOLVEPNP_SQPNP)`) → `refine_pose(K, rvec0, tvec0, point_obs,
   line_obs)` (warm-started from `seed.poses[frame]` when present). On `CalibError` /
   non-convergence, the frame is **not** an anchor.
4. Cache `anchor_h[frame] = frame_homography(K, rvec, tvec)` (normalized image→pitch).
5. `coverage_grade[frame]` via `_foreground_selfcheck` (below).

**`frame_homography(frame)`:**
- Anchor → `anchor_h[frame]`.
- Else find bracketing anchors `lo < frame < hi` (or the single nearest if one-sided).
  - If nearest-anchor gap > `gap_guard` → `None`.
  - Two-sided → `_bracket_h(frame, lo, hi)`; one-sided → `_shift_h(anchor, frame)`.

**`_shift_h(a, t) = anchor_h[a] @ inv(transforms[a]) @ transforms[t]`.**

**`_bracket_h(t, lo, hi)`** (blend realized as a valid single homography):
1. `H_lo_t = _shift_h(lo, t)`, `H_hi_t = _shift_h(hi, t)`.
2. Weight `w = (t - lo) / (hi - lo)`.
3. Over a `GRID_N × GRID_N` grid of normalized image points `g`, predict
   `p_lo = apply(H_lo_t, g)`, `p_hi = apply(H_hi_t, g)`, blended `p = (1-w)·p_lo + w·p_hi`.
4. Fit one homography `g → p` (`pitch.homography.fit_homography`, or `cv2.findHomography`).
   Return it. If the fit is degenerate, fall back to the nearer of `H_lo_t` / `H_hi_t`.

**`_foreground_selfcheck(frame, K, points_f, lines_f) -> str`:**
- If the frame has **no** `near_touchline` line click → `"yellow"` (foreground unverified —
  this is what keeps no-line frames like 193/134 out of green).
- Else refit the pose from the frame's points + non-near-touchline lines, predict its
  near-touchline clicks, take the median perpendicular error (feet). ≤ `FOREGROUND_OK_FT`
  → `"green"`, else `"yellow"`.

**`status(frame)`:**
- Anchor: `fold_count` out of `[FOLD_MIN, FOLD_MAX]` → `"red"`; else `coverage_grade[frame]`
  (`"green"` or `"yellow"`).
- Non-anchor: `frame_homography` is `None` (beyond gap) → `"red"`; `fold_count` out of range
  → `"red"`; else `"yellow"` (propagated, honestly under-verified).

### Acceptance metrics (in `physical_calib.py`, consumed by `validate_session`)

```python
@dataclass(frozen=True)
class GateReport:
    fg_median_ft: float; fg_p90_ft: float; fg_n: int          # near-TL held out
    prop_median_ft: float; prop_p90_ft: float; prop_n: int    # LOAO bracket, within-gap only
    passed_numeric: bool

def foreground_holdout(points, lines, size, transforms) -> tuple[float,float,int]
def propagation_holdout(points, lines, size, transforms, *, gap_guard=DEFAULT_GAP_GUARD)
    -> tuple[float,float,int]
def evaluate_gate(points, lines, size, transforms) -> GateReport
```

- **`foreground_holdout`:** over anchors with a near-touchline click, refit without their
  near-TL, predict it, aggregate median / p90 (feet). Pass: median ≤ 5, p90 ≤ 12.
- **`propagation_holdout`:** leave-one-anchor-out; if the held anchor's nearest remaining
  anchor is within `gap_guard`, bracket-predict its point clicks (feet). Aggregate over
  within-gap held anchors. Pass: median ≤ 5.
- `passed_numeric = foreground pass AND propagation pass`.

### `pitch/validate_session.py` (rewritten)

- Load chain + clicks (points **and** lines), build `transforms`.
- Print `GateReport`. Render a handful of spot-check overlays (spread across the clip +
  at least one sparse/no-line frame) to a directory, and instruct the user to review them
  and confirm pass/fail. Exit status reflects the numeric gate; the visual pass is the
  human's call (printed as REQUIRED).

### `labeler/state.py` (rewired)

- Replace `solve_bundle`/`BundleCalib` usage with `solve_session`/`PhysicalCalib`.
- Keep the non-blocking click model: clicks mark dirty + return; overlay drawn instantly
  from the cached calib; background `RefitWorker` recomputes `solve_session` warm-started
  from the previous `PhysicalCalib` (via the `seed=` param).
- Status per frame from `PhysicalCalib.status`. Export stays **blocks-until-idle** and
  **green-gated** (fix the audit's silent-partial-write: only write after the refit settles;
  never write on timeout).

### Overlay render (clipped)

Draw a projected field line only over its in-front (`w > 0`) + in-frame span (margin `M`),
per the validated prototype, so off-screen far field isn't drawn.

## Data flow

`clicks (points+lines) + chain transforms → solve_session → PhysicalCalib`
`→ frame_homography(f)  [anchor | bracket | None]`
`→ overlay (clipped) / export homographies.parquet / status`
`validate_session → evaluate_gate + spot-check renders`

## Error handling

- Anchor with too few points / `refine_pose` non-convergence / fold out of range → not an
  anchor (or red). Outlier point clicks dropped by `flag_outlier_clicks`.
- Bracket fit degenerate → fall back to nearest chain-shift; beyond gap guard → `None` → red.
- Export: wait for the background refit to settle, then write only non-red frames; on a
  wait timeout, **abort with an error** (no partial/stale write).

## What is deleted vs kept

- **Delete** from `pitch/global_calib.py`: `solve_bundle`, `BundleCalib`, `solve_global`,
  `GlobalCalib`, `cross_end_holdout`, `HoldoutReport`, `_solve_segment`, `_seed_hg`,
  `_seed_x0_for_segment`, `_interp_affine_params`, `_affine_from`, affine constants,
  `two_ended_segments`, `fold_of_norm`, `frame_status`, `OWN_END_IDX`/`OPP_END_IDX` (unless
  reused). If nothing remains, delete the file. Grep to confirm no dangling imports
  (`state.py`, `validate_session.py`, tests).
- **Keep:** `calib/calibrate.py`, `calib/validate.py::fold_count`, `pitch/calib_anchor.py`.

## File structure

- **Create:** `packages/soccer-vision/src/soccer_vision/pitch/physical_calib.py`
- **Create:** `packages/soccer-vision/tests/test_physical_calib.py`
- **Modify:** `labeler/state.py` (rewire), `pitch/validate_session.py` (gate + spot-check),
  overlay draw site (clipping).
- **Delete/trim:** `pitch/global_calib.py` and its tests; update any importers.

## Testing

- **Synthetic translating camera:** build known poses sharing K; verify `_shift_h` /
  `_bracket_h` recover an interior frame's homography to sub-pixel on a drift-free chain.
- **Anchor fidelity:** anchor poses reproduce their own clicks within tolerance (points
  ≤ ~5 ft, lines ≤ ~3 ft).
- **Status transitions:** anchor-with-near-TL-green, anchor-no-near-TL-yellow, anchor
  failing self-check → yellow, propagated within gap → yellow, beyond gap → red,
  fold-out-of-range → red.
- **Gate:** `evaluate_gate` on a synthetic pass/fail fixture hits the thresholds; on the
  real session reproduces foreground ~3.6 ft and propagation ~3.8 ft.
- **Overlay clipping:** points behind camera / off-frame are not drawn.
- **Regression:** full suite green; canonical `uv run mypy` (src+tests) + ruff clean; no
  references to the deleted bundle symbols remain.

## Acceptance criteria

- Live labeler runs entirely on `PhysicalCalib`; no `solve_bundle` in the live path.
- `evaluate_gate` on the training session: foreground median ≤ 5 / p90 ≤ 12 ft, propagation
  median ≤ 5 ft (within gap).
- Visual spot-check renders produced; user confirms foreground/overlay quality (incl. a
  sparse frame) before green is trusted.
- Full test suite + mypy + ruff clean.

## Risks / open items

- Bracket-fit-H on high-drift spans could be poor; mitigated by nearest-shift fallback and
  the gap guard. If long gaps are common in real sessions, the answer is **more anchors**,
  not a better interpolator (evidence: error is flat to ~200-frame gaps).
- The gate is not fully automated (visual step needs a human) — accepted by decision;
  numeric gate is CI-automatable, visual is a labeler-time confirmation.
- Sparse/no-line frames (193, 134) will be **yellow, not green** by design — the fix for
  them is line coverage (add a near-touchline click), not a model change.
