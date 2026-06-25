# Camera Calibration — Phase 2: Line Constraints (design)

**Date:** 2026-06-25
**Status:** Design approved, pending implementation plan
**Depends on:** Phase 1 (`soccer_vision.calib`: `field_model`, `calibrate`,
`CalibResult`), scipy (`scipy.optimize.least_squares`), OpenCV.

## Problem

Phase 1's calibration is fold-free and drift-free, but two pitch regions stay
under-constrained because they can't be pinned by point landmarks (Patrick's
overlay critique):
- **The near touchline (x=0)** is under the camera — its landmarks aren't
  cleanly clickable, so the near edge is weakly constrained.
- **The midline** can only be pinned at one end: `halfway_far` (idx 4) is
  clickable but `halfway_near` (idx 5) is hidden under the camera, so only the far
  portion of the visible midline is constrained.

Both are **lines we know exactly** even though their endpoint landmarks aren't
clickable. The established sports-vision fix (TVCalib, PnLCalib) is to add
**line correspondences** — a clicked pixel asserted to lie *on* a known field line
— to the calibration objective. This Phase builds that math.

## Goal (Phase 2)

The **line-constraint refinement**: given a camera pose seeded from Phase 1 and a
mix of point + line observations, refine the per-frame pose by minimizing point
reprojection error **and** line-to-projected-line distance. Validated rigorously
on **synthetic** data — show that a line constraint tightens an otherwise
under-constrained (shallow) view toward the true pose.

## Non-goals (Phase 3)

- Labeler UI for clicking lines (Phase 3 — the line-click mode).
- Real-data validation (Phase 3, once the UI provides real line clicks).
- Refining the shared focal (Phase 1's global focal is held fixed here; lines
  refine the per-frame pose only).
- Replacing the labeler's per-frame fit / per-frame propagation (Phase 3).

## Design

### Field-lines registry — `calib/field_model.py` (addition)
- `FIELD_LINES: dict[str, tuple[int, int]]` — named pitch lines as pairs of
  landmark indices whose 3D positions are the line's endpoints:
  - `near_touchline` = (0, 2)  (corner_own_left → corner_opp_left, x=0)
  - `far_touchline` = (1, 3)
  - `own_goal_line` = (0, 1)
  - `opp_goal_line` = (2, 3)
  - `midline` = (5, 4)  (halfway_near → halfway_far — the FULL halfway line, even
    though idx 5 isn't itself clickable; the line is still known exactly)
- `field_line_3d(line_id: str) -> tuple[NDArray(3,), NDArray(3,)]` — the two 3D
  metre endpoints (from `field_points_3d()` at the two landmark indices).
- Pure, tested. (Box edges can be added later; the five above cover the gap.)

### Line residual — `calib/calibrate.py` (addition)
- `line_residual(h_world, p1_3d, p2_3d, clicked_px) -> float`: project the line's
  two 3D endpoints through `h_world` to image points `a, b`; form the image line
  `ℓ = cross([a,1], [b,1])`; return the perpendicular distance from `clicked_px`
  to `ℓ` (`|ℓ·[u,v,1]| / hypot(ℓ0, ℓ1)`). Well-defined when endpoints project
  off-screen (the near-touchline/midline case). Pure, tested.

### Pose refinement — `calib/calibrate.py` (addition)
- `refine_pose(K, rvec0, tvec0, point_obs, line_obs, *, min_constraints=6) ->
  tuple[rvec, tvec]`:
  - `point_obs: list[(kp_idx, x_px, y_px)]`, `line_obs: list[(line_id, x_px, y_px)]`.
  - Residual vector: per point, the 2 reprojection components (`projectPoints`
    minus clicked); per line click, the 1 line-distance. Optimize `[rvec(3),
    tvec(3)]` with `scipy.optimize.least_squares` (method `'trf'`), seeded at
    `(rvec0, tvec0)`. `K` (focal) fixed.
  - Constraint count = `2*len(points) + len(lines)`. Raise `CalibError` if
    `< min_constraints` (a 6-DOF pose needs ≥6 residuals).
  - Returns the refined `(rvec, tvec)`. Pure (numpy/scipy/cv2), tested.

### How it composes (for Phase 3, not built here)
Phase 3 will: run Phase 1 `calibrate_camera` (points → shared focal + seed poses),
then `refine_pose` per frame with that frame's points + line clicks. `refine_pose`
is built standalone so Phase 3 just feeds it real line clicks.

## Error handling
- `field_line_3d` on an unknown `line_id` → `KeyError` with the valid names.
- `refine_pose` with `< min_constraints` → `CalibError`.
- `scipy.optimize.least_squares` fails to converge / returns non-finite → raise
  `CalibError` with the seed pose noted (caller can fall back to the seed).
- A line whose two endpoints project to the same pixel (degenerate projected
  line) → guard the `hypot(ℓ0, ℓ1) ≈ 0` case and contribute a **0 residual** for
  that line on that evaluation (NOT a dropped entry — `least_squares` needs a
  constant-length residual vector across iterations, so the count can't change),
  logged once. `line_residual` returns `0.0` in this degenerate case.

## Testing
- **Field lines:** `field_line_3d("near_touchline")` endpoints == field_points_3d
  at idx 0 and 2; unknown id raises; `midline` spans x=0 to x=WIDTH_M at y=L/2.
- **Line residual:** a pixel exactly on a projected line → ~0; a pixel a known
  perpendicular px off → that distance; endpoints projecting off-screen still
  yield a correct finite distance.
- **The key test — line constraint tightens a shallow view:** construct a known
  camera + a shallow/under-constrained point set (few points near one goal, like
  the folding frame). (a) `refine_pose` with points only recovers a pose whose
  held-out reprojection error is loose; (b) adding a near-touchline (or midline)
  line constraint recovers a pose markedly closer to the TRUE pose (lower
  held-out error / closer rvec,tvec). Assert (b) is meaningfully better than (a).
- **Round-trip:** with full point+line obs from a known camera, `refine_pose`
  recovers the true pose within tolerance and residuals ~0.
- **Constraint floor:** `< min_constraints` raises `CalibError`.
- **Degenerate line guard:** `line_residual` with coincident projected endpoints
  returns `0.0` (no crash, no division by zero); `refine_pose` keeps a
  constant-length residual vector.

## Deferred to Phase 3
- Labeler line-click UI + storing line clicks alongside point clicks.
- Real-data validation (does it tighten the near touchline/midline on Patrick's
  actual footage).
- Integration into the per-frame calibration + propagation for unclicked frames.
- Optionally folding a global line-aware bundle step (refine focal too) if the
  per-frame refinement proves insufficient.
