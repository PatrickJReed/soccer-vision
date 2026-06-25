# Drift-Free Camera-Calibration Field Registration — Phase 1 (design)

**Date:** 2026-06-25
**Status:** Design approved, pending implementation plan
**Depends on:** `pitch/spec.py` (`PitchSpec`, field dimensions), `pitch/landmarks.py`
(`PITCH_LANDMARKS`, 21-pt schema), the labeler's exported clicks
(`keypoints.parquet` / autosave sidecars), OpenCV (`cv2.calibrateCamera`,
`cv2.solvePnP`), scipy.

## Problem

The labeler fits a **free 8-DOF homography per frame** from whatever clicks fall
in a ±360-frame window. Two failure modes, both measured this session:

- **Folding.** When a frame's clicks span a shallow depth band (one goal area), the
  free homography is under-constrained and extrapolates the far field *into* the
  frame — mislabeling far landmarks as visible at folded positions (training
  audit: ≥2.6% of frames; the geometric cause is confirmed).
- **Drift.** A single global homography over a full game fails because the chained
  registration accumulates error — measured at **~25 ft on the 2-min clip, ~106 ft
  and a degenerate collapse on the full 58k-frame game**. The per-frame+window fit
  exists only to bound this drift to ±360 frames.

Both stem from inferring an unconstrained homography rather than solving the
**physically-constrained camera** against the **known rigid field**. The whole
sports-vision field (TVCalib, PnLCalib, SoccerNet winners) calibrates each frame
directly against the field model — drift-free (no chaining) and fold-free (a real
camera pose cannot fold). A proof-of-concept on the exact folding frame confirmed
it: a `SQPNP` camera pose kept 10/21 landmarks in view (the visible slice, not
21/21) and fit the clicked points *better* than the free homography (39 px vs
84 px).

## Goal (Phase 1)

Build and validate the **core drift-free calibration** on EXISTING labeled data
(`clip` + full `training` game clicks), proving it is **fold-free** and
**drift-free** across a full game, before any labeler integration. Decisive
go/no-go on real data.

Concretely: from the clicks (points) + the known 9v9 field model, estimate **one
shared camera intrinsic (focal)** across all frames + a **per-frame pose**, and
derive per-frame homographies that (a) don't fold on shallow frames and (b) keep
accuracy uniform across the whole game (no drift).

## Non-goals (later phases)

- Line-anchor input / line-to-line refinement (Phase 2 — the existing clicks are
  points; lines need a labeler UI change).
- Labeler integration / replacing the per-frame fit (Phase 3).
- Per-frame homography for UNCLICKED frames via pan/tilt propagation (Phase 3).
- Model-path (keypoint model → pose) (later).
- Lens-distortion modelling (assume none; Trace is a virtual-PTZ crop).

## Approach decisions (from research + PoC)

| Decision | Choice |
|---|---|
| Estimator | Physical **camera pose + shared focal**, NOT a free homography |
| Primitive | OpenCV `cv2.calibrateCamera` (shared K + per-frame poses) — permissive, robust, what SoccerNet winners used; `SQPNP` for single-frame pose |
| Field model | Our **9v9** dimensions from `PitchSpec` (3D planar, Z=0), NOT FIFA |
| Reuse | Build clean on OpenCV/scipy; PnLCalib/TVCalib as method references only (GPL / broadcast-trained, not vendored) |
| Drift control | Each frame solved directly against the field (no chaining) — drift-free by construction |
| Fixed-camera | Exploit it: ONE focal shared across all frames (the rigid camera), per-frame pose. Shared camera-centre constraint deferred (global focal alone is the key shared parameter). |

## Design

### Field model — `src/soccer_vision/calib/field_model.py`
- `field_points_3d(spec: PitchSpec) -> NDArray[(21, 3)]`: the 21 canonical
  landmarks as real-world metres on the Z=0 plane. `x = PITCH_LANDMARKS[:,0] *
  width_m`, `y = PITCH_LANDMARKS[:,1] * length_m`, `z = 0`. Field metres come from
  `PitchSpec` (9v9 default ≈ 45.7 m × 68.5 m); expose `width_m`/`length_m` (add to
  `PitchSpec` or a constant here, documented as the nominal 9v9 size).
- Pure, tested.

### Global calibration — `src/soccer_vision/calib/calibrate.py`
- `calibrate_camera(observations, frame_size, *, min_points=6) -> CalibResult`
  where `observations: Mapping[int, list[(kp_idx, x_px, y_px)]]` (clicks grouped
  by frame).
  1. Keep frames with `>= min_points` distinct landmarks (a calibration "view").
  2. Build per-view `objectPoints` (3D field metres) and `imagePoints` (px).
  3. `cv2.calibrateCamera` with an intrinsic guess, principal point fixed at the
     frame centre, and all lens-distortion coefficients fixed to zero → shared `K`
     (focal only) + per-view `rvec/tvec`. (Exact flag set is the plan's job; the
     intent is: estimate one focal across all frames, nothing else.)
  4. `CalibResult`: shared `K`, per-frame `(rvec, tvec)`, per-frame reprojection
     RMS, the list of calibrated frames, and `homography(frame) -> H` where
     `H = K @ [r1 | r2 | t]` (world-metres → pixel for the Z=0 plane).
- `K_init`: focal ≈ frame width (the PoC default), principal point at centre.
- Robustness: a per-view `SQPNP` pre-solve (the PoC showed SQPNP is the stable
  solver on shallow/near-collinear views) to seed/repair degenerate views before
  the joint `calibrateCamera`; drop any view whose reprojection RMS exceeds a
  threshold as an outlier, then refit.
- Pure except the cv2 calls; tested with synthetic cameras (below).

### Per-frame homography
- `CalibResult.homography(frame)` returns the world-metres→pixel homography. A
  pitch-normalised → pixel homography (for the existing `PitchMapper`) is
  `H @ diag(width_m, length_m, 1)` — provide `pitch_homography(frame)` returning
  the canonical-[0,1]² → pixel form so downstream code is unchanged.

### Validation harness — `examples/calib_validate.ipynb` + a CLI/script
Run on `clip` and `training` clicks (loaded from the autosave sidecars + the field
model). Reports, for each session:
1. **Fold-free:** per calibrated frame, reproject all 21 landmarks; count how many
   land in-frame. Histogram — must match slice sizes (~6–12), NOT 21. Directly
   compares to the free-homography fold (which gave 21/21 on shallow frames).
2. **Drift-free:** reprojection error (held-out, feet) vs frame number — must be
   FLAT across the full game, vs the chained homography's growth (4 ft → 95 ft).
3. **Accuracy:** leave-one-landmark-out per frame — fit the pose from the other
   clicks, measure the held-out landmark's reprojection error in feet (per-axis,
   `displacement_to_feet` from `eval/pitch_metrics`). Report median / 90th, and
   per-landmark. Compare head-to-head with the per-frame free homography on the
   same held-out clicks.
4. Rendered overlays (reference layout) for Patrick to assess — Claude renders,
   Patrick interprets ([[feedback-no-self-image-interpretation]]).

### Go/no-go criteria (Phase 1 success)
- Fold-free: shallow frames keep ≤ ~12 landmarks in-frame (no 21/21 folds).
- Drift-free: held-out reprojection error does NOT grow with frame number on the
  full game (flat within noise), unlike the chained homography.
- Accuracy: held-out median error competitive with (ideally better than) the
  per-frame free homography, with NO catastrophic frames.

## Error handling
- A frame with `< min_points` distinct landmarks → not calibratable in Phase 1
  (it's an unclicked/sparse frame; per-frame propagation is Phase 3). Counted,
  not crashed.
- `calibrateCamera` returns an implausible focal (e.g. negative, or > 50× frame
  width) → flag the session as a calibration failure with diagnostics (likely too
  few views or too little pose diversity).
- A degenerate view (`SQPNP`/`calibrateCamera` fails or RMS huge) → dropped from
  the global solve with a logged count.

## Testing
- **Field model:** corner/box/post metre positions match `PitchSpec` dimensions;
  shape `(21, 3)`, `z == 0`.
- **Synthetic round-trip (the key test):** construct a known camera (K, several
  poses), project the field points to synthetic "clicks", run `calibrate_camera`,
  assert it recovers the focal (within tolerance) and per-frame poses, and that
  reprojection error is ~0. Inject a shallow-band view and assert the recovered
  homography does NOT fold (far landmarks project off-screen, not in-frame).
- **Outlier view rejection:** add a degenerate/garbage view, assert it's dropped
  and the focal is still recovered.
- **Homography forms:** `pitch_homography` maps canonical [0,1]² landmarks to the
  synthetic pixels within tolerance.
- Validation notebook: structure/smoke only (the real run is on Patrick's data).

## Deferred to later phases
- Line constraints (Phase 2): add line correspondences to the scipy refinement
  (point reprojection + line-to-projected-line distance) — the line-anchor value.
- Shared camera-centre constraint (one fixed C, per-frame pan/tilt only) — a
  tighter model if global-focal-only proves insufficient on shallow frames.
- Labeler integration + per-frame pan/tilt propagation for unclicked frames
  (Phase 3).
