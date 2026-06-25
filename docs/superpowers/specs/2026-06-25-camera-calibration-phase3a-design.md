# Camera Calibration — Phase 3a: Calibration Registration Engine + Real-Data Comparison (design)

**Date:** 2026-06-25
**Status:** Design approved, pending implementation plan
**Depends on:** Phase 1 (`calib.calibrate.calibrate_camera`, `homography_from_pose`,
`pitch_homography`, `CalibResult`), Phase 2 (`calib.calibrate.refine_pose`,
`line_residual`, `field_line_3d`), Phase 1 (`calib.validate.fold_count`,
`leave_one_out_feet`), the labeler's registration chain
(`pitch.manual_anchor.build_segments`, `cumulative_transforms`, the
window-propagation in `fit_frame_homographies`), OpenCV, scipy.

## Problem

The labeler's per-frame engine (`manual_anchor.fit_frame_homographies`) fits a
**free 8-DOF homography per frame** from window-propagated clicks. Two measured
failures: **folding** on shallow-depth frames and **drift/collapse** over a full
game (~106 ft, degenerate at 58k frames). Phases 1–2 built a drift-free, fold-free
**camera-calibration** core (shared focal + per-frame pose, point + line
constraints), validated on synthetic cameras and (Phase 1) on the full-game clicks.

What's missing is the **production per-frame engine**: turning a session's clicks +
the registration chain into a calibrated homography for **every** frame of the
video (clicked and unclicked), in the labeler's output format — and a decisive
**real-data comparison** proving it beats the free fit before any labeler
integration (Phase 3b).

## Goal (Phase 3a)

Build **two** calibration-based per-frame engines and compare them head-to-head
against the current free fit on the **full training-game session**:

- **Engine A — propagate clicks, then calibrate** per frame (`refine_pose`, focal
  fixed).
- **Engine B — calibrate clicked frames, then propagate the pose** to unclicked
  frames via the chain's recovered camera rotation.

Decisive go/no-go: at least one engine is fold-free and drift-free across the whole
game, coverage ≥ the free fit, accuracy competitive/better, with a clear verdict to
carry into 3b. **No labeler integration, no UI** in 3a (those are 3b).

## Non-goals (Phase 3b and later)

- Wiring either engine into `LabelerState` / `server.py` / export (3b).
- The line-click frontend UI + storing line clicks (3b). 3a wires `line_obs` into
  the engine interface and tests it synthetically, but the **real full-game run is
  point-only** (no line clicks exist until the UI does).
- Real line-click validation (3b — needs the UI to produce line clicks).
- Replacing / deleting `manual_anchor.fit_frame_homographies` (3b decides the swap;
  3a is a parallel module).
- Model-path (keypoint model → pose) (later).
- Incremental / interactive recompute (3a computes whole-video batches; the
  interactive path is a 3b concern, informed by 3a's runtime numbers).

## Architecture

A new pure module **`src/soccer_vision/pitch/calib_anchor.py`** (parallel to
`manual_anchor.py`; nothing in `labeler/` changes in 3a). Three layers:

### Shared front-end
- `calibrate_clicked_frames(clicks, size, *, min_points=6) -> tuple[K, poses]`:
  group clicks by frame, keep frames with `>= min_points` distinct landmarks
  ("calibration views"), run Phase-1 `calibrate_camera` on those **directly-clicked**
  frames → one shared focal `K` (3x3) + `poses: dict[frame, (rvec, tvec)]` for the
  clicked frames. Clicks here are in the engine's working pixel space (see
  "Coordinate spaces"). Raises `CalibError` if too few/degenerate clicked views.
- Registration chain reused as-is: `build_segments`, `cumulative_transforms` from
  `manual_anchor` give `segment_of` and `M[f]` (frame `f` pixels → segment-reference
  pixels).

### Engine A — `poses_by_click_propagation(...) -> dict[frame, FramePose]`
For each target frame `g`:
1. Window-propagate clicks into `g` (shared helper, below): same segment,
   `|click.frame - g| <= window`, nearest click per landmark wins → propagated
   `point_obs` (and `line_obs` when present).
2. If `>= min_points` distinct propagated landmarks (constraints `>= 6` once lines
   count): `refine_pose(K, seed_rvec, seed_tvec, point_obs, line_obs)` with focal
   fixed. `seed` = nearest clicked frame's pose in the segment.
3. `refine_pose` raising `CalibError` → frame uncovered (counted, not crashed).

### Engine B — `poses_by_pose_propagation(...) -> dict[frame, FramePose]`
For each target frame `f`:
1. Nearest directly-clicked frame `c` in the same segment (else uncovered).
2. Chain transform `G_{c->f} = inv(M[f]) @ M[c]` (frame `c` px → frame `f` px),
   converted to working-pixel space.
3. Recover relative rotation: `Mr = K^-1 @ G_px @ K` ≈ `R_f R_c^T` up to scale;
   SVD `Mr = U S V^T`; nearest rotation `R_rel = U @ diag(1,1,det(U V^T)) @ V^T`.
   `R_f = R_rel @ R_c`.
4. Fixed optical centre: `C = -R_c^T @ t_c` (constant for a pure-rotation camera);
   `t_f = -R_f @ C`. `rvec_f = Rodrigues(R_f)`.

### Common back-end (both engines)
- `FramePose` dataclass: `rvec`, `tvec`, `residual_px` (calib reprojection RMS in
  **pixels** — the calib-native residual from `refine_pose`/the pose), `n_points`
  (propagated landmarks used; 0 for B / pose-propagated), `fold_count`.
- `frame_homography(K, rvec, tvec) -> H_img2pitch_px`: the **full-pixel image →
  pitch-[0,1]²** homography = `inv(pitch_homography(homography_from_pose(K, rvec,
  tvec)))`. This is exactly the labeler's **export** format (what
  `LabelerState.export` writes after denormalizing), so it is directly comparable
  to the free fit's export and drop-in for the downstream `PitchMapper` — no
  output-side normalization needed (`calib_anchor` works in full-pixel throughout).
- Confidence / status: `fold_count` + the residual; `frame_status` / coverage
  mirror `manual_anchor`'s helpers. **For the head-to-head, "green"/coverage is
  defined by a single uniform held-out-feet threshold applied to all three engines**
  (not each engine's native residual units), so the free fit and calib are compared
  on the same accuracy basis. (The labeler's own interactive green threshold is a
  3b concern.)

### Shared propagation helper (DRY)
Factor the vectorized window-propagation in `manual_anchor.fit_frame_homographies`
(lines ~131–188: project each click into each target frame, same-segment +
`|Δframe| <= window`, nearest-click-wins per landmark) into a reusable function
`propagate_clicks(clicks, transforms, segment_of, *, window, frames) ->
dict[frame, dict[kp_idx, (x, y)]]`. Both `fit_frame_homographies` (free fit) and
Engine A call it — one implementation, no copy-paste. The free fit's own behavior
must be unchanged (covered by its existing tests).

## Coordinate spaces (the integration's main hazard)

- The labeler works in **normalized [0,1] image space** (clicks are normalized
  canvas fractions; the chain `M[f]` is normalized) and exports **image-pixel →
  pitch-[0,1]²** homographies (denormalized on export).
- The calib core works in **full-pixel image space** against **world-metres** on
  Z=0 and produces **world-metres → pixel** (`homography_from_pose`) /
  **canonical-[0,1]² → pixel** (`pitch_homography`).
- **Decision:** `calib_anchor` works internally in **full-pixel** image space.
  Convert normalized clicks → pixel (`x_px = x * W`) on the way in, and convert the
  chain `M_norm` → pixel (`M_px = D @ M_norm @ D^-1`, `D = diag(W, H, 1)`) for
  Engine B's decomposition. The back-end emits the **full-pixel image → pitch-[0,1]²**
  homography, which is exactly the labeler's export format — so it is directly
  comparable to the free fit's export and drop-in for the downstream. Every
  conversion is unit-tested. (Only the input clicks and Engine B's chain need
  normalized→pixel conversion; the output needs none.)

## Comparison harness
A script/notebook (`examples/calib_anchor_compare.ipynb` + a spawn-safe `.py`
helper for the heavy run) the **user** runs on the full-game `.npz` chain + the
session's clicks. For each of {free-fit baseline, Engine A, Engine B} it reports,
over the full game:
1. **Coverage:** # / % frames with a homography; # green (residual ≤ threshold).
2. **Fold-free:** per covered frame, `fold_count`; histogram + count of folded
   frames (≫ plausible slice size).
3. **Drift:** held-out reprojection error (feet) vs frame number; must be FLAT.
4. **Accuracy:** leave-one-landmark-out held-out error (feet), median / 90th
   (reuse `validate.leave_one_out_feet`).
5. **Runtime:** wall-clock to compute all per-frame homographies per engine.
Plus rendered overlays at reference frames for the user to assess (Claude renders;
the **user interprets** — no self-assessment of images).

## Go/no-go criteria (Phase 3a success)
- **Fold-free:** ≥1 of A/B has ≈0 folded frames across the full game (vs the free
  fit's folds).
- **Drift-free:** ≥1 of A/B has held-out error flat across 58k frames (no growth /
  collapse), unlike the free fit.
- **Coverage:** ≥1 of A/B ≥ the free fit's covered/green count.
- **Accuracy:** held-out median competitive with or better than the free fit, no
  catastrophic frames.
- **Verdict:** a clear winner, or a principled "A near clicks, B to fill gaps"
  split, to carry into 3b.

## Error handling
- `calibrate_clicked_frames` with too few/degenerate clicked views → `CalibError`
  (propagated from `calibrate_camera`).
- Engine A: a frame with `< min_points` propagated landmarks (or `refine_pose`
  raising) → uncovered, counted.
- Engine B: a frame with no in-segment clicked neighbour → uncovered, counted; a
  degenerate / non-rotation chain transform → SVD nearest-rotation (graceful),
  flagged via its `fold_count` / residual; a clicked pose whose centre `C` is
  ill-defined (`t_c` ~ 0) → counted as a B failure for that frame.
- Empty clicks / empty chain → empty result (no crash).

## Testing
- **Field/back-end:** `frame_homography` round-trips a known pose to the full-pixel
  image→pitch form within tolerance (a canonical-[0,1]² point maps to its projected
  pixel and back); the normalized↔pixel chain conversion (`M_px = D M_norm D^-1`)
  is correct.
- **Engine A (synthetic):** a known camera + several poses → synthetic clicks on a
  subset of frames + a synthetic chain linking them; Engine A recovers the true
  per-frame poses on covered frames within tolerance; reprojection ~0; a shallow
  frame does NOT fold.
- **Engine B (synthetic, the key test):** build a synthetic chain from KNOWN
  inter-frame rotations of one fixed-centre camera; Engine B's rotation-from-chain
  (`K^-1 G K` → SVD → `R_rel`) recovers the true per-frame rotations and poses
  within tolerance (round-trip). A non-rotation perturbation → nearest rotation, no
  crash.
- **Shared propagation helper:** `propagate_clicks` reproduces the propagation the
  free fit used (a regression test pinning `fit_frame_homographies` output before/
  after the refactor is identical).
- **Coverage/status helpers:** mirror `manual_anchor`'s tested behavior.
- The **comparison harness**: structure / smoke only in CI (the real run is the
  user's, on real data).

## Deferred to Phase 3b
- Wire the chosen engine into `LabelerState` / server / export; the line-click UI;
  incremental/interactive recompute; real line-click validation; the
  free-fit-vs-calibration swap decision.
