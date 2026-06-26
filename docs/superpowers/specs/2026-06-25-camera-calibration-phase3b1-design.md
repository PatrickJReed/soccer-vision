# Camera Calibration — Phase 3b-1: Calibrated Labeler Backend + Per-Point RANSAC (design)

**Date:** 2026-06-25
**Status:** Design approved, pending implementation plan
**Depends on:** Phase 3a (`pitch/calib_anchor.py`: `poses_by_click_propagation`,
`calibrate_clicked_frames`, `frame_homography`, `FramePose`, `propagate_clicks`),
Phase 1 (`calib/calibrate.py`: `calibrate_camera`), the labeler
(`labeler/state.py:LabelerState`, `labeler/server.py`, `labeler/chain.py`), OpenCV
(`cv2.solvePnPRansac`), the validated A/B finding (Engine A — SQPNP-only — is the
robust per-frame engine; 8 s/full-game, fold-free, ~7 ft).

## Problem

The labeler still fits a **free per-frame homography** (`manual_anchor.fit_frame_homographies`),
which **folds** on shallow frames (361 folded frames on the full game) and drifts.
Phase 3a proved the fix: per-frame **camera calibration** (shared focal + per-frame
SQPNP pose) — fold-free, drift-free. Two integration issues remain (both confirmed
against the full-game data):
1. **No per-point robustness in the per-frame fit.** Engine A's per-frame `solvePnP`
   uses every click, so a gross mislabel corrupts that frame's pose, and the
   chain-drifted neighbour clicks that window-propagation pulls in add noise. (NB:
   the shared focal itself is *robust* — it comes out 1469 px with or without the two
   gross outliers, and Engine A's per-frame SQPNP already fits the shallow frames at
   the converged focal; calibrate's one-pass view-rejection drops ~16 of them from
   the *focal-seed set* but Engine A fits them anyway. So robustness is needed at the
   per-frame fit, not the focal.)
2. **No interactive recompute model** for a shared-focal engine yet (3a computed
   whole-game batches).

## Goal (Phase 3b-1)

Replace the labeler's free-fit engine with the **calibrated SQPNP engine (point-only)**
plus **per-point RANSAC**, under a **freeze-focal** incremental model — so the
interactive labeler produces fold-free, drift-free homographies with the same fast
windowed recompute it has today, and outlier clicks are dropped (not whole frames)
and flagged back to the user.

## Non-goals (Phase 3b-2 and later)

- The **line-click UI**, line-click storage/propagation, `refine_pose`-with-lines,
  and the **near touchline + midline** anchors (all Phase 3b-2).
- Landmark-schema additions (6-yard goal box, circle∩midline) — deferred accuracy
  pass; the synthetic test showed only ~5–9 % gain and they don't fix the rejections.
- Engine B / the free fit as the primary engine (out).
- The model-path (keypoints → pose) (later).

## Design

### Per-point RANSAC — `pitch/calib_anchor.py` (per-frame fit only)
Outlier clicks are dropped at the **per-frame fit** (the focal is already robust, so
there is no robust-bootstrap step — verified: the focal is 1469 px with or without
the outliers). `poses_by_click_propagation` switches its per-frame
`cv2.solvePnP(SQPNP)` to `cv2.solvePnPRansac` (with the frozen `K`): it returns the
pose **plus an inlier mask**, so the gross mislabels (8526, 56938) and the
chain-drifted propagated-neighbour clicks the window pulls in are dropped per frame.
The RANSAC reprojection threshold is a parameter (≈8–10 px). `poses_by_click_propagation`
also gains a `frames=` argument (restrict the targets) so the labeler can recompute
just the touched window. Per-frame **outlier clicks** (kp_idx of dropped points) are
returned for UI flagging, and the frame's `residual_px` / `fold_count` are computed on
the **inlier** set. `calibrate_clicked_frames` is unchanged — the 152-frame focal seed
is already clean.

### Freeze-focal recompute model — `labeler/state.py`
The Trace camera focal is **physically constant** (fixed lens, no zoom), so:
- **Bootstrap once.** When ≥ 3 calibratable anchor frames exist, estimate the shared
  focal (robust bootstrap above) and **freeze** it as `LabelerState._K`; set
  `_calibrated = True`.
- **Incremental recompute = windowed re-SQPNP with the frozen K.** `_refit(frames)`
  keeps the existing `_affected(frame)` window machinery but now calls
  `poses_by_click_propagation(frames=affected, k=self._K, ...)` (per-point RANSAC,
  fixed focal) instead of `fit_frame_homographies`. Same ~135 ms incremental cost.
- **`recalibrate()` action** re-estimates and re-freezes the focal (for when early
  anchors were poor); a full recompute follows.
- **Bootstrap gap:** with < 3 anchors, `_calibrated` is False → no calibrated
  homographies yet → the UI shows a "place more anchors" state.

The window default stays **360** (shallow goal-views depend on cross-pan click
aggregation; per-point RANSAC drops the chain-drifted neighbours the window pulls in
— window for coverage + RANSAC for robustness).

### `LabelerState` integration — keep the server/UI/export contract
- Per-frame record becomes a calibrated pose: store `FramePose` (rvec, tvec,
  `residual_px`, `n_points`, `fold_count`) per frame in `_fits`, plus a per-frame
  list of flagged outlier `kp_idx`.
- **Status / coverage:** green = covered (has a pose) AND `residual_px ≤ threshold`;
  red = uncovered. Calibration can't fold, so there's no fold-based red and the green
  rate rises. `frame_status` / `coverage_fraction` / the bucketed timeline adapt to
  the new record (residual in px, threshold tuned from the ~7 ft / held-out result).
- **Export unchanged downstream:** `export` writes `frame_homography(K, rvec, tvec)`
  (full-pixel image→pitch — the existing parquet format) for covered frames, plus
  `keypoints.parquet` as today; optionally also write the flagged-outlier list.
- **Resume / autosave:** unchanged click sidecar; on load, bootstrap the focal then
  one chunked recompute (mirrors `add_clicks`).

### Coordinate spaces
Same discipline as 3a: full-pixel internally; normalized clicks ×(w, h) and the
normalized chain converted in; output is full-pixel image→pitch (export format). No
change to the labeler's normalized click handling.

## Error handling
- < 3 calibratable anchor frames → `_calibrated = False`, no homographies (bootstrap
  gap), counted; UI prompts for more anchors.
- A target frame with < `min_points` propagated landmarks, or `solvePnPRansac` finding
  too few inliers / failing → that frame uncovered, counted (not crashed).
- `recalibrate()` with an implausible focal (Phase-1 guard) → keep the prior frozen K
  and surface a calibration-failure message.
- Empty clicks / pre-bootstrap → empty result.

## Testing
- **Per-point RANSAC (synthetic):** an anchor frame with one planted gross-outlier
  click → RANSAC drops it (inlier mask excludes it), the pose is recovered, and the
  bootstrapped focal matches the no-outlier focal (outlier doesn't poison it).
- **Freeze-focal model:** bootstrap from N diverse synthetic anchors → frozen K within
  tolerance; a subsequent `add_click` triggers windowed re-SQPNP with that K (not a
  re-calibration); `recalibrate()` re-estimates.
- **`LabelerState` behavior:** `add_click` recomputes exactly `_affected(frame)`;
  coverage/status reflect fold-free calibrated frames; `export` writes the full-pixel
  image→pitch parquet; resume-from-sidecar reproduces the session.
- **Regression:** the pure `manual_anchor` free-fit functions and their tests are left
  intact (the free fit stays available for the `compare_engines` tool); only
  `LabelerState` switches engines.
- **Real-data validation (the decisive run, on Patrick's full-game clicks):** the new
  `LabelerState` vs the free fit — folds 361 → 0; Engine A fits every frame (incl. the
  shallow goal-views); the two gross-mislabel frames (8526, 56938) get a clean pose
  via per-frame RANSAC with the bad click **flagged**; coverage ≥ the free fit;
  incremental recompute stays interactive. (Runnable locally — CPU, data on hand.)

## Go/no-go (Phase 3b-1 success)
- Fold-free in the actual labeler (≈0 folded frames vs 361).
- Engine A produces a pose for every covered frame (incl. the shallow goal-views);
  the 2 gross-mislabel frames get a clean per-frame-RANSAC pose with the bad click
  flagged (not the whole frame dropped).
- Coverage ≥ the free fit; green rate up; held-out accuracy ~7 ft.
- Incremental recompute stays interactive (windowed, ~free-fit cost); bootstrap +
  full recompute on load is acceptable (≈ Engine A's 8 s for the full game).

## Deferred to Phase 3b-2
Line-click UI + line storage/propagation + `refine_pose`-with-lines (SQPNP for point
frames, refine for line frames) for the **near touchline and midline**; real
line-anchor validation. (Schema additions remain a separate optional pass.)
