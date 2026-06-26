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
SQPNP pose) — fold-free, drift-free. The full-game A/B run also surfaced two
concrete issues to fix on integration:
1. **Outlier clicks poison the shared focal.** Two gross mislabels (frames 8526,
   56938) inflated the joint-calibration focal enough to false-reject ~16 otherwise-
   good frames (a whole-view rejection, no per-point robustness).
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

### Per-point RANSAC — `calib/` + `pitch/calib_anchor.py`
Outlier clicks must be dropped in **two** places:
- **Robust focal bootstrap.** `calibrate_clicked_frames` gains a per-point RANSAC
  pre-filter (a `robust=True` path): with an initial focal guess (frame width), run
  `cv2.solvePnPRansac` per anchor frame to get the **inlier clicks**, then estimate
  the shared focal via `calibrate_camera` on the **inliers only**. A gross mislabel
  can no longer inflate the focal and false-reject good frames.
- **Per-frame pose.** `poses_by_click_propagation` switches its per-frame
  `cv2.solvePnP(SQPNP)` to `cv2.solvePnPRansac` (frozen `K`): it returns the pose
  **plus an inlier mask**, so chain-drifted propagated neighbours and real mislabels
  are dropped per frame. The RANSAC reprojection threshold is a parameter (≈8–10 px).
  `poses_by_click_propagation` also gains a `frames=` argument (restrict the targets)
  so the labeler can recompute just the touched window. Per-frame **outlier clicks**
  (kp_idx of dropped points) are returned for UI flagging, and the frame's
  `residual_px` / `fold_count` are computed on the **inlier** set (not the dropped
  outliers).

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
  `LabelerState` vs the free fit — folds 361 → 0, the 18 previously-rejected frames
  recovered, the 2 real outlier clicks (8526, 56938) flagged, coverage ≥ the free fit,
  incremental recompute stays interactive. (Runnable locally — CPU, data on hand.)

## Go/no-go (Phase 3b-1 success)
- Fold-free in the actual labeler (≈0 folded frames vs 361).
- The 18 previously-rejected frames calibrate (per-point RANSAC un-poisons the focal);
  the 2 genuine outlier clicks are flagged, not whole frames.
- Coverage ≥ the free fit; green rate up; held-out accuracy ~7 ft.
- Incremental recompute stays interactive (windowed, ~free-fit cost); bootstrap +
  full recompute on load is acceptable (≈ Engine A's 8 s for the full game).

## Deferred to Phase 3b-2
Line-click UI + line storage/propagation + `refine_pose`-with-lines (SQPNP for point
frames, refine for line frames) for the **near touchline and midline**; real
line-anchor validation. (Schema additions remain a separate optional pass.)
