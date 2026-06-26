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

### Outlier-click flagging — a CLICKED-FRAME preprocessing step (`pitch/calib_anchor.py`)
Outlier clicks are detected **once, per clicked frame, on that frame's own clicks** —
NOT per-frame across the propagated set (verified why: a per-propagated-frame robust
fit drops chain-drifted neighbour clicks on 25-80% of the 58k frames — useless flags,
and the leave-one-out variant takes 140s; the focal is robust 1469px either way).

- `_robust_sqpnp(k, ids, img, *, thr=40, min_points=4)`: planar-safe robust PnP — NOT
  `cv2.solvePnPRansac`, which degenerates on the coplanar (Z=0) field (verified garbage).
  SQPNP the set; if the worst reprojection residual exceeds `thr`, drop a point
  (leave-one-out: try removing each candidate, drop the one whose removal gives the
  lowest max residual — avoids masking) and refit; repeat until the worst is within
  `thr` or `< min_points` remain. Returns `(rvec, tvec, inlier_ids, outlier_ids)`.
- `flag_outlier_clicks(clicks, k, size, *, thr=40, min_points=4) -> (clean_clicks, flagged)`:
  group by frame; per clicked frame run `_robust_sqpnp` on the OWN clicks; drop the
  outlier clicks from `clean_clicks` and record `flagged: {frame: [kp_idx]}`. Verified
  on real data: flags exactly the 2 gross mislabels (8526 circle_near, 56938 circle_far)
  plus ~38 imprecise clicks across 40 clicked frames (cheap — only ~170 frames).

`poses_by_click_propagation` stays the plain fast 3a engine (SQPNP per frame, ~7s for
the whole game on cleaned clicks) and only gains a `frames=` argument (restrict the
targets) for windowed recompute — it does NOT do per-frame outlier rejection. `FramePose`
is unchanged (no `outliers` field — flags are session-level). `calibrate_clicked_frames`
is unchanged — the 152-frame focal seed is already clean.

### Freeze-focal recompute model — `labeler/state.py`
The Trace camera focal is **physically constant** (fixed lens, no zoom), so:
- **Bootstrap once.** When ≥ 3 calibratable anchor frames exist, estimate the shared
  focal (robust bootstrap above) and **freeze** it as `LabelerState._K`; set
  `_calibrated = True`.
- **Outlier preprocessing on (re)calibration.** When the focal is (re)frozen, run
  `flag_outlier_clicks` on the clicks → a cleaned click list + `self._outliers:
  {frame: [kp_idx]}`. Engine A runs on the **cleaned** clicks; the flags are surfaced
  to the UI.
- **Incremental recompute = windowed re-SQPNP with the frozen K.** `_refit(frames)`
  keeps the existing `_affected(frame)` window machinery but now calls
  `poses_by_click_propagation(frames=affected, k=self._K, ...)` (plain fast SQPNP,
  fixed focal, on the cleaned clicks) instead of `fit_frame_homographies`. Same ~135 ms
  incremental cost.
- **`recalibrate()` action** re-estimates + re-freezes the focal and re-runs the
  outlier preprocessing (for when early anchors were poor); a full recompute follows.
- **Bootstrap gap:** with < 3 anchors, `_calibrated` is False → no calibrated
  homographies yet → the UI shows a "place more anchors" state.

The window default stays **360** (shallow goal-views depend on cross-pan click
aggregation; plain SQPNP averages the chain-noise the window pulls in — robustness is
the clicked-frame preprocessing, not per-frame).

### `LabelerState` integration — keep the server/UI/export contract
- Per-frame record becomes a calibrated pose: store `FramePose` (rvec, tvec,
  `residual_px`, `n_points`, `fold_count`) per frame in `_fits`, plus the session-level
  `self._outliers: {frame: [kp_idx]}` from the preprocessing step.
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
- A target frame with < `min_points` propagated landmarks, or SQPNP failing → that
  frame uncovered, counted (not crashed). `_robust_sqpnp` returning None on a clicked
  frame (preprocessing) → keep that frame's clicks unflagged.
- `recalibrate()` with an implausible focal (Phase-1 guard) → keep the prior frozen K
  and surface a calibration-failure message.
- Empty clicks / pre-bootstrap → empty result.

## Testing
- **Outlier flagging (synthetic):** `_robust_sqpnp` on a point set with one planted
  gross-outlier click drops exactly that landmark and returns a clean pose; on clean
  clicks it drops nothing. `flag_outlier_clicks` on a session with one planted mislabel
  at a frame removes that click from `clean_clicks` and records it in `flagged`.
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
