# Soccer-Vision — Codebase & Math Audit (2026-07-14)

**Method:** 5 parallel subsystem auditors (calibration/registration, Phase-4 metrics,
phase/possession, view-data engine, tracking/pipeline/export), each reading code + tests in
full, with key numerical claims verified empirically (RANSAC threshold behavior, findHomography
normalization, SQPNP on coplanar points, match-fraction symmetry, zone-tiling exactness).
Findings below were cross-checked against an independent manual read of the metrics core and
spot-verified in source. Baseline: prior audit `2026-06-26-codebase-audit.md`.

**Health snapshot:** 432 tests pass (3 deliberate skips), 89% coverage, `mypy --strict` clean,
ruff clean on `src/` + `tests/` (102 style hits confined to loose `scripts/` analysis files),
`master` in sync with `origin/master`.

---

## Bottom line

**The 2026-06-26 audit's findings were essentially all addressed — and addressed well.** All
11 concrete correctness bugs are verified FIXED-CORRECTLY (with tests), the two "unimplemented"
plan docs (own/opp grounding, pipeline-eval cleanup) turn out to be fully implemented, and the
product layer is no longer 0% built: Phase-4 team-shape + space-control metrics shipped
2026-07-01 with metrically sound core geometry.

The remaining problems concentrate in three places:

1. **The calibration engine still has the structural under-constraint it was built to kill**
   (per-anchor free translation — only the focal is shared, not the camera center), and the
   export layer stamps `confidence=1.0` on frames whose far end nothing ever measured.
2. **Phase-4 metrics have a validity hole, not a math hole**: Trace sees ~half the field, and
   shape/space numbers are computed from the visible subset but named as whole-team /
   whole-pitch quantities. Plus three genuine (smaller) math bugs.
3. **The view-data engine's within-game validation is near-tautological by construction**
   (adjacent-frame val split on a static background, eval includes training frames) — the
   embedding claim that matters is the held-out-views probe, which is clean but not headline.

---

## Part 1 — Prior findings: verified status (all checked against current code)

| 2026-06-26 finding | Status |
|---|---|
| Possession distances anisotropic | **FIXED** — `length_norm_xy` used for every distance; behavior-flip tests prove it (`possession.py:68-72`, `pitch/spec.py:65`) |
| defend split 0.5 vs 0.667 | **FIXED** — `OPP_THIRD_MIN_Y` live at `splitter.py:81`; boundary tests added, wrong test flipped not deleted |
| Transition misses own→contested→opp | **FIXED** — committed-label ffill turnover detection (`splitter.py:53-56`), tested |
| Agreement gate gameable by abstention | **FIXED (honest design)** — conditional agreement kept, but `pred_commit_rate`/`pred_contested_frac` exposed + anti-gaming test; direct `pct_contested` validation impossible with `{own,opp,none}` GT |
| Ball via `.iloc[0]` | **FIXED** — atomic `conf.idxmax()` row in both possession and pipeline |
| Halftime direction gap | **FIXED** — manual `halftime_frame` threaded through splitter, pipeline, metrics CLIs (but see F-M3: the *metrics* flip is wrong) |
| own/opp arbitrary cluster index | **FIXED on grounded path** — `analyze_video(own_kit=...)` chains hygiene kit-grounding (Lab color anchor, permutation-invariance test); ungrounded runs emit a loud "may be GLOBALLY INVERTED" warning. Grounding plan: implemented, all 13 tasks traced |
| Export partial-write after 30s timeout | **FIXED** — export drains worker fully, exports green-only, atomic autosave (but see F-P1: new infinite-hang risk) |
| `groupby().last()` Frankenstein ball | **FIXED** — `_highest_conf_ball_per_frame`, degenerate case tested |
| TrackingBackend protocol decorative | **FIXED** — `process_with_pitch` on the runtime-checkable protocol; MockBackend drives full `analyze_video` GPU-free |
| Frontend stale 0.05 threshold | **FIXED** — threshold served from state (60.0); readout now vestigial (always "—") |
| False-green status | **PARTIALLY FIXED** — session gate (`evaluate_gate`) is genuinely held-out and fails safe; per-frame green is by design foreground-observed + in-sample + fold-range, and export overclaims it (F-C2) |
| 8-DOF findHomography for ~2-DOF motion | **STILL PRESENT** — all three registration sites fit full projective H (`propagation.py:70,237`, `view_registration.py:84`); view_registration mitigates consequences (single hop) but doesn't restrict DOF |
| Legacy `manual_anchor.frame_status` 4-point false-green | **STILL PRESENT (dormant)** — off the live path, but exported API (F-C7) |

Refuted-and-still-refuted: `_infer_fps` rounding; eval 1.5/1.54 aspect ratio.

---

## Part 2 — New verified findings

### A. Calibration / registration (live accuracy ceiling)

**F-C1 (HIGH, confirmed): the physical engine shares the focal but NOT the camera center.**
`physical_calib.py:101` + `calibrate.py:220` — each anchor gets an independent 6-DOF
SQPNP+refine solve; nothing ties camera center C = −Rᵀt across frames. For a fixed-center
camera the true unknowns are one C (+ shared f) and per-frame rotation. With free translation,
a single-end anchor's solver trades tilt against height/depth along a near-null direction and
the unclicked far end swings — this is the surviving mechanism of "lines in the sky" and the
structural accuracy ceiling behind the measured 40–266 px long-span error.
**Fix (highest-leverage in the repo):** small bundle — one shared C (3 params, seeded from
median of per-anchor −RᵢᵀTᵢ) + per-anchor rotations (3 each) + shared f, over all anchors'
point/line clicks. This also makes single-end anchors solvable from 3 points and directly
implements the "global field-anchored solve" direction already chosen. At minimum: add a soft
prior penalizing camera-center dispersion in `refine_pose`.

**F-C2 (HIGH, confirmed): export stamps green frames "whole-field trustworthy, conf 1.0"; green
never measures the far end.** `state.py:386-393` vs `physical_calib.py:262-281`. Green =
foreground self-check + fold-count∈[4,15]; a pose whose far touchline flips behind the camera
(w<0) keeps 4–15 legitimate near landmarks and passes. The sky *rendering* is suppressed by
`clipped_polyline` (cosmetic). **Fix:** per-frame far-end gate (far touchline/goal-line
endpoints must project with w>0) + honest confidence derived from the session `GateReport`;
stop writing constant 1.0.

**F-C3 (MEDIUM, confirmed): propagated-frame green graded by distance to the *nearest green
anchor*, not the anchors actually used.** `physical_calib.py:276-281` — a frame built ~90%
from a yellow anchor can be green because a different green anchor is within GREEN_RADIUS.
**Fix:** grade from the bracketing anchors actually used; require distance-to-used ≤ radius.

**F-C4 (MEDIUM, confirmed): line clicks on non-anchor frames are silently inert.**
`solve_session` consumes only same-frame lines on frames with ≥4 points; `_line_obs` is defined
but never called; the UI implies effect (`state.py:223-229` docstring). Some of the 75 placed
line clicks are doing nothing. **Fix:** feed chain-propagated line observations into anchor
`refine_pose` (valid constraints), or notify when a line click lands on a non-anchor frame.

**F-C5 (MEDIUM, empirically confirmed): `fit_homography` default does ZERO outlier rejection.**
`homography.py:57-58` — RANSAC threshold defaults to 3.0 *in [0,1] pitch units*: 10/10 random
garbage correspondences ranked inliers in an empirical probe. Live on the ML-keypoint path via
`landmarks.py:136`; also `manual_anchor.py:273`. **Fix:** pass `ransac_thresh≈0.01–0.02`
(≈0.5–1 m) at these call sites, or fit in pixel space and invert.

**F-C6 (MEDIUM, confirmed): no behind-camera (w≤0) guard in live projection.**
`mapper.py:38` (`PitchMapper.transform`), `physical_calib.py:66`, `view_registration.py:228` —
near-horizon detections under a slightly-wrong H return finite, plausible, *mirrored* pitch
coords instead of NaN. Only `fold_count` guards w. **Fix:** NaN where `w ≤ eps` in
`PitchMapper.transform` at minimum.

**F-C7..C12 (LOW-MED):** legacy `manual_anchor.frame_status` exact-fit false-green (dormant —
delete/deprecate); `view_registration._nondegenerate` uses `abs(det)` so mirrored registrations
pass (drop `abs`, require det>0, normalize by h22); shared-K estimated from raw clicks *before*
outlier flagging (iterate once after flagging); `blend_homographies` element-blends
unnormalized chained H's (normalize by h22 first — `_bracket_h` already does it right);
focal gate 0.1w–50w near-vacuous (tighten to ~0.5w–5w); `cumulative_transforms` never
renormalizes h22 (free hygiene win).

**Verified correct (don't re-flag):** all composition orders (frame→rep→pitch, bracket shift,
chain forward/backward, normalize/denormalize); `homography_from_pose` ≡ projectPoints to 1e-6;
isotropic point/line feet metrics; genuinely held-out session gate (near-TL points AND lines
removed; leave-one-anchor-out; empty→inf→fail); `_robust_sqpnp` drop-worst; shared-f enforcement
flags in `calibrate_camera`; fold_count w-sign handling; view_registration bookkeeping,
honest gaps, coverage math; pipeline rep/registered coverage counting; export drain-then-write.

### B. Phase-4 metrics (the product layer)

**F-M1 (HIGH validity, confirmed): no visibility/coverage guard.** `shape.py:90`
(min_players=4), `space.py:108` (min_players=3) — Trace sees ~half the field, yet width/depth/
centroid/compactness are emitted as *team* shape from the visible subset, and Voronoi/influence
control is scored over **all** grid cells of the full pitch from visible players only. The
influence model sums Gaussians per player, so the more-visible team mechanically wins
everywhere. No camera-footprint mask exists although per-frame H is available.
**Fix:** project image bounds through H → score only visible cells + emit `pct_pitch_visible`;
shape: emit visible-fraction per frame/phase and either gate at n≥7 or rename outputs
visible-subset metrics. Every formula is correct *for the visible subset*; the names claim more.

**F-M2 (MEDIUM, confirmed + numerically demonstrated): per-third control = unweighted mean over
unequal-area zones.** `space.py:203,249` — channels are 0.14/0.23/0.26/0.23/0.14 of width;
wings-controlled third reads 40% vs true 28% (12 pp overstatement). **Fix:** area-weight
(`Σ pct_z·area_z / Σ area_z`); export n_cells per zone so consumers can't repeat the bug.

**F-M3 (MEDIUM, confirmed): halftime flip is a y-mirror, not a 180° rotation, in BOTH metrics
modules.** `shape.py:52-54`, `space.py:91`. End-swap = `(x,y)→(W−x, L−y)`; y-only reflection
mirrors left/right channels in H2, violating the zones' declared "own-team perspective" and
blending mirrored halves in per-phase means. Specs carry the same error (splitter's y-only flip
is fine — thirds don't depend on x). **Fix:** add `x_m → WIDTH_M − x_m` for frames ≥
halftime_frame in both files + spec errata. Width/depth/compactness/separation are x-flip
invariant; damage is channel-level space stats + `centroid_x_m`.

**F-M4 (LOW/decision):** metrics trust `homography_source != "none"` — contradicts
`pipeline.py:103-105` guidance (carry-forward frames have valid coords + source "none") and,
combined with F-C2, conf=1.0 manual sky-frames pass the gate. Decide the trust rule once:
e.g. `source != "none" AND confidence ≥ threshold`, after F-C2 makes confidence honest.

**Verified correct:** isotropy everywhere (both axes in metres before any distance — the old
anisotropy class did NOT recur); exact 15-zone tiling; mean-of-ratios-safe frame→phase control
aggregation (constant denominators); shape formulas match spec + hand computation; NaN
handling; GK handling per spec. Zone-boundary grid quantization (≤half-cell) noted, fine at 1m.

### C. Phase/possession — fresh findings

**F-P1 (MED-LOW, confirmed):** frames whose players are all team-`unknown` classify as
`loose_ball` (d_own=d_opp=inf) instead of `unknown`, inflating the loose share
(`possession.py:75-81`). **Fix:** early-return `unknown` when both masks empty.

**F-P2 (LOW latent, confirmed):** NaN player coords silently flip labels (NaN<x is False →
line 92 emits "opp"); safe in-pipeline (`filter_outside_pitch` drops NaN first) but
`classify_possession` is public API. **Fix:** `dropna(subset=["x_pitch","y_pitch"])`.

Notes: smoothing window 30→31 (centered windows must be odd — docstring nit); smoothing is
positional over detection-bearing frames (window spans >1s across gaps); splitter is an O(n)
Python loop (vectorize only if full-game perf ever hurts); team_mode tie→unknown verified sound.

### D. Tracking / pipeline / export — fresh findings

**F-T1 (MEDIUM, confirmed path): RefitWorker has no exception handling.** `refit_worker.py:94`
— any solver exception kills the daemon thread with `_inflight` stuck >0; `export()`'s new
no-timeout drain (`state.py:375-376`) then **hangs forever**, and all later clicks silently
stop refitting. **Fix:** try/except around compute+apply, error flag in `/api/state`,
`export()` raises if `not thread.is_alive()`.

**F-T2 (LOW-MED, plausible): `analyze_video` unconditionally overwrites `homographies.parquet`.**
`pipeline.py:312-313` — labeler (`state.py:394`) and view_registration write the same filename;
re-running the auto pipeline on a labeled game dir silently replaces hours of manual
calibration with the weaker auto source. **Fix:** refuse/warn when existing file has
source ∈ {manual, rep, registered}, or take a `homographies_path` override.

**F-T3..T8 (LOW):** `tracker_id is None` fallback collapses all detections to track 0 (skip
instead); `validate_trajectories` NaN-blind (add isna checks); dataset_export reports selected
not written counts, no atomic write; pipeline/deliverable parquets not tmp+rename (labeler
autosave already is); `PitchMapper` indexes arrays with DataFrame index labels (fine today,
misaligns on non-RangeIndex input — reset_index defensively); empty tracker output flows
through silently (warn).

**Verified correct:** grounded team-classification path (top-12 area crops, one batched
predict, modal vote, ties→unknown); foot-point geometry for ground-plane mapping; ball
interp math; hygiene reach-gate units; checkpoint/replay with loud fallback warning; eval
per-axis feet scaling; schema writer/reader agreement.

### E. View-data engine (digest / dataset / Colab)

**F-V1 (HIGH for interpretation, confirmed): within-game validation is near-tautological.**
Default split = per-view tail rows 5 frames (0.17s) from training frames of the same static
background; no test split exists; the main Colab ARI/purity/silhouette runs on ALL frames
including training (script self-flags this). What the Colab run proved: pseudo-labels are
learnable + views separable. What it didn't: generalization. The **held-out-views probe is
clean** (fresh model, blind views, no leakage) — make IT the headline within-game number, add
a time-gapped split, and treat tail-val accuracy as a smoke test. Cross-game remains untested.

**F-V2 (MEDIUM latent, confirmed): independent per-split class mappings in Colab datasets** —
a view present only in val (documented 1-frame-view case) crashes CE or silently shifts all
higher class indices. **Fix:** one mapping built from the full manifest, passed to both.

**F-V3 (MEDIUM, confirmed): smoothing/weight mismatch** — an override frame trains toward view
B weighted by its match to view A (`view_dataset.py:299`); `-1` votes in the mode filter and
wins min-tie-breaks. **Fix:** weight from the smoothed view's match column; exclude `-1` from
voting (abstain). Registration unaffected (view_registration re-matches per frame itself).

**F-V4..V9 (LOW-MED):** max-sum medoid doesn't bound min member↔rep match under average-linkage
chaining (emit min-sim stat, split below floor); `min()` denominator + min_keypoints=10 lets
sparse frames score high (raise floor, require absolute good-match count); digest keeps
undecodable frames → phantom singleton views with undecodable reps (drop them as 9e9feac did
in the exporter); cache key is path+size+mtime while three comments claim content-hash — an
mtime/size-preserving replace poisons cache *and passes* the Colab guard (fold the existing
`_content_hash` into the cache key; fix comments); `pos += 1` after failed mid-file read
desyncs indices (resync from CAP_PROP_POS_FRAMES); ReLU-terminated embedding (drop final ReLU).

**Verified correct:** match-fraction symmetry (empirical); sort-before-smoothing fix present +
regression-tested; smoothing window/tie semantics as documented; streaming truly sequential;
dropped-frame bookkeeping fully aligned; content-hash math identical both sides; held-out-views
probe leak-free; ARI/purity/silhouette formulas correct.

---

## Part 3 — Where to focus (recommended order)

**1. Finish calibration correctness at the root (P0, directly on the chosen "global solve" path):**
   a. Shared-camera-center bundle (F-C1) — one C + shared f + per-anchor rotations over all
      clicks. This is the minimal version of the global field-anchored solve and removes the
      structural cause of "lines in the sky" rather than patching symptoms.
   b. Honest export confidence + far-end w>0 gate (F-C2, F-C3) — stop poisoning
      `homographies.parquet` with conf=1.0 half-field frames.
   c. w≤0 → NaN in `PitchMapper` (F-C6) and RANSAC thresholds at `fit_homography` call sites
      (F-C5) — cheap, protects every downstream metric.

**2. Make Phase-4 metrics valid, not just correct (the product):**
   a. Visibility masking + coverage columns (F-M1) — biggest validity lever; the homography
      already gives the camera footprint for free.
   b. Area-weighted third aggregation (F-M2) and halftime x-flip (F-M3) — small, real fixes.
   c. One homography-trust rule (F-M4) once confidence is honest.

**3. Pre-labeling-session footguns (do before the next labeler run):**
   RefitWorker crash-safety + export liveness (F-T1); analyze_video clobber guard (F-T2);
   line-click inertness — wire in or notify (F-C4).

**4. Data-engine hygiene (before the next Colab round):**
   Held-out-views as headline + time-gapped split (F-V1); shared class map (F-V2);
   weight/−1 fixes (F-V3); content-hash cache keys (F-V7).

**5. Housekeeping:** commit the four untracked docs (their plans are implemented!);
   gitignore or relocate `data/`, `rejected_frames*`, `calib_drift_log.png`; delete/deprecate
   legacy `manual_anchor` status helpers; ruff the `scripts/` files or exclude them.

**Unchanged blockers from before:** Patrick's possession ground-truth CSV (≥80% gate) + kit
confirmation; cross-game transfer test for view embeddings; `--workers 1` macOS constraint.
