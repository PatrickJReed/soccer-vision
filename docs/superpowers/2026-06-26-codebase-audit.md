# Soccer-Vision — Codebase & Decisions Audit (2026-06-26)

**Method:** 6 parallel subsystem auditors + ~40 adversarial verifiers (each finding
re-checked against the real code; severities below are the *verified* severities, not the
raw finder claims). 2 findings were refuted outright on inspection. Cross-checked against an
independent manual read of the calibration core, eval, splitter, possession, validate, pipeline.

---

## Bottom line

High engineering hygiene wrapped around a strategic misdirection. 302 tests pass, 87%
coverage, `mypy --strict` + ruff clean across 86 files, zero TODO/stub markers — the code
that exists is clean and well-tested. **But ~1 month / 269 commits in, the actual product —
the §6 metrics layer the spec calls the core deliverable — is 0% built** (`metrics/` = one
docstring), while a Phase-3 task the spec scoped at *2 days with a manual fallback* (pitch
homography) consumed ~28 of ~30 elapsed days, ~3,400 LOC, and 96/269 commits — **and still
does not work** ("lines in the sky"). The root is an ill-posed camera model (per-frame
independent 6-DOF poses for a fixed camera), compounded by the absence of any product-tied
acceptance gate to make the calibration work *terminate*.

---

## Where the work is genuinely solid (keep)

- **Possession proxy** (`phase/possession.py`) faithfully implements the spec §6.1 5-state
  model + contested-preserving mode smoothing. The one finished slice of real product logic.
- **Track hygiene** (`hygiene/`) — principled, physics-based (8 m/s reach gate, isotropic
  length-normalized distances, tight-merge bias, boundary→unknown). Not clip-overfit.
- **Calibration math primitives** — `homography_from_pose`, `fold_count`, `line_residual`,
  `_robust_sqpnp` (planar-safe drop-worst PnP), `leave_one_out_feet`. Correct and tested.
  The *right* validation tools exist; they're just not wired into the status signal.
- **Labeler concurrency** (`refit_worker.py` + `state.py`) — RLock + revision cancellation +
  serial HTTPServer; the HIGH-severity clicks-snapshot race was correctly fixed. No remaining
  races/deadlocks found.
- **Pipeline glue / IO schema / checkpointing** — clean, pure, graceful degradation, exceeds
  the spec's "parquet between every stage." Exactly the seam the metrics layer needs.
- **The end-of-session self-diagnosis** (`2026-06-26-status-and-open-issues.md`) is technically
  correct and empirically grounded — the most valuable artifact in the repo.

---

## The central divergence from the project's goals

| Finding | Sev | Status |
|---|---|---|
| **Product (§6 metrics, Phases 4–9) is 0% built** while a 2-day phase consumed the month | **critical** | confirmed |
| **No product-tied acceptance gate** for homography accuracy → optimization couldn't terminate; it chased intermediate proxies (reproj px → fold_count → eval px → gated residual) forever | **high** | confirmed |
| **Failure-descent process pattern**: each calibration sub-failure spawned the next sub-project (13 of 16 specs on the homography/labeler thread); no circuit-breaker. The team *did* re-examine the model and *did* take the manual fallback — but never escalated to a project-altitude "ship metrics on a rough mapping" decision | **high** | confirmed |
| **Calibration over-engineering vs goal** — a 6-DOF SfM/calibration stack where the spec scoped a 2-day homography with a manual-corner fallback | **high** | confirmed |
| **Labeler scope creep** — a 977-LOC HTTP web-app IDE (+JS frontend) built ahead of the 0%-built product, and it still doesn't yield a working homography | **high** | confirmed |

---

## Ill-posed approaches (verified against the geometry)

| Finding | Sev | Status |
|---|---|---|
| **Per-frame INDEPENDENT 6-DOF SQPNP** is under-constrained for a fixed/translation-only camera. Each Trace view sees ~one end; the pose pins the near end and the unclicked end extrapolates to the sky. "Lines in the sky" is **structural to the model, not a bug.** `calibrate_camera` already shares one focal but every frame still gets a free `rvec`+`tvec`; for a fixed-center camera the optical center should be shared too | **critical** | confirmed |
| **False-green status** — `_status_of` gates green on `residual_px ≤ 60`, but `_reproj_rms_px` scores *only the gate-kept (near) landmarks*; a one-end frame fits to 8–20 px → green while the far end is off by hundreds of px. `fold_count` and `leave_one_out_feet` are computed/available but **never consulted** | **high** | confirmed |
| **False-green export poisons the product** — `export()` gates on the same residual and writes single-end "sky" homographies to `homographies.parquet` as `source="manual", conf≈0.87` (latent: no metric consumer yet) | **high** | confirmed |
| **Chain drift is structural AND self-inflicted** — `cumulative_transforms` composes per-pair transforms with no global anchor/loop-closure (error accumulates), and fits full **8-DOF `findHomography`** per pair when the measured motion is ~2-DOF translation | **high** | confirmed (causal magnitude tempered: the fitted transforms already sit near identity; degraded correspondences during the fast pan are the larger driver) |
| **Per-frame model was baked into the spec** (§3.1a) despite the camera being known-fixed | **high** | confirmed (down from critical: per-frame H from a per-frame *detector* is standard; it only became catastrophic after the pivot to sparse one-end *manual* clicks) |
| **Even the proposed fix risks over-engineering** — the status doc's action items still say "mosaic + bundle-adjust one focal + per-frame *rotations*," contradicting its own finding that the motion is translation not rotation. The minimal correct fix: translation-only inter-frame offsets + ONE global homography least-squares fit over all clicks (`H_f = offset_f ∘ H_global`). Prototype that before any SfM machinery | **medium** | confirmed |

---

## Concrete correctness bugs that need fixing (independent of the calibration rework)

| Finding | File | Sev |
|---|---|---|
| **own/opp assigned by arbitrary cluster index** (`_CLUSTER_TEAM={0:"own",1:"opp"}`) with no grounding in `analyze_video` — a whole game's "own-team" analytics can be silently **inverted**. The kit-grounded path exists only in the separate `hygiene` CLI, which the pipeline never calls | `tracking/roboflow.py:96`; `pipeline.py` | **high** |
| **Possession distances are anisotropic** — x not divided by aspect_ratio, so spec "pitch-length" thresholds are ~1.5× too tight in the width axis. `hygiene/core.py` does it right; `possession.py` doesn't | `phase/possession.py:52-54` | **medium** |
| **defend_low/high split at midfield (0.5) not the spec's opp-third (0.667)** — `OPP_THIRD_MIN_Y=0.667` declared but dead; a test locks in the wrong 0.5 boundary | `phase/splitter.py:13,63` | **medium** |
| **Transition windows miss own→contested→opp turnovers** — only direct own↔opp flips fire; youth turnovers pass through contested/loose, so the transition phase is near-dead in target footage | `phase/splitter.py:43-48` | **medium** |
| **`export()` proceeds after a 30s `wait_idle` timeout**, silently writing a partial/stale homography set with a success message (worst case: export within 30s of `recalibrate()` → near-empty parquet) | `labeler/state.py:351` | **medium** |
| **Halftime/attack-direction gap** — phase labels mirror-wrong for half of every full game; no `halftime_frame`/normalization. Bounded, acknowledged, cheap to fix; prerequisite for any season metric | `phase/splitter.py` | **medium** |
| **Possession agreement gate can't validate contested** — excludes contested/loose from num+denom and the GT schema has no contested state, so the headline `pct_contested` is unvalidatable and the gate is gameable by abstaining | `hygiene/core.py:351`; `hygiene/agreement.py` | **medium** |
| `groupby().last()` for highest-conf ball skips NaN per-column → latent Frankenstein x/y (currently safe via filter invariant); possession picks `ball.iloc[0]` (arbitrary), not highest-conf | `pipeline.py:99-100`; `possession.py:41` | **low** |
| Frontend residual readout uses stale normalized `0.05` threshold vs pixel residuals → always orange | `labeler/static/app.js:95` | **low** |
| `TrackingBackend` protocol is decorative — production calls un-protocoled `process_with_pitch`; `MockBackend` can't drive `analyze_video`; the "swap-later" promise is unverified | `tracking/base.py`, `pipeline.py:225` | **medium** |

---

## What the audit got wrong (self-corrected — reported for trust)

- **`_infer_fps` rounding** — *refuted.* `t_seconds = frame/fps` is stored float64, so `frame/t = fps` exactly; the "30.30 instead of 30.0" premise doesn't exist. Only the (documented, never-fires-on-real-clips) fallback-to-30 is real, and it's cosmetic.
- **Eval aspect-ratio 1.5 vs 1.54** — *refuted.* `standard_9v9()` uses 1.5 (default); 1.54 is `fifa_11v11`. 1.5 is correct for 9v9; no inconsistency.
- Several scope/cleanup findings (Engine B "doubly wrong", pitch-detection-in-tracker, orphaned eval/export, bake-off "1-point margin", web-app scope-guard) were **downgraded to low** by verification: real but minor, or already-deferred-by-design, or matters of taste.

---

## Recommended next moves (decision-oriented)

1. **Make the status honest now** (P1, cheap): gate green/export on whole-field sanity
   (`fold_count` in range and/or held-out projection of the 21 landmarks), not in-sample
   residual. `fold_count` is already computed and sitting unused.
2. **Replace the calibration core with the minimal global model** (the team's own reframe, done
   light): translation/similarity inter-frame offsets + one global homography fit over all
   clicks; `H_f = offset_f ∘ H_global`. Keep the labeler UI/transport/export seam; quarantine
   the gated engine, windowing, Engine B, `_rotation_from_chain`. **Do not** lead with the
   masking experiment or a rotation bundle-adjust.
3. **Define "good-enough" from the metrics, not from pixels.** Finest thresholds: contested
   margin 0.022 pu (~1.5 m), coverage cell 0.011 pu (~0.75 m); most headline metrics tolerate
   far more. Wire one homography into `pipeline.py`, build Phase-4 shape+space, and let
   end-to-end trend stability tell you whether calibration needs more.
4. **Fix own/opp grounding** before any metric is trusted — thread an `--own-kit` hint through
   `analyze_video`, or chain `analyze_video → hygiene` as the default path.
5. **Add a phase-gate circuit-breaker**: a sub-project that fails twice triggers a scope/fallback
   review, not a third sub-project. Tie every spec to a product deliverable.
