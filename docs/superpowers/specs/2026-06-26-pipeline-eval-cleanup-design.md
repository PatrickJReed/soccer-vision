---
name: pipeline-eval-cleanup
description: Cleanup + remaining LOW/MED findings on the pipeline glue, eval framework, and dataset-export — the deferred-ML flywheel infrastructure. Fix the assemble-recompute-path divergence and the ball highest-conf pick (code); make the eval match-threshold, single-end blind spot, deferred-ML status, and binding-constraint scope guard honest (docs). No calibration, no new product surface.
status: approved
date: 2026-06-26
---

# Pipeline / Eval / Export — Cleanup + Remaining Lows

Sub-project 4 of the post-audit program (see `docs/superpowers/2026-06-26-codebase-audit.md`).
This is the **cleanup batch**: the pipeline glue, the eval framework, and `dataset_export`
are clean and well-tested, but they are wired to a homography that didn't work
(closed by SP1, `2026-06-26-calibration-global-homography-design.md`) and to an empty metrics
consumer (the §6 product, SP5). This sub-project does not touch the calibration core or the
product. It (a) fixes two real correctness lows, (b) makes the eval/threshold/coverage signals
honest about what they do and do not measure, and (c) labels the deferred-ML infrastructure so it
stops reading as v1 work.

Two of the eight fixes are **code** (F1, F6) plus one small code addition (F3); the rest are
**documentation / marker** changes. Each fix is tagged CODE or DOCS below.

## 1. Problem

The audit found that "the glue serves the detour, not the product." Concretely:

- The documented **cheap-recompute path** (`assemble_from_parquet`) does *not* reproduce the
  production deliverable — it silently recomputes from a different homography source.
- The eval framework's **match threshold** is anchored to the labeler's *self-consistency*
  residual (its fit error on its own clicks), which is blind to whole-field accuracy — the exact
  blind spot SP1 exists to close.
- The eval **structurally cannot detect "lines in the sky"**: it scores only GT-visible
  (in-frame) landmarks, so an off-frame / unclicked-end landmark is never scored. The headline
  `accurate_coverage` is "match-the-labeler within the visible region," not whole-field truth.
- `eval/`, `dataset_export`, and the pitch-finetune notebooks are **deferred-ML infrastructure**
  with no live consumer (only tests + example notebooks), built before the product they serve.
- A latent **ball x/y "Frankenstein"** risk in the highest-conf ball pick, safe today only via an
  unasserted cross-module invariant.
- The spec's §8.2 **end-to-end regression gate** does not exist and cannot until the metrics layer
  is built; 87% coverage is subsystem-scoped, not product-readiness.

Findings closed by this sub-project: `pipeline-assemble-from-parquet-divergence` (LOW),
`eval-match-threshold-self-residual` (MED), `eval-single-end-blind-spot` (LOW),
`eval-export-orphaned-deferred-ml` (LOW), `ml-pitch-finetune-premature-deferred` (LOW),
`pipeline-ball-groupby-last-frankenstein` (LOW), `test-coverage-false-confidence-no-e2e` (LOW),
`glue-serves-detour-binding-constraints` (MED, documentation).

## 2. Goals & non-goals

**Goals**
- The recompute path reproduces the production deliverable, or says clearly that it doesn't.
- Every eval signal is honest about its scope: the match threshold is not derived from
  self-residual; `accurate_coverage` is documented as visible-region-only; a whole-field signal
  and a single-end regression test exist.
- The deferred-ML infrastructure (`eval/`, `dataset_export`, finetune notebooks, the
  `pitch_weights_path` slot) is labeled deferred and grep-confirmed off the v1 path.
- The ball highest-conf pick is atomic (no per-column NaN borrowing).
- A skipped e2e scaffold and a binding-constraint scope guard are in place so SP5 has a seam.

**Non-goals (YAGNI)**
- No calibration changes (SP1). No own/opp grounding (SP2). No possession/phase/halftime
  correctness (SP3). No metrics product (SP5).
- **No deletion** of `eval/` or `dataset_export` — they serve the (deferred, not cancelled) ML
  flywheel; they are labeled, not removed.
- No new pipeline / eval / export *surface* (no new public functions, no new report consumers).
  The audit's explicit guidance: add nothing here until SP1 or SP5 is unblocked.
- Do not re-litigate the two refuted findings (`_infer_fps` rounding; eval aspect-ratio 1.5 vs
  1.54). They are correct as written and out of scope.

## 3. Design (per finding)

### F1 — `assemble_from_parquet` recompute-path divergence  [CODE + DOCS, LOW]

**Where.** `pipeline.py`:
- `analyze_video` passes the production homographies into `assemble_phases`
  (`pipeline.py:236-238`, `homographies=homographies`).
- `assemble_from_homographies` replays `homographies.parquet`
  (`pipeline.py:332` reads it, `pipeline.py:336-338` passes it through).
- `assemble_from_parquet` (`pipeline.py:183-202`) calls `assemble_phases`
  **without** `homographies=` (`pipeline.py:198-200`), so it enters the keypoint branch
  (`pipeline.py:78-85`: `build_frame_homographies` + `smooth_homographies`). Its docstring
  (`pipeline.py:191-194`) calls itself "the cheap-recompute path," but it reproduces a *different
  homography source* (raw detected keypoint anchors) than the production deliverable (the
  labeler/propagation homographies written to `homographies.parquet`).
- Secondary, in the keypoint branch: `h_map = dict(smoothed)` (`pipeline.py:85`) includes
  carry-forward frames, so `PitchMapper` (`pipeline.py:89`) maps them, but `h_entries`
  (`pipeline.py:84`) holds only `raw_h` anchor frames. Carry-forward frames therefore get pitch
  coords but `homography_source="none"` (`pipeline.py:110-114`) — a consumer filtering
  `homography_source != "none"` would drop valid mapped frames.

**Fix (recommended: read `homographies.parquet` when present).** `assemble_from_parquet` already
runs inside a checkpoint directory where `analyze_video` wrote `homographies.parquet` alongside
`trajectories_px.parquet`/`keypoints.parquet`. Change it to accept an optional
`homographies_path` (default: sibling `homographies.parquet` of `trajectories_px_path`); when that
file exists, read it via `homographies_from_parquet` and pass it through (identical to
`assemble_from_homographies`), so the cheap-recompute path reproduces the production deliverable
while still allowing `**assemble_opts` threshold tweaks with no GPU. When it is absent, fall back
to the keypoint branch with a `logger.warning("homographies.parquet not found; recomputing from
raw keypoints — this is NOT the production homography source")`. Fix the docstring
(`pipeline.py:191-194`) to state exactly what each branch reproduces.

*Alternative (less code, recommended only if the keypoint branch has no remaining callers):*
deprecate `assemble_from_parquet` with a `DeprecationWarning` pointing at
`assemble_from_homographies` (which already does cheap threshold-only recompute *and* replays the
real homography). Functionally loses nothing; chosen against here only because it removes the
"recompute even when homographies.parquet is missing" path some callers may rely on.

**Carry-forward `source="none"`.** With the recommended fix the keypoint branch is only reached on
an explicit, logged fallback, so the mislabeling is visible rather than silent. Do not change the
coverage semantics (the `pipeline.py:81-83` comment deliberately excludes carry-forward frames from
anchor coverage). Minimal fix: add one docstring sentence to `assemble_phases` (near
`pipeline.py:81-85`) documenting that, in the keypoint fallback, carry-forward frames carry pitch
coords but `homography_source="none"`, and that downstream code must not treat `source=="none"` as
"no pitch coords." No behavior change. (No live consumer today; keep it minimal.)

### F2 — Eval match-threshold derived from self-consistency residual  [DOCS, MED]

**Where.** `examples/eval_pitch.ipynb` cell `cell-4`:
`NOISE_FLOOR_FT = float(np.percentile(residuals, 90))`; `MARGIN_FT = 1.0`;
`MATCH_THRESHOLD_FT = NOISE_FLOOR_FT + MARGIN_FT`. The residuals come from
`labeler_fit_residual_feet` (`pitch_metrics.py:98-115`), which is the labeler homography's median
error on its **own clicked anchors** — a self-consistency proxy. As the audit notes, this is blind
to whole-field accuracy: a single-end "sky" homography scores a tiny self-residual yet is wildly
wrong off-frame. Anchoring the match bar to it bakes that blind spot into the eval. `score_frame`
already takes `match_threshold_feet` as a parameter (`pitch_metrics.py:149`), so this is a
docs/notebook change only — no scorer behavior changes.

**Fix (docs).**
1. Add a docstring warning to `labeler_fit_residual_feet` (`pitch_metrics.py:98-110`): this is a
   **self-consistency proxy, NOT an accuracy bound**, and must not be used to set the match
   threshold until the labeler's whole-field accuracy is established.
2. Update `eval_pitch.ipynb` `cell-4` (markdown + the inline comment that currently says
   "labeler noise floor"): derive `MATCH_THRESHOLD_FT` from **SP1's cross-end held-out
   distribution** (the `HoldoutReport` from `cross_end_holdout`, acceptance bar overall median
   < 1.5 m — see `2026-06-26-calibration-global-homography-design.md` §7) or from an independent
   **test-retest**, not from self-residual. Keep the self-residual line only as a labeled
   diagnostic ("self-consistency floor; not the accuracy bar").

This unblocks honestly only once SP1 lands its held-out report; until then the notebook documents
that the threshold is provisional and self-consistency-based.

### F3 — Eval institutionalizes the single-end blind spot  [CODE (small) + DOCS, LOW]

**Where.** `score_frame` (`pitch_metrics.py:142-185`): `gt_kpts = project_landmarks(h_gt, ...)`
(`pitch_metrics.py:159`); `gt_visible` is in-frame only (`pitch_metrics.py:160-161`,
`gt_kpts[i, 2] > 0`). `project_landmarks` marks off-frame landmarks invisible
(`autolabel.py:60-63`, the `in_frame` mask). `keypoint_errors_feet` scores only GT-visible
landmarks (`pitch_metrics.py:67`). So the eval structurally cannot see "lines in the sky": a
landmark the model projects off-frame at the unclicked end is simply never scored. The headline
`accurate_coverage` (`EvalReport.accurate_coverage`, `pitch_metrics.py:133`, `:272`) is therefore
"match-the-labeler within the GT-visible region," not whole-field truth. Latent (no live consumer;
ML deferred), but it would mask the SP1 failure mode if a model were ever scored.

**Fix.**
1. **Whole-field signal (small code).** Add a field to `FrameScore` (`pitch_metrics.py:118-126`)
   and aggregate it in `EvalReport` (`pitch_metrics.py:129-139`): of the 20 scorable landmarks
   (21 minus `HIDDEN_IDX`), how many does the **model** place sanely — i.e. predicts with
   `conf >= conf_thr` at an in-frame pixel — computed directly from `model_kpts` independent of GT
   visibility. Report a per-frame count and an aggregated median (e.g.
   `model_in_frame_median`). This is the cheapest whole-field signal and needs no homography
   inversion. (More rigorous alternative, if a fitted `model_h` is available: reuse
   `fold_count(inv(model_h), frame_size)` from `calib/validate.py` — the same gate SP1 adopts — to
   count canonical landmarks the model H places in-frame. Use the cheap version unless the
   `calib/validate` dependency is already imported.)
2. **Single-end-GT regression test** (see §4): a GT homography that sees only one end; assert the
   far-end landmarks are unscored / the whole-field signal is low, documenting the blind spot as a
   known property rather than a silent gap.
3. **Docstring (docs).** On `EvalReport.accurate_coverage` (`pitch_metrics.py:133`) and the module
   docstring (`pitch_metrics.py:1-8`): state explicitly that `accurate_coverage` is
   "match-the-labeler within the GT-visible region," NOT whole-field accuracy, and point at the
   new whole-field signal + SP1's held-out validation for the whole-field check.

Keep it minimal: the test + docstring are the core; the new report field is a few lines and no new
public function.

### F4 — `eval/` and `dataset_export` are orphaned deferred-ML infrastructure  [DOCS, LOW]

**Where / confirmation.** Grep confirms the only consumers of `eval.pitch_metrics`,
`eval.benchmark`, and `dataset_export` are tests (`tests/test_eval_pitch_metrics.py`,
`tests/test_eval_benchmark.py`, `tests/test_dataset_export.py`) and example notebooks
(`examples/eval_pitch.ipynb`, `examples/finetune_pitch.ipynb`). The benchmark manifest
(`benchmark.py`) needs ~3 diverse labeled fields (the notebook expects `carlsbad`/`rebels`/`surf`)
that do not exist. **Caveat:** `autolabel.project_landmarks` is NOT orphaned — it is used live by
eval (`pitch_metrics.py:156-159`) and by `viz/pitch_overlay.py`; only the dataset-export /
finetune side is dormant.

**Fix (docs).** Add a one-line banner to the module docstrings:
- `eval/pitch_metrics.py` (`pitch_metrics.py:1-8`): "DEFERRED ML — scores a pitch-keypoint model
  that is not on the v1 path; the labeler is the per-game homography source. Kept for the ML
  flywheel (deferred, not cancelled)."
- `eval/benchmark.py` (`benchmark.py:1-5`): same banner + note the frozen benchmark fields do not
  yet exist.
- `dataset_export.py` (`dataset_export.py:1-11`): same banner.
- Note in `eval/pitch_metrics.py`'s banner that `autolabel.project_landmarks` it depends on is
  still live (so a future reader does not delete it as orphaned).

No deletion.

### F5 — ML pitch-finetune built before the product, now deferred  [DOCS, LOW — combine with F4]

**Where / confirmation.** Grep-confirmed: **no v1 path loads a finetuned pitch `.pt`.**
`RoboflowBackend.__init__` defaults `pitch_weights_path=None` (`roboflow.py:292`); the model load
is `pitch_weights = self.pitch_weights_path or weight_paths["pitch"]` (`roboflow.py:426`), so the
default v1 path uses the **base** `pitch_yolov8_v1.pt` (`roboflow.py:42`, `:54`), not a finetuned
model. A finetuned model loads only if a caller explicitly passes `pitch_weights_path`
(documented opt-in override, `roboflow.py:271-274`). `analyze_video` constructs
`RoboflowBackend(detect_pitch=True)` (`pipeline.py:223`) with no `pitch_weights_path`, so the v1
deliverable never depends on a finetune.

**Fix (docs, combined with F4).**
- `pitch/autolabel.py` docstring (`autolabel.py:1-11`, already references "Phase 3.5b"): add
  "DEFERRED ML — the active-learning loop is deferred; `project_landmarks` itself is still used
  live by eval and viz."
- Tag `examples/finetune_pitch.ipynb` (top markdown cell) as deferred-ML.
- Add one sentence to the `pitch_weights_path` parameter doc (`roboflow.py:271-274`): "Opt-in
  override; the default v1 path uses the base pitch model — no finetuned weights are loaded
  unless this is set."

### F6 — Ball `groupby().last()` Frankenstein risk  [CODE, LOW]

**Where.** `pipeline.py:98-100`:
```python
ball = enriched[enriched["class"] == "ball"]
ball_by_frame = ball.sort_values("conf").groupby("frame")[["x_pitch", "y_pitch"]].last()
```
`GroupBy.last()` skips NaN **per column independently**, so a high-conf ball row with `x_pitch`
valid but `y_pitch` NaN could take `y_pitch` from a *different, lower-conf* row — a Frankenstein
ball. This is safe today only by an unasserted cross-module invariant: `filter_outside_pitch` drops
rows with NaN in either pitch coord (`filter.py:12`, `:31-34`) and `PitchMapper` writes `x_pitch`
and `y_pitch` together (`mapper.py:39-40`), so a ball row never has exactly one of x/y NaN by the
time it reaches this line.

**Fix.** Select the highest-conf ball row **atomically** so x and y always come from the same row:
```python
ball = enriched[enriched["class"] == "ball"]
idx = ball.groupby("frame")["conf"].idxmax()
ball_by_frame = ball.loc[idx, ["frame", "x_pitch", "y_pitch"]].set_index("frame")
```
This removes the dependence on the filter invariant and makes the intent ("the single
highest-confidence ball detection per frame") explicit. Behavior is identical under the current
invariant; it only changes the degenerate case the invariant currently hides.

(Sibling note: `possession.py:41-42` picks `ball_rows[...].iloc[0]` — arbitrary, not highest-conf.
That is a possession-correctness issue and belongs to **SP3**; out of scope here. Cross-referenced
so it is not lost.)

### F7 — Missing e2e gate + coverage false-confidence  [DOCS + skipped scaffold, LOW]

**Where.** The spec's §8.2 (`2026-05-27-soccer-vision-design.md:539-546`) calls for
`tests/test_pipeline_e2e.py` to run the full pipeline on the bake-off clip and snapshot summary
statistics (`mean_area_own`, `pct_contested`, …) to a golden file with 5% drift tolerance. It does
**not exist** (the tests directory has `test_pipeline_assemble.py`,
`test_pipeline_homographies.py`, `test_pipeline_analyze_video.py`, but no e2e), and it **cannot**
exist until the metrics layer is built — `metrics/__init__.py` is a one-line docstring stub.

**Fix.**
- Add `tests/test_pipeline_e2e.py` as a **skipped scaffold**: a single
  `@pytest.mark.skip(reason="Phase 4 metrics not built — see SP5")` test whose body / docstring
  records the intended shape (run `analyze_video` on the bake-off clip → compute the §6 summary
  stats → assert within 5% of a golden JSON). This gives SP5 a ready seam and makes the gap
  visible in the test suite instead of invisible.
- Add a note (in this spec and as a one-liner in the e2e scaffold's module docstring): the repo's
  87% coverage is **subsystem-scoped**, not product-readiness — there is no end-to-end metric gate
  until SP5 fills this scaffold.

Lightweight by design: the real test lands with SP5 (Phase 4 metrics).

### F8 — Glue serves the detour, not the product (binding constraints + process)  [DOCS, MED]

**Where.** The pipeline glue is correct (the audit's "keep" list) but wired to a homography that
didn't work (SP1) and to an empty metrics consumer (`metrics/__init__.py` stub).

**Fix (short doc subsection).** Record the binding constraints and the process guardrails. Home:
the **"Process notes" subsection below** (the durable record) plus a one-line scope-guard pointer
added to the `pipeline.py` module docstring (`pipeline.py:1-6`) referencing this spec. Content:

> **Binding constraints.** The pipeline/eval/export glue is complete and not the bottleneck. The
> two binding constraints are (a) a **trustworthy homography** (SP1, calibration rework) and
> (b) the **metrics layer** (SP5, the §6 product). **Do not add any further pipeline, eval, or
> export surface** (no new public functions, report fields, or consumers) until one of those is
> unblocked. New glue without a product consumer is the failure mode this audit identified.

**Process notes** (from the audit's recommendations #3 and #5):
1. **Define "good-enough" from metric tolerance, not pixels.** The acceptance bar for a homography
   is a downstream-metric tolerance — the finest is the contested-possession margin 0.022 pu
   (≈ 1.5 m); the coverage cell is 0.011 pu (≈ 0.75 m); most headline metrics tolerate far more.
   SP1 adopts the 1.5 m held-out bound. Calibration work terminates when end-to-end metric
   stability is reached, not when a pixel residual is minimized.
2. **Circuit-breaker.** A sub-project that fails twice triggers a scope/fallback review at project
   altitude (e.g. "ship metrics on a rough mapping"), not a third sub-project. Tie every spec to a
   product deliverable.

## 4. Testing (TDD)

New/updated tests, written before the corresponding code change. Only F1, F3, F6 touch code; F7
adds a skipped scaffold; F2/F4/F5/F8 are documentation and are verified by reading, not tests
(plus the existing suite must still pass after docstring edits).

1. **F1 — recompute reproduces production** (`tests/test_pipeline_assemble.py`,
   extend `test_assemble_from_parquet_roundtrip` at `:101-124`):
   - Write `trajectories_px.parquet`, `keypoints.parquet`, **and** a `homographies.parquet`
     (non-identity, `source="manual"`) into one dir; call `assemble_from_parquet`; assert the
     resulting `phases`/`x_pitch` match `assemble_from_homographies` on the same
     `homographies.parquet` (i.e. the recompute path used the parquet, not the keypoint branch).
   - Separate test: omit `homographies.parquet`; assert the keypoint-branch fallback runs and emits
     the documented `logger.warning` (capture with `caplog`).

2. **F3 — single-end blind spot is surfaced** (`tests/test_eval_pitch_metrics.py`):
   - Construct an `h_gt` whose view sees only one end; assert `score_frame` scores only the
     in-frame (near-end) landmarks (existing behavior) **and** that the new whole-field signal
     (`model_in_frame_median` / per-frame count) is low for a model H that sends the far end
     off-frame — the "lines in the sky" regression the current suite never exercises.
   - Assert a physically-plausible model places most of the 20 landmarks in-frame (high signal).

3. **F6 — atomic highest-conf ball pick** (`tests/test_pipeline_assemble.py`):
   - Extract the ball pick into a tiny pure helper (or test through `assemble_phases`) with two
     ball rows for one frame: high-conf with `y_pitch=NaN, x_pitch` valid, and low-conf with both
     valid. With the filter invariant **bypassed** (to exercise the degenerate case directly),
     assert the chosen row's `y_pitch` is NaN (taken atomically from the high-conf row), not
     borrowed from the low-conf row. A second test confirms the normal case (both coords from the
     highest-conf row) is unchanged.

4. **F7 — e2e scaffold** (`tests/test_pipeline_e2e.py`, new): the skipped test collects (is
   discovered by pytest) and is marked skipped with the SP5 reason; assert nothing else.

5. **Regression.** The full existing suite (302 tests) passes after the docstring/notebook edits;
   no existing assertion changes except where F1's roundtrip test is extended.

## 5. Risks

- **F1 path resolution.** Auto-detecting the sibling `homographies.parquet` could pick up a stale
  file if a user points `trajectories_px_path` at a hand-assembled directory. Mitigation: the
  fallback warning makes the "no parquet" case explicit, and reading the parquet is exactly what
  the production deliverable did — staleness is the user's checkpoint hygiene, not a new failure
  mode. An explicit `homographies_path=None` override is provided.
- **F3 whole-field signal scope.** The cheap signal (count of in-frame model predictions) is a
  plausibility proxy, not an accuracy measure; it can be high for a confidently-wrong model.
  That's acceptable — the accuracy bound is SP1's held-out validation; this signal only catches the
  gross "off-frame" failure the current eval is blind to. Documented as such.
- **F6** is behavior-preserving under the current invariant; the only observable change is in a
  degenerate case the invariant currently prevents, so regression risk is minimal.
- **Docs drift.** F2/F4/F5/F8 are markers; if SP1/SP5 land and change the picture, these notes
  must be updated. Mitigation: they point at the sibling specs rather than restating their content.

## 6. Out of scope (handled elsewhere or excluded)

- **Calibration rework** (`H_global` + offsets, honest gate, held-out validation): SP1
  (`2026-06-26-calibration-global-homography-design.md`). F2 and F3 *reference* its held-out bar
  but do not implement it.
- **own/opp grounding** (SP2); **possession anisotropy / defend split / transition windows /
  halftime / possession `ball.iloc[0]` highest-conf pick** (SP3); **the §6 metrics product and the
  real e2e golden test** (SP5, which fills the F7 scaffold).
- **No deletion** of `eval/`, `dataset_export`, or the finetune notebooks — labeled deferred, not
  removed.
- **Refuted findings — explicitly excluded, not bugs:** `_infer_fps` silent-fallback/rounding
  (`pipeline.py:150-161`; `t = frame / fps` is exact float64, so `frame / t == fps`), and the eval
  aspect-ratio "1.5 vs 1.54" (`pitch_metrics.py:23`; 1.5 is correct for 9v9, 1.54 is
  `fifa_11v11`). These are correct as written.
