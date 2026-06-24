# Pitch-Model Evaluation Framework (design)

**Date:** 2026-06-24
**Status:** Design approved, pending implementation plan
**Depends on:** interactive anchor labeler (ground truth), `pitch/autolabel.py`
(`project_landmarks`), `pitch/landmarks.py` (`PITCH_LANDMARKS`), `PitchSpec`
(feet conversion), the YOLOv8-pose pitch model.

## Problem

The 2-game retrain's coverage gate (`anchor_cov` in `acceptance_pitch.ipynb`)
gave a false pass: 93% "combined coverage" on an unseen field (Chula Vista)
where the model's keypoints were actually off the markings (Patrick's dot-check).
`anchor_cov` counts frames that fit *a* homography, not an *accurate* one — it
carries no ground-truth accuracy signal and the eval logic shipped with no unit
tests, so it misled silently. We are now investing in the model (more fields,
more training); we need a rigorous, honest, test-covered way to quantify whether
the model is working, measured on **keypoint accuracy** — not coverage-that-lies,
not player positions.

## Goal

A pure, unit-tested metric module + a Colab notebook that scores the
pitch-keypoint model against **labeler ground truth** on a **frozen held-out
benchmark**, producing keypoint-accuracy metrics in real-world feet, a
per-landmark breakdown, an end-to-end homography check, and one false-pass-proof
headline (**accurate-coverage** = % of frames that match the labeler). It
**replaces** the `acceptance_pitch` coverage gate.

## Non-goals

- Player-position accuracy (conflates the separate player-detection model).
- Improving the labeler's absolute accuracy (the labeler is the GT *reference*
  here; tightening it — line-anchors — is a separate, later question).
- The training improvements / field labeling themselves — that is the iteration
  loop this framework *measures*, not part of this spec.
- A hard pass/fail threshold baked in now. We report distributions + the
  match-the-labeler headline; the official bar is set from real numbers later.

## Decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Primary metric | Per-keypoint position error, in **feet** + pixels, **per-landmark** |
| End-to-end backstop | Homography reprojection error (model H vs labeler H), feet |
| Headline | **Accurate-coverage** = % frames matching the labeler within its noise band |
| Ground truth | Labeler homographies on held-out fields (project canonical landmarks → GT keypoints) |
| Train/test split | Fixed held-out benchmark of ~3 diverse fields, never trained on |
| Accuracy bar | **Match-the-labeler**: model error within the labeler's own noise floor |
| Focus | Keypoints (model's direct output), NOT player positions |
| Packaging | Pure tested module + Colab notebook; replaces the `acceptance_pitch` gate |

## Design

### Benchmark (frozen, versioned)
- ~3 held-out fields, chosen for diversity (e.g. one turf/permanent-line field +
  two painted-grass), **never** in any training set.
- Each labeled with the labeler; a sampled set of frames per field is enough
  (target ~50–100 frames/field for statistical stability — not the whole game).
- Per-field artifacts: the labeler's `homographies.parquet` + the list of sampled
  frame indices. The labeler `H` per frame is the ground truth.
- A **manifest** (field → game id → frame indices → homography file path) freezes
  the set so every retrain scores the identical frames.

### Ground-truth keypoints
- For a benchmark frame, GT keypoints = project the 21 canonical
  `PITCH_LANDMARKS` through the labeler homography into pixels (the existing
  `project_landmarks`, which uses `inv(H)`), keeping those that land in-frame
  (visible). Landmark 5 (under-camera) is excluded (never visible).

### Labeler-noise characterization (defines the bar)
- **Primary proxy:** per-frame labeler reprojection residual — the median error
  of the labeler homography on its OWN visible clicked anchors (internal
  self-consistency), expressed in feet.
- **Optional validation:** a small test-retest — Patrick labels ~10 benchmark
  frames a second time; per-keypoint variation between the two labelings is the
  empirical noise floor, used to sanity-check the residual proxy.
- The **match bar** = the labeler noise band: a frame "matches the labeler" when
  the model's per-keypoint median error (feet) ≤ the labeler noise floor + a
  small margin. The band is *derived from the characterization and reported*, not
  hardcoded.

### Metrics module — `src/soccer_vision/eval/pitch_metrics.py` (pure, tested)
Inputs per frame: GT keypoints (px) + labeler `H`; model-predicted keypoints (px)
+ confidences; `frame_size`; `PitchSpec` (for feet). Pure functions — arrays in,
dataclass/dict out, no I/O. Computes:
- **`keypoint_error_feet[i]`** — for each landmark `i` visible in GT and predicted
  by the model (`conf ≥ thr`): map the model's predicted pixel through the labeler
  GT homography → pitch coords → distance to canonical `PITCH_LANDMARKS[i]`, scaled
  by field dimensions → feet (also pixels).
- **`per_landmark`** — aggregate `keypoint_error_feet` per landmark across frames
  (median, 90th pct, detection count) → exposes *which* of the 21 fail.
- **`homography_reproj_feet`** — fit the model homography from its keypoints,
  reproject a grid of pitch points through model `H` and labeler `H`, per-frame
  median disagreement in feet. End-to-end backstop (catches outlier keypoints that
  wreck `H` even when the average looks fine).
- **`match_labeler[frame]`** — bool: model within the labeler noise band.
- **`accurate_coverage`** — fraction of ALL sampled benchmark frames where
  `match_labeler` is true (frames where the model produced no usable homography
  count as not-matching). The headline; cannot false-pass.
- **distributions** — overall median and 90th percentile of keypoint feet error
  and reproj feet error (never just a mean).

### Feet conversion
`PitchSpec` gives field length/width in real units; normalized pitch error ×
field dims → feet. One small tested helper.

### Report + notebook — `examples/eval_pitch.ipynb`
- Loads model weights (Drive-aware, like the other notebooks) + the frozen
  benchmark (Drive).
- Runs model inference on the benchmark frames.
- Calls the metric module; prints: headline **accurate-coverage**; the
  **per-landmark feet table** (21 rows: median / 90th / detection-rate); overall
  distributions; homography-reproj distribution.
- Renders qualitative overlays (model keypoints vs GT keypoints on N sampled
  frames), saved for **Patrick** to assess — Claude renders, Patrick interprets.
- Output is comparable across retrains so the curve is visible as fields are added.

### Error handling
- A benchmark frame whose labeler `H` is degenerate (the rare `cond(H) > 1e8`
  case, ~1/3075) → excluded from GT with a logged count; a degenerate GT frame
  must not pollute the metric.
- Model produces no detection / < 4 keypoints on a frame → that frame counts as
  not-covered (not-matching), never silently skipped.
- A landmark visible in GT but not predicted (`conf < thr`) → counts as a miss in
  that landmark's detection-rate; excluded from its error-distance distribution
  (no prediction → no distance to measure).

## Testing
- Pure metric tests with **synthetic** ground truth: construct a known
  homography + canonical landmarks, inject controlled keypoint errors (e.g. shift
  landmark 3 by a known feet amount), assert `keypoint_error_feet`, `per_landmark`,
  `homography_reproj_feet`, and `accurate_coverage` match the injected values
  exactly. This is what guarantees the gate tells the truth.
- Feet-conversion test (normalized → feet with a known `PitchSpec`).
- Degenerate-GT-frame exclusion test.
- Missing-prediction handling test.
- Notebook structure/validation smoke.

## Deferred
- The official pass bar (set from the first real benchmark numbers).
- Labeler absolute-accuracy improvement (line-anchors) — only if match-the-labeler
  proves insufficient for Phase 4.
