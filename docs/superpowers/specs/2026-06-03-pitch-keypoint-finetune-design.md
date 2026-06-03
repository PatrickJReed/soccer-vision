# Phase 3.5b — Pitch-keypoint fine-tune (design)

**Date:** 2026-06-03
**Status:** Design approved, pending implementation plan
**Depends on:** Phase 3 (pitch homography + pipeline), Phase 3.5a (homography propagation)

## Problem

Pitch-homography coverage is the binding constraint on the entire analytics
layer. On the bake-off clip the broadcast-trained pitch-keypoint model yields a
valid homography on only **16.3%** of frames; it detects **0 keypoints** on
frames where corner flags and center circles are clearly visible — an
out-of-distribution recall failure on faint, patchy, sun-bleached youth-field
markings (and overlapping adjacent-game lines). Threshold tuning is exhausted
(0.5 → 0.1 moved coverage 16.3% → 16.8%); it is a detection-recall wall, not a
knob.

Phase 3.5a (temporal propagation) roughly doubled combined coverage to ~30–35%
at a flat 0.021 reprojection error, but cannot bridge the long landmark-free
stretches. The durable lever is **denser anchors**: fine-tune the keypoint model
on Trace footage so more frames produce confident landmarks. Denser anchors
compound with propagation — shorter gaps mean propagation fills them — so 3.5b +
3.5a multiply rather than add.

## Goal

Fine-tune the YOLOv8-pose pitch-keypoint model on labeled Trace footage with a
youth-9v9-appropriate landmark schema, producing denser, more reliable
homography anchors. Ship the weights as a GitHub release asset wired into
`RoboflowBackend`, gated on cross-field homography-coverage of a held-out
multi-game eval set.

## Non-goals

- The metrics/analytics layer (Phase 4) — still gated on coverage; not in scope.
- Per-game absolute-meter calibration — metrics stay dimensionless / pitch-relative.
- A line-segmentation auxiliary model — considered and rejected as YAGNI.
- 11v11 support — `PitchSpec.fifa_11v11()` stays as-is; this phase is youth 9v9.

## Decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Labeling budget | Large (600+ frames) |
| Landmark schema | 21-point youth set: corners + halfway + center + center-circle apexes + 8 penalty-box corners + 4 goal-post bases |
| Labeling method | Semi-automatic / active learning (propagation-projected pre-labels) |
| Footage | Several full games available at different fields |
| Canonical coordinates | Derived from `PitchSpec` + US Soccer published 9v9 proportions |
| Held-out gate | Multi-clip, multi-field set incl. an entirely-unseen field |

## Landmark schema (21 points)

Replaces the vendored 32-point FIFA `SoccerPitchConfiguration` in
`pitch/landmarks.py`. Index assignment:

| idx | group | points |
|----|----|----|
| 0–3 | field corners | 4 |
| 4–5 | halfway × touchline | 2 |
| 6 | center mark | 1 |
| 7–8 | center-circle apexes (far / near) | 2 |
| 9–12 | left penalty-box corners (2 outer + 2 on goal line) | 4 |
| 13–16 | right penalty-box corners (2 outer + 2 on goal line) | 4 |
| 17–18 | left goal-post bases | 2 |
| 19–20 | right goal-post bases | 2 |

**Camera-geometry note (domain knowledge):** the Trace camera sits on the near
sideline at midfield, so the **near halfway × touchline point (idx 5) sits
almost directly under the camera and is never visible.** Its index slot and
canonical coordinate are retained for schema regularity, but it is never
labeled; the model learns low confidence there and `build_frame_homographies`
filters it out by confidence. Watch whether the near center-circle apex (idx 8)
and stretches of the near touchline are similarly foreshortened in practice.

**Tier rationale:** corners + halfway + center (Tier 1, idx 0–8) survive
sun-bleaching and are unambiguous. Penalty-box corners (Tier 2, idx 9–16) roughly
double the points available per half, so end-zone pans (where Trace's virtual-PTZ
crop shows only one penalty area and no center line) can still anchor. Goal-post
bases (from Tier 3, idx 17–20) are crisp dark verticals — reliable despite being
"optional." The faint/ambiguous Tier 3 markings (goal-area lines, build-out
lines, penalty arcs) are deliberately excluded to avoid reintroducing label noise.

## Canonical coordinates (PitchSpec-derived)

Today `pitch/landmarks.py` hardcodes FIFA's 12000 × 7000 cm layout (aspect
1.714). For youth 9v9 (aspect ~1.5) this is wrong, distorting the homography
target space. New approach:

- Add `youth_landmarks(spec: PitchSpec) -> NDArray[np.float64]` that **computes**
  the 21 canonical [0,1]² coordinates from the spec's proportions.
- `PitchSpec.standard_9v9()` already carries `aspect_ratio=1.5`,
  `penalty_box_length_frac`, `penalty_box_width_frac`, `center_circle_radius_frac`.
  Add `goal_width_frac` (for goal-post bases); optional goal-area fracs reserved
  for future use but not part of the 21-point set.
- `PITCH_LANDMARKS = youth_landmarks(PitchSpec.standard_9v9())` — one source of
  truth shared by the homography target space and the metrics layer.

**Axis convention preserved:** y = goal-to-goal (the phase splitter relies on
`y < 0.333` = own third, `y > 0.667` = opp third), x = touchline-to-touchline.
The construction builds coordinates directly in this frame, replacing the
FIFA→ours swap.

**Coordinates accept mild distortion.** Real youth fields vary; published
proportions keep the canonical box/circle ratios roughly correct, which is what
keeps the homography fit (corners + box corners jointly) consistent.
`build_frame_homographies`, `mapper.py`, and `filter.py` are unchanged — they
already index `PITCH_LANDMARKS[kp_idx]` generically, so a 21-row table just works.

## Data pipeline & active-learning loop

### Footage split (leakage-safe, field-level)

Of the several available games, reserve whole games/fields for the held-out set
**before** sampling any training frames:

- **Train/val:** frames sampled from training games only.
- **Held-out gate:** the bake-off clip + clips from ≥2 other games at different
  fields, partitioned so that **at least one field is entirely absent from
  training** (strict unseen-field test) while other held-out clips come from
  fields seen in training but at **non-overlapping time ranges** (unseen-time,
  same field). No frame/clip overlap with training, ever.

### Frame sampling

A small extraction utility/notebook samples frames per training game,
**stratified to over-sample hard cases** — end-zone pans (one penalty area
visible, no center line), sun/shadow boundaries — rather than uniform sampling
that wastes labels on easy center-line frames.

### The loop (propagation-projected pre-labels)

1. **Seed (~100 frames, manual):** label across 2–3 training games in Roboflow
   with the 21-point skeleton → establishes pose-v0.
2. **Train pose-v0** on the seed.
3. **Propose** via the new `pitch/autolabel.py`: run pose-vN over a game, fit
   anchors with `build_frame_homographies`, propagate with `propagation.py`, then
   **project the 21 canonical landmarks through each frame's homography back into
   pixels**, keeping only points that land in-frame. Emit Roboflow
   pre-annotations (YOLO-pose / COCO-keypoint format) with visibility flags.
4. **Correct:** review/fix proposals in Roboflow — fast because points are
   pre-placed.
5. **Retrain & iterate:** fold corrected frames in, retrain, re-propose on the
   next batch of hard frames. Repeat toward 600+.

Why this loop: propagation carries a good homography *into* a frame where the
model saw nothing, so proposals land on the faint markings the model currently
misses — the exact OOD frames we need. The 3.5a "denser anchors → shorter gaps"
property does work during *labeling*, not just inference.

### `pitch/autolabel.py` (new module)

The one substantial new module and the reusable engine (notebooks merely drive
it). Pure-ish: takes frames + model predictions + propagation outputs, returns
annotations; testable on synthetic homographies without a GPU. Keep it separate
from the driving notebooks.

## Model & training

- **Architecture:** YOLOv8-pose, 1 class (`pitch`), `kpt_shape: [21, 3]`
  (x, y, visibility). Base `yolov8s-pose.pt`.
- **Recipe (mirror the ball fine-tune):** imgsz 1280, ~100 epochs, patience 20,
  mosaic/mixup/hsv/scale/fliplr augmentation.
- **`flip_idx` (gotcha):** YOLO-pose horizontal-flip augmentation requires a
  left↔right symmetric keypoint mapping (corner-L ↔ corner-R, box corners mirror,
  goal-post bases mirror, halfway/center map to themselves), or flip augmentation
  silently corrupts labels. The 21-point schema must define this and the
  `data.yaml` must carry it.
- **Visibility flags:** frames rarely show all 21 points; use v=0 (not labeled /
  out of frame), v=1 (occluded), v=2 (visible). The never-visible idx 5 is always
  v=0.
- **Inference parsing unchanged:** one pitch instance per frame yields kp_idx
  0–20; the existing loop in `roboflow.py`
  (`for kp_idx in range(kp.xy.shape[1])`) handles N keypoints generically.

## Weights hosting & code wiring

Mirror ball-v1:

- Publish trained weights as GitHub release asset `pitch_yolov8_v1.pt` under a
  `pitch-v1` release.
- Flip `WEIGHTS["pitch"]` in `tracking/roboflow.py` from the `gdrive` roboflow
  baseline to `("url", PITCH_V1_URL, "pitch_yolov8_v1.pt")`; add a `PITCH_V1_URL`
  constant. Auto-downloads through the existing `_download_weights` path.
- Add optional `pitch_weights_path` constructor arg for local override,
  paralleling `ball_weights_path`.

### Code touch-list

- `pitch/landmarks.py` — `youth_landmarks(spec)` + new 21-row `PITCH_LANDMARKS`.
- `pitch/spec.py` — add `goal_width_frac` (+ optional reserved goal-area fracs).
- `pitch/autolabel.py` — **new**, the proposal engine.
- `tracking/roboflow.py` — `WEIGHTS["pitch"]` → release URL; `PITCH_V1_URL`;
  `pitch_weights_path` arg.
- Tests: `test_pitch_landmarks.py` (21-pt table, PitchSpec-derived coords, axis
  convention, idx-5 reserved), new `test_pitch_autolabel.py` (projection
  round-trips on synthetic Hs), `test_tracking_roboflow*.py` (pitch weights
  wiring).

### Notebooks (run on Colab by Patrick)

1. `extract_pitch_frames.ipynb` — stratified frame sampling from training games.
2. `finetune_pitch.ipynb` — train pose-v0 → vN (mirrors `finetune_ball.ipynb`).
3. `autolabel_pitch.ipynb` — drive `autolabel.py`: predict → propagate → project
   → export Roboflow pre-annotations.
4. `acceptance_pitch.ipynb` — run the full pipeline on the held-out multi-field
   set; report the gate metrics below.

## Acceptance gate

Measured on the **held-out multi-clip, multi-field set** (never trained on).
Baseline to beat (bake-off clip): 16.3% anchor coverage, ~30–35%
combined-with-propagation, 0.021 held-out reprojection error.

**Primary gates (aggregate; also watch the worst field):**

1. **Anchor coverage** (frames with ≥4 confident keypoints) — **≥ 40%** (up from
   16.3%).
2. **Combined coverage** (anchors + 3.5a propagation at max_gap=45) — **≥ 65%**
   (up from ~30–35%).
3. **Held-out reprojection error** — **≤ 0.05** pitch units on every clip (must
   not regress the 3.5a gate).

**Diagnostics (reported, not hard gates), split by unseen-field vs unseen-time:**

- **Keypoint accuracy** on labeled held-out slices: median per-keypoint pixel
  error + PCK@(~1% of image diagonal). Expect corners > box corners > goal posts.
- **Per-landmark detection rate** — catches a systematically missed landmark
  (labeling or schema issue).
- **In-bounds rate** of mapped players — now finally measurable on enough frames
  to *confirm* the axis/team convention (per the 3.5a memo, do not flip
  axis/team blind; this is where we confirm).
- A large unseen-field vs unseen-time gap signals overfitting to specific fields
  → add field diversity to training.

**Honest caveats:**

- The 40% / 65% targets are **estimates, not guarantees**. The bet is sound (the
  model currently sees 0 keypoints on clearly-marked frames, so recall has
  nowhere to go but up), but a deeper OOD gap could land us lower and require more
  labeling rounds.
- **Early signal:** pose-v0 on the seed already shows whether the recall wall
  breaks. If anchor coverage stalls below ~25% after a couple of active-learning
  rounds, reassess rather than grind more labels.
- **Unseen-field coverage will likely come in below same-field** — that is the
  number that predicts next-season performance on new fields. Report it
  prominently, not buried.
- Proposal pre-labels inherit propagation error (a few px); the correction step
  is mandatory. Unreviewed proposals never enter the training set.

## Iteration economics

The checkpoint parquets make re-tuning cheap: after a new model, threshold and
propagation changes re-run on CPU (`build_homographies` / `assemble_from_parquet`)
in seconds; only re-detection needs GPU. This keeps the active-learning loop and
post-fine-tune gate sweeps fast.
