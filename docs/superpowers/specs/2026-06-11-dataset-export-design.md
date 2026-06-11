# Training-Dataset Export (design)

**Date:** 2026-06-11
**Status:** Design approved, pending implementation plan
**Depends on:** Interactive anchor labeler (exports), `pitch/autolabel.py` (3.5b),
`pitch/landmarks.py` (FLIP_IDX), local game videos on disk.

## Problem

The path off per-game clicking is retraining the YOLOv8-pose pitch-keypoint
model on labeler-generated data (the 3.5b flywheel). The pieces exist —
`autolabel.propose_labels` / `to_yolo_pose_line` convert labeler homographies
into YOLO-pose annotations (bake-off clip: 14,406 annotations across 1,832
frames from 273 clicks) — but there is no assembly step that turns N games'
labeler exports into one ready-to-train dataset, and no quality gate before
burning GPU time on machine-generated labels.

## Goal

A CLI that combines N `(video, homographies.parquet)` pairs into a single
YOLOv8-pose dataset (images + labels + complete `data.yaml`), with per-game QA
contact sheets and per-landmark statistics for a 2-minute human eyeball before
upload. Plus a `finetune_pitch.ipynb` path that consumes the zip directly
(no Roboflow — these labels are machine-generated; the correction step is the
QA eyeball, not a re-labeling pass).

## Non-goals

- No Roboflow upload/import.
- No human per-frame correction loop (that was the 3.5b model-bootstrap design;
  this flow projects from verified manual homographies instead).
- No automatic field-diversity balancing across games (report counts; the human
  decides what to label next).
- No change to training recipes or acceptance gates (existing notebooks).

## Decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Roboflow in the loop | No — direct-to-YOLO zip; QA replaces correction |
| Sampling | Stride 5 over covered frames (~6 fps) , CLI-tunable |
| Train/val split | Temporal: last 10% of each game's sampled frames → val (random would leak near-duplicates) |
| Multi-game | One CLI invocation, repeated `--game VIDEO HOMOGRAPHIES` pairs, one combined dataset |
| QA gate | Per-game contact sheet (~12 random frames, keypoints drawn) + per-landmark annotation counts |

## Design

**Module:** `src/soccer_vision/dataset_export.py` — pure assembly helpers + a
thin `main()`; runnable as `python -m soccer_vision.dataset_export`.

```
python -m soccer_vision.dataset_export \
  --game <video.mp4> <homographies.parquet> \
  [--game ...]... \
  --out-dir <dir> [--stride 5] [--min-confidence 0.5] [--val-frac 0.10] [--zip]
```

**Per game:**
1. `homographies_from_parquet` (full-pixel → pitch, as the labeler exports).
2. `propose_labels(entries, PITCH_LANDMARKS, (W, H), min_confidence)` — the
   existing residual-derived confidence gate.
3. Covered frames sorted; every `stride`-th selected; the last `val_frac` of the
   selection (temporally) goes to val, the rest to train.
4. One sequential video pass (grab/retrieve skipping, the established pattern)
   writes `images/{train,val}/<game>_f%06d.jpg` (JPEG quality 90) and
   `labels/{train,val}/<game>_f%06d.txt` via `to_yolo_pose_line`.

**`data.yaml`:** written complete and correct at export time: `path: .`,
`train: images/train`, `val: images/val`, `nc: 1`, `names: [pitch]`,
`kpt_shape: [21, 3]`, `flip_idx:` taken from `landmarks.FLIP_IDX` (the tested
constant — never hand-copied).

**QA artifacts (the gate before GPU):**
- `qa_<game>.jpg`: grid of ~12 randomly sampled exported frames with their
  projected keypoints drawn (dot + index). Human check: dots sit on the painted
  intersections; the known suspect is near-touchline drift in midfield views.
- Printed per-game stats: frames exported (train/val), total annotations, and
  **per-landmark annotation counts** (a landmark with anomalously few
  annotations signals a projection gap; idx 5 must be ~0).

**`--zip`:** writes `<out-dir>.zip` for the Colab upload.

**Notebook:** `finetune_pitch.ipynb` gains an "Option B — local dataset" cell:
upload/unzip `dataset.zip`, set `dataset_location` to it, skip the Roboflow
download; the existing data.yaml patch cell is kept but becomes a consistency
check (our yaml already carries kpt_shape/flip_idx).

**Determinism:** the QA frame sample uses a fixed seed so reruns produce the
same sheet.

## Error handling

- A `--game` video whose frame count is less than the homographies' max frame →
  hard error (wrong video for that parquet — mirrors the hygiene check).
- A game contributing zero covered frames after the confidence gate → loud
  warning, game skipped (not silently empty).
- Duplicate game stems (e.g. two `clip.mp4`) → hard error (filename collisions).

## Testing

- Pure: stride selection + temporal split arithmetic; label lines are 68 tokens;
  data.yaml contents incl. flip_idx == landmarks.FLIP_IDX; duplicate-stem error.
- End-to-end on a synthetic video + identity full-pixel homographies: dataset
  tree structure, image/label pairing 1:1, val fraction, QA sheet exists,
  per-landmark counts sane.
- CLI smoke (argparse wiring).
- **Real acceptance:** run on the bake-off export; Patrick eyeballs
  `qa_clip.jpg` before any upload.
