# Line-Segmentation Colab Trainer — Design (2026-07-15)

**Goal.** A self-contained Colab script (+ .ipynb twin) that trains SegFormer-B0 on the
line-mask dataset (`line_dataset.py` output) and evaluates it on the three honest tiers —
the model half of the dense-correspondence plan (spec
`2026-07-14-line-mask-autolabel-design.md`; Patrick chose SegFormer-B0, 2026-07-15).

**Pattern.** Mirrors `scripts/colab_view_embedding.py`: one cell-structured `.py` in
`packages/soccer-vision/scripts/` plus an `.ipynb`, fully self-contained (no soccer_vision
imports at runtime — Colab has only the uploaded dataset dir), config constants at top,
Patrick runs all GPU work.

## Inputs

`DATASET_DIR` (Drive/local): `manifest.parquet` (game_id, field_id, view_id, frame,
source, confidence, image, mask) + `images/` + `masks/` + `dataset_stats.json`. Pairs are
read **via the manifest only** (orphan-file safety); missing files are warned and dropped
with counts. LINE_CLASSES vocabulary hardcoded to match `pitch/line_masks.py`
(0 background, 1 touchline, 2 goal_line, 3 midline, 4 box_line, 5 center_circle) with a
comment tying the two.

## Splits (the honesty core)

Config: `HELDOUT_FIELDS: list[str]`, `HELDOUT_GAMES: list[str]`,
`HELDOUT_VIEW_FRACTION: float = 0.25`, `SEED`.

- **Tier 3 (field-held-out, the deployment claim):** every row whose `field_id` ∈
  HELDOUT_FIELDS → test set; never trained, never used for model selection; evaluated
  once, at the end.
- **Tier 2 (game-held-out):** rows with `game_id` ∈ HELDOUT_GAMES (games on TRAINING
  fields) → val₂; evaluated at the end.
- **Tier 1 (view-held-out):** within remaining training games, hold out a deterministic
  (seeded) `HELDOUT_VIEW_FRACTION` of each game's distinct `view_id`s → val₁; used for
  epoch-level model selection.
- **Degraded v0 mode:** if all remaining rows have `view_id == -1` (no view manifest),
  fall back to a TIME-BLOCKED split per game (first 85% of frames train, last 15% val₁)
  and print a loud banner: smoke-test only, adjacent-frame near-duplicates inflate val.
  NEVER a random frame split (the view-embedding tautology lesson).
- Empty tiers are reported as "not evaluable with this config", never silently skipped.

## Model / training

- `SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0")`, `num_labels=6`,
  id2label/label2id from the class vocabulary. Deps cell: `transformers`, `torch`,
  standard Colab stack.
- Data: random 512×512 crops from full-res (bias sampling toward crops containing ≥1
  line pixel, ~50%, so batches aren't all grass); masks NEAREST-resized/cropped only;
  horizontal flip (safe: classes are side-merged); mild color jitter. Val: center/full
  tiling at 512 or resize-with-NEAREST — deterministic, no augmentation.
- Loss: class-weighted cross-entropy, weights = inverse sqrt pixel frequency computed
  from the manifest's masks (background ≈ 99.9% — unweighted CE learns "grass").
- Optimizer AdamW, cosine or constant LR (config), `EPOCHS`/`BATCH`/`LR` config, seeded
  determinism (torch/np/random).
- Checkpointing: best-by-val₁ mean line-IoU saved to `OUT_DIR` (Drive) with the config
  echoed into a sidecar json.

## Metrics & evidence

- Per-class IoU + mean line-IoU (background EXCLUDED from the mean) per tier; printed
  as a table per epoch (tier 1) and final (tiers 2/3 where populated).
- Visual cell: N val frames as (frame | GT tint | prediction tint) rows — Claude renders
  the layout, Patrick assesses.
- Inference helper cell: single image path → predicted mask (+ tint), the seam the
  future per-frame-fit consumer will call.
- Smoke cell: tiny config (2 epochs, few batches, works on CPU/T4) against the 70-pair
  v0 dataset — validates the whole path in minutes.

## Repo-side verification (no GPU in repo)

The script gets a `--dry-run` CLI mode runnable in the repo venv WITHOUT torch/transformers
(lazy imports): loads the manifest, applies the split logic, prints per-tier row counts,
class-frequency table, and warnings (missing files, degraded mode). Evidence: dry-run
output against `~/sv-labeler/line_dataset_v0`. Ruff-clean (unlike the older scripts);
mypy not required for scripts/ (repo precedent).

## Out of scope

The mask→homography per-frame fit consumer + gate integration (next project once the
model shows transfer); distance-field targets; multi-GPU; augmentation search.
