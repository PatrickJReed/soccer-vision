# Line-Segmentation Colab Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Self-contained Colab trainer (SegFormer-B0) for the line-mask dataset with tiered honest evaluation, plus a repo-runnable `--dry-run` verification mode.

**Architecture:** One cell-structured script `packages/soccer-vision/scripts/colab_line_segmentation.py` (config → deps → data/splits → model → train → metrics → visuals → inference/smoke cells) + an `.ipynb` twin. Torch/transformers imported LAZILY inside cells so `--dry-run` (manifest + split verification) runs in the repo venv without them. Spec: `docs/superpowers/specs/2026-07-15-line-segmentation-colab-design.md` — implement it faithfully; the spec is the contract for splits, loss weighting, metrics, and the degraded-v0 banner.

**Tech Stack:** pandas/pyarrow + numpy (dry-run path); torch + transformers + PIL/cv2 (Colab cells). Follows `scripts/colab_view_embedding.py`'s single-file cell pattern.

### Task 1: The script, its .ipynb twin, and dry-run evidence

**Files:**
- Create: `packages/soccer-vision/scripts/colab_line_segmentation.py`
- Create: `packages/soccer-vision/scripts/colab_line_segmentation.ipynb`

- [ ] **Step 1:** Read the spec + `scripts/colab_view_embedding.py` (cell pattern, Drive conventions, .ipynb structure) + `line_dataset.py` (manifest schema) + `pitch/line_masks.py` (LINE_CLASSES vocabulary — hardcode with a sync comment).
- [ ] **Step 2:** Write the `.py` with `# %%` cell markers, sections per the spec: CONFIG (DATASET_DIR, OUT_DIR, HELDOUT_FIELDS, HELDOUT_GAMES, HELDOUT_VIEW_FRACTION=0.25, EPOCHS, BATCH, LR, SEED, SMOKE flags); pure split functions (`assign_tiers(manifest, ...) -> DataFrame with 'tier' column` — deterministic, degraded-mode banner when all view_id==-1); class-weight computation from masks; lazy-import training cells (SegFormer-B0, weighted CE, 512 crops with ≥1-line-pixel bias, hflip, NEAREST masks); per-epoch tier-1 IoU table + final tier-2/3; best-checkpoint save + config sidecar; visual grid cell; single-image inference cell; smoke cell; `--dry-run` CLI entry (argparse: --dataset, --heldout-field, --heldout-game repeatable) printing per-tier counts + class frequencies + warnings WITHOUT importing torch.
- [ ] **Step 3:** Verify dry-run in the repo venv: `cd packages/soccer-vision && uv run python scripts/colab_line_segmentation.py --dry-run --dataset ~/sv-labeler/line_dataset_v0` → expect: 70 rows, degraded-mode banner (all view_id==-1), time-blocked split ~59/11, class-frequency table, 0 missing files. Paste the output in your report.
- [ ] **Step 4:** Verify the torch cells are at least import-safe by static check: `uv run python -m py_compile scripts/colab_line_segmentation.py`; ruff-clean: `uv run ruff check scripts/colab_line_segmentation.py` (the NEW script must be clean even though older scripts are not).
- [ ] **Step 5:** Generate the `.ipynb` twin (cells split on `# %%`, markdown headers from cell comments — same approach as colab_view_embedding.ipynb; verify valid JSON via `python -c "import json; json.load(open(...))"`).
- [ ] **Step 6:** Commit:
```bash
git add packages/soccer-vision/scripts/colab_line_segmentation.py packages/soccer-vision/scripts/colab_line_segmentation.ipynb
git commit -m "feat(ml): SegFormer-B0 line-segmentation Colab trainer (tiered honest eval + repo dry-run)"
```

## Self-review checklist
Spec coverage (splits incl. degraded mode / weighted loss / metrics / visuals / smoke / dry-run) — all present; no placeholders; LINE_CLASSES values match pitch/line_masks.py exactly.
