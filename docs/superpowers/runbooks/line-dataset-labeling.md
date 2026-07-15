# Runbook: from a raw game video to line-segmentation training data

Per game, ~15–45 min of clicking + some unattended compute. All commands from
`packages/soccer-vision/`. Context: specs `2026-07-14-line-mask-autolabel-design.md`,
`2026-07-15-line-segmentation-colab-design.md`.

## 0. Prep the video (once per game)

```bash
ffmpeg -i game_raw.mp4 -g 1 -keyint_min 1 -x264-params scenecut=0 -c:a copy game.mp4
```
All-intra re-encode → ~5x faster frame scrubbing in the labeler (macOS hardware-decoder
gotcha). Full games beat clips: sun/shadow drift is training diversity.

## 1. View digest (automatic, no clicking)

```bash
uv run python -m soccer_vision.labeler.view_digest --video /path/game.mp4 --out /path/game_digest
```
Outputs `views_montage.png` (eyeball it: each tile = one distinct view + its
representative frame number), `view_digest.json`, `similarity.png`. Expect ~13–25 views
on a full game.

## 2. Label the representative frames (the only clicking)

```bash
uv run python -m soccer_vision.labeler --video /path/game.mp4 \
  --export-dir /path/game_export --port 8000 --workers 1
```
For EACH representative frame from the montage: navigate to that frame number, click
5+ point landmarks (spread them — corners, box corners, posts), plus near-touchline /
midline LINE clicks where visible (they drive the green foreground check). The frame
must come out **green** — export writes green frames only, and a rep that fails to
export drops its whole view from registration. Then hit Export →
`/path/game_export/homographies.parquet` (+ keypoints/line_clicks parquets, and
`calib_gate.json` in crop mode).

## 3. Register every frame to its rep (automatic)

```bash
uv run python -m soccer_vision.pitch.view_registration \
  --video /path/game.mp4 --digest-json /path/game_digest/view_digest.json \
  --rep-homographies /path/game_export/homographies.parquet \
  --out /path/game_session --validate
```
Writes `/path/game_session/homographies.parquet` (sources `rep`/`registered`) and prints
coverage + within-view drift stats. Coverage well under ~90%: some reps didn't export
green or register — check step 2 for those views.

## 4. View manifest (recommended — enables per-view caps + view-held-out training splits)

```bash
uv run python -m soccer_vision.labeler.view_dataset --video /path/game.mp4 \
  --digest-json /path/game_digest/view_digest.json --out /path/game_viewdata
cp /path/game_viewdata/view_dataset.parquet /path/game_session/view_manifest.parquet
```
(The generator reads `frame`/`view_id` from `view_manifest.parquet`; extra columns are
harmless.) Skipping this: that game's frames get `view_id = -1` — still trainable, but
no per-view dedup cap and no view-held-out membership.

## 5. Register the game and generate pairs

Append to `~/sv-labeler/games.toml` (paths relative to the toml):
```toml
[riverside_2026_05a]
field = "riverside"          # the FIELD id — powers field-held-out evaluation
video = "riverside/game.mp4"
session = "riverside/game_session"
```
```bash
uv run python -m soccer_vision.line_dataset --games ~/sv-labeler/games.toml \
  --out ~/sv-labeler/line_dataset_v1
```
Check the printed line: `N pairs written (... 0 empty masks)` — any empty masks or a
sparse-anchor WARNING means step 3's homographies need attention. Then eyeball
`contact_<game_id>.jpg` (masks tinted on frames) before trusting the game's labels.

## 6. Train (after all 3 games)

Upload `~/sv-labeler/line_dataset_v1/` to Drive. Open
`scripts/colab_line_segmentation.ipynb`, set `DATASET_DIR` (+ `HELDOUT_GAMES` /
`HELDOUT_FIELDS` when you have them), run the SMOKE cell first (minutes), then the full
run. Report the per-tier IoU tables back.

## Standing rules

- **Never train on the 2 reserved fields.** Their games get labeled only at evaluation
  time (ground truth to score against), never into the training dataset dir.
- Consumers read pairs via `manifest.parquet` — orphan files from shrinking re-runs are
  expected and harmless.
- Re-running the generator for one game is safe/idempotent (`--game <id>`).
