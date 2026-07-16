# Runbook: from a raw game video to line-segmentation training data (Colab-first)

Per game, ~15–45 min of clicking + unattended Colab compute. The full game video
never touches the local machine or Drive — only Colab's ephemeral disk. The single
local step is clicking the few-MB rep pack. Context: specs
`2026-07-16-colab-game-prep-design.md`, `2026-07-14-line-mask-autolabel-design.md`,
`2026-07-15-line-segmentation-colab-design.md`. Notebook:
`packages/soccer-vision/scripts/colab_game_prep.ipynb` (exact twin of the `.py`).

## 0. One-time Drive setup

Create `MyDrive/soccer-vision/` on Drive and put the unlisted Trace playlist URL in
`MyDrive/soccer-vision/trace_playlist_url.txt` (one URL per line). The URL is
intentionally **never** in the repo or notebook — the repo is public, the playlist is
unlisted family video. Mind Drive space: each game adds ~0.3–0.7 GB of dataset pairs.

## 1. Stage A — Colab: pull → digest → rep pack

Open `scripts/colab_game_prep.ipynb` in Colab (CPU runtime is fine). In the CONFIG
cell set per game: `PLAYLIST_LINE`, `GAME_NUMBER` (1-based playlist index),
`GAME_ID` (e.g. `riverside_2026_07a`), `FIELD_ID`. For a full game also raise
`DIGEST_STRIDE` (the digest cell warns and suggests a value — the ORB similarity
matrix is O(samples²)). Then Runtime → Run all: the Stage B cells self-skip until the
rep export exists, and Stage A self-skips once its rep pack is on Drive
(`FORCE_STAGE_A = True` to redo).

Check inline: the playlist listing (is `GAME_NUMBER` the right game?) and the view
montage (each tile = one distinct camera view; expect ~13–25 on a full game). Output:
`soccer-vision/<GAME_ID>/rep_pack/` on Drive — `rep_video.mp4` (one original-res
frame per view), `rep_map.json` (tiny index → original frame + view), the digest
json, and the montage. A few MB — the only download.

## 2. Local — click the rep frames (the only local work, unchanged in substance)

Download `rep_pack/` from Drive, then from `packages/soccer-vision/`:

```bash
uv run python -m soccer_vision.labeler --video rep_pack/rep_video.mp4 \
  --export-dir rep_export --workers 1
```

Label EVERY frame of the tiny video (frames are the view representatives, ordered by
original frame number — `rep_map.json` / the montage say which view is which). Each
frame must come out **green**: 5+ spread point landmarks (corners, box corners,
posts) + near-touchline / midline LINE clicks where visible. A frame that isn't green
drops its whole view from registration. Each rep is its own one-frame segment — the
physical engine handles that natively (the shared focal wants ≥3 diverse green
frames). Hit Export, then upload the `rep_export/` folder to Drive at
`soccer-vision/<GAME_ID>/rep_export/` (must contain `homographies.parquet`).

## 3. Stage B — Colab: registration → view manifest → dataset pairs

Back in the same notebook (after a Colab restart: run the CONFIG + setup cells
first), run the Stage B cells. The guard cell re-pulls the game onto the ephemeral
disk if the session was recycled; everything durable is already on Drive.

The cells: remap the labeled homographies tiny→original, register every frame to its
view rep (`session/homographies.parquet`), build the dense view manifest
(`session/view_manifest.parquet`), upsert the game into `soccer-vision/games.toml`
(idempotent per game id), run the generator into `soccer-vision/line_dataset_v1/`.

Check the printed output:
- registration coverage — a warning below ~90% means some reps didn't export green
  or register: redo step 2 for those views (the cell names them) and re-run Stage B;
- the generator line `N pairs written (... 0 empty masks)` — empty masks or a
  sparse-anchor warning mean the homographies need attention;
- the inline contact sheet (masks tinted on frames) — eyeball before trusting labels.

## 4. Train (after all 3 games)

The dataset is already on Drive — no upload. Open
`scripts/colab_line_segmentation.ipynb`, set `DATASET_DIR` to
`/content/drive/MyDrive/soccer-vision/line_dataset_v1` (+ `HELDOUT_GAMES` /
`HELDOUT_FIELDS` when you have them), run the SMOKE cell first (minutes), then the
full run. Report the per-tier IoU tables back.

## Standing rules

- **Never train on the 2 reserved fields.** Their games get labeled only at
  evaluation time (ground truth to score against), never into the training dataset.
- Consumers read pairs via `manifest.parquet` — orphan files from shrinking re-runs
  are expected and harmless.
- Re-running the generator for one game is safe/idempotent (`--game <id>`).
- The playlist URL lives only in the Drive text file — never commit it anywhere.

## Appendix — fallback: fully local (the pre-Colab flow)

Use only when Colab is unavailable AND the machine can hold a full game video.
All commands from `packages/soccer-vision/`.

### A0. Prep the video (once per game)

```bash
ffmpeg -i game_raw.mp4 -g 1 -keyint_min 1 -x264-params scenecut=0 -c:a copy game.mp4
```
All-intra re-encode → ~5x faster frame scrubbing in the labeler (macOS
hardware-decoder gotcha). Full games beat clips: sun/shadow drift is training
diversity.

### A1. View digest (automatic, no clicking)

```bash
uv run python -m soccer_vision.labeler.view_digest --video /path/game.mp4 --out /path/game_digest
```
Outputs `views_montage.png` (eyeball it: each tile = one distinct view + its
representative frame number), `view_digest.json`, `similarity.png`. Expect ~13–25
views on a full game.

### A2. Label the representative frames (the only clicking)

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

### A3. Register every frame to its rep (automatic)

```bash
uv run python -m soccer_vision.pitch.view_registration \
  --video /path/game.mp4 --digest-json /path/game_digest/view_digest.json \
  --rep-homographies /path/game_export/homographies.parquet \
  --out /path/game_session --validate
```
Writes `/path/game_session/homographies.parquet` (sources `rep`/`registered`) and
prints coverage + within-view drift stats. Coverage well under ~90%: some reps didn't
export green or register — check step A2 for those views.

### A4. View manifest (recommended)

```bash
uv run python -m soccer_vision.labeler.view_dataset --video /path/game.mp4 \
  --digest-json /path/game_digest/view_digest.json --out /path/game_viewdata
cp /path/game_viewdata/view_dataset.parquet /path/game_session/view_manifest.parquet
```
(The generator reads `frame`/`view_id` from `view_manifest.parquet`; extra columns
are harmless.) Skipping this: that game's frames get `view_id = -1` — still
trainable, but no per-view dedup cap and no view-held-out membership.

### A5. Register the game and generate pairs

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
Check the printed line: `N pairs written (... 0 empty masks)`, then eyeball
`contact_<game_id>.jpg` before trusting the game's labels. Upload the dataset dir to
Drive for training (step 4 above).
