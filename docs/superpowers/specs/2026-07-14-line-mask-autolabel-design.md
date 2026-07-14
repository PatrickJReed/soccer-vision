# Line-Mask Auto-Label Generator — Design (2026-07-14)

**Goal.** Turn every trusted-homography frame in a labeled session into a training pair
(frame JPEG + per-pixel field-line class mask) with zero human drawing — the data engine
for a per-frame line-segmentation model (the dense-correspondence fix to the project's
core calibration problem; see `docs/superpowers/2026-07-14-global-crop-task11-verdict.md`
§"where this points" and the 2026-07-14 session decision).

**Why this design (context).** Every calibration failure to date traces to per-frame
correspondence scarcity (sparse one-end clicks), not to the projection math. A model that
finds the painted lines densely in EVERY frame makes each frame independently solvable —
no chains, no clusters, no per-game clicking after the model generalizes. Training data is
bootstrapped from what already exists: any frame whose homography we trust can have the
known field structure projected into it as a pixel-accurate mask. Patrick's decision
(2026-07-14): new games get homographies via the **view-rep route** (view_digest reps →
rep clicks → `view_registration.register_clip`, ~15–45 min/game); the generator also
accepts labeler exports so existing sessions work today.

## 1. Inputs

Per game, a session directory containing:
- `homographies.parquet` — pipeline schema (`frame, h00..h22, source, confidence`),
  full-pixel image→pitch. Accepted sources: `rep`, `registered` (view-rep route),
  `manual` (labeler export). Trust gate: `confidence >= min_confidence` (default 0.6).
- the game video (path given on the CLI; ALL-INTRA re-encode recommended for seek speed,
  as with the labeler).
- OPTIONAL `view_manifest.parquet` (Slice 1.5 view_dataset output) — provides
  `frame → view_id` for per-view sampling caps and later view-held-out splits. Absent →
  `view_id = -1` and the per-view cap is skipped (a warning notes weaker dedup).

Plus one registry file the user maintains at the dataset root, `games.yaml`:
```yaml
oceanside_2026_06:   { field: oceanside,  video: /path/to/oceanside_clip.mp4 }
riverside_2026_05a:  { field: riverside,  video: ... }
```
`game_id` (the key) and `field_id` flow into every manifest row — they are what make
game-held-out and **field-held-out** evaluation possible later. The generator never
assigns splits; it records identifiers so the Colab loader can re-split freely.

## 2. Mask generation (pure core: `pitch/line_masks.py`)

For a frame with pixel→pitch homography `H`, rasterize the field structure through
`P = inv(H)` (pitch→pixel) into a uint8 class mask, all drawing behind-camera-clipped
(reuse `viz.pitch_overlay.clipped_polyline` segment-wise, as validate_session renders do,
so unphysical projections can never poison a label):

| class id | name | geometry (pitch [0,1]² coords, from the field model / PitchSpec 9v9) |
|---|---|---|
| 0 | background | everything else |
| 1 | touchline | x=0 and x=1 edges (landmarks 0–2, 1–3) |
| 2 | goal_line | y=0 and y=1 edges (0–1, 2–3) |
| 3 | midline | y=0.5 (landmarks 5–4) |
| 4 | box_line | own box 11→9→10→12 and opp box 15→13→14→16 polylines |
| 5 | center_circle | centre (0.5, 0.5); pitch-frame ellipse with y-radius r=0.087 and x-radius r·(LENGTH_M/WIDTH_M), sampled as a 72-segment polyline |

Rationale for merged left/right classes: the model's job is dense correspondence
geometry, not side identity — the downstream geometric fit disambiguates side from
layout. (A `distance-field` regression target and side-split classes are noted follow-ups,
out of scope v1.)

**Line thickness:** physically scaled. Real paint is ~0.12 m wide; per polyline segment,
convert to pixels via the local projection scale (px length of the segment ÷ pitch-metre
length of the segment), clamped to [2, 7] px. Drawn with `cv2.polylines`
(`LINE_8`, no anti-aliasing — masks are class ids, not images). Later classes draw over
earlier ones at intersections (midline/circle over touchlines) — order is fixed and
documented; intersections are a negligible pixel fraction.

Pure-core API:
```python
def line_mask(h_img_to_pitch: NDArray, size: tuple[int, int],
              *, spec: PitchSpec = PitchSpec.standard_9v9(),
              paint_width_m: float = 0.12) -> NDArray[np.uint8]   # (H, W) class ids
```
plus `LINE_CLASSES: dict[int, str]` and a `mask_overlay(frame, mask) -> NDArray` helper
(color-tints classes onto the frame) for the contact sheet. No I/O in this module.

## 3. Sampling & orchestration (`line_dataset.py`, top level beside `dataset_export.py`)

`build_line_dataset(games.yaml entries or one session, out_dir, *, stride_s=1.0,
per_view_cap=120, min_confidence=0.6, jpeg_quality=90)`:

1. Load `homographies.parquet`; keep frames with accepted source AND confidence ≥ gate.
2. Stride-sample to ~`stride_s` seconds (frame stride = round(fps·stride_s); fps from the
   video); then apply `per_view_cap` per `view_id` (near-duplicate control).
3. Decode selected frames SEQUENTIALLY (single pass, ascending — the view_dataset
   streaming lesson; never per-frame seek), skip undecodable frames (drop from manifest,
   count in stats — the 9e9feac lesson).
4. Write `images/{game_id}_{frame:06d}.jpg`, `masks/{game_id}_{frame:06d}.png`
   (PNG, uint8, lossless — masks must never be JPEG).
5. Append manifest rows: `game_id, field_id, view_id, frame, source, confidence,
   image, mask` → `manifest.parquet` at the dataset root (one manifest across games;
   re-running a game REPLACES its rows — idempotent per game).
6. `dataset_stats.json`: per game/field/view frame counts, per-class pixel fractions,
   confidence histogram, undecodable count, and the generator's config. This file is the
   go/no-go evidence for the Colab run.
7. Contact sheet `contact_{game_id}.jpg`: `mask_overlay` on ~16 frames sampled across
   views/time — for Patrick's visual assessment of label quality (Claude renders, Patrick
   assesses; the generator never self-certifies).

CLI: `python -m soccer_vision.line_dataset --games games.yaml --out <dir>
[--game <id> ...] [--stride-s 1.0] [--per-view-cap 120] [--min-confidence 0.6]`
(`--game` limits to named entries; default = all entries whose inputs exist).

## 4. Dataset plan (agreed 2026-07-14)

- ~16 available games across ≥7 fields. **Reserve 2 fields entirely** (never in
  training) as the tier-3 field-held-out test; label one game on each only when
  evaluation time comes.
- v1 training: 3 games on 3 distinct fields via the view-rep route (~5–8k frames after
  stride/cap), grow along the learning curve (1→2→3 games) rather than guessing.
- Split tiers live in the Colab loader (view-held-out sanity / game-held-out /
  field-held-out deployment claim); the generator only guarantees the identifiers.
- v0 development dataset: oceanside clip via `register_clip` (works today; ~90–150
  frames after sampling — enough to exercise every code path end-to-end).

## 5. Testing

Pure core against the proven synthetic camera (test_global_crop's `_h_g_true` recipe):
- Mask pixels lie where the homography says: for each class, sample mask pixels, map
  through `H` to pitch space, assert distance-to-that-line < paint_width (metres).
- Behind-camera safety: a pose whose far end has w<0 yields a mask with NO far-line
  pixels (clipping) — never wrapped/mirrored pixels.
- Thickness scaling: near-field line thicker than far-field line in a perspective view;
  clamps honored.
- Ellipse correctness: circle points map back to radius r·LENGTH_M metres from centre
  (isotropy check — the x/y radius asymmetry in pitch units).
- Orchestrator: gating by source/confidence; stride+cap counts; idempotent re-run
  replaces a game's rows; undecodable-frame accounting; manifest/stat schema; masks are
  PNG-lossless round-trip.

## 6. Out of scope (v1)

Training code/notebook (follow-up, patterned on `colab_view_embedding`), augmentation
(Colab-side), distance-field targets, side-split classes, per-frame fit from predicted
lines (the consumer, next project after the model exists), packaging beyond jpg/png/parquet.
