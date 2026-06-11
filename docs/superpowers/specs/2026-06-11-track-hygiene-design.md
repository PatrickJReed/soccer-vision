# Track Hygiene & Team Re-clustering (design)

**Date:** 2026-06-11
**Status:** Design approved, pending implementation plan
**Depends on:** Phase 3 pipeline (trajectories_px checkpoint), interactive anchor
labeler (homographies at high coverage), local game video on disk.

## Problem

The first analytics-grade run (82.4% manual homography coverage on the bake-off
clip) exposed that the team/tracking layer — not homography — is now the binding
constraint on possession semantics:

- **Tracking is shredded:** ByteTrack produced 1,088 tracks for ~20 people;
  median track lives 8 frames (0.27 s); zero tracks span the clip. Modal-team
  smoothing is powerless over quarter-second fragments.
- **Team clustering is skewed:** 2.9 "own" vs 10.4 "opp" player detections per
  frame — physically impossible (both teams keep ~9 on the field). The SigLIP
  jersey clustering runs during detection, BEFORE any pitch filter exists, so
  adjacent-field players (tournament footage) pollute the 2-cluster fit.
- Net effect: possession attributes 100% of team frames to "opp"; `build`,
  `attack`, and `transition` phases never occur. Every possession-conditioned
  Phase 4 metric is gated on fixing this.

## Goal

A post-hoc, CPU-only **hygiene stage** that consumes existing artifacts
(`trajectories_px.parquet`, `homographies.parquet`, the local video) and emits a
cleaned `trajectories_px_clean.parquet` (same schema) that drops adjacent-field
detections, stitches track fragments in pitch space, re-clusters teams from
jersey/shorts color on stitched tracks, and assigns goalkeepers positionally —
feeding the existing `assemble_from_homographies` unchanged.

## Non-goals

- No GPU work, no re-detection, no ByteTrack re-tuning (upstream backend
  untouched; a future fold-upstream is noted as optional, not designed).
- No recovery of detections the detector never made.
- No player re-identification across long absences (substitutions, long
  off-crop periods may produce multiple stitched tracks per person — acceptable;
  metrics are team-level).
- No halftime/attack-direction normalization (separate known gap; GK assignment
  here is deliberately direction-agnostic).

## Decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Where the fix lives | Post-hoc CPU stage on existing artifacts (not upstream GPU backend) |
| Own-team identification | `--own-kit <color>` hint matched to cluster centroid + contact-sheet images for verification |
| Goalkeeper teams | Positional: nearest team's outfield players on average (clustering excludes GKs) |
| Acceptance | Hard balance gate (own:opp per-frame ratio) + hand-labeled possession CSV → frame agreement, target ≥80% |

## Architecture

New `soccer_vision/hygiene/` subpackage mirroring the labeler pattern:
- `hygiene/core.py` — pure functions (DataFrames in/out, no I/O): stitching,
  feature clustering, own-kit mapping, GK assignment, balance/agreement math.
- `hygiene/run.py` — impure driver: reads parquets, extracts crops from the
  video (cv2), calls the core, writes artifacts.
- `hygiene/__main__.py` — CLI.

**Data flow:** trajectories_px + homographies + video → on-pitch filter →
stitching → team re-clustering → GK assignment → `trajectories_px_clean.parquet`
→ existing `assemble_from_homographies`. Ball and referee rows pass through
untouched.

## Algorithms

### Step 1 — On-pitch filter
Map detections to pitch coords with the existing `PitchMapper`; drop player/GK
rows outside `[−margin, 1+margin]²` (existing `filter_outside_pitch`, default
margin 0.05). This kills adjacent-field pollution before anything downstream
sees it. Rows on frames without a homography (~18% on the bake-off clip) are
kept but flagged: they contribute no clustering features and no stitch
endpoints, and inherit their track's team at the end.

### Step 2 — Pitch-space fragment stitching
Per class (player, goalkeeper separately):
- Sort fragments by start frame. Greedily chain: fragment B continues chain A if
  B starts within `max_gap_s` (default 2.0 s) after A ends AND B's start pitch
  position is within physical reach of A's end: `dist ≤ max_speed × gap + slack`
  (defaults: max_speed 8 m/s — youth sprint; slack 0.02 pitch-length units
  covers detection/box-center jitter).
  Distance is computed in length-normalized pitch units, correcting the x-axis
  by the PitchSpec aspect ratio (x is width-normalized).
- Nearest-in-space wins among time-eligible candidates; each fragment joins at
  most one chain. Endpoints lacking pitch coords do not stitch (conservative).
- Output: `track_id` becomes the stitched id; `orig_track_id` preserved.
- **Bias: tight thresholds.** A missed stitch leaves two clean fragments
  (harmless); a wrong stitch fuses two people (poison for modal team).

### Step 3 — Team re-clustering on stitched tracks
- For each stitched player track, sample up to ~10 crops evenly across its life
  from the local video. Shirt region = upper ~40% of bbox, center ~60% width;
  shorts region = lower ~40%, center ~60%.
- Feature per track: robust median Lab color of shirt + shorts regions (6-dim).
  Both are used deliberately — kits are often doubly separable (here:
  white/blue vs dark-blue/yellow).
- K-means k=2 over track features, tracks weighted by length. A track is near the decision
  boundary when its distances to the two centroids are within 20% of each
  other (d_near/d_far > 0.8) → `team="unknown"` (better unknown than wrong).
- `--own-kit <color>`: a small word→Lab anchor table ("white", "blue",
  "dark blue", "yellow", "red", "green", "black", "orange", "purple"); the
  cluster whose shirt centroid is nearest the hint becomes "own".
- **Contact sheets** `team_cluster_0.png` / `team_cluster_1.png` (grid of sample
  crops per cluster; the own-mapped one suffixed `_OWN`) for 2-second human
  verification.

### Step 4 — Goalkeeper assignment
GKs wear neither kit; excluded from clustering. Each stitched GK track is
assigned the team whose outfield players are on average nearest to it across
its frames. Direction-agnostic, halftime-safe.

## Interface

```
python -m soccer_vision.hygiene \
  --traj trajectories_px.parquet --homographies homographies.parquet \
  --video game.mp4 --own-kit white --out-dir hygiene_out/
```
Tunables: `--max-gap-s 2.0`, `--max-speed 8.0`, `--margin 0.05`. The core is
importable for notebooks/pipeline use without the CLI.

## Artifacts (written to --out-dir)

- `trajectories_px_clean.parquet` — same schema + `orig_track_id`; drop-in for
  `assemble_from_homographies`.
- `team_cluster_0.png` / `team_cluster_1.png` (own-mapped gets `_OWN`).
- `hygiene_report.json` — tracks before→after, span-histogram summary, per-frame
  own/opp count means, unknown fraction, cluster centroids, balance gate
  PASS/FAIL.

## Acceptance

1. **Balance gate (hard, automatic):** mean per-frame own:opp detection ratio in
   [0.6, 1.6]. Printed PASS/FAIL; in the report.
2. **Semantic gate (ground truth):** Patrick watches the clip once and writes a
   change-point CSV — `t_seconds,possession` with `own/opp/none`, one row per
   possession change. An `agreement` function expands it to frames, compares
   with `phases.parquet`'s `possession_state` on frames where both attribute a
   team, and reports % agreement + a disagreement timeline. **Target ≥80%.**
   The same CSV then doubles as calibration data for the possession thresholds
   (margin/clump/loose radii) via cheap recompute — tuning iterations are
   seconds, no GPU.

## Error handling

- Missing-homography rows: flagged, feature-excluded, inherit track team.
- Degenerate clustering (near-identical centroids or extreme membership
  imbalance): loud warning, all players `team="unknown"` — confident garbage is
  never emitted.
- `--own-kit` matching neither centroid well: warning; contact sheets arbitrate.
- Videos whose frame count disagrees with the trajectories' frame range: hard
  error (wrong video supplied).

## Testing

- **Pure core (TDD):** stitching on synthetic fragments — known identities
  re-joined, gap/speed limits respected, wrong-stitch refusal cases; clustering
  on synthetic two-color features incl. boundary→unknown; GK positional cases;
  own-kit mapping incl. no-good-match warning path; balance + agreement math.
- **Driver:** crop-region extraction on a tiny synthetic video; CLI arg smoke.
- **Real-data acceptance** (Patrick, local): run on the bake-off artifacts →
  balance gate → contact-sheet check → re-assemble → possession breakdown +
  agreement vs the ground-truth CSV.
