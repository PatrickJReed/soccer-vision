# Bake-off Results — 2026-05-28

**Source clip:** `data/bakeoff_clip.mp4` (2 minutes, 1920×1080, 30 fps, H.264 —
extracted from a Trace-camera recording of a 9v9 youth tournament game).

**Note on homography reference:** the spec originally called for a manual
4-corner annotation per the clip. We dropped that during Phase 1: Trace is a
virtual-PTZ that pans to follow the ball, so no single frame contains all four
pitch corners. Each candidate's per-frame pitch keypoint detection (or absence
thereof) is what was actually scored under "Homography fidelity."

## Scoring matrix

Scores 1 (worst) — 5 (best), assigned by visual inspection of each candidate's
`annotated.mp4` in `data/bakeoff_outputs/`.

| Axis                          | Roboflow | ATFA      | TZ      |
| ----------------------------- | -------- | --------- | ------- |
| Player detection recall       |    3     |    3      |    1    |
| Ball detection rate           |    2     |    2      |    0    |
| Track stability               |    2     |  **4**    |  —      |
| Team classification           |    3     |    3      |    2    |
| Homography fidelity           |  **4**   |    1      |    1    |
| 9v9 handling                  |    3     |    3      |    1    |
| **Total**                     | **17**   |   16      |    5    |

Player + ball detection were near-identical between Roboflow and ATFA because
both ran the same roboflow-trained `football-player-detection.pt` model. ATFA
provides its own `Tracker` wrapper which produced visibly less ID flicker than
Roboflow's bare `supervision.ByteTrack`, hence ATFA's higher Track stability
score. TZ used its own `old_data.pt` weights, which were trained on
broadcast/eagle-eye footage and degraded sharply on Trace's variable-scale pan
— the TZ README explicitly warns that "the project should only be used with
eagle-eye 1080p or higher match videos."

Runtime per minute of input video (Colab T4, indicative):
- Roboflow: streaming (no full-clip RAM cost)
- ATFA: ~1.0–1.5× clip-length, **loads all frames into memory** before tracking
  (problematic for full-game footage)
- TZ: similar in-memory model to ATFA, irrelevant given detection quality

## Winner

**Chosen backend: `roboflow/sports`**

**Why:**

1. **Only candidate with per-frame pitch keypoint detection.** Given Trace's
   virtual-PTZ pan, every frame needs a fresh homography computed from
   detected pitch landmarks. ATFA only supports manual 4-corner perspective
   (which we dropped); TZ has no pitch model at all. This single axis is
   architecturally disqualifying for ATFA and TZ in our setting.
2. **Streaming-friendly.** Processes frames one at a time. ATFA reads the
   entire clip into a `video_frames` list before tracking — fine for 2-minute
   bake-off clips, bad for full-game footage and untenable for the planned
   season-wide processing.
3. **Best-supported.** Active maintenance (most recent commits), clear
   reference implementation in `examples/soccer/main.py`, public model weights,
   built-in SoccerPitchConfiguration for the radar overlay.

ATFA's tracker wrapper produced visibly cleaner IDs than Roboflow's, which we
will revisit as a v1 enhancement: keep Roboflow as the detection + pitch
backend, but consider grafting ATFA-style ByteTrack tuning into our adapter.

## Identified weakest stage (informs Phase 2)

**Weakest: ball detection** across all three candidates. Roboflow and ATFA
share the same model (≈1–2 / 5); TZ scores 0.

**Severity check:** the spec's Phase 2 trigger condition is "ball detection
≤ 2 / 5 in *all three* candidate repos." This condition is met. **Phase 2
ball-detector fine-tune is locked in** as the next phase after the Roboflow
adapter (Task 17) lands.

Empirical evidence: the ball was visible to the human eye in most frames of
the clip but detected with high confidence in only an estimated 40–60% of
frames by the Roboflow/ATFA shared model. Sustained-detection windows broke
apart during fast camera pans and when the ball was small at the far
touchline.

## Notes for downstream phases (Plan B)

Observations during the bake-off that should shape Plan B Phase 3+ work:

1. **Background games visible in the clip.** Trace footage from tournaments
   includes detectable players on adjacent fields. The model dutifully boxes
   them — pitch-boundary filtering (drop detections that project outside the
   focal pitch via the per-frame homography) is a **v1 requirement**, not a
   v1+ enhancement. Captured in
   `~/.claude/projects/-Users-patrickreed/memory/project_soccer_vision_tournament_context.md`.

2. **Per-frame team prediction is too noisy.** Players flicker between
   own/opp from frame to frame even when the underlying clustering is correct
   on average. Fix: after tracking, take each track's modal team assignment
   over its lifetime and apply uniformly. Trivial pandas operation in the
   `pitch/` or `phase/` layer. Add as a v1 step in Plan B.

3. **Near-camera players missed.** The roboflow-trained model expects players
   at broadcast-camera scale. Trace's PTZ produces dramatic scale variance as
   the camera pans, and near-camera players (large pixel footprint) get missed
   more than far-touchline ones. This is the secondary fine-tune target if
   ball-detection fine-tuning still leaves us short.

4. **Goalkeeper handling is uneven.** Roboflow's model has a `goalkeeper`
   class but the team classifier mis-clusters GKs (different kit color from
   field players). ATFA folds GKs into players entirely. The spec's
   `resolve_goalkeepers_team_id` (proximity-to-team-centroid heuristic in
   roboflow/sports/main.py) is worth porting in.

5. **TZ team-classifier was brittle.** TZ's `assign_team_color` single-frame
   seeding clustered intra-team variance instead of inter-team. A
   pooled-across-the-clip KMeans fit worked. Both Roboflow's SigLIP+UMAP and
   ATFA's KMeans-on-pixels seeded on more representative data and didn't have
   this failure mode. Worth ensuring our adapter pools across at least 60–120
   sampled frames when fitting the team classifier.

## Deliverables

- `examples/bakeoff_roboflow.ipynb`, `bakeoff_atfa.ipynb`, `bakeoff_tz.ipynb`
  — Colab runners (committed)
- `examples/colab_bakeoff.ipynb` — synthesis notebook (committed; not run for
  this scoring round, since the verdict was decisive from individual review)
- `data/bakeoff_outputs/{roboflow,atfa,tz}/{trajectories.parquet,annotated.mp4}`
  — gitignored; live on Patrick's laptop
- `docs/superpowers/bakeoff-results.md` — this file
- `packages/soccer-vision/src/soccer_vision/tracking/roboflow.py` — adapter
  for the chosen backend (Task 17, dispatching next)
