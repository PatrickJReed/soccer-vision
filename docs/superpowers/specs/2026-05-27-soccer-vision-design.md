---
name: soccer-vision-design
description: Design spec for soccer-vision — team-level positional analytics for 9v9 youth soccer from Trace camera footage. Notebook-driven Python toolkit built on an existing detection backend chosen via structured bake-off, with metrics expressed in pitch-relative units to handle variable field sizes.
status: approved
date: 2026-05-27
---

# soccer-vision — Design Specification

## 1. Goals & Scope

### 1.1 What this is

A notebook-driven Python toolkit that ingests Trace camera footage of 9v9 youth
soccer games and emits **team-level positional analytics**: shape, space and
pitch ownership, gaps, ball-relative positioning, dynamics, zonal occupation,
and youth-specific metrics like a swarm index. The output is a Python package
plus Colab-runnable notebooks that produce parquet data tables, static plots, a
per-game HTML report, video overlays, and season-level analyses.

### 1.2 What this is not (v1 scope guards)

- **No passing or event detection.** Parked. The frame-level possession proxy
  in §6 is sufficient for phase splits.
- **No player-specific analysis.** All metrics aggregate at the team level. No
  player re-identification, jersey-number recognition, or individual stat
  tracking.
- **No tactical/coaching prescriptions.** The toolkit reports metrics; humans
  interpret them.
- **No realtime processing.** Trace footage is post-game.
- **No web dashboard.** Notebook-driven only.

### 1.3 Target use case

Patrick Reed's 9v9 youth team, full-season analysis. Formations encountered:
3-2-3, 3-3-2, 2-3-3. ~10+ games of Trace footage available. Compute: Colab Pro
with GPU. The system must work across different fields whose absolute
dimensions vary game-to-game.

### 1.4 Selected approach (Approach C)

Structured bake-off of three existing repos on a canonical 2-minute Trace clip;
commit to one as the detection backend; wrap behind a `TrackingBackend` adapter
protocol; build the analytics layer as the core product; fine-tune only the
weakest stage identified by the bake-off (almost certainly ball detection).
Alternatives considered: adopt one repo wholesale (too entangled with upstream
choices); composite pipeline from best repo per stage (too much glue, brittle).

## 2. Repository Layout & Data Flow

### 2.1 Layout

```
soccer-vision/                          # uv workspace
├── packages/
│   └── soccer-vision/
│       ├── src/soccer_vision/
│       │   ├── tracking/               # adapter layer over upstream repo(s)
│       │   │   ├── base.py             # TrackingBackend protocol
│       │   │   └── <winner>.py         # concrete adapter from bake-off
│       │   ├── pitch/                  # homography + pitch coords
│       │   │   ├── spec.py             # PitchSpec dataclass
│       │   │   └── homography.py
│       │   ├── phase/                  # possession & phase splitter
│       │   ├── metrics/
│       │   │   ├── shape.py
│       │   │   ├── space.py
│       │   │   ├── gaps.py
│       │   │   ├── ball_relative.py
│       │   │   ├── dynamics.py
│       │   │   ├── zones.py
│       │   │   └── youth.py
│       │   ├── io/                     # frame iter + parquet serialization
│       │   └── viz/                    # mplsoccer-based plotting
│       └── pyproject.toml
├── examples/
│   ├── colab_bakeoff.ipynb
│   ├── colab_pipeline_demo.ipynb
│   └── colab_season_analysis.ipynb
├── data/
│   ├── bakeoff_clip.mp4                # 2-min canonical clip
│   ├── labeled/                        # 400-500 frame ball dataset
│   └── games/                          # gitignored; pulled from Drive in Colab
├── docs/
│   └── superpowers/
│       ├── specs/                      # this doc lives here
│       └── bakeoff-results.md          # written at end of Phase 1
└── pyproject.toml
```

### 2.2 Data flow per game

```
input.mp4
   │
   ▼
TrackingBackend.process()              ← adapter, swappable
   │
   ▼
trajectories.parquet                   ← (frame, track_id, x_px, y_px, team, conf)
   │
   ├─► PitchMapper                     ← homography to normalized pitch coords
   │
   ▼
pitch_trajectories.parquet             ← (frame, track_id, x_pitch, y_pitch, team)
   │
   ▼
PhaseSplitter.label()                  ← adds possession_state + phase
   │
   ▼
phased.parquet
   │
   ▼
metrics/*.py (parallel)
   │
   ▼
game_<id>/
   ├── shape.parquet
   ├── space.parquet
   ├── gaps.parquet
   ├── ball_relative.parquet
   ├── dynamics.parquet
   ├── zones.parquet
   ├── youth.parquet
   ├── report.html
   ├── report.pdf
   └── viz/*.png + overlays/*.mp4
```

### 2.3 Key design choices

- **Parquet between every stage.** Avoids re-tracking when only downstream
  metrics change.
- **`TrackingBackend` protocol.** Single `process(video_path) -> trajectories
  DataFrame` method with stable column contract. Approach C's swap-later
  promise lives here.
- **Pitch coordinates are canonical.** All metrics consume normalized
  `(x_pitch, y_pitch) ∈ [0, 1]²`. Pixels never leave the tracking module.
- **Each game produces a directory of parquets.** Season analysis is
  `pd.concat` over directories.

## 3. Pitch & Units

### 3.1 Pitch-relative units (no absolute meters)

Field dimensions vary game-to-game at 9v9 (different venues, different
recommended ranges of 64–73 m × 41–50 m). Rather than measure each field, every
frame's homography maps visible pitch landmarks to a normalized `[0, 1] × [0, 1]`
canonical pitch frame, and all metrics are expressed in **pitch-units**
(fractions of pitch length). 1 pitch-unit = 1 × pitch length.

This makes metrics directly comparable across games regardless of field size,
and eliminates a brittle calibration step.

Optionally, a user can supply `pitch_length_m` per game in a one-line config
and reports will display secondary axes in approximate meters. Default is
pitch-units everywhere.

### 3.1a Per-frame homography (Trace virtual-PTZ implication)

Trace cameras synthesize a follow-the-ball pan from a wider physical capture,
so no single frame contains all four pitch corners. The pitch keypoint
detection runs **per frame**, producing a fresh homography `H_t` for each
frame. Player positions transform through `H_t` into the canonical pitch frame.
Over the clip, every player's full trajectory accumulates in canonical pitch
coordinates even though no individual frame ever shows the whole pitch — we
never construct a stitched panoramic image. Robustness measures applied to
`H_t`:

- **Temporal smoothing** — exponential moving average on the homography
  matrix elements (or Kalman) to suppress per-frame detection jitter
- **Insufficient-keypoint fallback** — when fewer than 4 well-distributed
  landmarks are detected, `H_t` is interpolated from the nearest good frames
- **Validity check** — reject `H_t` if it projects any tracked player more
  than ~10% off-pitch; substitute the interpolated value

### 3.2 `PitchSpec` (dimensionless)

```python
@dataclass(frozen=True)
class PitchSpec:
    aspect_ratio: float = 1.5
    n_outfield_per_team: int = 8
    penalty_box_length_frac: float = 0.187
    penalty_box_width_frac: float = 0.720
    center_circle_radius_frac: float = 0.106
    coverage_cell_frac: float = 0.011

PitchSpec.STANDARD_9V9 = PitchSpec()
PitchSpec.FIFA_11V11   = PitchSpec(aspect_ratio=1.54, n_outfield_per_team=10, ...)
```

Every metric module takes `pitch_spec: PitchSpec` as a parameter.

### 3.3 Distance thresholds (pitch-units)

| Threshold | Value | Equivalent at 68.5 m pitch |
|---|---|---|
| Contested-possession margin | 0.022 | ~1.5 m |
| Clump radius | 0.044 | ~3 m |
| Players-near-ball (mid) | 0.073 | ~5 m |
| Players-near-ball (outer) | 0.146 | ~10 m |
| Pressure-on-carrier radius | 0.044 | ~3 m |
| Gap (horizontal in-line) | 0.146 | ~10 m |
| Gap (between lines, vertical) | 0.175 | ~12 m |
| Pitch coverage cell | 0.011 | ~0.75 m |
| Pitch grid for zones | 5 × 3 | ~14 m × 14 m cells |
| Vertical channels | 3 (L/C/R) | half-spaces dropped at 9v9 scale |

## 4. Tracking Backend & Bake-Off

### 4.1 `TrackingBackend` protocol

```python
class TrackingBackend(Protocol):
    name: str
    version: str

    def process(self, video_path: Path) -> pd.DataFrame:
        """Return a DataFrame with columns:
        frame, t_seconds, track_id, x_px, y_px,
        bbox_x1, bbox_y1, bbox_x2, bbox_y2,
        class, team, conf
        Class is one of {player, goalkeeper, referee, ball}.
        Team is one of {own, opp, ref, unknown}.
        """
```

### 4.2 Bake-off candidates

| Candidate | Why it's in |
|---|---|
| `roboflow/sports` | Most actively maintained; SigLIP team-ID; pitch keypoints |
| `abdullahtarek/football_analysis` | Simpler, color-based; optical-flow camera comp; pedagogical |
| `AbdelrahmanAtef01/Tactic_Zone` | Feature-complete (events, GK class); detector-only assessment |

Narya excluded (TF1.x, dated). SoccerNet repos are research baselines, not
end-to-end candidates.

### 4.3 Canonical bake-off clip

`data/bakeoff_clip.mp4` — 2 minutes from the available Trace footage, structured as:
- ~60 s build-up play (slow, structured, all players visible)
- ~60 s transition + counterattack (fast motion, occlusion)

Committed to repo. Same clip fed to all three candidates. **No manual
homography reference.** Trace's virtual-PTZ pan means no single frame contains
all four pitch corners, so a one-shot manual annotation isn't viable. Each
candidate's per-frame pitch keypoint detector is exercised on the clip and the
resulting top-down radar view is judged by inspection. If a quantitative
ground-truth metric becomes necessary later, sparse multi-frame keypoint
annotation can be added as a v1+ enhancement.

### 4.4 Scoring rubric

Five axes scored 1–5 by inspection of side-by-side panels in
`colab_bakeoff.ipynb`:

| Axis | "5" looks like | "1" looks like |
|---|---|---|
| Player detection recall | All 16 outfield + 2 GK detected nearly every frame | Players missed at far touchline, GK absent |
| Ball detection rate | Ball detected >80% of frames, smooth trajectory | Ball detected <30%, jittery |
| Track stability | IDs persist through occlusions | IDs flip frequently |
| Team classification | All players cleanly assigned to one of two teams | Players flipping teams frame-to-frame |
| Homography fidelity | Per-frame top-down radar maintains plausible player spacing through camera pans; few visible warp artefacts | Players warp, positions implausible, or homography unstable across frames |
| **9v9 handling (new)** | Degrades gracefully on 9-per-team and youth pitch | Hard-coded 11v11 assumptions break |

Plus a non-scored runtime / Colab GPU cost measurement.

### 4.5 Failure modes the bake-off should surface

The point is to discover these before metrics work begins:

1. Ball goes missing for long stretches → triggers Phase 2 ball-detector fine-tune
2. Homography breaks at field edges → upgrade or accept truncation
3. Team-ID poor on multicolored or similar kits → switch SigLIP↔KMeans, hand-seed per game
4. Track IDs unusable beyond ~30s → consider grafting on sn-reid weights

### 4.6 Deliverables (end of Phase 1)

- `examples/colab_bakeoff.ipynb` with reproducible side-by-side runs
- `data/bakeoff_clip.mp4` canonical clip
- `docs/superpowers/bakeoff-results.md` with scoring matrix and written rationale
- First concrete `TrackingBackend` adapter at
  `packages/soccer-vision/src/soccer_vision/tracking/<winner>.py`

## 5. Fine-Tuning Plan (Phase 2, Conditional)

### 5.1 Trigger criteria

Phase 2 starts only if **all three** bake-off candidates score ≤2/5 on ball
detection. If even one scores ≥3/5, Phase 2 is deferred to backlog and we move
directly to metrics work.

### 5.2 Scope (priority order)

1. **Ball detector fine-tune** — primary target. Reuses JuggleNet workflow.
2. Player detector — only if recall < 90%. Likely unnecessary.
3. Team-color templates per game — 5-min manual seed when KMeans drifts.
   Trivial work, not a model fine-tune.
4. Pitch keypoint model — fallback is manual 4-corner annotation per game.

### 5.3 Ball detector workflow

| Stage | Detail |
|---|---|
| Data | ~500 frames from 5 games (100/game), sampled to cover ball states (rolling, in-flight, occluded, behind net, near touchline) |
| Labeling | Roboflow free tier, single class `ball`, tight boxes |
| Base model | YOLOv8n or YOLOv8s, 1280×1280 input |
| Augmentations | Mosaic, mixup, HSV jitter, motion blur, ±50 % scale |
| Training | ~100 epochs on Colab T4/L4 (~30–60 min/cycle) |
| Holdout | 50 frames from an unseen game |
| Acceptance gate | ≥75 % sustained ball detection on bake-off clip (vs ~30–50 % pretrained) |
| Integration | `models/ball_yolov8.pt` consumed by tracking adapter |

Total effort estimate: ~one weekend.

## 6. Metrics

### 6.1 Cross-cutting: possession proxy (`phase/possession.py`)

Five-state model that explicitly accommodates youth ball-clumping:

```python
states: Literal["own", "opp", "contested", "loose_ball", "unknown"]
```

Logic:
- `unknown` when ball is missing.
- `loose_ball` when no player within 0.073 pitch-units of ball.
- `contested` when:
  - `|nearest_own_dist - nearest_opp_dist| < 0.022`, OR
  - both teams have ≥1 player within 0.044 of ball with the two counts differing by ≤1.
- `own` / `opp` otherwise.

Smoothed over a 30-frame (1-second) window using mode, **preserving
`contested`** rather than masking it.

`phase` extends this:
- `attack` = own + ball in opp 2/3
- `build` = own + ball in own 1/3
- `defend_low` = opp + ball in own 2/3
- `defend_high` = opp + ball in opp 1/3
- `transition` = 5-second window after each possession change
- `contested`, `loose_ball` retain their own phase labels

Metrics dependent on phase get NaN for `contested` / `loose_ball` / `unknown`
frames — these don't pollute "in possession" statistics.

### 6.2 Shape (`metrics/shape.py`)

Per (frame, team):

| Metric | Formula |
|---|---|
| Centroid | mean(x, y) of outfield players |
| Length | percentile-trimmed `max(y) − min(y)` |
| Width | percentile-trimmed `max(x) − min(x)` |
| Area (hull) | `scipy.spatial.ConvexHull(positions).volume` |
| Area (α-shape) | `alphashape.alphashape(positions, α=0.7).area` |
| Convexity score | `area_hull / area_alpha` |
| Stretch index | `mean(‖player − centroid‖)` (Bourbousson) |
| Block height | mean y of "back line" — see below |
| L:W ratio | `length / width` |

**Formation-agnostic block height:** exclude the goalkeeper first
(lowest-y on own team, highest-y on opp team), then k-means the remaining 8
outfield players' y-coordinates into K=3 clusters (defenders / mids /
forwards), use the back cluster's mean y. Robust to 3-2-3, 3-3-2, and 2-3-3
within a game since all three formations are 3-line shapes. If a future
formation has only 2 lines, K=3 will produce one near-empty cluster and the
metric remains well-defined via the back cluster's mean.

### 6.3 Space / pitch control (`metrics/space.py`)

**Basic Voronoi (v1):**
- `scipy.spatial.Voronoi` over all 16 outfield positions
- Clip cells to pitch bounding box via `shapely`
- Per-frame, per-team: total area, plus area in defensive / middle / attacking
  thirds

**Weighted (motion-aware) Voronoi (v1+):**
- Spearman pitch-control model: `P_i(p) = 1 / (1 + exp((t_i(p) - t_min(p))/σ))`
- Constant-acceleration motion model from 5-frame smoothed velocities
- σ = 0.45 s

Aggregated per game: `space_heatmap.npy` on a normalized pitch grid.

### 6.4 Gaps (`metrics/gaps.py`)

Three event types:

| Type | Detection | Event signature |
|---|---|---|
| Largest opponent cell | Biggest opp Voronoi cell in our def third > rolling-90th-percentile | `(frame, "opp_cell", x, y, size)` |
| Line gap (horizontal) | Within each y-cluster, gap between consecutive own players in x > 0.146 pitch-units | `(frame, "line_h", x_mid, y_band, size)` |
| Between-line gap | y-distance between cluster means > 0.175 pitch-units | `(frame, "between_lines", y_top, y_bot, size)` |

Each event records `(t_seconds, video_frame_no)` for clip extraction.

### 6.5 Ball-relative (`metrics/ball_relative.py`)

Per (frame, team):

| Metric | Formula |
|---|---|
| `mean_dist_to_ball` | mean over outfield players of distance to ball |
| `players_within_5m` (0.073 pu) | count where dist ≤ 0.073 |
| `players_within_10m` (0.146 pu) | count where dist ≤ 0.146 |
| `pressure_on_carrier` | opp players within 0.044 of current ball-carrier |
| `cover_count` | (own only) players between ball and own goal |
| `numerical_balance_5m` | (own within 0.073) − (opp within 0.073) |
| `numerical_balance_10m` | same at 0.146 |

NaN when ball is missing.

### 6.6 Dynamics (`metrics/dynamics.py`)

| Metric | Window | Formula |
|---|---|---|
| `area_rolling_var` | 30 s | rolling variance of hull area |
| `length_rolling_var` | 30 s | rolling variance of length |
| `centroid_velocity` | per-frame | Savitzky-Golay-smoothed derivative of centroid y |
| `centroid_coupling` | 60 s rolling | Pearson r between own and opp centroid y |
| `reorg_time_post_turnover` | event | seconds until area returns within 10 % of rolling-60s mean after each possession change |

Reorganization-time events go to `dynamics_events.parquet`.

### 6.7 Zones (`metrics/zones.py`)

- **Pitch grid:** 5 longitudinal × 3 transverse cells (~14 m × 14 m equivalent)
- **Channels:** 3 vertical (left / central / right) — half-spaces dropped at
  9v9 scale
- Per frame, per team: `players_in_cell[r][c]`, `players_in_channel[ch]`
- `field_tilt = (players in opp half) / (total players)`
- `verticality = mean(y of own team) / pitch length`

Aggregated per game: time-in-zone fractions, channel occupation percentages.

### 6.8 Youth (`metrics/youth.py`)

| Metric | Formula |
|---|---|
| `swarm_index` | mean over **all 16 outfield players** of distance to ball |
| `swarm_index_own` | own team only |
| `pitch_coverage_cum[t]` | cumulative count of 0.011-pitch-unit grid cells any own player has visited by time t |
| `shape_cluster_id` | per-frame assignment of `(length, width, area, stretch)` to K=5 KMeans clusters learned within the game |
| `shape_diversity` | Shannon entropy of the per-game cluster distribution |
| `pct_contested` | rolling and per-game fraction of frames in `contested` state |

Swarm index and pct_contested are headline youth metrics. Both appear on the
per-game report.

### 6.9 Baselines

Absolute pitch-unit values are not calibrated to age-group norms (literature
values are from small-sided games on small pitches and don't transfer cleanly
to 9v9). Approach:

1. **Per-team rolling baseline:** for each metric, the median over the
   team's first 3 *complete-game* processings (games with ≥85 % of frames
   yielding usable tracking; games failing this threshold are flagged in the
   season summary and excluded from baseline computation).
2. **Trend over season** is the primary signal — e.g., "swarm index decreasing
   game over game" is the actionable read.
3. Reports include a clear caveat that absolute levels are uncalibrated;
   trends and within-game phase splits are the reliable signal.

## 7. Outputs

### 7.1 Tabular (per game)

`shape.parquet`, `space.parquet`, `gaps.parquet`, `ball_relative.parquet`,
`dynamics.parquet`, `zones.parquet`, `youth.parquet`, `phased.parquet`,
`game_summary.json` (means/medians/quantiles per metric, split by phase).

### 7.2 Static plots (core, v1)

- Average shape with confidence ellipses, split by phase
- Convex hull overlay at 4 representative moments
- Pitch control heatmap (in-poss vs out-of-poss panels)
- Gap location heatmap (KDE, sized by magnitude)
- Zonal occupation 5 × 3 grid
- Channel occupation 3-bar chart
- Pitch coverage raster (cumulative)
- Swarm-index time series with rolling band
- Centroid-coupling timeline with correlation annotation
- Shape time series 4-panel grid (length, width, area, stretch)

Plus per-metric time series with phase shading, phase-split distributions, and
reorganization-time histogram.

Visual style: `mplsoccer` pitch backgrounds, team colors from KMeans output.

### 7.3 Video overlays (core, v1)

1. Annotated tracking (boxes + team colors + IDs + trails)
2. Top-down radar minimap composited on original video
3. Voronoi overlay via inverse homography
4. Shape envelope (translucent convex hull)
5. Gap-event clips (auto-cut 10-s windows around top-N events)

Stretch: coach-cut composite combining all overlays + metric ticker.

### 7.4 Per-game HTML report (core)

Single page per game (`game_<id>/report.html`), self-contained with embedded
images and no external assets:

- Header: game ID, date, opponent, result, possession split, **time in
  contested play**
- 6-panel grid: avg shape, pitch-control heatmap, gap heatmap, swarm-index
  curve, channel occupation, centroid coupling
- Footer table: top-3 largest-gap moments (with thumbnails), top-3
  longest-stretched moments, reorganization-time stats

PDF generation is deferred to v1+. Browsers can print-to-PDF for now.

### 7.5 Season-level outputs

In `colab_season_analysis.ipynb`, reading all `game_*/` directories:

- Per-game summary matrix (heatmap of z-scores vs season mean)
- Season trend lines (swarm index, pitch coverage, area, stretch over game number)
- Phase-conditional comparison (in-poss vs out-of-poss per game)
- Score-state breakdown (leading / tied / trailing)
- Opponent comparison (when metadata available)
- Game-vs-game radar charts

## 8. Testing & Validation

### 8.1 Unit tests on metrics

Synthetic fixtures with known geometric properties exercise every metric
module. Coverage target: ≥80% per module, ≥70% overall (gate). Fixtures
include line teams, two-line formations, clumped teams, contested-ball
configurations, and explicit phase-transition sequences.

### 8.2 End-to-end regression on bake-off clip

`tests/test_pipeline_e2e.py` runs the full pipeline on `bakeoff_clip.mp4` and
snapshots summary statistics (`mean_area_own`, `pct_contested`, etc.) to a
golden file with 5% drift tolerance. Catches unintentional changes to our
metrics code or pipeline glue. Upstream model weights are pinned, so model
drift is not in scope; intentional upstream version bumps regenerate the
golden file.

### 8.3 Visual validation

Per-run `validation.html` with 8 stratified-random frames + all overlays.
Manual eyeball sweep (~60 s) catches dramatic failures.

### 8.4 CI

GitHub Actions: ruff, mypy strict on the metrics package, pytest with
coverage. Heavy pipeline runs happen in Colab manually; CI only runs the unit
tests + e2e regression on the bake-off clip.

## 9. Phasing

| Phase | Days | Deliverable |
|---|---|---|
| 0 — Scaffolding | 1–2 | Empty package, CI green, uv workspace |
| 1 — Bake-off | 2–3 | Chosen backend, scoring matrix, first adapter |
| 2 — Fine-tune (conditional) | 2 | `models/ball_yolov8.pt` if triggered |
| 3 — Pitch + phase | 2 | Homography + 5-state possession |
| 4 — Shape + space | 3 | `shape.parquet`, `space.parquet` |
| 5 — Gaps + ball-relative | 2 | `gaps.parquet`, `ball_relative.parquet` |
| 6 — Dynamics + zones + youth | 2 | All metric parquets |
| 7 — Viz + report | 3 | Static plots + HTML report |
| 8 — Video overlays | 3–4 | Five overlay outputs |
| 9 — Season analysis | 2 | Season notebook + per-team baselines |

**~25 focused workdays to v1.** Phase-gate: each phase merges only with green
CI and produced deliverable artifact.

A second `TrackingBackend` adapter (real or mock) is added at the end of
Phase 4 to verify the swap-later promise of Approach C.

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| All three bake-off candidates fail badly on Trace's wide fixed camera | Phase 2 fine-tuning addresses the dominant failure; deeper failure triggers an unplanned Phase 1.5 (component recombination, sn-reid) |
| 9v9 pitch keypoint detection unreliable on youth fields | Smooth across frames; interpolate from neighbors when keypoints sparse; sparse multi-frame manual keypoint annotation as a v1+ ground-truth reference if quantitative evaluation is needed |
| Team color KMeans fails on similar kits | Per-game manual color seeding documented in the pipeline notebook |
| Ball detection so bad that ball-relative metrics are useless | All ball-dependent metrics report NaN gracefully; PathCRF as a v1+ ball-free fallback if needed |
| Literature-derived thresholds don't transfer to 9v9 | Pitch-relative units sidestep absolute scale; team rolling baseline replaces literature norms |
| Trace footage quality varies across the season | Each game gets its own validation.html sweep; outlier games flagged in the season summary |

## 11. Open Questions (deferred to v1+)

- Whether to bolt on a passing-event layer using SoccerNet ball-action spotting
  or PathCRF
- Whether weighted (Spearman) pitch control beats basic Voronoi enough to
  justify the motion model
- Multi-team / age-group benchmarking once more datasets exist
- The coach-cut composite video overlay

## 12. Sources

Key academic and code references that informed this design:

- [Folgado et al. — Length, width, centroid distance in youth football](https://onlinelibrary.wiley.com/doi/10.1080/17461391.2012.730060)
- [Aguiar et al. — Tactical Analysis Across Age Groups (4v4 SSG)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7084353/)
- [The older, the wider — Folgado et al.](https://www.sciencedirect.com/science/article/abs/pii/S0167945715000238)
- [Bourbousson — Stretch index original formulation](https://pubmed.ncbi.nlm.nih.gov/20183549/)
- [Spearman — Pitch control model](https://www.sfu.ca/~tswartz/papers/pitch_control.pdf)
- [Fernandez & Bornn — Wide Open Spaces](https://www.lukebornn.com/papers/fernandez_ssac_2018.pdf)
- [SoccerCPD (Kim et al., KDD 2022)](https://github.com/hyunsungkim-ds/soccercpd)
- [roboflow/sports — soccer example](https://github.com/roboflow/sports/tree/main/examples/soccer)
- [abdullahtarek/football_analysis](https://github.com/abdullahtarek/football_analysis)
- [AbdelrahmanAtef01/Tactic_Zone](https://github.com/AbdelrahmanAtef01/Tactic_Zone)
- [US Soccer 9v9 field dimensions](https://soccer-fields.com/soccer-field-dimensions-by-age-format/)
