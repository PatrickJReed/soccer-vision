# Team-Shape Metrics (Phase 4, v1) — Design

**Date:** 2026-07-01
**Status:** Approved (brainstorm complete) — ready for implementation plan
**Sub-project:** SP5 (Phase 4 metrics product), first vertical slice.

## Goal

Build the first Phase-4 metric end-to-end: **team shape / compactness** for both teams plus
inter-team relations, computed from the pipeline's pitch-space trajectories, summarized per
phase, and rendered as plots. This is a thin vertical slice (one metric family, compute →
parquet → summary → plots) to establish the metrics module and let downstream need drive
precision — not the full §6 taxonomy.

## Background

`metrics/` is empty (0 % built). Calibration is now trustworthy (physical per-frame engine,
validated), which unblocks metrics. The pipeline already produces pitch-space player
positions per frame, so shape is directly computable.

## Inputs (real pipeline schema)

- **`trajectories.parquet`**: per-detection rows. Used columns: `frame` (int), `track_id`,
  `class` ∈ {`ball`, `player`, `goalkeeper`, `referee`}, `team` ∈ {`own`, `opp`, `ref`,
  `unknown`}, `conf`, **`x_pitch`, `y_pitch`** = canonical pitch coords in **[0,1]**
  (`x` = touchline-axis / width fraction, `y` = goal-to-goal / length fraction; small ±0.05
  spill at boundaries is normal).
- **`phases.parquet`**: per-frame rows: `frame`, `phase` ∈ {unknown, loose_ball, transition,
  attack, contested, defend_high, defend_low, build}, `possession_state`,
  `ball_x_pitch`/`ball_y_pitch`, **`homography_source`** (`manual` vs `none`), `homography_conf`.

Constants: `WIDTH_M = 45.7`, `LENGTH_M = 68.5` (from `calib.field_model`). Convert
`x_m = x_pitch * WIDTH_M`, `y_m = y_pitch * LENGTH_M`.

## Filtering / frame inclusion

- **Rows:** keep `team ∈ {own, opp}` and `class == "player"` (exclude ball, referee; and the
  goalkeeper unless `drop_goalkeeper=False`, in which case also keep `class == "goalkeeper"`).
- **Frames:** only frames whose `phases.homography_source != "none"` (a trusted homography
  exists). For each such frame, a team contributes a shape row only if it has
  **≥ `min_players` (default 4)** kept players; otherwise that (frame, team) is omitted.
- Report **coverage**: fraction of homography frames with a valid shape row per team.

## Computed metrics

**Per (frame, team)** — all in metres:
- `centroid_x_m`, `centroid_y_m` = mean of the team's player `x_m`, `y_m`.
- `width_m` = `max(x_m) − min(x_m)` (touchline-axis extent).
- `depth_m` = `max(y_m) − min(y_m)` (goal-axis extent).
- `compactness_m` = mean Euclidean distance of players to the centroid (lower = tighter).
- `n_players`.

**Per frame (inter-team)** — only frames where BOTH teams have a shape row:
- `centroid_separation_m` = distance between own and opp centroids.
- `vertical_offset_m` = `own.centroid_y_m − opp.centroid_y_m` (which block is higher up the
  length axis; sign interpreted against the fixed pitch frame / after halftime reflection).
- `width_diff_m` = `own.width_m − opp.width_m`.

**Per phase (summary):** group `per_frame` by (`phase`, `team`); report the mean of
`width_m`, `depth_m`, `compactness_m`, `centroid_y_m` and `n_frames`.

## Attacking direction / halftime

`y` is a fixed goal-to-goal axis. v1 assumes a single attacking direction (one clip). Optional
`halftime_frame`: for frames `>= halftime_frame`, reflect `y_m → LENGTH_M − y_m` before
computing y-dependent quantities (`centroid_y_m`, `depth_m` is reflection-invariant,
`vertical_offset_m`), so attacking direction is consistent across halves. Width is unaffected.
Default `None` (no reflection).

## Outputs

- **`shape.parquet`** — the `per_frame` table (frame, team, the five metrics, n_players),
  plus the `inter_team` columns joined by frame (wide) OR written as a sibling
  `shape_interteam.parquet` (implementer's choice; keep `per_frame` clean).
- **Per-phase summary** — returned as a DataFrame and printed by the CLI.
- **Plots** (PNGs; rendered for the user to interpret, numbers reported as facts):
  1. `shape_timeseries.png` — `width_m`, `depth_m`, `compactness_m` vs time (`t_seconds`),
     both teams (own solid / opp dashed), with phase bands lightly shaded.
  2. `avg_positions.png` — a pitch outline with each team's average player-position density
     (2D histogram/heat) or mean positions.
  3. `separation.png` — `centroid_separation_m` and `vertical_offset_m` vs time.

## API / interfaces

`packages/soccer-vision/src/soccer_vision/metrics/shape.py`:

```python
@dataclass(frozen=True)
class ShapeResult:
    per_frame: pd.DataFrame    # frame, t_seconds, team, centroid_x_m, centroid_y_m,
                               # width_m, depth_m, compactness_m, n_players
    inter_team: pd.DataFrame   # frame, t_seconds, centroid_separation_m, vertical_offset_m, width_diff_m
    per_phase: pd.DataFrame    # phase, team, width_m, depth_m, compactness_m, centroid_y_m, n_frames

def compute_shape(
    trajectories: pd.DataFrame,
    phases: pd.DataFrame,
    *,
    min_players: int = 4,
    drop_goalkeeper: bool = True,
    halftime_frame: int | None = None,
) -> ShapeResult: ...

def plot_shape(result: ShapeResult, trajectories: pd.DataFrame, out_dir: Path) -> list[Path]: ...
```

CLI `python -m soccer_vision.metrics.shape --session <dir> [--out <dir>] [--min-players N]
[--keep-goalkeeper] [--halftime-frame F]`: reads `<session>/trajectories.parquet` +
`phases.parquet`, writes `shape.parquet` + plots to `--out` (default `<session>`), prints the
per-phase summary + coverage. `compute_shape` is pure (no I/O); the CLI and `plot_shape` are
the only I/O.

## File structure

- Create `metrics/shape.py` (compute + dataclass + plot + CLI `main()`).
- Create `metrics/__main__.py` delegating to `shape.main` (so `python -m soccer_vision.metrics`
  works) OR run via `python -m soccer_vision.metrics.shape` (implementer picks; document it).
- Create `tests/test_metrics_shape.py`.
- `matplotlib` (3.10 present) — confirm it's a declared dependency; add to `pyproject` if not.

## Error handling / edge cases

- Empty/short frames (< min_players) → omitted, not errored; coverage reflects it.
- A frame with only one team meeting min_players → that team gets a shape row; no inter-team row.
- `x_pitch`/`y_pitch` NaN or far out of [−0.2, 1.2] → drop those detections (bad projection).
- No valid frames at all → empty result + a clear message (don't crash).

## Testing

- **Exactness:** synthetic trajectories with players at known pitch positions → assert
  `centroid`, `width`, `depth`, `compactness` equal hand-computed values (metres).
- **Inter-team:** two teams at known centroids → `centroid_separation_m`, `vertical_offset_m`.
- **Filtering:** referee/ball/unknown rows ignored; goalkeeper excluded by default, included with
  `drop_goalkeeper=False`; frames with `homography_source == none` excluded; `< min_players`
  omitted.
- **Per-phase:** two phases with different synthetic shapes → correct grouped means.
- **Halftime:** `halftime_frame` reflects y (centroid_y flips about LENGTH_M/2; depth unchanged).
- **Plots:** `plot_shape` writes the expected PNG files without error on a small fixture.

## Acceptance criteria

- `compute_shape` on the real session (`~/sv-labeler/analysis3/`) returns non-empty per_frame,
  inter_team, per_phase with plausible metres (widths ~10–40 m, depths ~10–50 m) and sensible
  coverage; CLI writes `shape.parquet` + 3 plots + prints the summary.
- Full suite + `uv run mypy` (src+tests) + `ruff` clean.

## Out of scope (future slices)

Space control, defensive lines, gaps, ball-relative, dynamics/velocity, zones, youth-specific
metrics (§6 taxonomy). Per-player role/line assignment. GK auto-detection beyond `class`.
