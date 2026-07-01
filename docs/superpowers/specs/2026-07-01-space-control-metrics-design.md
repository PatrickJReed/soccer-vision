# Space-Control Metrics (Phase 4, v1) — Design

**Date:** 2026-07-01
**Status:** Approved — ready to build
**Sub-project:** SP5 (Phase 4 metrics), second slice (after team-shape).

## Goal
Compute pitch-control ("space control") from pitch-space player positions, with **two models**
(hard Voronoi + soft Gaussian influence), and report **control % per standard tactical zone**
(3 thirds × 5 channels = 15 zones, incl. half-spaces), summarized per phase, with plots.

## Inputs
Same as shape: `trajectories.parquet` (x_pitch/y_pitch canonical [0,1], team, class) ⋈
`phases.parquet` (phase, homography_source). `WIDTH_M=45.7`, `LENGTH_M=68.5`.

## Players / frames
Keep `team ∈ {own, opp}`, `class ∈ {player, goalkeeper}` (**GK included** — it occupies space;
`drop_goalkeeper=True` to exclude), coords in [−0.2, 1.2]. Frame used only if
`homography_source != none` and **both teams have ≥ `min_players` (default 3)**.

## Computation (fine grid → dominance)
- Grid of cell centres over `[0,WIDTH_M]×[0,LENGTH_M]` at `grid_m` spacing (default 1.0 m).
  Control % is robust to resolution; the grid is only a substrate.
- Per frame, per model, each cell is assigned to `own` or `opp`:
  - **voronoi:** `own` iff `min dist to an own player < min dist to an opp player`.
  - **influence:** `own` iff `Σ exp(−d²/2σ²)` over own players > that over opp (`sigma_m` def 7).
- `own` control for a region = fraction of its cells assigned to `own` (×100). `opp` = complement.

## Zones (own-team perspective; `halftime_frame`-aware)
Reflect player `y_m → LENGTH_M − y_m` for frames `≥ halftime_frame` so own always attacks toward
`y = LENGTH_M`. Zones are fixed pitch regions:
- **Thirds (length):** `defensive` (y_frac < 1/3), `middle` (< 2/3), `attacking` (else).
- **Channels (width), from penalty-box landmarks (x-fraction boundaries, configurable):**
  `left_wing` (<0.14), `left_halfspace` (<0.37), `central` (<0.63), `right_halfspace` (<0.86),
  `right_wing` (else).
- Zone label = `{third}_{channel}` → 15 zones. Each cell gets a zone once (precomputed).

## Outputs
- `SpaceResult`:
  - `per_frame`: long — `frame, t_seconds, model, zone, own_control_pct` (15×2 rows/frame).
  - `overall`: `frame, t_seconds, model, own_control_pct, opp_control_pct` (whole pitch).
  - `per_phase`: `phase, model, zone, own_control_pct` (mean), `n_frames`.
- `space.parquet` (per_frame) + `space_overall.parquet`; plots:
  1. **15-zone control grid** — pitch drawn as 3×5 zones, each shaded by mean own control
     (diverging 0–100, 50 neutral); one panel per model.
  2. **overall control % over time** (own, per model).
  3. **per-third control over time** (zones aggregated to thirds, own %).
- CLI prints a compact **per-third × model** table + overall means.

## API
`metrics/space.py`:
```python
@dataclass(frozen=True)
class SpaceResult:
    per_frame: pd.DataFrame
    overall: pd.DataFrame
    per_phase: pd.DataFrame

def compute_space_control(
    trajectories, phases, *, grid_m=1.0, sigma_m=7.0, min_players=3,
    drop_goalkeeper=False, halftime_frame=None) -> SpaceResult: ...
def plot_space(result: SpaceResult, out_dir: Path) -> list[Path]: ...
def main(argv=None) -> None: ...   # CLI: python -m soccer_vision.metrics.space --session <dir>
```
`compute_space_control` is pure (no I/O). Zone boundaries / grid / sigma are module constants +
kwargs.

## Edge cases
Frames with < min_players on either team → skipped. No valid frames → empty result, no crash.
Cells always assigned to a team (both models), so own+opp = 100 % per region.

## Testing (synthetic exactness)
- All own players clustered on one half, opp on the other → ~100 % own control on own's side,
  per zone; voronoi and influence agree on clean separation.
- A lone own player (opp absent) can't be scored (min_players) → frame skipped; with both teams
  present but one player each, the nearer team owns cells near it.
- Zone assignment: a player placed in the left-half-space region drives that zone's own control up.
- Attacking-third accounting; `halftime_frame` flips thirds; filtering (GK/ball/ref); per-phase;
  plots write.

## Acceptance
On `~/sv-labeler/analysis3`: non-empty results, control ∈ [0,100], sensible per-third pattern
(e.g., own controls more of its attacking third during `attack`). Full suite + mypy + ruff clean.

## Out of scope
Velocity-aware control, off-ball run value, expected-threat weighting — later slices.
