"""Space-control (pitch-control) metrics (Phase 4, v1).

Per frame, two models decide which team controls each cell of a fine pitch grid:
  - `voronoi`  : the cell's nearest player's team (hard dominant region),
  - `influence`: the team with larger summed Gaussian influence (soft contest).
Cells are aggregated into the 15 standard tactical zones (3 thirds x 5 width channels incl.
half-spaces) and reported as control %, summarized per phase. Pure compute; plot/CLI are I/O.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]

from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M

_TEAMS = ("own", "opp")
_COORD_LO, _COORD_HI = -0.2, 1.2
MODELS = ("voronoi", "influence")
# width-channel boundaries as fractions of pitch width (from the penalty-box landmarks).
_CHANNEL_EDGES = (0.14, 0.37, 0.63, 0.86)
_CHANNELS = ("left_wing", "left_halfspace", "central", "right_halfspace", "right_wing")
_THIRD_EDGES = (1.0 / 3.0, 2.0 / 3.0)
_THIRDS = ("defensive", "middle", "attacking")
ZONES = tuple(f"{t}_{c}" for t in _THIRDS for c in _CHANNELS)  # 15, own-team perspective


@dataclass(frozen=True)
class SpaceResult:
    per_frame: pd.DataFrame   # frame, t_seconds, model, zone, own_control_pct
    overall: pd.DataFrame     # frame, t_seconds, model, own_control_pct, opp_control_pct
    per_phase: pd.DataFrame   # phase, model, zone, own_control_pct, n_frames


def _grid(grid_m: float) -> tuple[NDArray[np.float64], NDArray[np.str_]]:
    """Cell centres (G,2) in metres + each cell's zone label (G,)."""
    nx = max(1, round(WIDTH_M / grid_m))
    ny = max(1, round(LENGTH_M / grid_m))
    xs = (np.arange(nx) + 0.5) * WIDTH_M / nx
    ys = (np.arange(ny) + 0.5) * LENGTH_M / ny
    gx, gy = np.meshgrid(xs, ys)
    cells = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)
    ch = np.digitize(cells[:, 0] / WIDTH_M, _CHANNEL_EDGES)
    th = np.digitize(cells[:, 1] / LENGTH_M, _THIRD_EDGES)
    zones = np.array([f"{_THIRDS[t]}_{_CHANNELS[c]}" for t, c in zip(th, ch, strict=True)])
    return cells, zones


def _dominance(
    cells: NDArray[np.float64], own: NDArray[np.float64], opp: NDArray[np.float64],
    model: str, sigma_m: float,
) -> NDArray[np.bool_]:
    """Boolean (G,): True where `own` controls the cell under the given model."""
    d_own = cdist(cells, own)
    d_opp = cdist(cells, opp)
    if model == "voronoi":
        return np.asarray(d_own.min(1) < d_opp.min(1))
    twosig2 = 2.0 * sigma_m * sigma_m
    infl_own = np.exp(-(d_own ** 2) / twosig2).sum(1)
    infl_opp = np.exp(-(d_opp ** 2) / twosig2).sum(1)
    return np.asarray(infl_own > infl_opp)


def compute_space_control(
    trajectories: pd.DataFrame,
    phases: pd.DataFrame,
    *,
    grid_m: float = 1.0,
    sigma_m: float = 7.0,
    min_players: int = 3,
    drop_goalkeeper: bool = False,
    halftime_frame: int | None = None,
) -> SpaceResult:
    classes = {"player"} if drop_goalkeeper else {"player", "goalkeeper"}
    good = set(phases.loc[phases["homography_source"] != "none", "frame"].tolist())
    df = trajectories[
        trajectories["team"].isin(_TEAMS)
        & trajectories["class"].isin(classes)
        & trajectories["frame"].isin(good)
        & trajectories["x_pitch"].between(_COORD_LO, _COORD_HI)
        & trajectories["y_pitch"].between(_COORD_LO, _COORD_HI)
    ].copy()
    df["x_m"] = df["x_pitch"].to_numpy() * WIDTH_M
    y = df["y_pitch"].to_numpy() * LENGTH_M
    if halftime_frame is not None:
        y = np.where(df["frame"].to_numpy() >= halftime_frame, LENGTH_M - y, y)
    df["y_m"] = y

    empty_pf = pd.DataFrame(columns=["frame", "t_seconds", "model", "zone", "own_control_pct"])
    empty_ov = pd.DataFrame(
        columns=["frame", "t_seconds", "model", "own_control_pct", "opp_control_pct"])
    empty_pp = pd.DataFrame(columns=["phase", "model", "zone", "own_control_pct", "n_frames"])
    if df.empty:
        return SpaceResult(empty_pf, empty_ov, empty_pp)

    cells, zones = _grid(grid_m)
    zone_masks = {z: (zones == z) for z in ZONES}
    pf_rows: list[tuple[int, float, str, str, float]] = []
    ov_rows: list[tuple[int, float, str, float, float]] = []
    for frame, g in df.groupby("frame", sort=True):
        own = g.loc[g["team"] == "own", ["x_m", "y_m"]].to_numpy()
        opp = g.loc[g["team"] == "opp", ["x_m", "y_m"]].to_numpy()
        if len(own) < min_players or len(opp) < min_players:
            continue
        t_s = float(g["t_seconds"].iloc[0])
        fr = int(frame)  # type: ignore[arg-type]  # groupby key
        for model in MODELS:
            owns = _dominance(cells, own, opp, model, sigma_m)
            own_pct = 100.0 * float(owns.mean())
            ov_rows.append((fr, t_s, model, own_pct, 100.0 - own_pct))
            for z, m in zone_masks.items():
                if m.any():
                    pf_rows.append((fr, t_s, model, z, 100.0 * float(owns[m].mean())))

    per_frame = pd.DataFrame(pf_rows,
                             columns=["frame", "t_seconds", "model", "zone", "own_control_pct"])
    overall = pd.DataFrame(
        ov_rows, columns=["frame", "t_seconds", "model", "own_control_pct", "opp_control_pct"])
    per_phase = _per_phase(per_frame, phases) if not per_frame.empty else empty_pp
    return SpaceResult(per_frame, overall, per_phase)


def _per_phase(per_frame: pd.DataFrame, phases: pd.DataFrame) -> pd.DataFrame:
    pf = per_frame.merge(phases[["frame", "phase"]], on="frame", how="left")
    return pf.groupby(["phase", "model", "zone"], sort=True).agg(
        own_control_pct=("own_control_pct", "mean"),
        n_frames=("frame", "nunique"),
    ).reset_index()


# --------------------------------------------------------------------------- plotting + CLI
def _zone_grid_means(per_frame: pd.DataFrame, model: str) -> NDArray[np.float64]:
    """3x5 array (rows: defensive->attacking, cols: channels) of mean own control for a model."""
    sub = per_frame[per_frame["model"] == model]
    means = sub.groupby("zone")["own_control_pct"].mean()
    grid = np.full((3, 5), np.nan)
    for ti, t in enumerate(_THIRDS):
        for ci, c in enumerate(_CHANNELS):
            z = f"{t}_{c}"
            if z in means.index:
                grid[ti, ci] = means[z]
    return grid


def plot_space(result: SpaceResult, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save(fig: Figure, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    # 1. 15-zone control grid per model (attacking third at top)
    fig, axs = plt.subplots(1, len(MODELS), figsize=(11, 5))
    for ax, model in zip(np.atleast_1d(axs), MODELS, strict=True):
        grid = _zone_grid_means(result.per_frame, model)
        im = ax.imshow(grid[::-1], cmap="RdBu", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels(_CHANNELS, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(3))
        ax.set_yticklabels(_THIRDS[::-1], fontsize=8)
        ax.set_title(f"{model}: mean own control %")
        for i in range(3):
            for j in range(5):
                v = grid[::-1][i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, "space_zone_grid.png")

    # 2. overall control % over time, per model
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for model in MODELS:
        s = result.overall[result.overall["model"] == model].sort_values("t_seconds")
        if not s.empty:
            ax.plot(s["t_seconds"], s["own_control_pct"], label=f"own ({model})", alpha=0.8)
    ax.axhline(50, color="#555", lw=0.6)
    ax.set_ylim(0, 100)
    ax.set_xlabel("t_seconds")
    ax.set_ylabel("own control %")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "space_overall.png")

    # 3. per-third own control over time (voronoi)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    vf = result.per_frame[result.per_frame["model"] == "voronoi"].copy()
    if not vf.empty:
        vf["third"] = vf["zone"].str.split("_").str[0]
        per_third = vf.groupby(["t_seconds", "third"])["own_control_pct"].mean().reset_index()
        for third in _THIRDS:
            s = per_third[per_third["third"] == third].sort_values("t_seconds")
            if not s.empty:
                ax.plot(s["t_seconds"], s["own_control_pct"], label=third, alpha=0.8)
    ax.axhline(50, color="#555", lw=0.6)
    ax.set_ylim(0, 100)
    ax.set_xlabel("t_seconds")
    ax.set_ylabel("own control % (voronoi)")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "space_thirds.png")
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Space-control metrics (Phase 4 v1)")
    ap.add_argument("--session", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--grid-m", type=float, default=1.0)
    ap.add_argument("--sigma-m", type=float, default=7.0)
    ap.add_argument("--min-players", type=int, default=3)
    ap.add_argument("--drop-goalkeeper", action="store_true")
    ap.add_argument("--halftime-frame", type=int, default=None)
    args = ap.parse_args(argv)

    session = args.session
    out = args.out or session
    traj = pd.read_parquet(session / "trajectories.parquet")
    phases = pd.read_parquet(session / "phases.parquet")
    result = compute_space_control(
        traj, phases, grid_m=args.grid_m, sigma_m=args.sigma_m, min_players=args.min_players,
        drop_goalkeeper=args.drop_goalkeeper, halftime_frame=args.halftime_frame)

    Path(out).mkdir(parents=True, exist_ok=True)
    result.per_frame.to_parquet(Path(out) / "space.parquet", index=False)
    result.overall.to_parquet(Path(out) / "space_overall.parquet", index=False)
    paths = plot_space(result, Path(out))

    n_frames = result.overall["frame"].nunique()
    print(f"scored frames: {n_frames}")
    if not result.overall.empty:
        ov = result.overall.groupby("model")["own_control_pct"].mean()
        print("overall mean own control %:", {m: round(v, 1) for m, v in ov.items()})
    if not result.per_phase.empty:
        vt = result.per_phase[result.per_phase["model"] == "voronoi"].copy()
        vt["third"] = vt["zone"].str.split("_").str[0]
        tbl = vt.groupby(["phase", "third"])["own_control_pct"].mean().unstack().round(1)
        print("\nown control % by phase x third (voronoi):")
        print(tbl.to_string())
    print(f"\nwrote space.parquet + {len(paths)} plots -> {out}")


if __name__ == "__main__":
    main()
