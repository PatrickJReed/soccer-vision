"""Team-shape metrics (Phase 4, v1): per-frame team centroid / width / depth / compactness
for both teams + inter-team relations, summarized per phase, from the pipeline's pitch-space
trajectories. Pure compute (DataFrame in -> DataFrame out); plotting + CLI are the only I/O.

x_pitch/y_pitch are canonical [0,1] (x = width/touchline axis, y = length/goal axis); metrics
are reported in METRES via WIDTH_M / LENGTH_M.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M

_TEAMS = ("own", "opp")
_COORD_LO, _COORD_HI = -0.2, 1.2  # drop detections whose pitch coords are implausibly far out


@dataclass(frozen=True)
class ShapeResult:
    per_frame: pd.DataFrame    # frame, t_seconds, team, centroid_x_m, centroid_y_m,
                               # width_m, depth_m, compactness_m, n_players
    inter_team: pd.DataFrame   # frame, t_seconds, centroid_separation_m, vertical_offset_m,
                               # width_diff_m
    per_phase: pd.DataFrame    # phase, team, width_m, depth_m, compactness_m,
                               # centroid_y_m, n_frames


def _player_frame(
    trajectories: pd.DataFrame,
    phases: pd.DataFrame,
    *,
    drop_goalkeeper: bool,
    halftime_frame: int | None,
) -> pd.DataFrame:
    """Filtered per-detection players in metres on frames with a trusted homography."""
    classes = {"player"} if drop_goalkeeper else {"player", "goalkeeper"}
    good_frames = set(phases.loc[phases["homography_source"] != "none", "frame"].tolist())
    df = trajectories[
        trajectories["team"].isin(_TEAMS)
        & trajectories["class"].isin(classes)
        & trajectories["frame"].isin(good_frames)
        & trajectories["x_pitch"].between(_COORD_LO, _COORD_HI)
        & trajectories["y_pitch"].between(_COORD_LO, _COORD_HI)
    ].copy()
    df["x_m"] = df["x_pitch"].to_numpy() * WIDTH_M
    y = df["y_pitch"].to_numpy() * LENGTH_M
    if halftime_frame is not None:
        flip = df["frame"].to_numpy() >= halftime_frame
        y = np.where(flip, LENGTH_M - y, y)  # normalise attacking direction across halftime
    df["y_m"] = y
    return df


def compute_shape(
    trajectories: pd.DataFrame,
    phases: pd.DataFrame,
    *,
    min_players: int = 4,
    drop_goalkeeper: bool = True,
    halftime_frame: int | None = None,
) -> ShapeResult:
    """Compute team-shape metrics. See module docstring / spec for definitions."""
    df = _player_frame(
        trajectories, phases, drop_goalkeeper=drop_goalkeeper, halftime_frame=halftime_frame)

    empty_pf = pd.DataFrame(columns=[
        "frame", "t_seconds", "team", "centroid_x_m", "centroid_y_m",
        "width_m", "depth_m", "compactness_m", "n_players"])
    empty_it = pd.DataFrame(columns=[
        "frame", "t_seconds", "centroid_separation_m", "vertical_offset_m", "width_diff_m"])
    empty_pp = pd.DataFrame(columns=[
        "phase", "team", "width_m", "depth_m", "compactness_m", "centroid_y_m", "n_frames"])
    if df.empty:
        return ShapeResult(empty_pf, empty_it, empty_pp)

    grouped = df.groupby(["frame", "team"], sort=True)
    per = grouped.agg(
        t_seconds=("t_seconds", "first"),
        centroid_x_m=("x_m", "mean"),
        centroid_y_m=("y_m", "mean"),
        x_min=("x_m", "min"), x_max=("x_m", "max"),
        y_min=("y_m", "min"), y_max=("y_m", "max"),
        n_players=("x_m", "size"),
    ).reset_index()
    per = per[per["n_players"] >= min_players].copy()
    if per.empty:
        return ShapeResult(empty_pf, empty_it, empty_pp)
    per["width_m"] = per["x_max"] - per["x_min"]
    per["depth_m"] = per["y_max"] - per["y_min"]

    # compactness = mean player distance to its team centroid (only kept (frame, team) groups)
    merged = df.merge(per[["frame", "team", "centroid_x_m", "centroid_y_m"]],
                      on=["frame", "team"], how="inner")
    merged["_d"] = np.hypot(merged["x_m"] - merged["centroid_x_m"],
                            merged["y_m"] - merged["centroid_y_m"])
    comp = merged.groupby(["frame", "team"])["_d"].mean().reset_index(name="compactness_m")
    per = per.merge(comp, on=["frame", "team"], how="left")

    per_frame = per[[
        "frame", "t_seconds", "team", "centroid_x_m", "centroid_y_m",
        "width_m", "depth_m", "compactness_m", "n_players"]].sort_values(
        ["frame", "team"]).reset_index(drop=True)

    inter_team = _inter_team(per_frame)
    per_phase = _per_phase(per_frame, phases)
    return ShapeResult(per_frame, inter_team, per_phase)


def _inter_team(per_frame: pd.DataFrame) -> pd.DataFrame:
    own = per_frame[per_frame["team"] == "own"].set_index("frame")
    opp = per_frame[per_frame["team"] == "opp"].set_index("frame")
    common = own.index.intersection(opp.index)
    if len(common) == 0:
        return pd.DataFrame(columns=[
            "frame", "t_seconds", "centroid_separation_m", "vertical_offset_m", "width_diff_m"])
    o, p = own.loc[common], opp.loc[common]
    out = pd.DataFrame({
        "frame": common,
        "t_seconds": o["t_seconds"].to_numpy(),
        "centroid_separation_m": np.hypot(
            o["centroid_x_m"].to_numpy() - p["centroid_x_m"].to_numpy(),
            o["centroid_y_m"].to_numpy() - p["centroid_y_m"].to_numpy()),
        "vertical_offset_m": o["centroid_y_m"].to_numpy() - p["centroid_y_m"].to_numpy(),
        "width_diff_m": o["width_m"].to_numpy() - p["width_m"].to_numpy(),
    })
    return out.sort_values("frame").reset_index(drop=True)


def _per_phase(per_frame: pd.DataFrame, phases: pd.DataFrame) -> pd.DataFrame:
    pf = per_frame.merge(phases[["frame", "phase"]], on="frame", how="left")
    agg = pf.groupby(["phase", "team"], sort=True).agg(
        width_m=("width_m", "mean"),
        depth_m=("depth_m", "mean"),
        compactness_m=("compactness_m", "mean"),
        centroid_y_m=("centroid_y_m", "mean"),
        n_frames=("frame", "nunique"),
    ).reset_index()
    return agg


# --------------------------------------------------------------------------- plotting + CLI
def plot_shape(result: ShapeResult, out_dir: Path) -> list[Path]:
    """Render the three shape plots as PNGs. Returns the written paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pf = result.per_frame

    def _save(fig: Figure, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    # 1. width / depth / compactness over time, both teams
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, metric in zip(axes, ("width_m", "depth_m", "compactness_m"), strict=True):
        for team, style in (("own", "-"), ("opp", "--")):
            s = pf[pf["team"] == team].sort_values("t_seconds")
            if not s.empty:
                ax.plot(s["t_seconds"], s[metric], style, label=team, alpha=0.8)
        ax.set_ylabel(metric)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t_seconds")
    fig.suptitle("Team shape over time (metres)")
    _save(fig, "shape_timeseries.png")

    # 2. team centroid positions on a pitch outline
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, team in zip(axs, _TEAMS, strict=True):
        s = pf[pf["team"] == team]
        if not s.empty:
            ax.scatter(s["centroid_x_m"], s["centroid_y_m"], s=4, alpha=0.15)
        ax.add_patch(Rectangle((0, 0), WIDTH_M, LENGTH_M, fill=False, edgecolor="#2a8a55"))
        ax.set_xlim(-2, WIDTH_M + 2)
        ax.set_ylim(-2, LENGTH_M + 2)
        ax.set_aspect("equal")
        ax.set_title(f"{team} centroids")
        ax.set_xlabel("width (m)")
        ax.set_ylabel("length (m)")
    _save(fig, "avg_positions.png")

    # 3. inter-team separation + vertical offset over time
    it = result.inter_team
    fig, ax = plt.subplots(figsize=(11, 3.2))
    if not it.empty:
        ax.plot(it["t_seconds"], it["centroid_separation_m"], label="centroid separation")
        ax.plot(it["t_seconds"], it["vertical_offset_m"], label="vertical offset (own-opp)")
    ax.axhline(0, color="#555", lw=0.6)
    ax.set_xlabel("t_seconds")
    ax.set_ylabel("metres")
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Inter-team relations")
    _save(fig, "separation.png")
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Team-shape metrics (Phase 4 v1)")
    ap.add_argument("--session", required=True, type=Path,
                    help="dir with trajectories.parquet + phases.parquet")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: --session)")
    ap.add_argument("--min-players", type=int, default=4)
    ap.add_argument("--keep-goalkeeper", action="store_true")
    ap.add_argument("--halftime-frame", type=int, default=None)
    args = ap.parse_args(argv)

    session = args.session
    out = args.out or session
    traj = pd.read_parquet(session / "trajectories.parquet")
    phases = pd.read_parquet(session / "phases.parquet")
    result = compute_shape(
        traj, phases, min_players=args.min_players,
        drop_goalkeeper=not args.keep_goalkeeper, halftime_frame=args.halftime_frame)

    Path(out).mkdir(parents=True, exist_ok=True)
    result.per_frame.to_parquet(Path(out) / "shape.parquet", index=False)
    result.inter_team.to_parquet(Path(out) / "shape_interteam.parquet", index=False)
    paths = plot_shape(result, Path(out))

    n_hom = int((phases["homography_source"] != "none").sum())
    own_cov = result.per_frame[result.per_frame["team"] == "own"]["frame"].nunique()
    print(f"homography frames: {n_hom}")
    print(f"shape coverage: own {own_cov}/{n_hom} frames "
          f"({100 * own_cov / n_hom:.1f}%)" if n_hom else "no homography frames")
    print("\nper-phase summary (metres):")
    with pd.option_context("display.width", 120, "display.max_rows", None):
        print(result.per_phase.round(1).to_string(index=False))
    print(f"\nwrote shape.parquet + {len(paths)} plots -> {out}")


if __name__ == "__main__":
    main()
