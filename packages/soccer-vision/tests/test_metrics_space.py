"""Tests for space-control metrics (Phase 4 v1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from soccer_vision.metrics.space import SpaceResult, compute_space_control, plot_space


def _row(frame: int, tid: int, team: str, x: float, y: float, cls: str = "player") -> dict[str, Any]:
    return {"frame": frame, "t_seconds": frame / 30.0, "track_id": tid, "class": cls,
            "team": team, "conf": 0.9, "x_pitch": x, "y_pitch": y}


def _session(own_xy: list[tuple[float, float]], opp_xy: list[tuple[float, float]],
             *, frames: tuple[int, ...] = (10,), phase: str = "attack",
             extra: list[dict[str, Any]] | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    tid = 0
    for fr in frames:
        for x, y in own_xy:
            rows.append(_row(fr, tid := tid + 1, "own", x, y))
        for x, y in opp_xy:
            rows.append(_row(fr, tid := tid + 1, "opp", x, y))
    if extra:
        rows.extend(extra)
    traj = pd.DataFrame(rows)
    phases = pd.DataFrame([
        {"frame": fr, "phase": phase, "homography_source": "manual"} for fr in frames])
    return traj, phases


def _zone(res: SpaceResult, model: str, zone: str) -> float:
    pf = res.per_frame
    s = pf[(pf["model"] == model) & (pf["zone"] == zone)]["own_control_pct"]
    return float(s.mean())


OWN_HIGH = [(0.3, 0.85), (0.4, 0.85), (0.5, 0.85), (0.6, 0.85)]   # attacking end (high y)
OPP_LOW = [(0.3, 0.15), (0.4, 0.15), (0.5, 0.15), (0.6, 0.15)]    # defensive end (low y)


def test_thirds_separation() -> None:
    traj, phases = _session(OWN_HIGH, OPP_LOW)
    res = compute_space_control(traj, phases, grid_m=2.0)
    for model in ("voronoi", "influence"):
        assert _zone(res, model, "attacking_central") > 80   # own owns its attacking end
        assert _zone(res, model, "defensive_central") < 20   # opp owns the defensive end


def test_channel_separation() -> None:
    own = [(0.2, y) for y in (0.35, 0.45, 0.55, 0.65)]       # left side, middle third
    opp = [(0.8, y) for y in (0.35, 0.45, 0.55, 0.65)]       # right side
    res = compute_space_control(*_session(own, opp), grid_m=2.0)
    assert _zone(res, "voronoi", "middle_left_wing") > 80
    assert _zone(res, "voronoi", "middle_right_wing") < 20


def test_overall_complement() -> None:
    res = compute_space_control(*_session(OWN_HIGH, OPP_LOW), grid_m=2.0)
    ov = res.overall.iloc[0]
    assert abs(ov["own_control_pct"] + ov["opp_control_pct"] - 100.0) < 1e-6


def test_min_players_skips_frame() -> None:
    traj, phases = _session(OWN_HIGH[:2], OPP_LOW)            # only 2 own < min 3
    res = compute_space_control(traj, phases, grid_m=2.0, min_players=3)
    assert res.overall.empty


def test_referee_and_ball_are_ignored() -> None:
    extra = [_row(10, 900, "ref", 0.5, 0.5, cls="referee"),
             _row(10, 901, "own", 0.5, 0.5, cls="ball")]
    with_noise = compute_space_control(*_session(OWN_HIGH, OPP_LOW, extra=extra), grid_m=2.0)
    clean = compute_space_control(*_session(OWN_HIGH, OPP_LOW), grid_m=2.0)
    assert abs(with_noise.overall.iloc[0]["own_control_pct"]
               - clean.overall.iloc[0]["own_control_pct"]) < 1e-9


def test_per_phase_present() -> None:
    res = compute_space_control(*_session(OWN_HIGH, OPP_LOW, frames=(10, 20)), grid_m=2.0)
    pp = res.per_phase
    row = pp[(pp["phase"] == "attack") & (pp["model"] == "voronoi")
             & (pp["zone"] == "attacking_central")].iloc[0]
    assert row["n_frames"] == 2 and row["own_control_pct"] > 80


def test_halftime_flips_thirds() -> None:
    traj, phases = _session(OWN_HIGH, OPP_LOW)
    res = compute_space_control(traj, phases, grid_m=2.0, halftime_frame=10)
    # reflected: own (was high y) now defends low y -> attacking third flips to opp
    assert _zone(res, "voronoi", "attacking_central") < 20
    assert _zone(res, "voronoi", "defensive_central") > 80


def test_empty_when_no_homography() -> None:
    traj, phases = _session(OWN_HIGH, OPP_LOW)
    phases = phases.assign(homography_source="none")
    res = compute_space_control(traj, phases, grid_m=2.0)
    assert res.per_frame.empty and res.overall.empty and res.per_phase.empty


def test_plot_space_writes_files(tmp_path: Path) -> None:
    res = compute_space_control(*_session(OWN_HIGH, OPP_LOW), grid_m=2.0)
    paths = plot_space(res, tmp_path)
    assert len(paths) == 3 and all(p.exists() for p in paths)
