"""Tests for team-shape metrics (Phase 4 v1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M
from soccer_vision.metrics.shape import ShapeResult, compute_shape, plot_shape

OWN = [(0.2, 0.4), (0.4, 0.4), (0.6, 0.4), (0.8, 0.4)]   # centroid (0.5,0.4), width 0.6, depth 0
OPP = [(0.3, 0.6), (0.4, 0.6), (0.5, 0.6), (0.6, 0.6)]   # centroid (0.45,0.6), width 0.3, depth 0


def _row(frame: int, tid: int, team: str, x: float, y: float, cls: str = "player") -> dict[str, Any]:
    return {"frame": frame, "t_seconds": frame / 30.0, "track_id": tid, "class": cls,
            "team": team, "conf": 0.9, "x_pitch": x, "y_pitch": y}


def _trajectories() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tid = 0
    for fr in (10, 20):                                  # two 'attack' frames w/ homography
        for x, y in OWN:
            rows.append(_row(fr, tid := tid + 1, "own", x, y))
        for x, y in OPP:
            rows.append(_row(fr, tid := tid + 1, "opp", x, y))
    # frame 10 noise that must be filtered out
    rows.append(_row(10, tid := tid + 1, "ref", 0.5, 0.5, cls="referee"))
    rows.append(_row(10, tid := tid + 1, "own", 0.5, 0.5, cls="ball"))
    rows.append(_row(10, tid := tid + 1, "own", 0.05, 0.05, cls="goalkeeper"))
    rows.append(_row(10, tid := tid + 1, "unknown", 0.5, 0.5))
    # frame 30 has NO homography -> excluded; frame 40 own has < 4 players -> omitted
    for x, y in OWN:
        rows.append(_row(30, tid := tid + 1, "own", x, y))
    for x, y in OWN[:3]:
        rows.append(_row(40, tid := tid + 1, "own", x, y))
    return pd.DataFrame(rows)


def _phases() -> pd.DataFrame:
    return pd.DataFrame([
        {"frame": 10, "phase": "attack", "homography_source": "manual"},
        {"frame": 20, "phase": "attack", "homography_source": "manual"},
        {"frame": 30, "phase": "attack", "homography_source": "none"},
        {"frame": 40, "phase": "build", "homography_source": "manual"},
    ])


def _own10(res: ShapeResult) -> pd.Series:
    pf = res.per_frame
    return pf[(pf["frame"] == 10) & (pf["team"] == "own")].iloc[0]


def test_core_metrics_exact() -> None:
    res = compute_shape(_trajectories(), _phases())
    r = _own10(res)
    assert r["n_players"] == 4
    assert abs(r["centroid_x_m"] - 0.5 * WIDTH_M) < 0.05      # 22.85
    assert abs(r["centroid_y_m"] - 0.4 * LENGTH_M) < 0.05     # 27.4
    assert abs(r["width_m"] - 0.6 * WIDTH_M) < 0.05           # 27.42
    assert abs(r["depth_m"]) < 1e-6                           # all same y
    assert abs(r["compactness_m"] - 0.2 * WIDTH_M) < 0.05     # mean |offset| = 9.14 = 0.2*W


def test_inter_team() -> None:
    res = compute_shape(_trajectories(), _phases())
    it = res.inter_team
    row = it[it["frame"] == 10].iloc[0]
    import math
    exp_sep = math.hypot((0.5 - 0.45) * WIDTH_M, (0.4 - 0.6) * LENGTH_M)
    assert abs(row["centroid_separation_m"] - exp_sep) < 0.05
    assert abs(row["vertical_offset_m"] - (0.4 - 0.6) * LENGTH_M) < 0.05   # -13.7
    assert abs(row["width_diff_m"] - (0.6 - 0.3) * WIDTH_M) < 0.05         # 13.71


def test_filtering_frames_and_classes() -> None:
    res = compute_shape(_trajectories(), _phases())
    own_frames = set(res.per_frame[res.per_frame["team"] == "own"]["frame"])
    assert own_frames == {10, 20}                 # 30 (no homography) + 40 (<4 players) excluded
    # goalkeeper/ball/referee/unknown excluded -> own frame-10 has exactly 4 players
    assert _own10(res)["n_players"] == 4


def test_keep_goalkeeper() -> None:
    res = compute_shape(_trajectories(), _phases(), drop_goalkeeper=False)
    assert _own10(res)["n_players"] == 5          # GK now counted


def test_per_phase_means() -> None:
    res = compute_shape(_trajectories(), _phases())
    pp = res.per_phase
    row = pp[(pp["phase"] == "attack") & (pp["team"] == "own")].iloc[0]
    assert row["n_frames"] == 2                    # frames 10 and 20
    assert abs(row["width_m"] - 0.6 * WIDTH_M) < 0.05


def test_halftime_reflects_length_axis() -> None:
    res = compute_shape(_trajectories(), _phases(), halftime_frame=10)
    r = _own10(res)
    assert abs(r["centroid_y_m"] - (LENGTH_M - 0.4 * LENGTH_M)) < 0.05  # reflected 27.4 -> 41.1
    assert abs(r["depth_m"]) < 1e-6                                     # depth reflection-invariant


def test_empty_when_no_homography() -> None:
    phases = _phases().assign(homography_source="none")
    res = compute_shape(_trajectories(), phases)
    assert res.per_frame.empty and res.inter_team.empty and res.per_phase.empty


def test_plot_shape_writes_files(tmp_path: Path) -> None:
    res = compute_shape(_trajectories(), _phases())
    paths = plot_shape(res, tmp_path)
    assert len(paths) == 3 and all(p.exists() for p in paths)
