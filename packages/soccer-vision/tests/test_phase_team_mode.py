"""Tests for per-track modal team smoothing."""

from __future__ import annotations

import pandas as pd
from soccer_vision.phase.team_mode import apply_modal_team_per_track


def test_majority_own_overrides_minority_opp() -> None:
    df = pd.DataFrame({
        "frame": [0, 1, 2, 3, 4],
        "track_id": [1, 1, 1, 1, 1],
        "class": ["player"] * 5,
        "team": ["own", "own", "opp", "own", "own"],
    })
    out = apply_modal_team_per_track(df)
    assert (out["team"] == "own").all()


def test_per_track_independent() -> None:
    df = pd.DataFrame({
        "frame": [0, 0, 1, 1],
        "track_id": [1, 2, 1, 2],
        "class": ["player", "player", "player", "player"],
        "team": ["own", "opp", "own", "opp"],
    })
    out = apply_modal_team_per_track(df)
    assert out[out["track_id"] == 1]["team"].iloc[0] == "own"
    assert out[out["track_id"] == 2]["team"].iloc[0] == "opp"


def test_referees_unchanged() -> None:
    df = pd.DataFrame({
        "frame": [0, 1],
        "track_id": [10, 10],
        "class": ["referee", "referee"],
        "team": ["ref", "ref"],
    })
    out = apply_modal_team_per_track(df)
    assert (out["team"] == "ref").all()


def test_unknown_when_no_clear_majority() -> None:
    """Equal counts → 'unknown'."""
    df = pd.DataFrame({
        "frame": [0, 1],
        "track_id": [1, 1],
        "class": ["player", "player"],
        "team": ["own", "opp"],
    })
    out = apply_modal_team_per_track(df)
    assert (out["team"] == "unknown").all()
