"""Tests for 5-state possession proxy."""

from __future__ import annotations

import pandas as pd
from soccer_vision.phase.possession import classify_possession


def _make_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_own_when_clearly_closest() -> None:
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.51, "y_pitch": 0.51},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.70, "y_pitch": 0.70},
    ])
    states = classify_possession(df)
    assert states[0] == "own"


def test_contested_when_close_margin() -> None:
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.51, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.515, "y_pitch": 0.50},
    ])
    states = classify_possession(df)
    assert states[0] == "contested"


def test_contested_when_clump_balanced() -> None:
    """Both teams have >=1 within clump radius; counts differ by <=1."""
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.52, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.51, "y_pitch": 0.51},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.49, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.50, "y_pitch": 0.51},
    ])
    states = classify_possession(df)
    assert states[0] == "contested"


def test_loose_ball_when_no_one_close() -> None:
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.80, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.20, "y_pitch": 0.50},
    ])
    states = classify_possession(df)
    assert states[0] == "loose_ball"


def test_unknown_when_no_ball() -> None:
    df = _make_frame([
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.50, "y_pitch": 0.50},
    ])
    states = classify_possession(df)
    assert states[0] == "unknown"
