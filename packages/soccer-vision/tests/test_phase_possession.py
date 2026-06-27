"""Tests for 5-state possession proxy."""

from __future__ import annotations

import pandas as pd
from soccer_vision.phase.possession import classify_possession
from soccer_vision.pitch.spec import PitchSpec


def _make_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_own_when_clearly_closest() -> None:
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.51, "y_pitch": 0.51, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.70, "y_pitch": 0.70, "conf": 0.9},
    ])
    states = classify_possession(df)
    assert states[0] == "own"


def test_contested_when_close_margin() -> None:
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.51, "y_pitch": 0.50, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.515, "y_pitch": 0.50, "conf": 0.9},
    ])
    states = classify_possession(df)
    assert states[0] == "contested"


def test_contested_when_clump_balanced() -> None:
    """Both teams have >=1 within clump radius; counts differ by <=1."""
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.52, "y_pitch": 0.50, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.51, "y_pitch": 0.51, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.49, "y_pitch": 0.50, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.50, "y_pitch": 0.51, "conf": 0.9},
    ])
    states = classify_possession(df)
    assert states[0] == "contested"


def test_loose_ball_when_no_one_close() -> None:
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.80, "y_pitch": 0.50, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.20, "y_pitch": 0.50, "conf": 0.9},
    ])
    states = classify_possession(df)
    assert states[0] == "loose_ball"


def test_unknown_when_no_ball() -> None:
    df = _make_frame([
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.50, "y_pitch": 0.50, "conf": 0.9},
    ])
    states = classify_possession(df)
    assert states[0] == "unknown"


# --- §3.1 anisotropy / length-normalization ---------------------------------

def test_width_axis_offset_is_not_loose_after_normalization() -> None:
    """Ball and lone own player 0.10 apart on the WIDTH axis: raw 0.10 > 0.073
    (would be loose_ball), but length-normalized 0.10/1.5 = 0.067 < 0.073 -> own."""
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.60, "y_pitch": 0.50, "conf": 0.9},
    ])
    states = classify_possession(df)
    assert states[0] == "own"  # was "loose_ball" before normalization


def test_width_axis_clump_counts_only_after_normalization() -> None:
    """Opp player at width-offset 0.06 (raw 0.06 > 0.044 clump, normalized
    0.06/1.5 = 0.04 <= 0.044). own is clearly nearest (margin 0.03 > 0.022, so
    the margin path does NOT fire) -> contested only via the clump path, only
    after normalization (raw -> 'own')."""
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.50, "y_pitch": 0.51, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.56, "y_pitch": 0.50, "conf": 0.9},
    ])
    states = classify_possession(df)
    assert states[0] == "contested"  # was "own" before normalization


def test_pitch_spec_aspect_ratio_is_honored() -> None:
    """Width-offset 0.111: under 9v9 (/1.5 = 0.074 > 0.073) -> loose_ball;
    under 11v11 (/1.54 = 0.0721 < 0.073) -> own. Proves the param is used."""
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.611, "y_pitch": 0.50, "conf": 0.9},
    ])
    assert classify_possession(df)[0] == "loose_ball"
    assert classify_possession(df, pitch_spec=PitchSpec.fifa_11v11())[0] == "own"


# --- §3.7 highest-confidence ball -------------------------------------------

def test_uses_highest_confidence_ball_not_first_row() -> None:
    """Low-conf FIRST ball row near opp, high-conf ball near own -> own."""
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.90, "conf": 0.10},
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.10, "conf": 0.90},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.50, "y_pitch": 0.10, "conf": 0.9},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.50, "y_pitch": 0.90, "conf": 0.9},
    ])
    states = classify_possession(df)
    assert states[0] == "own"  # iloc[0] would have given "opp"


def test_highest_conf_ball_valid_when_lower_conf_row_is_nan() -> None:
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": float("nan"), "y_pitch": float("nan"), "conf": 0.10},
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.10, "conf": 0.90},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.50, "y_pitch": 0.10, "conf": 0.9},
    ])
    states = classify_possession(df)
    assert states[0] == "own"


def test_unknown_when_highest_conf_ball_is_nan() -> None:
    """Highest-conf ball NaN; we must NOT fall back to the lower-conf valid row."""
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": float("nan"), "y_pitch": float("nan"), "conf": 0.90},
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.90, "conf": 0.10},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.50, "y_pitch": 0.90, "conf": 0.9},
    ])
    states = classify_possession(df)
    assert states[0] == "unknown"
