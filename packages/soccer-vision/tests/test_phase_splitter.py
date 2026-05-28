"""Tests for phase splitter (combines possession + ball location)."""

from __future__ import annotations

import pandas as pd
from soccer_vision.phase.splitter import label_phase


def test_attack_when_own_with_ball_in_opp_two_thirds() -> None:
    states = pd.Series(["own", "own", "own"], index=[0, 1, 2])
    ball_y = pd.Series([0.5, 0.7, 0.8], index=[0, 1, 2])
    fps = 30.0
    phases = label_phase(states, ball_y, fps=fps)
    assert phases.loc[2] == "attack"


def test_build_when_own_with_ball_in_own_third() -> None:
    states = pd.Series(["own", "own"], index=[0, 1])
    ball_y = pd.Series([0.2, 0.25], index=[0, 1])
    phases = label_phase(states, ball_y, fps=30.0)
    assert phases.loc[1] == "build"


def test_defend_low_when_opp_with_ball_in_own_half() -> None:
    states = pd.Series(["opp", "opp"], index=[0, 1])
    ball_y = pd.Series([0.3, 0.4], index=[0, 1])
    phases = label_phase(states, ball_y, fps=30.0)
    assert phases.loc[1] == "defend_low"


def test_transition_window_after_possession_change() -> None:
    """5-second window after a state change → 'transition'."""
    fps = 30.0
    states_list = ["own"] * 30 + ["opp"] * 60  # turnover at frame 30
    ball_y = [0.6] * 90
    states = pd.Series(states_list, index=range(90))
    ball_y_s = pd.Series(ball_y, index=range(90))
    phases = label_phase(states, ball_y_s, fps=fps, transition_seconds=1.0)
    # Frame 30 turnover -> transition window 30..59
    assert phases.loc[35] == "transition"
    # Outside window: regular phase
    assert phases.loc[80] == "defend_high"


def test_contested_passes_through() -> None:
    states = pd.Series(["contested", "own"], index=[0, 1])
    ball_y = pd.Series([0.5, 0.7], index=[0, 1])
    phases = label_phase(states, ball_y, fps=30.0)
    assert phases.loc[0] == "contested"
