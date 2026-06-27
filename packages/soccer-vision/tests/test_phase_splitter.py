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


def test_defend_low_when_opp_with_ball_in_own_two_thirds() -> None:
    states = pd.Series(["opp", "opp"], index=[0, 1])
    ball_y = pd.Series([0.3, 0.4], index=[0, 1])
    phases = label_phase(states, ball_y, fps=30.0)
    assert phases.loc[1] == "defend_low"


def test_defend_split_at_opp_third_not_midfield() -> None:
    """§6.1: defend_low = own 2/3 (y < 0.667); defend_high = opp 1/3 (y > 0.667).
    y=0.6 must be defend_low (it was defend_high under the wrong 0.5 split)."""
    states = pd.Series(["opp", "opp", "opp", "opp"], index=[0, 1, 2, 3])
    ball_y = pd.Series([0.60, 0.70, 0.666, 0.668], index=[0, 1, 2, 3])
    phases = label_phase(states, ball_y, fps=30.0)
    assert phases.loc[0] == "defend_low"    # 0.60 < 0.667
    assert phases.loc[1] == "defend_high"   # 0.70 > 0.667
    assert phases.loc[2] == "defend_low"    # 0.666 < 0.667
    assert phases.loc[3] == "defend_high"   # 0.668 > 0.667


def test_transition_window_after_possession_change() -> None:
    """5-second window after a state change -> 'transition'."""
    fps = 30.0
    states_list = ["own"] * 30 + ["opp"] * 60  # turnover at frame 30
    ball_y = [0.6] * 90
    states = pd.Series(states_list, index=range(90))
    ball_y_s = pd.Series(ball_y, index=range(90))
    phases = label_phase(states, ball_y_s, fps=fps, transition_seconds=1.0)
    # Frame 30 turnover -> transition window 30..59
    assert phases.loc[35] == "transition"
    # Outside window: opp + y=0.6 is defend_low under the §6.1 0.667 split.
    assert phases.loc[80] == "defend_low"


def test_contested_passes_through() -> None:
    states = pd.Series(["contested", "own"], index=[0, 1])
    ball_y = pd.Series([0.5, 0.7], index=[0, 1])
    phases = label_phase(states, ball_y, fps=30.0)
    assert phases.loc[0] == "contested"


def test_turnover_fires_through_contested() -> None:
    """own -> contested -> opp is a real possession change; transition must fire
    at the first confirmed opp frame, and the contested frames keep 'contested'."""
    states = pd.Series(
        ["own"] * 10 + ["contested"] * 5 + ["opp"] * 20, index=range(35)
    )
    ball_y = pd.Series([0.2] * 35, index=range(35))  # own->build, opp->defend_low
    phases = label_phase(states, ball_y, fps=30.0, transition_seconds=0.2)  # 6-frame window
    assert phases.loc[15] == "transition"   # first committed opp frame
    assert phases.loc[10] == "contested"    # intervening frames untouched
    assert phases.loc[12] == "contested"
    assert phases.loc[9] == "build"         # before the turnover, no transition


def test_no_false_turnover_when_committed_label_unchanged() -> None:
    """own -> loose_ball -> own stays own (ffill), so NO transition fires."""
    states = pd.Series(
        ["own"] * 10 + ["loose_ball"] * 5 + ["own"] * 10, index=range(25)
    )
    ball_y = pd.Series([0.2] * 25, index=range(25))
    phases = label_phase(states, ball_y, fps=30.0, transition_seconds=1.0)
    assert "transition" not in set(phases.to_numpy())
    assert phases.loc[12] == "loose_ball"
