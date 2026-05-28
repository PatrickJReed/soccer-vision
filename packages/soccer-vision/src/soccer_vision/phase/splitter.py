"""Phase splitter — labels each frame with one of:
build / attack / defend_low / defend_high / transition / contested / loose_ball / unknown.

Operates on the output of classify_possession() plus per-frame ball y-coord.
"""

from __future__ import annotations

import pandas as pd

OWN_THIRD_MAX_Y = 0.333
OPP_HALF_MIN_Y = 0.5
OPP_THIRD_MIN_Y = 0.667


def label_phase(
    possession_state: pd.Series,
    ball_y_pitch: pd.Series,
    fps: float,
    transition_seconds: float = 5.0,
) -> pd.Series:
    """Combine per-frame possession state + ball y-coord into a phase label.

    Parameters
    ----------
    possession_state
        Series of PossessionState literals indexed by frame.
    ball_y_pitch
        Series of y_pitch (in [0, 1]) of the ball, indexed by frame. NaN if missing.
    fps
        Frame rate; used to size the transition window in frames.
    transition_seconds
        Window after each possession change labeled 'transition'.

    Returns
    -------
    Series of phase labels indexed by frame.
    """
    frames = possession_state.index
    transition_frames = round(transition_seconds * fps)

    # Detect transitions: where state changes from own↔opp (not into/out of contested/loose/unknown)
    state_prev = possession_state.shift(1)
    is_turnover = (
        ((possession_state == "own") & (state_prev == "opp"))
        | ((possession_state == "opp") & (state_prev == "own"))
    )
    turnover_frames = frames[is_turnover.fillna(False).to_numpy()]

    phases = pd.Series("unknown", index=frames)
    for fi in frames:
        st = possession_state.loc[fi]
        if st in ("contested", "loose_ball", "unknown"):
            phases.loc[fi] = st
            continue
        by = ball_y_pitch.loc[fi] if fi in ball_y_pitch.index else float("nan")
        if pd.isna(by):
            phases.loc[fi] = "unknown"
            continue
        if st == "own":
            phases.loc[fi] = "build" if by < OWN_THIRD_MAX_Y else "attack"
        else:  # "opp"
            phases.loc[fi] = "defend_high" if by > OPP_HALF_MIN_Y else "defend_low"

    # Overlay transition windows
    for to_frame in turnover_frames:
        end = int(to_frame) + transition_frames
        affected = frames[(frames >= int(to_frame)) & (frames < end)]
        for fi in affected:
            # Don't overwrite contested/loose_ball/unknown
            if phases.loc[fi] in ("build", "attack", "defend_low", "defend_high"):
                phases.loc[fi] = "transition"

    return phases
