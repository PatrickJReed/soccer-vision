"""Phase splitter — labels each frame with one of:
build / attack / defend_low / defend_high / transition / contested / loose_ball / unknown.

Operates on the output of classify_possession() plus per-frame ball y-coord.
"""

from __future__ import annotations

import pandas as pd

OWN_THIRD_MAX_Y = 0.333
OPP_THIRD_MIN_Y = 0.667


def label_phase(
    possession_state: pd.Series,
    ball_y_pitch: pd.Series,
    fps: float,
    transition_seconds: float = 5.0,
    halftime_frame: int | None = None,
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
    halftime_frame
        Manual half-time frame (None = single-half clip, no flip). For frames at
        or after it, the thirds-mapping y is reflected (y -> 1-y) so "own goal at
        y=0" holds in both halves after the teams switch ends. Only the phase
        sub-labels are direction-dependent; possession_state and the homography
        are NOT (so they are unaffected). NaN stays NaN through the reflection.

    Returns
    -------
    Series of phase labels indexed by frame.
    """
    frames = possession_state.index
    transition_frames = round(transition_seconds * fps)

    # A turnover is a change of the LAST COMMITTED own/opp label: forward-fill the
    # own/opp labels through intervening contested/loose_ball/unknown, then diff. So
    # own -> contested -> opp fires at the first confirmed opp frame (a real youth
    # turnover), while own -> loose_ball -> own does NOT (committed stays own). The
    # notna() guards prevent firing on the first committed label (no prior possession).
    committed = possession_state.where(possession_state.isin(["own", "opp"])).ffill()
    prev_committed = committed.shift(1)
    is_turnover = committed.ne(prev_committed) & committed.notna() & prev_committed.notna()
    turnover_frames = frames[is_turnover.to_numpy()]

    # Attack-direction normalization: reflect the thirds-mapping y for the second
    # half so build<->attack and defend_low<->defend_high are correct after the
    # teams switch ends. 1.0 - NaN is NaN, so missing-ball frames stay 'unknown'.
    eff_ball_y = ball_y_pitch
    if halftime_frame is not None:
        flip = ball_y_pitch.index >= halftime_frame
        eff_ball_y = ball_y_pitch.mask(flip, 1.0 - ball_y_pitch)

    phases = pd.Series("unknown", index=frames)
    for fi in frames:
        st = possession_state.loc[fi]
        if st in ("contested", "loose_ball", "unknown"):
            phases.loc[fi] = st
            continue
        by = eff_ball_y.loc[fi] if fi in eff_ball_y.index else float("nan")
        if pd.isna(by):
            phases.loc[fi] = "unknown"
            continue
        if st == "own":
            phases.loc[fi] = "build" if by < OWN_THIRD_MAX_Y else "attack"
        else:  # "opp"
            # §6.1: defend_high = opp 1/3 (y > 0.667); defend_low = own 2/3.
            # Symmetric with the own-third build/attack split at OWN_THIRD_MAX_Y.
            phases.loc[fi] = "defend_high" if by > OPP_THIRD_MIN_Y else "defend_low"

    # Overlay transition windows
    for to_frame in turnover_frames:
        end = int(to_frame) + transition_frames
        affected = frames[(frames >= int(to_frame)) & (frames < end)]
        for fi in affected:
            # Don't overwrite contested/loose_ball/unknown
            if phases.loc[fi] in ("build", "attack", "defend_low", "defend_high"):
                phases.loc[fi] = "transition"

    return phases
