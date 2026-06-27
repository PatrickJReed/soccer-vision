"""5-state per-frame possession classifier (own/opp/contested/loose_ball/unknown).

Distances are isotropic length-normalized pitch units (x_len = x_pitch /
aspect_ratio, y_len = y_pitch), the same convention hygiene/core.py uses, via
the shared pitch.spec.length_norm_xy helper. The §3.3 thresholds are already in
pitch-LENGTH units, so only the metric they apply to is corrected.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd

from soccer_vision.pitch.spec import PitchSpec, length_norm_xy

PossessionState = Literal["own", "opp", "contested", "loose_ball", "unknown"]


@dataclass(frozen=True)
class PossessionThresholds:
    """Thresholds in pitch-units (fractions of pitch length)."""

    margin: float = 0.022  # contested if |nearest_own - nearest_opp| < margin
    clump_radius: float = 0.044  # players within this radius count toward clump
    loose_ball_radius: float = 0.073  # ball is loose if no one within this radius


def classify_possession(
    detections: pd.DataFrame,
    thresholds: PossessionThresholds | None = None,
    pitch_spec: PitchSpec | None = None,
) -> pd.Series:
    """Classify possession state for each frame in `detections`.

    Distances are length-normalized (isotropic pitch-LENGTH units) via the
    aspect_ratio of `pitch_spec` (default 9v9). The ball used per frame is the
    single highest-confidence ball detection (multiple low-conf balls are
    possible at the conf floor); the same ball the pipeline uses for phase.

    Returns a pd.Series indexed by frame, values from PossessionState literals.
    """
    th = thresholds or PossessionThresholds()
    spec = pitch_spec or PitchSpec.standard_9v9()
    states: dict[int, PossessionState] = {}

    for frame_idx, group in detections.groupby("frame", sort=False):
        fkey: int = cast(int, frame_idx)
        ball_rows = group[group["class"] == "ball"]
        if ball_rows.empty:
            states[fkey] = "unknown"
            continue
        ball = ball_rows.loc[ball_rows["conf"].idxmax()]  # atomic highest-conf row
        bx = ball["x_pitch"]
        by = ball["y_pitch"]
        if pd.isna(bx) or pd.isna(by):
            states[fkey] = "unknown"
            continue

        players = group[group["class"].isin(["player", "goalkeeper"])]
        if players.empty:
            states[fkey] = "unknown"
            continue

        px_n, py_n = length_norm_xy(
            players["x_pitch"].to_numpy(), players["y_pitch"].to_numpy(), spec
        )
        bx_n, by_n = length_norm_xy(float(bx), float(by), spec)
        dists = np.hypot(px_n - bx_n, py_n - by_n)  # isotropic pitch-LENGTH units
        teams = players["team"].to_numpy()

        own_mask = teams == "own"
        opp_mask = teams == "opp"
        d_own = dists[own_mask].min() if own_mask.any() else np.inf
        d_opp = dists[opp_mask].min() if opp_mask.any() else np.inf

        if min(d_own, d_opp) > th.loose_ball_radius:
            states[fkey] = "loose_ball"
            continue

        clump_own = int((dists[own_mask] <= th.clump_radius).sum() if own_mask.any() else 0)
        clump_opp = int((dists[opp_mask] <= th.clump_radius).sum() if opp_mask.any() else 0)
        if abs(d_own - d_opp) < th.margin or (
            clump_own >= 1 and clump_opp >= 1 and abs(clump_own - clump_opp) <= 1
        ):
            states[fkey] = "contested"
            continue

        states[fkey] = "own" if d_own < d_opp else "opp"

    return pd.Series(states, name="possession_state")


def smooth_possession(possession_state: pd.Series, window_frames: int) -> pd.Series:
    """Mode-smooth the per-frame possession series over a centered window.

    A frame whose raw state is 'contested' is preserved as 'contested'
    (spec §6.1). Every other frame takes the modal state of the surrounding
    window_frames-wide window, computed over non-'contested' neighbours only
    (so a clear-possession frame beside a contested scramble keeps its label,
    while contested frames still pass through untouched). If the window has no
    non-'contested' neighbours, the frame is left as-is. Ties break by first
    occurrence. Operates positionally, so callers should pass a series sorted
    by frame.
    """
    if window_frames <= 1 or len(possession_state) == 0:
        return possession_state.copy()
    states = possession_state.to_list()
    n = len(states)
    half = window_frames // 2
    out: list[str] = []
    for i in range(n):
        if states[i] == "contested":
            out.append("contested")
            continue
        window = [
            s for s in states[max(0, i - half):min(n, i + half + 1)]
            if s != "contested"
        ]
        out.append(Counter(window).most_common(1)[0][0] if window else states[i])
    return pd.Series(out, index=possession_state.index, name=possession_state.name)
