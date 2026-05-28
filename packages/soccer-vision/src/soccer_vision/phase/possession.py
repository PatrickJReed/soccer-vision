"""5-state per-frame possession classifier (own/opp/contested/loose_ball/unknown)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd

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
) -> pd.Series:
    """Classify possession state for each frame in `detections`.

    Returns a pd.Series indexed by frame, values from PossessionState literals.
    """
    th = thresholds or PossessionThresholds()
    states: dict[int, PossessionState] = {}

    for frame_idx, group in detections.groupby("frame", sort=False):
        fkey: int = cast(int, frame_idx)
        ball_rows = group[group["class"] == "ball"]
        if ball_rows.empty:
            states[fkey] = "unknown"
            continue
        bx = ball_rows["x_pitch"].iloc[0]
        by = ball_rows["y_pitch"].iloc[0]
        if pd.isna(bx) or pd.isna(by):
            states[fkey] = "unknown"
            continue

        players = group[group["class"].isin(["player", "goalkeeper"])]
        if players.empty:
            states[fkey] = "unknown"
            continue

        dx = players["x_pitch"].to_numpy() - bx
        dy = players["y_pitch"].to_numpy() - by
        dists = np.sqrt(dx * dx + dy * dy)
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
