"""Per-track modal team aggregation — smooths per-frame team prediction noise."""

from __future__ import annotations

import pandas as pd

PLAYER_CLASSES = frozenset({"player", "goalkeeper"})


def apply_modal_team_per_track(detections: pd.DataFrame) -> pd.DataFrame:
    """Replace each player/GK row's team with the modal value for its track_id.

    Referees and ball rows are left untouched. Ties produce 'unknown'.

    Parameters
    ----------
    detections
        DataFrame with at least `track_id`, `class`, `team` columns.

    Returns
    -------
    A new DataFrame with team smoothed per-track for player/GK rows.
    """
    out = detections.copy()
    player_mask = out["class"].isin(PLAYER_CLASSES)

    def modal_or_unknown(group: pd.Series) -> str:
        counts = group.value_counts()
        if len(counts) >= 2 and counts.iloc[0] == counts.iloc[1]:
            return "unknown"
        return str(counts.idxmax())

    modes = (
        out[player_mask]
        .groupby("track_id")["team"]
        .apply(modal_or_unknown)
    )
    out.loc[player_mask, "team"] = out.loc[player_mask, "track_id"].map(modes)
    return out
