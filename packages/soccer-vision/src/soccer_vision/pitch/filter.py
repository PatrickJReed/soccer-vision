"""Pitch-boundary filter — drops detections whose projected pitch coords are off-pitch."""

from __future__ import annotations

import pandas as pd


def filter_outside_pitch(
    detections: pd.DataFrame,
    margin: float = 0.05,
) -> pd.DataFrame:
    """Drop rows whose (x_pitch, y_pitch) is outside [-margin, 1+margin]².

    Rows with NaN in either pitch coord are dropped (homography missing for the frame).

    Parameters
    ----------
    detections
        DataFrame with x_pitch and y_pitch columns.
    margin
        Slack around the unit square; tolerates near-boundary detections.

    Returns
    -------
    A new DataFrame with the out-of-bounds and NaN rows removed.
    """
    if "x_pitch" not in detections.columns or "y_pitch" not in detections.columns:
        raise ValueError("detections must have x_pitch and y_pitch columns")
    lo = -margin
    hi = 1.0 + margin
    mask = (
        detections["x_pitch"].between(lo, hi, inclusive="both")
        & detections["y_pitch"].between(lo, hi, inclusive="both")
    )
    return detections[mask].reset_index(drop=True)
