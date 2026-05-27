"""Schema for the canonical trajectories DataFrame.

Every TrackingBackend.process() return value MUST validate against this schema
before downstream code consumes it. Validation is fail-fast and explicit.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

REQUIRED_COLUMNS: Final = (
    "frame",
    "t_seconds",
    "track_id",
    "x_px",
    "y_px",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "class",
    "team",
    "conf",
)

ALLOWED_CLASSES: Final = frozenset({"player", "goalkeeper", "referee", "ball"})
ALLOWED_TEAMS: Final = frozenset({"own", "opp", "ref", "unknown"})


class TrajectorySchemaError(ValueError):
    """Raised when a DataFrame fails trajectories-schema validation."""


def validate_trajectories(df: pd.DataFrame) -> None:
    """Validate that df conforms to the trajectories schema.

    Raises TrajectorySchemaError on the first violation found.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise TrajectorySchemaError(f"trajectories DataFrame missing columns: {sorted(missing)}")

    if (df["frame"] < 0).any():
        raise TrajectorySchemaError("frame column contains negative values")

    if (df["t_seconds"] < 0).any():
        raise TrajectorySchemaError("t_seconds column contains negative values")

    bad_classes = set(df["class"].unique()) - ALLOWED_CLASSES
    if bad_classes:
        raise TrajectorySchemaError(
            f"class column contains unknown values: {sorted(bad_classes)}; "
            f"allowed: {sorted(ALLOWED_CLASSES)}"
        )

    bad_teams = set(df["team"].unique()) - ALLOWED_TEAMS
    if bad_teams:
        raise TrajectorySchemaError(
            f"team column contains unknown values: {sorted(bad_teams)}; "
            f"allowed: {sorted(ALLOWED_TEAMS)}"
        )

    if ((df["conf"] < 0) | (df["conf"] > 1)).any():
        raise TrajectorySchemaError("conf column contains values outside [0, 1]")
