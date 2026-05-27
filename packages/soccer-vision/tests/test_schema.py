"""Tests for the trajectories DataFrame schema."""

from __future__ import annotations

import pandas as pd
import pytest
from soccer_vision.io.schema import REQUIRED_COLUMNS, TrajectorySchemaError, validate_trajectories


def _good_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "t_seconds": [0.0, 0.0, 0.033],
            "track_id": [1, 2, 1],
            "x_px": [100.0, 200.0, 102.0],
            "y_px": [300.0, 350.0, 303.0],
            "bbox_x1": [90.0, 190.0, 92.0],
            "bbox_y1": [280.0, 330.0, 283.0],
            "bbox_x2": [110.0, 210.0, 112.0],
            "bbox_y2": [320.0, 370.0, 323.0],
            "class": ["player", "player", "player"],
            "team": ["own", "opp", "own"],
            "conf": [0.95, 0.92, 0.94],
        }
    )


def test_required_columns_listed_exhaustively() -> None:
    assert set(REQUIRED_COLUMNS) == {
        "frame", "t_seconds", "track_id",
        "x_px", "y_px",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "class", "team", "conf",
    }


def test_validate_accepts_good_df() -> None:
    validate_trajectories(_good_df())  # should not raise


def test_validate_rejects_missing_column() -> None:
    df = _good_df().drop(columns=["conf"])
    with pytest.raises(TrajectorySchemaError, match=r"missing.*conf"):
        validate_trajectories(df)


def test_validate_rejects_unknown_class() -> None:
    df = _good_df()
    df.loc[0, "class"] = "horse"
    with pytest.raises(TrajectorySchemaError, match=r"class.*horse"):
        validate_trajectories(df)


def test_validate_rejects_unknown_team() -> None:
    df = _good_df()
    df.loc[0, "team"] = "blue"
    with pytest.raises(TrajectorySchemaError, match=r"team.*blue"):
        validate_trajectories(df)


def test_validate_rejects_negative_frame() -> None:
    df = _good_df()
    df.loc[0, "frame"] = -1
    with pytest.raises(TrajectorySchemaError, match="frame"):
        validate_trajectories(df)


def test_validate_rejects_bad_conf() -> None:
    df = _good_df()
    df.loc[0, "conf"] = 1.5
    with pytest.raises(TrajectorySchemaError, match="conf"):
        validate_trajectories(df)
