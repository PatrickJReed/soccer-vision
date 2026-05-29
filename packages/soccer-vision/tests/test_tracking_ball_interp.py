"""Tests for interpolate_ball_gaps() — bridging short ball-trajectory holes."""

from __future__ import annotations

import pandas as pd
import pytest
from soccer_vision.io.schema import validate_trajectories
from soccer_vision.tracking.roboflow import RoboflowBackend, interpolate_ball_gaps

FPS = 30.0


def _ball_row(frame: int, x: float, y: float, conf: float = 0.5) -> dict[str, object]:
    """A schema-complete ball detection row at integer pixel center (x, y)."""
    return {
        "frame":     frame,
        "t_seconds": frame / FPS,
        "track_id":  -1_000_000 - frame,
        "x_px":      x,
        "y_px":      y,
        "bbox_x1":   x - 1.0,
        "bbox_y1":   y - 1.0,
        "bbox_x2":   x + 1.0,
        "bbox_y2":   y + 1.0,
        "class":     "ball",
        "team":      "unknown",
        "conf":      conf,
    }


def _df(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_fills_single_frame_gap() -> None:
    df = _df([_ball_row(0, 0.0, 0.0), _ball_row(2, 10.0, 20.0)])
    out = interpolate_ball_gaps(df, fps=FPS)
    assert set(out["frame"]) == {0, 1, 2}
    mid = out[out["frame"] == 1].iloc[0]
    assert mid["x_px"] == pytest.approx(5.0)
    assert mid["y_px"] == pytest.approx(10.0)


def test_interpolated_rows_are_marked_conf_zero() -> None:
    df = _df([_ball_row(0, 0.0, 0.0, conf=0.3), _ball_row(4, 8.0, 0.0, conf=0.3)])
    out = interpolate_ball_gaps(df, fps=FPS)
    filled = out[~out["frame"].isin({0, 4})]
    assert (filled["conf"] == 0.0).all()
    # Real detections retain their confidence.
    assert (out[out["frame"].isin({0, 4})]["conf"] == 0.3).all()


def test_linear_interpolation_is_correct_across_multi_frame_gap() -> None:
    df = _df([_ball_row(10, 0.0, 100.0), _ball_row(14, 40.0, 100.0)])
    out = interpolate_ball_gaps(df, fps=FPS).sort_values("frame")
    xs = out.set_index("frame")["x_px"]
    assert xs[11] == pytest.approx(10.0)
    assert xs[12] == pytest.approx(20.0)
    assert xs[13] == pytest.approx(30.0)
    # t_seconds stays consistent with frame index.
    assert out[out["frame"] == 12].iloc[0]["t_seconds"] == pytest.approx(12 / FPS)


def test_gap_longer_than_max_is_left_as_hole() -> None:
    # gap of 20 missing frames, max=15 → untouched
    df = _df([_ball_row(0, 0.0, 0.0), _ball_row(21, 0.0, 0.0)])
    out = interpolate_ball_gaps(df, fps=FPS, max_gap_frames=15)
    assert set(out["frame"]) == {0, 21}


def test_gap_exactly_at_max_is_filled() -> None:
    # 15 missing frames (1..15) with frames 0 and 16 present
    df = _df([_ball_row(0, 0.0, 0.0), _ball_row(16, 16.0, 0.0)])
    out = interpolate_ball_gaps(df, fps=FPS, max_gap_frames=15)
    assert len(out) == 17
    assert set(out["frame"]) == set(range(17))


def test_non_ball_rows_pass_through_untouched() -> None:
    player = {
        "frame": 1, "t_seconds": 1 / FPS, "track_id": 7,
        "x_px": 500.0, "y_px": 600.0,
        "bbox_x1": 490.0, "bbox_y1": 580.0, "bbox_x2": 510.0, "bbox_y2": 620.0,
        "class": "player", "team": "own", "conf": 0.9,
    }
    df = _df([_ball_row(0, 0.0, 0.0), player, _ball_row(2, 10.0, 0.0)])
    out = interpolate_ball_gaps(df, fps=FPS)
    players = out[out["class"] == "player"]
    assert len(players) == 1
    assert players.iloc[0]["track_id"] == 7
    # Exactly one interpolated ball row was added at frame 1.
    assert len(out[(out["class"] == "ball") & (out["frame"] == 1)]) == 1


def test_multiple_detections_in_a_frame_anchor_on_highest_conf() -> None:
    df = _df([
        _ball_row(0, 0.0, 0.0, conf=0.2),
        _ball_row(0, 100.0, 0.0, conf=0.8),  # higher conf → the anchor
        _ball_row(2, 0.0, 0.0, conf=0.5),
    ])
    out = interpolate_ball_gaps(df, fps=FPS)
    mid = out[(out["frame"] == 1) & (out["conf"] == 0.0)].iloc[0]
    # Anchors are (100, 0) and (0, 0) → midpoint x = 50, not 0.
    assert mid["x_px"] == pytest.approx(50.0)
    # Both original frame-0 detections are preserved.
    assert len(out[(out["class"] == "ball") & (out["frame"] == 0)]) == 2


def test_result_is_schema_conformant() -> None:
    df = _df([_ball_row(0, 1.0, 2.0), _ball_row(5, 11.0, 12.0)])
    out = interpolate_ball_gaps(df, fps=FPS)
    validate_trajectories(out)  # must not raise


def test_empty_df_passes_through() -> None:
    empty = RoboflowBackend  # sanity: import works
    assert empty is not None
    df = pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in ("x_px", "y_px")}
        | {"frame": pd.Series(dtype="int64"), "class": pd.Series(dtype="object")}
    )
    # An empty frame with the columns present should pass straight through.
    out = interpolate_ball_gaps(df, fps=FPS)
    assert out.empty


def test_single_ball_frame_is_a_noop() -> None:
    df = _df([_ball_row(3, 5.0, 5.0)])
    out = interpolate_ball_gaps(df, fps=FPS)
    assert len(out) == 1


# --- constructor knobs --------------------------------------------------------


def test_default_ball_inference_knobs() -> None:
    backend = RoboflowBackend()
    assert backend.ball_imgsz == 1280
    assert backend.ball_conf == 0.05
    assert backend.ball_max_gap_frames == 15


def test_ball_knobs_are_overridable() -> None:
    backend = RoboflowBackend(ball_imgsz=960, ball_conf=0.1, ball_max_gap_frames=8)
    assert backend.ball_imgsz == 960
    assert backend.ball_conf == 0.1
    assert backend.ball_max_gap_frames == 8
