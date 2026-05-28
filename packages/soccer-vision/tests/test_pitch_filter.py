"""Tests for pitch-boundary filter."""

from __future__ import annotations

import pandas as pd
from soccer_vision.pitch.filter import filter_outside_pitch


def test_default_keeps_in_bounds_rows() -> None:
    df = pd.DataFrame({
        "frame": [0, 0, 0],
        "x_pitch": [0.1, 0.5, 0.9],
        "y_pitch": [0.2, 0.5, 0.8],
    })
    out = filter_outside_pitch(df)
    assert len(out) == 3


def test_drops_clearly_out_of_bounds() -> None:
    df = pd.DataFrame({
        "frame": [0, 0, 0],
        "x_pitch": [0.5, -0.5, 1.5],
        "y_pitch": [0.5, 0.5, 0.5],
    })
    out = filter_outside_pitch(df)
    assert len(out) == 1
    assert out["x_pitch"].iloc[0] == 0.5


def test_margin_allows_slightly_off() -> None:
    df = pd.DataFrame({
        "frame": [0],
        "x_pitch": [-0.05],
        "y_pitch": [0.5],
    })
    out_no_margin = filter_outside_pitch(df, margin=0.0)
    out_margin_10 = filter_outside_pitch(df, margin=0.1)
    assert len(out_no_margin) == 0
    assert len(out_margin_10) == 1


def test_nan_rows_dropped() -> None:
    df = pd.DataFrame({
        "frame": [0, 0],
        "x_pitch": [0.5, float("nan")],
        "y_pitch": [0.5, 0.5],
    })
    out = filter_outside_pitch(df)
    assert len(out) == 1
