"""Phase 2 tests: field-lines registry, line residual, line-constrained pose refine."""

from __future__ import annotations

import numpy as np
import pytest
from soccer_vision.calib.field_model import (
    FIELD_LINES,
    LENGTH_M,
    WIDTH_M,
    field_line_3d,
    field_points_3d,
)


def test_field_line_3d_near_touchline_endpoints() -> None:
    p1, p2 = field_line_3d("near_touchline")
    fp = field_points_3d()
    assert np.allclose(p1, fp[0])  # corner_own_left
    assert np.allclose(p2, fp[2])  # corner_opp_left
    assert p1[2] == 0.0 and p2[2] == 0.0  # on the Z=0 plane


def test_field_line_3d_midline_spans_full_width() -> None:
    # midline = (5, 4): halfway_near (x=0) -> halfway_far (x=WIDTH), both at y=L/2
    p1, p2 = field_line_3d("midline")
    assert np.allclose(p1, [0.0, LENGTH_M / 2, 0.0])
    assert np.allclose(p2, [WIDTH_M, LENGTH_M / 2, 0.0])


def test_field_lines_registry_has_five_named_lines() -> None:
    assert set(FIELD_LINES) == {
        "near_touchline", "far_touchline", "own_goal_line", "opp_goal_line", "midline",
    }


def test_field_line_3d_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        field_line_3d("not_a_line")
