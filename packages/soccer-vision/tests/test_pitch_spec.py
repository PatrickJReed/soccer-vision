"""Tests for PitchSpec."""

from __future__ import annotations

import numpy as np
from soccer_vision.pitch.spec import PitchSpec, length_norm_xy


def test_default_is_9v9() -> None:
    spec = PitchSpec()
    assert spec.aspect_ratio == 1.5
    assert spec.n_outfield_per_team == 8


def test_standard_9v9_classmethod() -> None:
    spec = PitchSpec.standard_9v9()
    assert spec == PitchSpec()


def test_fifa_11v11_classmethod() -> None:
    spec = PitchSpec.fifa_11v11()
    assert spec.n_outfield_per_team == 10
    assert abs(spec.aspect_ratio - 1.54) < 0.01


def test_immutable() -> None:
    """PitchSpec is frozen."""
    spec = PitchSpec()
    try:
        spec.aspect_ratio = 2.0  # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("PitchSpec should be frozen")


def test_standard_9v9_has_goal_width_frac() -> None:
    spec = PitchSpec.standard_9v9()
    # US Soccer 9v9: ~6.4 m goal on a ~45.7 m wide field -> ~0.14 of width.
    assert 0.10 < spec.goal_width_frac < 0.20


def test_fifa_11v11_has_goal_width_frac() -> None:
    spec = PitchSpec.fifa_11v11()
    # 7.32 m goal on a 68 m wide field -> ~0.108 of width.
    assert 0.08 < spec.goal_width_frac < 0.14


def test_length_norm_xy_scales_x_by_aspect_ratio_leaves_y() -> None:
    spec = PitchSpec.standard_9v9()  # aspect_ratio = 1.5
    nx, ny = length_norm_xy(0.30, 0.40, spec)
    assert abs(float(nx) - 0.30 / 1.5) < 1e-12
    assert ny == 0.40


def test_length_norm_xy_honors_spec_aspect_ratio() -> None:
    nx_9, _ = length_norm_xy(0.30, 0.40, PitchSpec.standard_9v9())   # /1.5
    nx_11, _ = length_norm_xy(0.30, 0.40, PitchSpec.fifa_11v11())    # /1.54
    assert nx_11 < nx_9  # larger aspect_ratio -> smaller normalized x


def test_length_norm_xy_vectorizes_over_numpy_arrays() -> None:
    spec = PitchSpec.standard_9v9()
    x = np.array([0.0, 0.30, 0.75])
    y = np.array([0.10, 0.50, 0.90])
    nx, ny = length_norm_xy(x, y, spec)
    assert np.allclose(nx, x / 1.5)
    assert np.allclose(ny, y)


def test_length_norm_xy_makes_distance_isotropic() -> None:
    """A pure-x offset and the same-magnitude pure-y offset are physically equal
    only after normalization: x is width (1/aspect of y)."""
    spec = PitchSpec.standard_9v9()
    nx, _ = length_norm_xy(0.15, 0.0, spec)   # 0.15 of WIDTH
    _, ny = length_norm_xy(0.0, 0.10, spec)   # 0.10 of LENGTH == 0.15 of width physically
    assert abs(float(nx) - float(ny)) < 1e-12
