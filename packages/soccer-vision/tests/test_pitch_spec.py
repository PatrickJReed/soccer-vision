"""Tests for PitchSpec."""

from __future__ import annotations

from soccer_vision.pitch.spec import PitchSpec


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
