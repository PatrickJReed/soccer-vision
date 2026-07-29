"""Tests for the pure per-frame focal search (pitch/focal.py)."""

from __future__ import annotations

import math
from collections.abc import Callable

from soccer_vision.pitch.focal import (
    MIN_ACCEPTED_FITS,
    MIN_FOCAL_GAIN_FT,
    N_COARSE,
    FocalFit,
    fit_frame_focal,
    session_focal,
)


def _bowl(f_true: float, floor: float = 0.2) -> Callable[[float], float]:
    """Quadratic error bowl in log-focal with a clear minimum at f_true."""
    def err_at(f: float) -> float:
        return floor + 40.0 * math.log(f / f_true) ** 2
    return err_at


def test_recovers_bowl_minimum_within_tolerance() -> None:
    fit = fit_frame_focal(_bowl(1450.0), f_init=1300.0)
    assert fit is not None and fit.constrained
    assert abs(fit.f - 1450.0) / 1450.0 < 0.005
    assert fit.err_ft < 0.25


def test_edge_pinned_curve_is_unconstrained() -> None:
    # Monotone decreasing toward the high edge: minimum sits at the sweep boundary.
    fit = fit_frame_focal(lambda f: 10.0 - f / 1000.0, f_init=1000.0)
    assert fit is not None and not fit.constrained


def test_flat_curve_below_gain_is_unconstrained() -> None:
    fit = fit_frame_focal(_bowl(1000.0, floor=1.0), f_init=1001.0)
    # err(f_init) is within MIN_FOCAL_GAIN_FT of the minimum -> no real gain
    assert fit is not None and not fit.constrained


def test_none_regions_are_skipped() -> None:
    def err_at(f: float) -> float | None:
        return None if f < 1200.0 else _bowl(1450.0)(f)
    fit = fit_frame_focal(err_at, f_init=1300.0)
    assert fit is not None and fit.constrained
    assert abs(fit.f - 1450.0) / 1450.0 < 0.005


def test_all_none_returns_none() -> None:
    assert fit_frame_focal(lambda f: None, f_init=1300.0) is None


def test_evaluation_budget_is_bounded() -> None:
    calls = 0
    bowl = _bowl(1450.0)

    def counting(f: float) -> float:
        nonlocal calls
        calls += 1
        return bowl(f)

    fit_frame_focal(counting, f_init=1300.0)
    assert calls <= N_COARSE + 30  # coarse sweep + bounded golden refine + init eval


def test_session_focal_ladder_fit_and_median() -> None:
    fits = {
        0: FocalFit(f=1400.0, constrained=True, err_ft=0.5),
        1: FocalFit(f=1500.0, constrained=True, err_ft=0.5),
        2: FocalFit(f=1600.0, constrained=True, err_ft=0.5),
        3: FocalFit(f=999.0, constrained=False, err_ft=9.9),   # unconstrained
        4: None,                                                # < 6 ids
    }
    out = session_focal(fits, f_shared=1234.0)
    assert out[0] == (1400.0, "fit") and out[2] == (1600.0, "fit")
    assert out[3] == (1500.0, "median") and out[4] == (1500.0, "median")


def test_session_focal_falls_back_to_shared_below_min_fits() -> None:
    assert MIN_ACCEPTED_FITS == 3
    fits = {
        0: FocalFit(f=1400.0, constrained=True, err_ft=0.5),
        1: FocalFit(f=1500.0, constrained=True, err_ft=0.5),
        2: None,
    }
    out = session_focal(fits, f_shared=1234.0)
    assert out == {0: (1234.0, "shared"), 1: (1234.0, "shared"), 2: (1234.0, "shared")}


def test_gain_threshold_constant() -> None:
    assert MIN_FOCAL_GAIN_FT == 0.15
