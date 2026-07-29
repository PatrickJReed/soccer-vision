"""Per-frame focal search for the physical engine: a pure 1-D minimizer over focal
length, decoupled from pose solving (the caller supplies the error function, so this
module never imports physical_calib — no cycle). Spec:
docs/superpowers/specs/2026-07-28-per-frame-focal-design.md §1.3-§1.4, §2. No I/O."""
from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass

SWEEP_LO_FRAC = 0.6
SWEEP_HI_FRAC = 1.6
N_COARSE = 9
REFINE_TOL_PX = 1.0
MIN_FOCAL_GAIN_FT = 0.15
MIN_ACCEPTED_FITS = 3

_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class FocalFit:
    """Result of one frame's focal sweep. `f` is the sweep's best focal; whether it
    is USED is the fallback ladder's call (session_focal), keyed on `constrained`."""

    f: float
    constrained: bool
    err_ft: float


def _golden(
    err_at: Callable[[float], float | None], lo: float, hi: float, tol: float
) -> tuple[float, float]:
    """Golden-section minimize on [lo, hi]; None evaluations count as +inf."""

    def ev(f: float) -> float:
        e = err_at(f)
        return math.inf if e is None else e

    a, b = lo, hi
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc, fd = ev(c), ev(d)
    while (b - a) > tol:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = ev(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = ev(d)
    return (c, fc) if fc <= fd else (d, fd)


def fit_frame_focal(
    err_at: Callable[[float], float | None], f_init: float
) -> FocalFit | None:
    """Coarse log sweep over [SWEEP_LO_FRAC, SWEEP_HI_FRAC]*f_init then golden-section
    refine. Accepted (`constrained=True`) only for an interior coarse minimum whose
    improvement over err_at(f_init) is >= MIN_FOCAL_GAIN_FT (spec §1.3). None if
    err_at solves nowhere on the sweep."""
    lo, hi = SWEEP_LO_FRAC * f_init, SWEEP_HI_FRAC * f_init
    cands = [lo * (hi / lo) ** (i / (N_COARSE - 1)) for i in range(N_COARSE)]
    errs = [err_at(f) for f in cands]
    pairs = [(e, i) for i, e in enumerate(errs) if e is not None]
    if not pairs:
        return None
    best_e, i_best = min(pairs)
    best_f = cands[i_best]
    interior = 0 < i_best < N_COARSE - 1
    if interior:
        f_star, e_star = _golden(err_at, cands[i_best - 1], cands[i_best + 1], REFINE_TOL_PX)
        if e_star <= best_e:
            best_f, best_e = f_star, e_star
    e_init = err_at(f_init)
    gain = (math.inf if e_init is None else e_init) - best_e
    return FocalFit(f=best_f, constrained=interior and gain >= MIN_FOCAL_GAIN_FT,
                    err_ft=best_e)


def session_focal(
    fits: Mapping[int, FocalFit | None], f_shared: float
) -> dict[int, tuple[float, str]]:
    """Fallback ladder (spec §1.4): constrained fits keep their focal ("fit"); the
    rest use the median of accepted fits ("median"); with < MIN_ACCEPTED_FITS
    accepted, EVERY frame uses f_shared ("shared") — exact pre-change behavior."""
    accepted = [ft.f for ft in fits.values() if ft is not None and ft.constrained]
    if len(accepted) < MIN_ACCEPTED_FITS:
        return {f: (f_shared, "shared") for f in fits}
    f_med = float(statistics.median(accepted))
    return {f: ((ft.f, "fit") if ft is not None and ft.constrained else (f_med, "median"))
            for f, ft in fits.items()}
