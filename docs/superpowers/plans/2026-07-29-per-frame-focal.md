# Per-Frame Focal Engine Change Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the physical engine's single shared focal with per-anchor focal fitting and fix the K-before-outlier-flagging ordering bug, per `docs/superpowers/specs/2026-07-28-per-frame-focal-design.md` (the spec is the contract — read it first).

**Architecture:** New pure 1-D search module `pitch/focal.py` (caller supplies an error closure — no import cycle); `solve_session` reorders to two-pass outlier flagging then per-frame focal selection with a fallback ladder; `foreground_holdout` re-selects focal from held-out evidence only; focal metadata flows into `PhysicalCalib`, `calib_gate.json`, and the `validate_session` report. Evidence-gated against a frozen real sidecar and the shipped oceanside session.

**Tech Stack:** Python 3.12, numpy, OpenCV (SQPNP), pytest; repo gates: bare `uv run mypy` (strict; 58 pre-existing errors allowed, no new ones), `uv run ruff check`, full pytest suite (baseline 488 passed, 3 skipped). All commands run from `packages/soccer-vision/`. Work lands directly on `master` (repo convention).

**Conventions the engineer must know:**
- Clicks are stored NORMALIZED ([0,1]); pixel = `c.x * w, c.y * h`. `anchor_h` matrices map NORMALIZED image coords → pitch [0,1]².
- `physical_calib.py` helpers used throughout: `_apply(h, pts)` (projective apply), `_point_feet(q, kp_idx)`, `_line_perp_feet(q, line_id)`, `_group(items)` (by `.frame`), `_anchor_pose(k, po, lo, seed)`, `frame_homography(k, rvec, tvec)` (from `calib_anchor`).
- `po` = `list[tuple[int, float, float]]` of (kp_idx, x_px, y_px); `lo` = `list[tuple[str, float, float]]` of (line_id, x_px, y_px).
- Landmark ids: 0–20 (`pitch/landmarks.py` LANDMARK_NAMES); near-touchline point ids are {0, 2}.

---

### Task 0: Baseline evidence capture (NO engine code — must land before Task 2)

**Files:**
- Create: `docs/superpowers/2026-07-29-per-frame-focal-evidence.md`
- Create (scratchpad, NOT committed): `/private/tmp/claude-501/-Users-patrickreed-Sandbox-soccer-vision/5ae1a3e6-7ea8-45b2-87b4-3aabfa8c623d/scratchpad/engine_metrics.py`

- [ ] **Step 1: Write the metrics script** (works unchanged before AND after the engine change — it only uses `solve_session` outputs):

```python
"""Frozen-sidecar engine metrics: anchors, green count, median in-sample error (ft)."""
import json

import numpy as np

from soccer_vision.pitch import physical_calib as pc
from soccer_vision.pitch.manual_anchor import Click, LineClick

SIDE = "/Users/patrickreed/sv-labeler/home_g4_oceanside/rep_video.clicks.frozen-2026-07-24.json"
W, H = SIZE = (1920, 1080)

d = json.load(open(SIDE))
points = [Click(c["frame"], c["kp_idx"], c["x"], c["y"]) for c in d["clicks"]]
lines = [LineClick(c["frame"], c["line_id"], c["x"], c["y"]) for c in d["line_clicks"]]
calib = pc.solve_session(points, lines, SIZE, {})
greens = sum(1 for g in calib.coverage_grade.values() if g == "green")
by_pt: dict[int, list[Click]] = {}
for c in points:
    by_pt.setdefault(c.frame, []).append(c)
by_ln: dict[int, list[LineClick]] = {}
for lc in lines:
    by_ln.setdefault(lc.frame, []).append(lc)
errs: list[float] = []
for f, hmat in calib.anchor_h.items():
    for c in by_pt.get(f, []):
        errs.append(pc._point_feet(pc._apply(hmat, np.array([[c.x, c.y]]))[0], c.kp_idx))
    for lc in by_ln.get(f, []):
        errs.append(pc._line_perp_feet(pc._apply(hmat, np.array([[lc.x, lc.y]]))[0], lc.line_id))
print(f"anchors: {len(calib.anchor_h)}  green: {greens}  "
      f"median in-sample err: {float(np.median(errs)):.2f} ft  K: {calib.K[0,0]:.1f}px")
if getattr(calib, "focal_of", None):
    vals = sorted(calib.focal_of.values())
    src = getattr(calib, "focal_source", {})
    print(f"focal: {sum(1 for s in src.values() if s=='fit')} fit / "
          f"{sum(1 for s in src.values() if s=='median')} median / "
          f"{sum(1 for s in src.values() if s=='shared')} shared | "
          f"spread p90/p10 {np.percentile(vals, 90)/np.percentile(vals, 10):.3f}")
```

- [ ] **Step 2: Run it (BEFORE any engine change):**

Run: `cd packages/soccer-vision && uv run python /private/tmp/claude-501/-Users-patrickreed-Sandbox-soccer-vision/5ae1a3e6-7ea8-45b2-87b4-3aabfa8c623d/scratchpad/engine_metrics.py`
Expected: one line like `anchors: 112  green: 86  median in-sample err: ~3.3-4.0 ft  K: 1444.1px` (the 2026-07-24 audit found 86 green and shared K 1444.1 with the poison clicks in). Record the EXACT output.

- [ ] **Step 3: Run the shipped-session baseline:**

Run: `cd packages/soccer-vision && uv run python -m soccer_vision.pitch.validate_session --chain ~/sv-labeler/.sv_labeler_cache/da63d2bb640cc974.npz --clicks ~/sv-labeler/.sv_labeler_cache/oceanside_clip.clicks.json --engine physical`
If the chain npz doesn't match (loud error), try the other npz files in `~/sv-labeler/.sv_labeler_cache/` (candidates: `50bf81d981ceac3f.npz`, `95c1bd4c8ceaa267.npz`, `553c4a1859e6fd60.npz`, `e03fb2ca017e29a5.npz`, `ef2546eaddd5e6fc.npz`) until one loads. Record the fg/prop gate numbers printed.

- [ ] **Step 4: Write the evidence doc** `docs/superpowers/2026-07-29-per-frame-focal-evidence.md` with a `## Baseline (shared-K engine)` section containing both verbatim outputs and the exact commands used (including which chain npz matched). Leave a `## After (per-frame focal)` section header with the text "(filled by Task 5)".

- [ ] **Step 5: Commit:**

```bash
git add docs/superpowers/2026-07-29-per-frame-focal-evidence.md
git commit -m "docs(pitch): baseline evidence for per-frame focal change"
```

---

### Task 1: `pitch/focal.py` — pure 1-D focal search

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/focal.py`
- Test: `packages/soccer-vision/tests/test_focal.py`

- [ ] **Step 1: Write the failing tests:**

```python
"""Tests for the pure per-frame focal search (pitch/focal.py)."""

from __future__ import annotations

import math

from soccer_vision.pitch.focal import (
    MIN_ACCEPTED_FITS,
    MIN_FOCAL_GAIN_FT,
    N_COARSE,
    FocalFit,
    fit_frame_focal,
    session_focal,
)


def _bowl(f_true: float, floor: float = 0.2):
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
```

- [ ] **Step 2: Run to verify failure:**

Run: `cd packages/soccer-vision && uv run pytest tests/test_focal.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'soccer_vision.pitch.focal'`

- [ ] **Step 3: Implement `src/soccer_vision/pitch/focal.py`:**

```python
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
```

- [ ] **Step 4: Run tests, lint, types:**

Run: `cd packages/soccer-vision && uv run pytest tests/test_focal.py -q && uv run ruff check src/soccer_vision/pitch/focal.py tests/test_focal.py && uv run mypy src/soccer_vision/pitch/focal.py`
Expected: all tests PASS; ruff clean; mypy clean.

- [ ] **Step 5: Commit:**

```bash
git add src/soccer_vision/pitch/focal.py tests/test_focal.py
git commit -m "feat(pitch): pure per-frame focal search (focal.py) — sweep + golden refine + fallback ladder"
```

---

### Task 2: `solve_session` reorder + per-frame focal + `PhysicalCalib` surface

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/physical_calib.py` (imports; `PhysicalCalib` dataclass ~line 206; `solve_session` ~line 284; new `_frame_err_at` helper)
- Test: `packages/soccer-vision/tests/test_physical_calib.py` (append a new section)

- [ ] **Step 1: Read first:** the spec §1/§3; `physical_calib.py` in full; `tests/test_physical_calib.py` lines 1–80 (the synthetic world helpers `_pose_clicks`, `SIZE`, the shared-focal world at 1460) and lines 141–160 (`_near_tl_clicks`, `_pose_h`).

- [ ] **Step 2: Write the failing tests** (append to `tests/test_physical_calib.py`; reuse its existing imports — add `from soccer_vision.pitch.focal import FocalFit` and `import soccer_vision.pitch.physical_calib as pc_mod` if not present):

```python
# ---- per-frame focal (spec 2026-07-28) ----
FOCALS_MZ = {0: 1330.0, 1: 1450.0, 2: 1580.0}


def _mz_k(focal: float) -> NDArray[np.float64]:
    return np.array([[focal, 0, SIZE[0] / 2], [0, focal, SIZE[1] / 2], [0, 0, 1.0]])


def _mz_clicks(frame: int, focal: float, rvec: NDArray[np.float64],
               tvec: NDArray[np.float64]) -> list[Click]:
    fp = field_points_3d()
    px = cv2.projectPoints(fp, rvec, tvec, _mz_k(focal), np.zeros(5))[0].reshape(-1, 2)
    return [Click(frame, j, float(px[j, 0]) / SIZE[0], float(px[j, 1]) / SIZE[1])
            for j in range(21)
            if j != 5 and 0 < px[j, 0] < SIZE[0] and 0 < px[j, 1] < SIZE[1]]


def _mz_lines(frame: int, focal: float, rvec: NDArray[np.float64],
              tvec: NDArray[np.float64], n: int = 3) -> list[LineClick]:
    obj = np.array([[0.0, y, 0.0] for y in np.linspace(5.0, LENGTH_M - 5.0, n)])
    px = cv2.projectPoints(obj, rvec, tvec, _mz_k(focal), np.zeros(5))[0].reshape(-1, 2)
    return [LineClick(frame, "near_touchline", float(x) / SIZE[0], float(y) / SIZE[1])
            for x, y in px if 0 < x < SIZE[0] and 0 < y < SIZE[1]]


def _mz_session() -> tuple[list[Click], list[LineClick]]:
    """Three distinct poses, each rendered at a DIFFERENT true focal."""
    clicks: list[Click] = []
    lines: list[LineClick] = []
    for f, (eye_dy, look_dy) in enumerate([(-6.0, -8.0), (0.0, 0.0), (6.0, 8.0)]):
        rvec, tvec = _look_at((-8.0, 34.0 + eye_dy, 9.0), (22.85, 34.0 + look_dy, 0.0))
        clicks += _mz_clicks(f, FOCALS_MZ[f], rvec, tvec)
        lines += _mz_lines(f, FOCALS_MZ[f], rvec, tvec)
    return clicks, lines


def test_per_frame_focal_recovery_multizoom() -> None:
    clicks, lines = _mz_session()
    calib = solve_session(clicks, lines, SIZE, {})
    assert set(calib.focal_of) == {0, 1, 2}
    n_fit = sum(1 for s in calib.focal_source.values() if s == "fit")
    assert n_fit >= 2  # the middle frame may legitimately match the shared estimate
    for f, src in calib.focal_source.items():
        if src == "fit":
            assert abs(calib.focal_of[f] - FOCALS_MZ[f]) / FOCALS_MZ[f] < 0.02
    assert all(g == "green" for g in calib.coverage_grade.values())


def test_frame_k_accessor() -> None:
    clicks, lines = _mz_session()
    calib = solve_session(clicks, lines, SIZE, {})
    k0 = calib.frame_K(0)
    assert k0[0, 0] == calib.focal_of[0] and k0[0, 2] == SIZE[0] / 2
    assert np.array_equal(calib.frame_K(999), calib.K)  # unknown frame -> nominal


def _poison(clicks: list[Click]) -> list[Click]:
    """Click corner_own_left (id 0) of frame 0 at the pixel where corner_opp_right
    (id 3) actually is — a catastrophic identity swap, hundreds of px wrong."""
    src = next(c for c in clicks if c.frame == 0 and c.kp_idx == 3)
    return [*clicks, Click(0, 0, src.x, src.y)]


def test_poison_click_does_not_move_other_frames() -> None:
    clicks, lines = _mz_session()
    clicks = [c for c in clicks if not (c.frame == 0 and c.kp_idx == 0)]
    clean_calib = solve_session(clicks, lines, SIZE, {})
    poisoned_calib = solve_session(_poison(clicks), lines, SIZE, {})
    for f in (1, 2):
        rel = abs(poisoned_calib.focal_of[f] - clean_calib.focal_of[f]) / clean_calib.focal_of[f]
        assert rel < 0.01
        assert poisoned_calib.coverage_grade[f] == clean_calib.coverage_grade[f]


def test_ordering_fix_poisoned_k_equals_clean_k() -> None:
    clicks, lines = _mz_session()
    clicks = [c for c in clicks if not (c.frame == 0 and c.kp_idx == 0)]
    k_clean = solve_session(clicks, lines, SIZE, {}).K[0, 0]
    k_poisoned = solve_session(_poison(clicks), lines, SIZE, {}).K[0, 0]
    assert abs(k_poisoned - k_clean) <= 1.0  # spec §6(c): within 1 px


def test_fallback_ladder_wiring(monkeypatch: Any) -> None:
    # Force every sweep unconstrained -> ladder must yield all-"shared" at f1 exactly
    # (pre-change behavior). Patch at the physical_calib import site.
    import soccer_vision.pitch.physical_calib as pc_mod

    def unconstrained(err_at: Any, f_init: float) -> FocalFit:
        return FocalFit(f=f_init * 1.11, constrained=False, err_ft=1.0)

    monkeypatch.setattr(pc_mod, "fit_frame_focal", unconstrained)
    clicks, lines = _mz_session()
    calib = solve_session(clicks, lines, SIZE, {})
    assert set(calib.focal_source.values()) == {"shared"}
    assert len(set(calib.focal_of.values())) == 1
    assert calib.K[0, 0] == next(iter(calib.focal_of.values()))  # nominal == f1


def test_few_point_frame_gets_median_focal() -> None:
    clicks, lines = _mz_session()
    # Strip frame 1 down to 5 unique ids (below the 6-id focal threshold, above
    # min_points=4 so it still gets a pose).
    keep_ids = sorted({c.kp_idx for c in clicks if c.frame == 1})[:5]
    clicks = [c for c in clicks if c.frame != 1 or c.kp_idx in keep_ids]
    calib = solve_session(clicks, lines, SIZE, {})
    assert 1 in calib.anchor_h
    assert calib.focal_source[1] in ("median", "shared")
```

- [ ] **Step 3: Run to verify failure:**

Run: `cd packages/soccer-vision && uv run pytest tests/test_physical_calib.py -q -k "focal or poison or ordering or ladder or median_focal or frame_k"`
Expected: FAIL — `AttributeError: 'PhysicalCalib' object has no attribute 'focal_of'` (and similar).

- [ ] **Step 4: Implement in `physical_calib.py`.**

4a. Add imports (top of file, with the other `soccer_vision` imports):

```python
from soccer_vision.pitch.focal import (
    MIN_ACCEPTED_FITS,
    FocalFit,
    fit_frame_focal,
    session_focal,
)
```
Also add `Callable` to the `collections.abc` import.

4b. Add to the `PhysicalCalib` dataclass (AFTER `segment_of`, both with defaults):

```python
    # Per-anchor focal (spec 2026-07-28): frame -> focal px and its source
    # ("fit" | "median" | "shared"). Empty for the pre-focal empty-calib path.
    focal_of: dict[int, float] = field(default_factory=dict)
    focal_source: dict[int, str] = field(default_factory=dict)

    def frame_K(self, frame: int) -> NDArray[np.float64]:
        """This frame's intrinsics (per-anchor focal); nominal K for unknown frames."""
        f = self.focal_of.get(frame)
        if f is None:
            return self.K
        w, h = self.size
        return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
```

4c. Add the module-level error-closure builder (near `_grade`):

```python
def _frame_err_at(
    po: list[tuple[int, float, float]],
    lo: list[tuple[str, float, float]],
    size: tuple[int, int],
) -> Callable[[float], float | None]:
    """Median in-sample residual (ft) of one frame's clicks as a function of focal —
    the same residuals _grade uses. None where no pose solves. This is the closure
    fed to focal.fit_frame_focal (focal.py never imports this module)."""
    w, h = size
    diag = np.diag([float(w), float(h), 1.0])

    def err_at(f: float) -> float | None:
        k = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
        pose = _anchor_pose(k, po, lo, None)
        if pose is None:
            return None
        h_norm = np.asarray(frame_homography(k, *pose), dtype=np.float64) @ diag
        errs = [_point_feet(_apply(h_norm, np.array([[x / w, y / h]]))[0], kp)
                for kp, x, y in po]
        errs += [_line_perp_feet(_apply(h_norm, np.array([[x / w, y / h]]))[0], lid)
                 for lid, x, y in lo]
        return float(np.median(errs)) if errs else None

    return err_at
```

4d. Rewrite the body of `solve_session` from the `K = calibrate_camera(...)` line onward (keep the signature and the preamble building `tf`, `seg_of`, `by_pt`, `by_ln`, `obs`):

```python
    try:
        k0 = calibrate_camera(obs, size, min_points=6).K
    except CalibError:
        # A physical calibration needs a shared focal from >= 3 diverse views. With fewer,
        # there is no physical solution yet -> return an empty calib (no anchors); the
        # labeler bootstrap simply waits for more clicked frames. We deliberately do NOT
        # fall back to a free per-frame homography -- that is exactly the model this engine
        # replaces (it is non-physical and folds the field into the sky).
        return PhysicalCalib(np.eye(3), {}, {}, {}, tf, size, gap_guard, seg_of)
    # Two-pass outlier flagging (spec §1.2): K0 only seeds pass 1; K1 is refit on the
    # pass-1 clean set and re-flags the ORIGINAL clicks, so a click that fails the
    # final flagging never touches any focal used for poses.
    clean1, _ = flag_outlier_clicks(points, k0, size)
    obs1 = {f: [(int(c.kp_idx), float(c.x * w), float(c.y * h)) for c in cs]
            for f, cs in _group(clean1).items()}
    try:
        k1 = calibrate_camera(obs1, size, min_points=6).K
    except CalibError:
        k1 = k0
    clean, _flagged = flag_outlier_clicks(points, k1, size)
    by_clean = _group(clean)
    f1 = float(k1[0, 0])
    diag = np.diag([float(w), float(h), 1.0])
    # Per-frame observations for every frame that will get a pose
    frame_obs: dict[int, tuple[list[tuple[int, float, float]],
                               list[tuple[str, float, float]]]] = {}
    for f in sorted(by_clean):
        pcs = by_clean[f]
        if len({c.kp_idx for c in pcs}) < min_points:
            continue
        frame_obs[f] = (
            [(int(c.kp_idx), float(c.x * w), float(c.y * h)) for c in pcs],
            [(str(lc.line_id), float(lc.x * w), float(lc.y * h))
             for lc in by_ln.get(f, [])],
        )
    # Per-anchor focal (spec §1.3): >= 6 unique ids buys a frame its own 1-D search
    fits: dict[int, FocalFit | None] = {
        f: (fit_frame_focal(_frame_err_at(po, lo, size), f1)
            if len({kp for kp, _, _ in po}) >= 6 else None)
        for f, (po, lo) in frame_obs.items()
    }
    focal = session_focal(fits, f1)
    poses: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    anchor_h: dict[int, NDArray[np.float64]] = {}
    grade: dict[int, str] = {}
    for f, (po, lo) in frame_obs.items():
        f_px, _src = focal[f]
        k_f = np.array([[f_px, 0.0, w / 2.0], [0.0, f_px, h / 2.0], [0.0, 0.0, 1.0]])
        pose = _anchor_pose(k_f, po, lo, seed.poses.get(f) if seed else None)
        if pose is None:
            continue
        rv, tv = pose
        poses[f] = (rv, tv)
        anchor_h[f] = np.asarray(frame_homography(k_f, rv, tv), dtype=np.float64) @ diag
        grade[f] = _grade(k_f, rv, tv, po, by_ln.get(f, []), size)
    # Nominal K (spec §3): the session-consensus focal — median of accepted fits when
    # the ladder is active, else f1 (identical to the pre-change shared K).
    accepted = [v[0] for f, v in focal.items() if v[1] == "fit"]
    f_nom = float(np.median(accepted)) if len(accepted) >= MIN_ACCEPTED_FITS else f1
    k_nom = np.array([[f_nom, 0.0, w / 2.0], [0.0, f_nom, h / 2.0], [0.0, 0.0, 1.0]])
    return PhysicalCalib(
        k_nom, poses, anchor_h, grade, tf, size, gap_guard, seg_of,
        focal_of={f: v[0] for f, v in focal.items() if f in poses},
        focal_source={f: v[1] for f, v in focal.items() if f in poses},
    )
```

CAREFUL: `_grade` is called with `by_ln.get(f, [])` (LineClick objects, normalized) exactly as before — do NOT pass `lo` (pixel tuples) to `_grade`.

- [ ] **Step 5: Run the new tests:**

Run: `cd packages/soccer-vision && uv run pytest tests/test_physical_calib.py -q`
Expected: ALL pass — the pre-existing tests in this file (shared-focal world at 1460) must still pass; per-frame fits on a same-focal world recover ~1460 per frame and grades stay green. If `test_per_frame_focal_recovery_multizoom` fails on the 2% tolerance, do NOT widen the tolerance — report BLOCKED with the observed recovery errors (the spec fixes 2%).

- [ ] **Step 6: Full-suite sanity + gates:**

Run: `cd packages/soccer-vision && uv run pytest -q 2>&1 | tail -3 && uv run ruff check src tests && cd .. && uv run mypy 2>&1 | tail -2`
Expected: no NEW failures vs the 488-passed baseline (`test_labeler_*`, `test_pitch_*` consume solve_session — they must stay green); ruff clean; mypy shows only the 58 pre-existing errors (count them: `uv run mypy 2>&1 | grep -c "error:"`).

- [ ] **Step 7: Commit:**

```bash
git add src/soccer_vision/pitch/physical_calib.py tests/test_physical_calib.py
git commit -m "feat(pitch): per-anchor focal in solve_session + two-pass outlier flagging (kills the poisoned-K path)"
```

---

### Task 3: `foreground_holdout` honesty — focal re-selected from held-out evidence

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/physical_calib.py` (`_foreground_errors` ~line 119; `foreground_holdout` ~line 353)
- Test: `packages/soccer-vision/tests/test_physical_calib.py`

- [ ] **Step 1: Read** the current `_foreground_errors` and `foreground_holdout` and the gate fixture (`_gate_fixture`, tests ~line 209–260). `evaluate_gate` calls `foreground_holdout` — its signature must not change.

- [ ] **Step 2: Write the failing test** (append to `tests/test_physical_calib.py`):

```python
def test_holdout_focal_has_no_near_touchline_leak() -> None:
    """Displace the near-TL line clicks; if the holdout's pose/focal were influenced
    by near-TL evidence, the per-click error deltas would not track the displacement.
    With an honest holdout the fit is IDENTICAL in both runs, so each error moves by
    exactly the displacement's perpendicular feet."""
    clicks, lines = _mz_session()
    base = foreground_holdout(clicks, lines, SIZE)
    assert base  # fixture must be holdout-evaluable
    dx_px = 30.0
    moved = [LineClick(lc.frame, lc.line_id, lc.x + dx_px / SIZE[0], lc.y)
             if lc.line_id == "near_touchline" else lc for lc in lines]
    shifted = foreground_holdout(clicks, moved, SIZE)
    assert len(shifted) == len(base)
    deltas = [abs(b - s) for b, s in zip(base, shifted)]
    # Every click moved by the same +x pixel shift; under an UNCHANGED fit the error
    # change per click is bounded by the projected shift (a few ft) and is strictly
    # positive for clicks that started near-perfect. A leaked (refit) focal would
    # leave some deltas ~0 while shrinking the reported errors instead.
    assert all(d > 0.05 for d in deltas)
    assert max(shifted) > max(base)  # displaced evidence must WORSEN the claim, never improve it
```

- [ ] **Step 3: Run to verify failure or verify the property is untested:** with the current code this test may PASS already (shared-K holdout is also leak-free w.r.t. lines). That is fine — it is a regression guard for the new re-selection path. Confirm it passes BEFORE the change, then again AFTER (Step 5). The real change-detector is `test_foreground_holdout_counts` (existing) continuing to pass plus the evidence numbers in Task 5.

- [ ] **Step 4: Implement.**

4a. `_foreground_errors` — change the first parameter from a K matrix to a default focal, and re-select focal from the held-out fit set (spec §4):

```python
def _foreground_errors(
    f_default: float,
    po: list[tuple[int, float, float]],
    line_clicks: Sequence[LineClick],
    size: tuple[int, int],
) -> list[float] | None:
    """Held-out near-touchline error (feet) for ONE frame: refit the pose WITHOUT any
    near-touchline evidence -- both the near-touchline LINE clicks AND the point landmarks
    that lie on it (its endpoints, x=0) -- then measure how far the near-touchline clicks
    land from the x=0 line. The focal is RE-SELECTED from the held-out fit set (spec §4:
    a focal chosen using near-touchline clicks must not leak into a near-touchline
    claim); frames whose held-out set cannot constrain a focal use f_default (their
    session focal). None if the frame has no near-touchline click (foreground
    unverifiable) or too few remaining points to refit a pose."""
    if not any(lc.line_id == "near_touchline" for lc in line_clicks):
        return None
    w, h = size
    lo_fit = [(str(lc.line_id), lc.x * w, lc.y * h)
              for lc in line_clicks if lc.line_id != "near_touchline"]
    po_fit = [obs for obs in po if obs[0] not in _NEAR_TL_POINT_IDS]
    if len(po_fit) < 4:
        return None  # not enough off-near-touchline points to genuinely hold it out
    f_use = f_default
    if len({kp for kp, _, _ in po_fit}) >= 6:
        fit = fit_frame_focal(_frame_err_at(po_fit, lo_fit, size), f_default)
        if fit is not None and fit.constrained:
            f_use = fit.f
    k = np.array([[f_use, 0.0, w / 2.0], [0.0, f_use, h / 2.0], [0.0, 0.0, 1.0]])
    pose = _anchor_pose(k, po_fit, lo_fit, None)
    if pose is None:
        return None
    rv, tv = pose
    h_norm = np.asarray(frame_homography(k, rv, tv), dtype=np.float64) @ np.diag(
        [float(w), float(h), 1.0])
    errs = [_line_perp_feet(_apply(h_norm, np.array([[lc.x, lc.y]]))[0], "near_touchline")
            for lc in line_clicks if lc.line_id == "near_touchline"]
    return errs or None
```

4b. `foreground_holdout` — run the same two-pass + per-frame-focal pipeline, then hand each frame its session focal as the default:

```python
def foreground_holdout(
    points: Sequence[Click],
    lines: Sequence[LineClick],
    size: tuple[int, int],
    *,
    min_points: int = 4,
) -> list[float]:
    """Per-anchor held-out near-touchline error (feet), pooled across all anchors that have
    a near-touchline click. Empty if the session can't calibrate a shared focal. Runs the
    same two-pass-flagging + per-frame-focal pipeline as solve_session; each frame's
    holdout re-selects its focal from held-out evidence only (spec §4)."""
    calib = solve_session(points, lines, size, {}, min_points=min_points)
    if not calib.anchor_h:
        return []
    w, h = size
    by_pt = _group(points)
    by_ln = _group(lines)
    errs: list[float] = []
    for f in calib.anchor_h:
        po = [(int(c.kp_idx), float(c.x * w), float(c.y * h)) for c in by_pt.get(f, [])]
        fe = _foreground_errors(calib.focal_of.get(f, float(calib.K[0, 0])),
                                po, by_ln.get(f, []), size)
        if fe:
            errs.extend(fe)
    return errs
```

NOTE this reuses `solve_session` (empty transforms) instead of duplicating the pipeline — the previous version duplicated the calibrate-then-solve flow inline. `po` here is built from ALL of the frame's clicks (as before); outlier robustness inside `_anchor_pose`/`refine_pose` is unchanged.

- [ ] **Step 5: Run the focused tests, then the file:**

Run: `cd packages/soccer-vision && uv run pytest tests/test_physical_calib.py -q`
Expected: ALL pass, including `test_foreground_holdout_counts`, `test_evaluate_gate_passes_on_clean_session`, `test_gate_fails_without_foreground`, and the new leak test.

- [ ] **Step 6: Full gates:**

Run: `cd packages/soccer-vision && uv run pytest -q 2>&1 | tail -3 && uv run ruff check src tests && cd .. && uv run mypy 2>&1 | grep -c "error:"`
Expected: no new failures; ruff clean; mypy error count == the count recorded in Task 2 Step 6.

- [ ] **Step 7: Commit:**

```bash
git add src/soccer_vision/pitch/physical_calib.py tests/test_physical_calib.py
git commit -m "fix(pitch): foreground holdout re-selects focal from held-out evidence (no near-TL leak)"
```

---

### Task 4: Focal reporting — physical `calib_gate.json` + `validate_session` summary

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py` (`export`, ~line 437–495: add a physical-engine branch beside the existing crop-only `calib_gate.json` write)
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/validate_session.py` (physical-engine report section)
- Test: `packages/soccer-vision/tests/test_labeler_state.py`

- [ ] **Step 1: Read** `state.py` `export()` (the crop branch writing `calib_gate.json`) and `_solve()`/`_last_calib`; `validate_session.py`'s physical report section (find where the physical engine's gate metrics are printed); `tests/test_labeler_state.py` fixtures `_pan_session`, `_near_tl_clicks`, `_spread_anchors` (lines 1–80) and `test_export_writes_only_green_frames` (~line 164).

- [ ] **Step 2: Write the failing test** (append to `tests/test_labeler_state.py`):

```python
def test_export_physical_gate_json_has_focal_block(tmp_path: Path) -> None:
    import json

    interframe, poses, clicks = _pan_session(9)
    anchors = _spread_anchors(9)
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        st.add_clicks(clicks)
        st.add_line_clicks(_near_tl_clicks(poses, anchors))
        st.wait_idle(timeout=10)
        st.export(tmp_path)
        gate = json.loads((tmp_path / "calib_gate.json").read_text())
        assert gate["engine"] == "physical"
        focal = gate["focal"]
        assert set(focal) == {"per_frame", "source", "spread_p90_p10"}
        assert set(focal["per_frame"]) == set(focal["source"])
        assert len(focal["per_frame"]) >= 3  # the session's anchors
        assert all(s in ("fit", "median", "shared") for s in focal["source"].values())
        assert focal["spread_p90_p10"] >= 1.0
    finally:
        st.stop_worker()
```

- [ ] **Step 3: Run to verify failure:**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py::test_export_physical_gate_json_has_focal_block -q`
Expected: FAIL — `FileNotFoundError` (physical mode writes no `calib_gate.json` today).

- [ ] **Step 4: Implement.**

4a. In `state.py` `export()`, after the existing `if self._engine == "crop":` gate-json block, add the physical branch (same indentation level as the `if`):

```python
        elif self._engine == "physical":
            with self._lock:
                calib = self._last_calib
            if calib is not None and calib.focal_of:
                vals = sorted(calib.focal_of.values())
                spread = (float(np.percentile(vals, 90) / np.percentile(vals, 10))
                          if len(vals) > 1 else 1.0)
                (out / "calib_gate.json").write_text(json.dumps({
                    "engine": "physical",
                    # Focal transparency (spec 2026-07-28 §5): which frames run on a
                    # fitted vs fallback focal, and how wide the session's zoom range is.
                    "focal": {
                        "per_frame": {str(f): v for f, v in sorted(calib.focal_of.items())},
                        "source": {str(f): s for f, s in sorted(calib.focal_source.items())},
                        "spread_p90_p10": spread,
                    },
                }, indent=2))
```
(`self._last_calib` is a `PhysicalCalib` in physical mode; export already ran `wait_idle`, so it reflects the final clicks. If `_last_calib` is typed as an optional union, `assert`/narrow accordingly for mypy.)

4b. In `validate_session.py`, inside the physical-engine report section (immediately after its gate-metrics print; use the local `PhysicalCalib` variable in scope there — read the function to find its name), add:

```python
    if calib.focal_source:
        srcs = list(calib.focal_source.values())
        vals = sorted(calib.focal_of.values())
        spread = float(np.percentile(vals, 90) / np.percentile(vals, 10)) if len(vals) > 1 else 1.0
        print(f"  focal: {srcs.count('fit')} fit / {srcs.count('median')} median / "
              f"{srcs.count('shared')} shared | {min(vals):.0f}-{max(vals):.0f}px | "
              f"spread p90/p10 {spread:.3f}")
```
(Adapt the variable name `calib` to the actual local; keep the print format exactly.)

- [ ] **Step 5: Run tests + gates:**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py tests/test_labeler_state_crop.py -q && uv run pytest -q 2>&1 | tail -3 && uv run ruff check src tests && cd .. && uv run mypy 2>&1 | grep -c "error:"`
Expected: all pass (crop export tests unchanged); mypy count unchanged.

- [ ] **Step 6: Commit:**

```bash
git add src/soccer_vision/labeler/state.py src/soccer_vision/pitch/validate_session.py tests/test_labeler_state.py
git commit -m "feat(pitch): focal transparency — physical calib_gate.json block + validate_session summary"
```

---

### Task 5: Acceptance evidence vs Task 0 baselines (spec §6 — the merge gate)

**Files:**
- Modify: `docs/superpowers/2026-07-29-per-frame-focal-evidence.md` (fill the `## After` section)

- [ ] **Step 1: Frozen-sidecar after-numbers:** re-run the Task 0 metrics script UNCHANGED:

Run: `cd packages/soccer-vision && uv run python /private/tmp/claude-501/-Users-patrickreed-Sandbox-soccer-vision/5ae1a3e6-7ea8-45b2-87b4-3aabfa8c623d/scratchpad/engine_metrics.py`
MUST-PASS (spec §6a): green count > 86 AND median in-sample err < the Task 0 baseline value. The focal line should show most frames "fit" and a spread well above 1.0.

- [ ] **Step 2: Shipped-session after-numbers:** re-run the exact Task 0 Step 3 `validate_session` command (same chain npz).
MUST-PASS (spec §6b): fg and prop med/p90 each within +0.3 ft of the Task 0 baseline or better. The report should include the new `focal:` summary line.

- [ ] **Step 3: If either gate FAILS:** do not "fix" numbers — STOP and report BLOCKED with both before/after tables. (The agreed fallback is scoping to rep-pack-only sessions, which is a design change the controller must take back to Patrick.)

- [ ] **Step 4: Fill `## After (per-frame focal)`** in the evidence doc with both verbatim outputs and a 4-row summary table (frozen-sidecar green / frozen-sidecar median-err / oceanside fg / oceanside prop; before vs after).

- [ ] **Step 5: Final gates, exactly:**

Run: `cd packages/soccer-vision && uv run pytest -q 2>&1 | tail -3 && uv run ruff check src tests && cd .. && uv run mypy 2>&1 | grep -c "error:"`
Expected: >= 488+new passed, 3 skipped, 0 failed; ruff clean; mypy error count == Task 2 baseline count.

- [ ] **Step 6: Commit:**

```bash
git add docs/superpowers/2026-07-29-per-frame-focal-evidence.md
git commit -m "docs(pitch): per-frame focal acceptance evidence — frozen sidecar + oceanside gates"
```

---

## Self-review checklist (run before handoff)

- Spec §1 order → Task 2 Step 4d; §1.3 acceptance → Task 1; §1.4 ladder → Task 1 + Task 2 tests; §2 API → Task 1 (session_focal returns (focal, source) pairs per the amended spec); §3 surface → Task 2 4b; §4 holdout → Task 3; §5 reporting → Task 4; §6 evidence → Tasks 0 + 5; §7 exclusions respected (no distortion/PP/crop/UI changes).
- No placeholders; every code step shows the code; type names consistent (`FocalFit`, `fit_frame_focal(err_at, f_init)`, `session_focal(fits, f_shared) -> dict[int, tuple[float, str]]`, `PhysicalCalib.focal_of/focal_source/frame_K`).
- Known judgment calls for the implementer to flag (not silently change): Task 2 middle-frame "fit" count (>= 2 of 3); Task 3 leak test passing pre-change is expected; Task 5 BLOCKED path is a real outcome, not a failure of the implementer.
