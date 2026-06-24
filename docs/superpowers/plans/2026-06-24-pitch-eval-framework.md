# Pitch-Model Evaluation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure, unit-tested metric module + Colab notebook that scores the pitch-keypoint model against labeler ground truth on a frozen held-out benchmark, in real-world feet, with a false-pass-proof headline (accurate-coverage).

**Architecture:** New `soccer_vision/eval/` subpackage: `pitch_metrics.py` (pure feet/keypoint/homography/aggregation functions) + `benchmark.py` (frozen-benchmark manifest). A `examples/eval_pitch.ipynb` runs the model on the benchmark and calls the runner. Replaces the `acceptance_pitch` coverage gate.

**Tech Stack:** Python 3.11, numpy, pandas. pytest, mypy strict (bare `uv run mypy` from REPO ROOT), ruff.

---

## CRITICAL conventions for every task

- **GIT (ABSOLUTE):** commit only on the current branch (`feat/pitch-eval-framework`) via `git add <paths> && git commit`. NEVER checkout/switch/reset/stash/rebase. Read-only git is fine.
- **mypy:** bare `uv run mypy` from the REPO ROOT only (inside `packages/soccer-vision` gives bogus duplicate-module errors). Zero new errors; annotate test helpers; `tmp_path: Path`.
- **ruff:** imports at top (sorted), no `;`-joined statements; lint changed src AND tests.
- Existing APIs (read the files):
  - `soccer_vision.pitch.landmarks.PITCH_LANDMARKS` — `(21, 2)` float64 canonical coords; BOTH axes are fractions of pitch LENGTH (uniform scale). `NEAR_HALFWAY_IDX == 5` (under-camera, never visible).
  - `soccer_vision.pitch.autolabel.project_landmarks(H, landmarks, frame_size) -> (N,3)` (x_px, y_px, visible∈{0,2}); uses `inv(H)`; `H` maps image→pitch.
  - `soccer_vision.pitch.homography.fit_homography(image_points, pitch_points) -> (3,3)` image→pitch; raises `HomographyError` (<4 pts or degenerate).
  - `soccer_vision.pipeline.homographies_from_parquet(path) -> dict[int, HomographyEntry]` (`.H`, `.confidence`, `.source`).

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/soccer_vision/eval/__init__.py` | subpackage marker | Create |
| `src/soccer_vision/eval/pitch_metrics.py` | feet conv, keypoint/homography errors, labeler noise, frame scoring, benchmark aggregation | Create |
| `src/soccer_vision/eval/benchmark.py` | frozen-benchmark manifest (load/save) | Create |
| `tests/test_eval_pitch_metrics.py` | synthetic-GT metric tests | Create |
| `tests/test_eval_benchmark.py` | manifest round-trip | Create |
| `examples/eval_pitch.ipynb` | Colab runner + report + overlays | Create |

---

## Task 1: eval subpackage + feet conversion

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/eval/__init__.py`
- Create: `packages/soccer-vision/src/soccer_vision/eval/pitch_metrics.py`
- Test: `packages/soccer-vision/tests/test_eval_pitch_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `packages/soccer-vision/tests/test_eval_pitch_metrics.py`:

```python
"""Synthetic-ground-truth tests for the pitch-model eval metrics."""

from __future__ import annotations

import numpy as np
from soccer_vision.eval.pitch_metrics import DEFAULT_PITCH_LENGTH_FT, canonical_to_feet


def test_canonical_to_feet_scalar() -> None:
    # both pitch axes are fractions of length, so a 0.1 canonical distance is
    # 0.1 * length_ft feet.
    assert canonical_to_feet(0.1) == DEFAULT_PITCH_LENGTH_FT * 0.1


def test_canonical_to_feet_array() -> None:
    out = canonical_to_feet(np.array([0.0, 0.5, 1.0]), length_ft=200.0)
    assert np.allclose(out, [0.0, 100.0, 200.0])
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_eval_pitch_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: soccer_vision.eval`

- [ ] **Step 3: Implement**

Create `packages/soccer-vision/src/soccer_vision/eval/__init__.py` (empty file):

```python
```

Create `packages/soccer-vision/src/soccer_vision/eval/pitch_metrics.py`:

```python
"""Pure metrics for scoring the pitch-keypoint model against labeler ground truth.

Everything here is numpy in / dataclass out, no I/O — so the eval logic itself is
unit-testable (the lesson of anchor_cov, which shipped untested and gave a false
pass). Errors are reported in real-world FEET: both pitch axes are fractions of
pitch length, so a canonical Euclidean distance scales by one constant.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Nominal US Soccer 9v9 pitch length: ~68.5 m. Youth fields vary; this is a fixed
# nominal scale so feet errors are interpretable and comparable across retrains.
DEFAULT_PITCH_LENGTH_FT: float = 224.7
HIDDEN_IDX: int = 5  # under-camera landmark, never ground-truth-visible


def canonical_to_feet(
    distance: float | NDArray[np.floating],
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
) -> float | NDArray[np.floating]:
    """Convert a canonical-pitch distance (fraction of length) to feet."""
    return distance * length_ft
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_eval_pitch_metrics.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Gate + commit**

REPO ROOT `uv run mypy 2>&1 | tail -1` → Success; ruff both files → clean.

```bash
git add packages/soccer-vision/src/soccer_vision/eval/__init__.py packages/soccer-vision/src/soccer_vision/eval/pitch_metrics.py packages/soccer-vision/tests/test_eval_pitch_metrics.py
git commit -m "feat(eval): eval subpackage + canonical->feet conversion"
```

---

## Task 2: per-keypoint error in feet

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/eval/pitch_metrics.py`
- Test: `packages/soccer-vision/tests/test_eval_pitch_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to the test file (add imports at top, sorted: `from soccer_vision.eval.pitch_metrics import keypoint_errors_feet`; `from soccer_vision.pitch.homography import fit_homography`; `from soccer_vision.pitch.landmarks import PITCH_LANDMARKS`; `from soccer_vision.pitch.autolabel import project_landmarks`):

```python
_W, _H = 1920, 1080


def _gt_homography() -> np.ndarray:
    # map the full frame to the pitch region [0,0.6]x[0,1.0]; image->pitch.
    img = np.array([[0, 0], [_W, 0], [_W, _H], [0, _H]], dtype=float)
    pitch = np.array([[0.0, 0.0], [0.6, 0.0], [0.6, 1.0], [0.0, 1.0]])
    return fit_homography(img, pitch)


def _pitch_to_px(h_gt: np.ndarray, pt: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(h_gt)
    v = inv @ np.array([pt[0], pt[1], 1.0])
    return v[:2] / v[2]


def test_keypoint_errors_feet_known_offset() -> None:
    h_gt = _gt_homography()
    gt = project_landmarks(h_gt, PITCH_LANDMARKS, (_W, _H))  # (21,3) px+vis
    # model = perfect, except landmark 3 nudged by a known pitch offset.
    model = gt.copy()
    model[:, 2] = 2.0  # treat 'visible' column as confidence >= thr
    offset = np.array([0.02, 0.0])  # canonical -> 0.02 * length feet
    j = 3
    model[j, :2] = _pitch_to_px(h_gt, PITCH_LANDMARKS[j] + offset)
    errs = keypoint_errors_feet(h_gt, gt, model, conf_thr=0.5)
    assert abs(errs[j] - 0.02 * 224.7) < 0.1
    for i, v in errs.items():
        if i != j:
            assert v < 0.01  # perfect elsewhere


def test_keypoint_errors_feet_skips_hidden_and_lowconf() -> None:
    h_gt = _gt_homography()
    gt = project_landmarks(h_gt, PITCH_LANDMARKS, (_W, _H))
    model = gt.copy(); model[:, 2] = 2.0
    model[7, 2] = 0.1  # below conf threshold -> excluded
    errs = keypoint_errors_feet(h_gt, gt, model, conf_thr=0.5)
    assert 5 not in errs   # hidden idx never scored
    assert 7 not in errs   # low confidence excluded
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_eval_pitch_metrics.py -k keypoint_errors -q`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement**

Append to `pitch_metrics.py` (add this import at the top of the file, sorted with the existing imports):

```python
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS


def _apply_h(h: NDArray[np.floating], pts_px: NDArray[np.floating]) -> NDArray[np.float64]:
    """Map (N,2) image pixels -> (N,2) pitch coords through image->pitch H."""
    pts = np.asarray(pts_px, dtype=np.float64)
    homog = np.column_stack([pts, np.ones(len(pts))])
    proj = homog @ np.asarray(h, dtype=np.float64).T
    return np.asarray(proj[:, :2] / proj[:, 2:3], dtype=np.float64)


def keypoint_errors_feet(
    h_gt: NDArray[np.floating],
    gt_kpts: NDArray[np.floating],
    model_kpts: NDArray[np.floating],
    *,
    conf_thr: float = 0.5,
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
) -> dict[int, float]:
    """Per-landmark feet error: map the model's predicted pixel through the GT
    homography into pitch space, compare to the canonical landmark.

    gt_kpts: (21,3) px + visibility (from project_landmarks). model_kpts: (21,3)
    px + confidence. Scores landmarks that are GT-visible, not the hidden idx,
    and predicted with conf >= conf_thr.
    """
    out: dict[int, float] = {}
    for i in range(len(PITCH_LANDMARKS)):
        if i == HIDDEN_IDX or gt_kpts[i, 2] <= 0 or model_kpts[i, 2] < conf_thr:
            continue
        pitch_pred = _apply_h(h_gt, model_kpts[i:i + 1, :2])[0]
        d = float(np.hypot(*(pitch_pred - PITCH_LANDMARKS[i])))
        out[i] = float(canonical_to_feet(d, length_ft))
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_eval_pitch_metrics.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Gate + commit**

REPO ROOT mypy → Success; ruff → clean.

```bash
git add packages/soccer-vision/src/soccer_vision/eval/pitch_metrics.py packages/soccer-vision/tests/test_eval_pitch_metrics.py
git commit -m "feat(eval): per-keypoint feet error vs labeler ground truth"
```

---

## Task 3: homography reprojection error (end-to-end backstop)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/eval/pitch_metrics.py`
- Test: `packages/soccer-vision/tests/test_eval_pitch_metrics.py`

- [ ] **Step 1: Write the failing test**

Append (add import `reproj_error_feet` to the metrics import line):

```python
def test_reproj_error_zero_when_model_h_equals_gt() -> None:
    h_gt = _gt_homography()
    gt = project_landmarks(h_gt, PITCH_LANDMARKS, (_W, _H))
    assert reproj_error_feet(h_gt, h_gt, gt) < 1e-6  # identical H -> ~0 ft


def test_reproj_error_none_when_no_visible() -> None:
    h_gt = _gt_homography()
    gt = project_landmarks(h_gt, PITCH_LANDMARKS, (_W, _H))
    gt = gt.copy(); gt[:, 2] = 0.0  # nothing visible
    assert reproj_error_feet(h_gt, h_gt, gt) is None
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_eval_pitch_metrics.py -k reproj -q`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement**

Append to `pitch_metrics.py`:

```python
def reproj_error_feet(
    h_gt: NDArray[np.floating],
    model_h: NDArray[np.floating],
    gt_kpts: NDArray[np.floating],
    *,
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
) -> float | None:
    """End-to-end check: at each GT-visible landmark's pixel, how far (feet) does
    the MODEL homography place it from the canonical truth? Median over visible
    landmarks. None if no GT-visible landmarks. Catches outlier keypoints that
    wreck the fitted homography even when average keypoint error looks fine.
    """
    idx = [i for i in range(len(PITCH_LANDMARKS)) if i != HIDDEN_IDX and gt_kpts[i, 2] > 0]
    if not idx:
        return None
    px = gt_kpts[idx, :2]
    pitch_via_model = _apply_h(model_h, px)
    d = np.hypot(*(pitch_via_model - PITCH_LANDMARKS[idx]).T)
    return float(canonical_to_feet(float(np.median(d)), length_ft))
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_eval_pitch_metrics.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Gate + commit**

```bash
git add packages/soccer-vision/src/soccer_vision/eval/pitch_metrics.py packages/soccer-vision/tests/test_eval_pitch_metrics.py
git commit -m "feat(eval): homography reprojection error in feet"
```

---

## Task 4: labeler-noise floor (the match bar)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/eval/pitch_metrics.py`
- Test: `packages/soccer-vision/tests/test_eval_pitch_metrics.py`

- [ ] **Step 1: Write the failing test**

Append (import `labeler_fit_residual_feet`):

```python
def test_labeler_fit_residual_zero_for_exact_clicks() -> None:
    h_gt = _gt_homography()
    # clicks placed exactly where the GT homography says the landmarks are.
    idx = np.array([0, 3, 13, 16, 19, 20])
    clicks_px = np.array([_pitch_to_px(h_gt, PITCH_LANDMARKS[i]) for i in idx])
    r = labeler_fit_residual_feet(h_gt, clicks_px, idx)
    assert r < 0.05  # perfectly consistent -> ~0 ft


def test_labeler_fit_residual_known_error() -> None:
    h_gt = _gt_homography()
    idx = np.array([0, 3, 13, 16])
    clicks_px = np.array([_pitch_to_px(h_gt, PITCH_LANDMARKS[i]) for i in idx])
    # nudge one click so it maps 0.03 canonical off -> 0.03*224.7 ft, median of [0,0,0,that]
    clicks_px[1] = _pitch_to_px(h_gt, PITCH_LANDMARKS[idx[1]] + np.array([0.03, 0.0]))
    r = labeler_fit_residual_feet(h_gt, clicks_px, idx)
    assert 0.0 <= r <= 0.03 * 224.7 + 0.5
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_eval_pitch_metrics.py -k labeler_fit -q`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement**

Append:

```python
def labeler_fit_residual_feet(
    h_gt: NDArray[np.floating],
    clicked_px: NDArray[np.floating],
    kp_indices: NDArray[np.integer],
    *,
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
) -> float:
    """Labeler self-consistency: median feet error of its homography on its OWN
    clicked anchors. This is the noise-floor proxy that defines the
    match-the-labeler bar. (clicked_px: (k,2) image pixels; kp_indices: (k,)
    landmark ids for each click.)
    """
    idx = np.asarray(kp_indices)
    pitch_pred = _apply_h(h_gt, np.asarray(clicked_px, dtype=np.float64))
    d = np.hypot(*(pitch_pred - PITCH_LANDMARKS[idx]).T)
    return float(canonical_to_feet(float(np.median(d)), length_ft))
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_eval_pitch_metrics.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Gate + commit**

```bash
git add packages/soccer-vision/src/soccer_vision/eval/pitch_metrics.py packages/soccer-vision/tests/test_eval_pitch_metrics.py
git commit -m "feat(eval): labeler fit-residual noise floor in feet"
```

---

## Task 5: frame scoring + benchmark aggregation (the headline)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/eval/pitch_metrics.py`
- Test: `packages/soccer-vision/tests/test_eval_pitch_metrics.py`

- [ ] **Step 1: Write the failing test**

Append (import `FrameScore, EvalReport, score_frame, score_benchmark`):

```python
def _perfect_model(h_gt: np.ndarray) -> np.ndarray:
    gt = project_landmarks(h_gt, PITCH_LANDMARKS, (_W, _H))
    m = gt.copy(); m[:, 2] = np.where(gt[:, 2] > 0, 2.0, 0.0)
    return m


def test_score_frame_perfect_matches_labeler() -> None:
    h_gt = _gt_homography()
    fs = score_frame(0, h_gt, _perfect_model(h_gt), frame_size=(_W, _H),
                     match_threshold_feet=2.0)
    assert fs.reproj_feet is not None and fs.reproj_feet < 0.01
    assert fs.matches is True
    assert fs.median_feet is not None and fs.median_feet < 0.01


def test_score_benchmark_accurate_coverage_and_exclusions() -> None:
    h_gt = _gt_homography()
    gt_homs = {0: h_gt, 1: h_gt, 2: h_gt}
    perfect = _perfect_model(h_gt)
    # frame 0 perfect (match); frame 1 missing model prediction (not covered);
    # frame 2 model present but all far off (no match).
    bad = perfect.copy()
    bad[:, :2] += 500.0  # shove every keypoint 500 px -> large feet error
    preds = {0: perfect, 2: bad}
    rep = score_benchmark(gt_homs, preds, frame_size=(_W, _H),
                          match_threshold_feet=2.0)
    assert rep.n_frames == 3
    assert rep.n_matched == 1
    assert abs(rep.accurate_coverage - 1 / 3) < 1e-9
    assert rep.per_landmark[0]["detect_rate"] <= 1.0


def test_score_benchmark_excludes_degenerate_gt() -> None:
    h_gt = _gt_homography()
    degenerate = np.full((3, 3), 1e-9); degenerate[2, 2] = 1.0
    gt_homs = {0: h_gt, 1: degenerate}
    preds = {0: _perfect_model(h_gt), 1: _perfect_model(h_gt)}
    rep = score_benchmark(gt_homs, preds, frame_size=(_W, _H),
                          match_threshold_feet=2.0, degenerate_cond=1e8)
    assert rep.n_excluded_degenerate == 1
    assert rep.n_frames == 1  # the degenerate GT frame is not scored
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_eval_pitch_metrics.py -k "score_" -q`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement**

Append to `pitch_metrics.py` (add `from dataclasses import dataclass` at top, sorted):

```python
from dataclasses import dataclass


@dataclass
class FrameScore:
    frame: int
    per_kp_feet: dict[int, float]
    median_feet: float | None      # None if model produced no scorable keypoints
    reproj_feet: float | None      # None if model H not fittable / no GT-visible
    gt_visible: list[int]
    predicted: list[int]
    matches: bool


@dataclass
class EvalReport:
    n_frames: int                  # scored frames (excludes degenerate GT)
    n_matched: int
    accurate_coverage: float       # n_matched / n_frames  (the headline)
    keypoint_feet_median: float | None
    keypoint_feet_p90: float | None
    reproj_feet_median: float | None
    reproj_feet_p90: float | None
    per_landmark: dict[int, dict[str, float]]  # idx -> {median, p90, detect_rate}
    n_excluded_degenerate: int


def score_frame(
    frame: int,
    h_gt: NDArray[np.floating],
    model_kpts: NDArray[np.floating] | None,
    *,
    frame_size: tuple[int, int],
    match_threshold_feet: float,
    conf_thr: float = 0.5,
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
) -> FrameScore:
    """Score one benchmark frame. A None / <4-confident model prediction yields a
    not-covered (non-matching) frame, never a skip."""
    from soccer_vision.pitch.autolabel import project_landmarks
    from soccer_vision.pitch.homography import HomographyError, fit_homography

    gt_kpts = project_landmarks(h_gt, PITCH_LANDMARKS, frame_size)
    gt_visible = [i for i in range(len(PITCH_LANDMARKS))
                  if i != HIDDEN_IDX and gt_kpts[i, 2] > 0]

    if model_kpts is None:
        return FrameScore(frame, {}, None, None, gt_visible, [], False)

    errs = keypoint_errors_feet(h_gt, gt_kpts, model_kpts, conf_thr=conf_thr,
                                length_ft=length_ft)
    predicted = sorted(errs)
    median_feet = float(np.median(list(errs.values()))) if errs else None

    # fit the model homography from its confident keypoints for the reproj check
    reproj_feet: float | None = None
    conf_mask = model_kpts[:, 2] >= conf_thr
    conf_idx = [i for i in np.nonzero(conf_mask)[0] if i != HIDDEN_IDX]
    if len(conf_idx) >= 4:
        try:
            model_h = fit_homography(model_kpts[conf_idx, :2], PITCH_LANDMARKS[conf_idx])
            reproj_feet = reproj_error_feet(h_gt, model_h, gt_kpts, length_ft=length_ft)
        except HomographyError:
            reproj_feet = None

    matches = median_feet is not None and median_feet <= match_threshold_feet
    return FrameScore(frame, errs, median_feet, reproj_feet, gt_visible, predicted, matches)


def _cond(h: NDArray[np.floating]) -> float:
    return float(np.linalg.cond(np.asarray(h, dtype=np.float64)))


def score_benchmark(
    gt_homographies: dict[int, NDArray[np.floating]],
    model_predictions: dict[int, NDArray[np.floating]],
    *,
    frame_size: tuple[int, int],
    match_threshold_feet: float,
    conf_thr: float = 0.5,
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
    degenerate_cond: float = 1e8,
) -> EvalReport:
    """Score the model over the whole frozen benchmark. Degenerate-GT frames are
    excluded with a count; missing predictions count as not-covered."""
    scores: list[FrameScore] = []
    n_excluded = 0
    for frame in sorted(gt_homographies):
        h_gt = gt_homographies[frame]
        if _cond(h_gt) > degenerate_cond:
            n_excluded += 1
            continue
        scores.append(score_frame(
            frame, h_gt, model_predictions.get(frame), frame_size=frame_size,
            match_threshold_feet=match_threshold_feet, conf_thr=conf_thr,
            length_ft=length_ft))

    n = len(scores)
    n_matched = sum(s.matches for s in scores)
    kp_all = [v for s in scores for v in s.per_kp_feet.values()]
    reproj_all = [s.reproj_feet for s in scores if s.reproj_feet is not None]

    per_landmark: dict[int, dict[str, float]] = {}
    for i in range(len(PITCH_LANDMARKS)):
        if i == HIDDEN_IDX:
            continue
        vals = [s.per_kp_feet[i] for s in scores if i in s.per_kp_feet]
        gt_vis = sum(i in s.gt_visible for s in scores)
        per_landmark[i] = {
            "median": float(np.median(vals)) if vals else float("nan"),
            "p90": float(np.percentile(vals, 90)) if vals else float("nan"),
            "detect_rate": (len(vals) / gt_vis) if gt_vis else float("nan"),
        }

    def _med(x: list[float]) -> float | None:
        return float(np.median(x)) if x else None

    def _p90(x: list[float]) -> float | None:
        return float(np.percentile(x, 90)) if x else None

    return EvalReport(
        n_frames=n,
        n_matched=n_matched,
        accurate_coverage=(n_matched / n) if n else 0.0,
        keypoint_feet_median=_med(kp_all),
        keypoint_feet_p90=_p90(kp_all),
        reproj_feet_median=_med(reproj_all),
        reproj_feet_p90=_p90(reproj_all),
        per_landmark=per_landmark,
        n_excluded_degenerate=n_excluded,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_eval_pitch_metrics.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Gate + commit**

REPO ROOT mypy → Success; ruff → clean; full suite `uv run pytest -q | tail -1`.

```bash
git add packages/soccer-vision/src/soccer_vision/eval/pitch_metrics.py packages/soccer-vision/tests/test_eval_pitch_metrics.py
git commit -m "feat(eval): frame scoring, accurate-coverage headline, benchmark aggregation"
```

---

## Task 6: frozen-benchmark manifest

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/eval/benchmark.py`
- Test: `packages/soccer-vision/tests/test_eval_benchmark.py`

- [ ] **Step 1: Write the failing test**

Create `packages/soccer-vision/tests/test_eval_benchmark.py`:

```python
"""Round-trip tests for the frozen-benchmark manifest."""

from __future__ import annotations

from pathlib import Path

from soccer_vision.eval.benchmark import BenchmarkField, BenchmarkManifest, load_manifest


def test_manifest_round_trip(tmp_path: Path) -> None:
    m = BenchmarkManifest(fields=[
        BenchmarkField(field="chula_vista", game_id="game8",
                       homographies="chula/homographies.parquet",
                       frame_indices=[100, 200, 300]),
        BenchmarkField(field="carlsbad", game_id="game3",
                       homographies="carlsbad/homographies.parquet",
                       frame_indices=[50, 60]),
    ])
    p = tmp_path / "benchmark.json"
    m.save(p)
    loaded = load_manifest(p)
    assert loaded == m
    assert [f.field for f in loaded.fields] == ["chula_vista", "carlsbad"]
    assert loaded.fields[0].frame_indices == [100, 200, 300]
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_eval_benchmark.py -q`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement**

Create `packages/soccer-vision/src/soccer_vision/eval/benchmark.py`:

```python
"""Frozen-benchmark manifest: which held-out fields/frames define the eval set.

Versioned so every retrain scores the identical frames. Paths are relative to the
manifest file's directory (portable across machines/Colab).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkField:
    field: str               # human label, e.g. "chula_vista"
    game_id: str             # source game identifier
    homographies: str        # path to the labeler homographies.parquet (relative)
    frame_indices: list[int]  # the sampled frames scored for this field


@dataclass(frozen=True)
class BenchmarkManifest:
    fields: list[BenchmarkField]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(
            {"fields": [asdict(f) for f in self.fields]}, indent=2))


def load_manifest(path: Path) -> BenchmarkManifest:
    data = json.loads(Path(path).read_text())
    return BenchmarkManifest(fields=[
        BenchmarkField(
            field=f["field"], game_id=f["game_id"],
            homographies=f["homographies"], frame_indices=list(f["frame_indices"]),
        )
        for f in data["fields"]
    ])
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_eval_benchmark.py -q`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

REPO ROOT mypy → Success; ruff → clean.

```bash
git add packages/soccer-vision/src/soccer_vision/eval/benchmark.py packages/soccer-vision/tests/test_eval_benchmark.py
git commit -m "feat(eval): frozen-benchmark manifest (load/save)"
```

---

## Task 7: eval_pitch.ipynb notebook

**Files:**
- Create: `examples/eval_pitch.ipynb`

- [ ] **Step 1: Build the notebook**

Create the notebook with a python script (DELETE the script after; do not commit it). `json.dump(nb, indent=1)`, `nbformat=4`, `nbformat_minor=0`. Cells:

Cell 0 (markdown):
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PatrickJReed/soccer-vision/blob/master/examples/eval_pitch.ipynb)

# Pitch-model evaluation (keypoint accuracy vs labeler ground truth)
Scores a trained pitch model on the FROZEN held-out benchmark. Headline:
accurate-coverage = % of frames matching the labeler within its noise floor.
Replaces the old anchor_cov coverage gate (which gave a false pass).
```

Cell 1 (code) — install + Drive-aware staging of weights + benchmark:
```python
!pip install -q "soccer-vision @ git+https://github.com/PatrickJReed/soccer-vision.git#subdirectory=packages/soccer-vision"
!pip install -q "ultralytics>=8.2"
from pathlib import Path
WEIGHTS = Path("/content/pitch_v1.pt")
BENCH = Path("/content/benchmark")        # dir with benchmark.json + per-field homographies + clips
if not WEIGHTS.exists() or not BENCH.exists():
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = Path("/content/drive/MyDrive/soccer-vision")
    if not WEIGHTS.exists():
        for cand in ("last.pt", "best.pt", "pitch_v1.pt"):
            if (DRIVE / cand).exists():
                !cp "{DRIVE/cand}" /content/pitch_v1.pt
                print("weights:", cand); break
    if not BENCH.exists() and (DRIVE / "benchmark").exists():
        !cp -r "{DRIVE/'benchmark'}" /content/benchmark
assert WEIGHTS.exists() and (BENCH / "benchmark.json").exists()
```

Cell 2 (code) — run model on benchmark frames, build predictions + GT:
```python
import cv2
import numpy as np
from ultralytics import YOLO
from soccer_vision.eval.benchmark import load_manifest
from soccer_vision.pipeline import homographies_from_parquet

manifest = load_manifest(BENCH / "benchmark.json")
model = YOLO(str(WEIGHTS))
FRAME_SIZE = (1920, 1080)

# gt_homographies/model_predictions keyed by a global (field, frame) id flattened to int
gt_homs, preds, key = {}, {}, 0
keymap = {}
for fld in manifest.fields:
    homs = homographies_from_parquet(BENCH / fld.homographies)
    # the per-field clip is expected at BENCH/<field>.mp4
    cap = cv2.VideoCapture(str(BENCH / f"{fld.field}.mp4"))
    for fi in fld.frame_indices:
        if fi not in homs:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, frame = cap.read()
        if not ok:
            continue
        gt_homs[key] = np.asarray(homs[fi].H, float)
        res = model.predict(frame, imgsz=1280, verbose=False)[0]
        kp = res.keypoints
        if kp is not None and kp.xy is not None and len(kp.xy):
            xy = kp.xy[0].cpu().numpy(); cf = kp.conf[0].cpu().numpy()
            preds[key] = np.column_stack([xy, cf])
        keymap[key] = (fld.field, fi); key += 1
    cap.release()
print(f"benchmark frames with GT: {len(gt_homs)}; with model prediction: {len(preds)}")
```

Cell 3 (code) — derive the match-the-labeler bar from the labeler noise floor:
```python
from soccer_vision.eval.pitch_metrics import labeler_fit_residual_feet
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS, NEAR_HALFWAY_IDX

# labeler noise floor: median per-frame fit residual (feet), using the keypoints
# parquet clicks if available; else fall back to projecting visible landmarks.
# Patrick: set MARGIN from a test-retest if you ran one, else keep the default.
residuals = []
for fld in manifest.fields:
    homs = homographies_from_parquet(BENCH / fld.homographies)
    kpt_path = BENCH / fld.homographies.replace("homographies", "keypoints")
    if kpt_path.exists():
        import pandas as pd
        kdf = pd.read_parquet(kpt_path)
        for fi in fld.frame_indices:
            sub = kdf[kdf["frame"] == fi]
            if fi in homs and len(sub) >= 1:
                residuals.append(labeler_fit_residual_feet(
                    np.asarray(homs[fi].H, float),
                    sub[["x_px", "y_px"]].to_numpy(float),
                    sub["kp_idx"].to_numpy(int)))
NOISE_FLOOR_FT = float(np.percentile(residuals, 90)) if residuals else 5.0
MARGIN_FT = 1.0
MATCH_THRESHOLD_FT = NOISE_FLOOR_FT + MARGIN_FT
print(f"labeler noise floor (p90): {NOISE_FLOOR_FT:.2f} ft  -> match threshold {MATCH_THRESHOLD_FT:.2f} ft")
```

Cell 4 (code) — score + print the report:
```python
from soccer_vision.eval.pitch_metrics import score_benchmark
from soccer_vision.pitch.landmarks import LANDMARK_NAMES

rep = score_benchmark(gt_homs, preds, frame_size=FRAME_SIZE,
                      match_threshold_feet=MATCH_THRESHOLD_FT)
print(f"=== HEADLINE: accurate-coverage {rep.accurate_coverage:.1%} "
      f"({rep.n_matched}/{rep.n_frames} frames match the labeler) ===")
print(f"keypoint feet error  median {rep.keypoint_feet_median:.2f}  p90 {rep.keypoint_feet_p90:.2f}")
print(f"homography reproj ft median {rep.reproj_feet_median:.2f}  p90 {rep.reproj_feet_p90:.2f}")
print(f"excluded degenerate-GT frames: {rep.n_excluded_degenerate}\n")
print(f"{'idx':>3} {'landmark':22} {'median_ft':>9} {'p90_ft':>8} {'detect':>7}")
for i, s in rep.per_landmark.items():
    print(f"{i:>3} {LANDMARK_NAMES[i]:22} {s['median']:>9.2f} {s['p90']:>8.2f} {s['detect_rate']:>7.0%}")
```

Cell 5 (markdown):
```
## Qualitative overlays (Patrick assesses)
The next cell renders model keypoints (green) vs labeler GT keypoints (red) on
sampled benchmark frames and saves a contact sheet. Claude does not interpret it
— review it yourself.
```

Cell 6 (code) — render overlay sheet for the worst-scoring frames:
```python
import cv2, numpy as np
from soccer_vision.pitch.autolabel import project_landmarks
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS, NEAR_HALFWAY_IDX

cells = []
for k in list(gt_homs)[:8]:
    fld, fi = keymap[k]
    cap = cv2.VideoCapture(str(BENCH / f"{fld}.mp4")); cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, img = cap.read(); cap.release()
    if not ok: continue
    gt = project_landmarks(gt_homs[k], PITCH_LANDMARKS, FRAME_SIZE)
    for i in range(len(PITCH_LANDMARKS)):
        if i == NEAR_HALFWAY_IDX: continue
        if gt[i,2] > 0:
            cv2.circle(img,(int(gt[i,0]),int(gt[i,1])),12,(0,0,255),2)  # GT red ring
    if k in preds:
        m = preds[k]
        for i in range(len(PITCH_LANDMARKS)):
            if i != NEAR_HALFWAY_IDX and m[i,2] >= 0.5:
                cv2.circle(img,(int(m[i,0]),int(m[i,1])),8,(60,220,120),-1)  # model green dot
    cells.append(img)
cw=640
tiles=[cv2.resize(c,(cw,int(c.shape[0]*cw/c.shape[1]))) for c in cells]
rows=[np.hstack(tiles[i:i+4]+[np.zeros_like(tiles[0])]*(4-len(tiles[i:i+4]))) for i in range(0,len(tiles),4)]
cv2.imwrite("/content/eval_overlay.jpg", np.vstack(rows))
from IPython.display import Image, display
display(Image("/content/eval_overlay.jpg"))
```

- [ ] **Step 2: Validate + commit**

Run: `python3 -c "import json; nb=json.load(open('examples/eval_pitch.ipynb')); assert nb['nbformat']==4; assert len(nb['cells'])==7; s=''.join(x for c in nb['cells'] for x in c['source']); assert 'accurate-coverage' in s and 'score_benchmark' in s; print('OK', len(nb['cells']))"`
Expected: `OK 7`. Confirm only the notebook is added (`git status`), throwaway script deleted.

```bash
git add examples/eval_pitch.ipynb
git commit -m "docs(examples): eval_pitch notebook — keypoint-accuracy report on the frozen benchmark"
```

---

## Task 8: deprecate the false-pass gate + acceptance

**Files:**
- Modify: `examples/acceptance_pitch.ipynb`

- [ ] **Step 1: Mark acceptance_pitch deprecated**

Edit `examples/acceptance_pitch.ipynb` (throwaway script, delete after): insert a NEW markdown cell at index 0 (before the existing badge cell):
```
> **DEPRECATED — anchor_cov gave a false pass (high coverage, off-the-markings
> keypoints). Use `eval_pitch.ipynb` (keypoint accuracy vs labeler ground truth)
> instead.**
```

Validate: `python3 -c "import json; nb=json.load(open('examples/acceptance_pitch.ipynb')); assert 'DEPRECATED' in ''.join(nb['cells'][0]['source']); print('ok')"`

```bash
git add examples/acceptance_pitch.ipynb
git commit -m "docs(examples): deprecate acceptance_pitch (anchor_cov false-pass) in favor of eval_pitch"
```

- [ ] **Step 2 (controller, not subagent): synthetic end-to-end smoke**

Run the metric module end-to-end locally on synthetic data to confirm a perfect model scores 100% accurate-coverage and a shifted model scores low — already covered by Task 5 tests; re-run the full suite:
`cd packages/soccer-vision && uv run pytest -q` — all pass.

The real benchmark run is Patrick's (label ~3 held-out fields, assemble the `benchmark/` dir in Drive, run `eval_pitch.ipynb`).

---

## Final verification

- [ ] `cd packages/soccer-vision && uv run pytest -q` — all pass.
- [ ] REPO ROOT `uv run mypy 2>&1 | tail -1` — Success.
- [ ] `cd packages/soccer-vision && uv run ruff check src/soccer_vision/eval/ tests/test_eval_pitch_metrics.py tests/test_eval_benchmark.py` — clean.
