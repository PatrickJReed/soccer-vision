# Pipeline Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing (but unintegrated) `pitch/` and `phase/` modules into an end-to-end pipeline that turns a Trace video into enriched per-detection trajectories + per-frame possession/phase labels, persisted as parquet.

**Architecture:** A pure `assemble_phases()` runs the whole chain (homographies → pitch-map → boundary-filter → modal-team → possession → smooth → phase) and is fully unit-testable without a GPU. A thin `analyze_video()` wraps it with model invocation + parquet checkpoints; `assemble_from_parquet()` re-runs the pure stage cheaply from the checkpoint. Two new pieces the chain needs: `pitch/landmarks.py` (keypoint-index → pitch coordinate + per-frame homography fitter) and `phase/possession.smooth_possession`.

**Tech Stack:** pandas, numpy, OpenCV (`cv2.findHomography`, already used in `pitch/homography.py`), pyarrow (parquet), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-29-pipeline-orchestrator-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `packages/soccer-vision/src/soccer_vision/pitch/landmarks.py` | **New.** `PITCH_LANDMARKS` constant + `build_frame_homographies`. |
| `packages/soccer-vision/src/soccer_vision/phase/possession.py` | **Modify.** Add `smooth_possession`. |
| `packages/soccer-vision/src/soccer_vision/pipeline.py` | **New.** `PipelineResult`, `assemble_phases`, `assemble_from_parquet`, `analyze_video`, parquet writers. |
| `packages/soccer-vision/tests/test_pitch_landmarks.py` | **New.** Tests for Task 1. |
| `packages/soccer-vision/tests/test_phase_possession_smoothing.py` | **New.** Tests for Task 2. |
| `packages/soccer-vision/tests/test_pipeline_assemble.py` | **New.** Tests for Tasks 3–4. |
| `packages/soccer-vision/tests/test_pipeline_analyze_video.py` | **New.** Stub-backend wiring test for Task 5. |

Run all tests with: `uv run pytest packages/soccer-vision/tests/ -q`
Lint/type: `uv run ruff check packages/soccer-vision/` and `uv run mypy packages/soccer-vision/src`

---

## Task 1: `pitch/landmarks.py` — landmark table + per-frame homographies

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/landmarks.py`
- Test: `packages/soccer-vision/tests/test_pitch_landmarks.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/soccer-vision/tests/test_pitch_landmarks.py`:

```python
"""Tests for PITCH_LANDMARKS and build_frame_homographies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS, build_frame_homographies


def test_landmarks_shape_and_range() -> None:
    assert PITCH_LANDMARKS.shape == (32, 2)
    assert PITCH_LANDMARKS.min() >= 0.0
    assert PITCH_LANDMARKS.max() <= 1.0


def _keypoints_for_identity(frame: int, conf: float = 0.9) -> pd.DataFrame:
    """6 well-spread landmarks whose image points equal their pitch coords
    (so the fitted homography is identity)."""
    idxs = [0, 5, 13, 16, 24, 29]
    pts = PITCH_LANDMARKS[idxs]
    return pd.DataFrame({
        "frame": [frame] * len(idxs),
        "kp_idx": idxs,
        "x_px": pts[:, 0],
        "y_px": pts[:, 1],
        "conf": [conf] * len(idxs),
    })


def test_build_recovers_known_transform() -> None:
    idxs = [0, 5, 13, 16, 24, 29]
    pitch = PITCH_LANDMARKS[idxs]
    image = pitch * 1000.0 + np.array([50.0, 30.0])  # known affine
    kp = pd.DataFrame({
        "frame": [7] * len(idxs),
        "kp_idx": idxs,
        "x_px": image[:, 0],
        "y_px": image[:, 1],
        "conf": [0.9] * len(idxs),
    })
    homographies = build_frame_homographies(kp)
    assert set(homographies) == {7}
    pts = np.column_stack([image[:, 0], image[:, 1], np.ones(len(idxs))])
    mapped = (homographies[7] @ pts.T).T
    mapped /= mapped[:, 2:3]
    assert np.allclose(mapped[:, :2], pitch, atol=1e-6)


def test_frames_below_min_points_are_skipped() -> None:
    kp = _keypoints_for_identity(3).head(3)  # only 3 points
    assert build_frame_homographies(kp) == {}


def test_low_confidence_keypoints_filtered() -> None:
    kp = _keypoints_for_identity(3, conf=0.1)  # all below default 0.5
    assert build_frame_homographies(kp) == {}


def test_empty_keypoints_returns_empty() -> None:
    empty = pd.DataFrame(columns=["frame", "kp_idx", "x_px", "y_px", "conf"])
    assert build_frame_homographies(empty) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_landmarks.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'soccer_vision.pitch.landmarks'`

- [ ] **Step 3: Write the implementation**

Create `packages/soccer-vision/src/soccer_vision/pitch/landmarks.py`:

```python
"""Canonical pitch landmarks + per-frame homography fitting from keypoints.

PITCH_LANDMARKS maps each pitch-model keypoint index to its canonical pitch
coordinate, normalized to [0, 1]^2. Coordinates are vendored from roboflow's
SoccerPitchConfiguration (sports/configs/soccer.py) — a 12000 x 7000 cm pitch
with 32 vertices, whose order matches the football-pitch-detection keypoint
indices.

Axis convention: roboflow's length axis (x, goal-to-goal) maps to our y, and
roboflow's width axis (y) maps to our x. So landmark = (raw_y / width,
raw_x / length). This makes y the goal-to-goal axis the phase splitter expects
(y < 0.333 = own third, y > 0.667 = opp third).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.pitch.homography import HomographyError, fit_homography

_PITCH_LENGTH_CM: Final = 12000.0
_PITCH_WIDTH_CM: Final = 7000.0

# roboflow SoccerPitchConfiguration vertices in cm, index order == keypoint order.
_RAW_VERTICES_CM: Final = [
    (0, 0), (0, 1450), (0, 2584), (0, 4416), (0, 5550), (0, 7000),
    (550, 2584), (550, 4416), (1100, 3500),
    (2015, 1450), (2015, 2584), (2015, 4416), (2015, 5550),
    (6000, 0), (6000, 2585), (6000, 4415), (6000, 7000),
    (9985, 1450), (9985, 2584), (9985, 4416), (9985, 5550),
    (10900, 3500), (11450, 2584), (11450, 4416),
    (12000, 0), (12000, 1450), (12000, 2584), (12000, 4416), (12000, 5550), (12000, 7000),
    (5085, 3500), (6915, 3500),
]

# Normalized [0,1]^2 landmarks: (x, y) = (raw_y / width, raw_x / length).
PITCH_LANDMARKS: Final[NDArray[np.float64]] = np.array(
    [(ry / _PITCH_WIDTH_CM, rx / _PITCH_LENGTH_CM) for (rx, ry) in _RAW_VERTICES_CM],
    dtype=np.float64,
)


def build_frame_homographies(
    keypoints: pd.DataFrame,
    *,
    conf_threshold: float = 0.5,
    min_points: int = 4,
) -> dict[int, NDArray[np.floating]]:
    """Fit a per-frame homography from detected pitch keypoints.

    For each frame, keypoints with conf >= conf_threshold and a known kp_idx
    contribute (x_px, y_px) -> PITCH_LANDMARKS[kp_idx] correspondences. Frames
    with at least min_points such correspondences get a homography mapping image
    pixels to canonical pitch coords; other frames are skipped (no entry).

    Returns a sparse {frame: H} dict suitable for smooth_homographies().
    """
    homographies: dict[int, NDArray[np.floating]] = {}
    if keypoints.empty:
        return homographies
    n_landmarks = len(PITCH_LANDMARKS)
    for frame_idx, group in keypoints.groupby("frame", sort=True):
        sel = group[(group["conf"] >= conf_threshold) & (group["kp_idx"] < n_landmarks)]
        if len(sel) < min_points:
            continue
        image_points = sel[["x_px", "y_px"]].to_numpy(dtype=np.float64)
        pitch_points = PITCH_LANDMARKS[sel["kp_idx"].to_numpy()]
        try:
            homographies[int(frame_idx)] = fit_homography(image_points, pitch_points)
        except HomographyError:
            continue
    return homographies
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_landmarks.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check packages/soccer-vision/src/soccer_vision/pitch/landmarks.py packages/soccer-vision/tests/test_pitch_landmarks.py && uv run mypy packages/soccer-vision/src/soccer_vision/pitch/landmarks.py`
Expected: All checks pass / Success.

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/landmarks.py packages/soccer-vision/tests/test_pitch_landmarks.py
git commit -m "feat(pitch): landmark table + build_frame_homographies"
```

---

## Task 2: `smooth_possession` — temporal mode smoothing (spec §6.1)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/phase/possession.py`
- Test: `packages/soccer-vision/tests/test_phase_possession_smoothing.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/soccer-vision/tests/test_phase_possession_smoothing.py`:

```python
"""Tests for smooth_possession (30-frame modal smoothing, preserving contested)."""

from __future__ import annotations

import pandas as pd
from soccer_vision.phase.possession import smooth_possession


def test_removes_single_frame_flicker() -> None:
    s = pd.Series(["own", "opp", "own", "own", "own"], index=[0, 1, 2, 3, 4])
    out = smooth_possession(s, window_frames=3)
    assert out.tolist() == ["own", "own", "own", "own", "own"]


def test_preserves_contested() -> None:
    s = pd.Series(["own", "contested", "own", "own", "own"], index=[0, 1, 2, 3, 4])
    out = smooth_possession(s, window_frames=3)
    assert out.iloc[1] == "contested"


def test_small_window_is_noop() -> None:
    s = pd.Series(["own", "opp"], index=[0, 1])
    out = smooth_possession(s, window_frames=1)
    assert out.tolist() == ["own", "opp"]


def test_preserves_index_and_name() -> None:
    s = pd.Series(["own", "opp", "own"], index=[10, 11, 12], name="possession_state")
    out = smooth_possession(s, window_frames=3)
    assert list(out.index) == [10, 11, 12]
    assert out.name == "possession_state"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/soccer-vision/tests/test_phase_possession_smoothing.py -q`
Expected: FAIL — `ImportError: cannot import name 'smooth_possession'`

- [ ] **Step 3: Add the implementation**

In `packages/soccer-vision/src/soccer_vision/phase/possession.py`, add `from collections import Counter` to the imports at the top of the file, then append this function at the end of the module:

```python
def smooth_possession(possession_state: pd.Series, window_frames: int) -> pd.Series:
    """Mode-smooth the per-frame possession series over a centered window.

    A frame whose raw state is 'contested' is preserved as 'contested'
    (spec §6.1); every other frame takes the modal state of the surrounding
    window_frames-wide window (ties broken by first occurrence). Operates
    positionally on the series as given, so callers should pass a series sorted
    by frame.
    """
    if window_frames <= 1 or len(possession_state) == 0:
        return possession_state.copy()
    states = possession_state.to_list()
    n = len(states)
    half = window_frames // 2
    out: list[str] = []
    for i in range(n):
        if states[i] == "contested":
            out.append("contested")
            continue
        window = states[max(0, i - half):min(n, i + half + 1)]
        out.append(Counter(window).most_common(1)[0][0])
    return pd.Series(out, index=possession_state.index, name=possession_state.name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/soccer-vision/tests/test_phase_possession_smoothing.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check packages/soccer-vision/src/soccer_vision/phase/possession.py packages/soccer-vision/tests/test_phase_possession_smoothing.py && uv run mypy packages/soccer-vision/src/soccer_vision/phase/possession.py`
Expected: All checks pass / Success.

```bash
git add packages/soccer-vision/src/soccer_vision/phase/possession.py packages/soccer-vision/tests/test_phase_possession_smoothing.py
git commit -m "feat(phase): smooth_possession 30-frame modal smoothing"
```

---

## Task 3: `pipeline.py` — `PipelineResult` + `assemble_phases` (pure chain)

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pipeline.py`
- Test: `packages/soccer-vision/tests/test_pipeline_assemble.py`

- [ ] **Step 1: Write the failing integration test**

Create `packages/soccer-vision/tests/test_pipeline_assemble.py`:

```python
"""End-to-end tests for assemble_phases (the integration that was missing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from soccer_vision.pipeline import PipelineResult, assemble_phases
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

FPS = 1.0  # window_frames = round(1.0) = 1 -> smoothing is a no-op, so per-frame states show through


def _identity_keypoints(n_frames: int) -> pd.DataFrame:
    """6 landmarks per frame with image points == pitch coords -> identity H."""
    idxs = [0, 5, 13, 16, 24, 29]
    pts = PITCH_LANDMARKS[idxs]
    rows = []
    for f in range(n_frames):
        for k, (x, y) in zip(idxs, pts):
            rows.append({"frame": f, "kp_idx": k, "x_px": float(x), "y_px": float(y), "conf": 0.9})
    return pd.DataFrame(rows)


def _det(frame, track_id, x, y, cls, team, conf=0.9):
    # Identity homography -> x_px/y_px ARE the pitch coords.
    return {
        "frame": frame, "t_seconds": frame / FPS, "track_id": track_id,
        "x_px": x, "y_px": y,
        "bbox_x1": x - 0.01, "bbox_y1": y - 0.01, "bbox_x2": x + 0.01, "bbox_y2": y + 0.01,
        "class": cls, "team": team, "conf": conf,
    }


def _scene() -> pd.DataFrame:
    rows = []
    # own player track 1: own third, team flickers own/opp/own -> modal "own"
    for f, team in zip(range(3), ["own", "opp", "own"]):
        rows.append(_det(f, 1, 0.50, 0.25, "player", team))
    # opp player track 101: opp end, always "opp"
    for f in range(3):
        rows.append(_det(f, 101, 0.50, 0.85, "player", "opp"))
    # ball: f0 near own (build), f1 mid loose, f2 near opp (defend_high)
    rows.append(_det(0, -1, 0.50, 0.27, "ball", "unknown"))
    rows.append(_det(1, -2, 0.50, 0.55, "ball", "unknown"))
    rows.append(_det(2, -3, 0.50, 0.80, "ball", "unknown"))
    # adjacent-game (off-pitch) detection in f0: x_pitch 1.40 > 1+margin -> dropped
    rows.append(_det(0, 999, 1.40, 0.50, "player", "unknown"))
    df = pd.DataFrame(rows)
    return df.astype({"frame": "int64", "track_id": "int64"})


def test_assemble_phases_end_to_end() -> None:
    traj = _scene()
    kp = _identity_keypoints(3)
    result = assemble_phases(traj, kp, fps=FPS, total_frames=3)

    assert isinstance(result, PipelineResult)
    # Enriched trajectories carry pitch coords.
    assert "x_pitch" in result.trajectories.columns
    assert "y_pitch" in result.trajectories.columns
    # Off-pitch adjacent-game detection was dropped.
    assert (result.trajectories["track_id"] == 999).sum() == 0
    # Team flicker on track 1 resolved to modal "own".
    own_rows = result.trajectories[result.trajectories["track_id"] == 1]
    assert set(own_rows["team"]) == {"own"}
    # Per-frame phases over the full [0, 3) range.
    phases = result.phases.set_index("frame")
    assert list(result.phases["frame"]) == [0, 1, 2]
    assert phases.loc[0, "possession_state"] == "own"
    assert phases.loc[0, "phase"] == "build"
    assert phases.loc[1, "possession_state"] == "loose_ball"
    assert phases.loc[1, "phase"] == "loose_ball"
    assert phases.loc[2, "possession_state"] == "opp"
    assert phases.loc[2, "phase"] == "defend_high"
    # Coverage stats.
    assert result.homography_coverage == 1.0
    assert result.ball_coverage == 1.0


def test_assemble_phases_no_homography_degrades_to_unknown() -> None:
    traj = _scene()
    empty_kp = pd.DataFrame(columns=["frame", "kp_idx", "x_px", "y_px", "conf"])
    result = assemble_phases(traj, empty_kp, fps=FPS, total_frames=3)
    assert result.homography_coverage == 0.0
    assert set(result.phases["possession_state"]) == {"unknown"}
    assert set(result.phases["phase"]) == {"unknown"}


def test_assemble_phases_fills_full_frame_range() -> None:
    traj = _scene()
    kp = _identity_keypoints(3)
    result = assemble_phases(traj, kp, fps=FPS, total_frames=5)  # 2 trailing empty frames
    assert list(result.phases["frame"]) == [0, 1, 2, 3, 4]
    assert result.phases.set_index("frame").loc[4, "possession_state"] == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/soccer-vision/tests/test_pipeline_assemble.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'soccer_vision.pipeline'`

- [ ] **Step 3: Write the implementation**

Create `packages/soccer-vision/src/soccer_vision/pipeline.py`:

```python
"""Pipeline orchestrator: chains pitch + phase modules into enriched outputs.

assemble_phases is pure (no models, no GPU, no ultralytics/sports import) so the
integration logic is testable without a GPU. analyze_video / assemble_from_parquet
add model invocation and parquet I/O around it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from soccer_vision.io.schema import validate_trajectories
from soccer_vision.phase.possession import (
    PossessionThresholds,
    classify_possession,
    smooth_possession,
)
from soccer_vision.phase.splitter import label_phase
from soccer_vision.phase.team_mode import apply_modal_team_per_track
from soccer_vision.pitch.filter import filter_outside_pitch
from soccer_vision.pitch.homography import smooth_homographies
from soccer_vision.pitch.landmarks import build_frame_homographies
from soccer_vision.pitch.mapper import PitchMapper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Enriched outputs of the pipeline plus coverage diagnostics."""

    trajectories: pd.DataFrame   # per-detection, +x_pitch/+y_pitch, team modal-cleaned
    phases: pd.DataFrame         # per-frame over [0, total_frames)
    homography_coverage: float   # fraction of frames with a smoothed H
    ball_coverage: float         # fraction of frames with a non-NaN ball pitch coord


def assemble_phases(
    trajectories_px: pd.DataFrame,
    keypoints: pd.DataFrame,
    fps: float,
    total_frames: int,
    *,
    kp_conf_threshold: float = 0.5,
    homography_alpha: float = 0.5,
    filter_margin: float = 0.05,
    possession_thresholds: PossessionThresholds | None = None,
    transition_seconds: float = 5.0,
) -> PipelineResult:
    """Run the full pitch + phase chain on tracker output. Pure; no I/O."""
    raw_h = build_frame_homographies(keypoints, conf_threshold=kp_conf_threshold)
    homographies = smooth_homographies(raw_h, alpha=homography_alpha)
    if not homographies:
        logger.warning("No homographies fitted; pitch coords NaN, phases all 'unknown'.")

    enriched = PitchMapper().transform(trajectories_px, homographies)
    enriched = filter_outside_pitch(enriched, margin=filter_margin)
    enriched = apply_modal_team_per_track(enriched)
    validate_trajectories(enriched)

    poss = classify_possession(enriched, possession_thresholds).sort_index()
    window = max(1, round(fps))
    poss_smoothed = smooth_possession(poss, window_frames=window)

    ball = enriched[enriched["class"] == "ball"]
    ball_by_frame = ball.groupby("frame")[["x_pitch", "y_pitch"]].first()

    full_index = pd.RangeIndex(0, total_frames, name="frame")
    poss_full = poss_smoothed.reindex(full_index, fill_value="unknown")
    ball_x_full = ball_by_frame["x_pitch"].reindex(full_index)
    ball_y_full = ball_by_frame["y_pitch"].reindex(full_index)

    phase_series = label_phase(
        poss_full, ball_y_full, fps, transition_seconds=transition_seconds
    )

    phases = pd.DataFrame({
        "frame": full_index,
        "t_seconds": full_index.to_numpy() / fps,
        "possession_state": poss_full.to_numpy(),
        "phase": phase_series.to_numpy(),
        "ball_x_pitch": ball_x_full.to_numpy(),
        "ball_y_pitch": ball_y_full.to_numpy(),
    }).astype({
        "frame": "int64",
        "t_seconds": "float64",
        "possession_state": "object",
        "phase": "object",
        "ball_x_pitch": "float64",
        "ball_y_pitch": "float64",
    })

    hom_cov = len(homographies) / total_frames if total_frames else 0.0
    ball_cov = float(ball_y_full.notna().sum()) / total_frames if total_frames else 0.0

    return PipelineResult(
        trajectories=enriched,
        phases=phases,
        homography_coverage=hom_cov,
        ball_coverage=ball_cov,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/soccer-vision/tests/test_pipeline_assemble.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check packages/soccer-vision/src/soccer_vision/pipeline.py packages/soccer-vision/tests/test_pipeline_assemble.py && uv run mypy packages/soccer-vision/src/soccer_vision/pipeline.py`
Expected: All checks pass / Success.

```bash
git add packages/soccer-vision/src/soccer_vision/pipeline.py packages/soccer-vision/tests/test_pipeline_assemble.py
git commit -m "feat(pipeline): assemble_phases pure chain + PipelineResult"
```

---

## Task 4: `assemble_from_parquet` + parquet writers

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pipeline.py`
- Test: `packages/soccer-vision/tests/test_pipeline_assemble.py` (add to existing file)

- [ ] **Step 1: Write the failing test**

Append to `packages/soccer-vision/tests/test_pipeline_assemble.py`:

```python
def test_assemble_from_parquet_roundtrip(tmp_path) -> None:
    from soccer_vision.pipeline import assemble_from_parquet

    traj = _scene()
    kp = _identity_keypoints(3)
    traj_path = tmp_path / "trajectories_px.parquet"
    kp_path = tmp_path / "keypoints.parquet"
    traj.to_parquet(traj_path, index=False)
    kp.to_parquet(kp_path, index=False)
    out_dir = tmp_path / "out"

    result = assemble_from_parquet(traj_path, kp_path, out_dir)

    # fps inferred from t_seconds (= 1.0 here), total_frames from max(frame)+1 (= 3).
    assert list(result.phases["frame"]) == [0, 1, 2]
    # Deliverables written.
    assert (out_dir / "trajectories.parquet").exists()
    assert (out_dir / "phases.parquet").exists()
    reloaded = pd.read_parquet(out_dir / "trajectories.parquet")
    assert "x_pitch" in reloaded.columns
    reloaded_phases = pd.read_parquet(out_dir / "phases.parquet")
    assert list(reloaded_phases.columns) == [
        "frame", "t_seconds", "possession_state", "phase", "ball_x_pitch", "ball_y_pitch",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/soccer-vision/tests/test_pipeline_assemble.py::test_assemble_from_parquet_roundtrip -q`
Expected: FAIL — `ImportError: cannot import name 'assemble_from_parquet'`

- [ ] **Step 3: Add the implementation**

Add `from pathlib import Path` to the imports in `pipeline.py`, then append:

```python
def _infer_fps(trajectories_px: pd.DataFrame) -> float:
    """Recover fps from a row's frame / t_seconds (t = frame / fps). Defaults to 30."""
    nonzero = trajectories_px[trajectories_px["t_seconds"] > 0]
    if nonzero.empty:
        return 30.0
    row = nonzero.iloc[0]
    return float(row["frame"]) / float(row["t_seconds"])


def _write_deliverables(result: PipelineResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.trajectories.to_parquet(out_dir / "trajectories.parquet", index=False)
    result.phases.to_parquet(out_dir / "phases.parquet", index=False)


def assemble_from_parquet(
    trajectories_px_path: Path,
    keypoints_path: Path,
    out_dir: Path,
    *,
    fps: float | None = None,
    **assemble_opts: object,
) -> PipelineResult:
    """Re-run the pure assembly stage from a Stage-1 checkpoint and write deliverables.

    This is the cheap-recompute path: tweak thresholds without re-running GPU tracking.
    """
    trajectories_px = pd.read_parquet(trajectories_px_path)
    keypoints = pd.read_parquet(keypoints_path)
    resolved_fps = fps if fps is not None else _infer_fps(trajectories_px)
    total_frames = int(trajectories_px["frame"].max()) + 1 if not trajectories_px.empty else 0
    result = assemble_phases(
        trajectories_px, keypoints, fps=resolved_fps, total_frames=total_frames, **assemble_opts  # type: ignore[arg-type]
    )
    _write_deliverables(result, Path(out_dir))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/soccer-vision/tests/test_pipeline_assemble.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check packages/soccer-vision/src/soccer_vision/pipeline.py packages/soccer-vision/tests/test_pipeline_assemble.py && uv run mypy packages/soccer-vision/src/soccer_vision/pipeline.py`
Expected: All checks pass / Success.

```bash
git add packages/soccer-vision/src/soccer_vision/pipeline.py packages/soccer-vision/tests/test_pipeline_assemble.py
git commit -m "feat(pipeline): assemble_from_parquet + parquet writers"
```

---

## Task 5: `analyze_video` — thin GPU wrapper + Stage-1 checkpoint

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pipeline.py`
- Test: `packages/soccer-vision/tests/test_pipeline_analyze_video.py`

- [ ] **Step 1: Write the failing stub-backend test**

Create `packages/soccer-vision/tests/test_pipeline_analyze_video.py`:

```python
"""Wiring test for analyze_video using a stub backend (no GPU / no real video)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from soccer_vision.pipeline import PipelineResult, analyze_video
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

FPS = 1.0


def _identity_keypoints(n_frames: int) -> pd.DataFrame:
    idxs = [0, 5, 13, 16, 24, 29]
    pts = PITCH_LANDMARKS[idxs]
    rows = []
    for f in range(n_frames):
        for k, (x, y) in zip(idxs, pts):
            rows.append({"frame": f, "kp_idx": k, "x_px": float(x), "y_px": float(y), "conf": 0.9})
    return pd.DataFrame(rows)


def _trajectories() -> pd.DataFrame:
    rows = []
    for f in range(3):
        rows.append({
            "frame": f, "t_seconds": f / FPS, "track_id": 1,
            "x_px": 0.5, "y_px": 0.25,
            "bbox_x1": 0.49, "bbox_y1": 0.24, "bbox_x2": 0.51, "bbox_y2": 0.26,
            "class": "player", "team": "own", "conf": 0.9,
        })
        rows.append({
            "frame": f, "t_seconds": f / FPS, "track_id": -1 - f,
            "x_px": 0.5, "y_px": 0.27,
            "bbox_x1": 0.49, "bbox_y1": 0.26, "bbox_x2": 0.51, "bbox_y2": 0.28,
            "class": "ball", "team": "unknown", "conf": 0.9,
        })
    return pd.DataFrame(rows).astype({"frame": "int64", "track_id": "int64"})


class _StubBackend:
    name = "stub"
    version = "0"

    def __init__(self, traj: pd.DataFrame, kp: pd.DataFrame) -> None:
        self._traj = traj
        self._kp = kp

    def process_with_pitch(self, video_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._traj, self._kp


def test_analyze_video_writes_four_parquets(tmp_path: Path) -> None:
    backend = _StubBackend(_trajectories(), _identity_keypoints(3))
    out_dir = tmp_path / "game1"

    result = analyze_video(Path("unused.mp4"), out_dir, backend=backend)

    assert isinstance(result, PipelineResult)
    for name in ("trajectories_px.parquet", "keypoints.parquet",
                 "trajectories.parquet", "phases.parquet"):
        assert (out_dir / name).exists(), f"missing {name}"
    # Checkpoint is the verbatim tracker output.
    checkpoint = pd.read_parquet(out_dir / "trajectories_px.parquet")
    assert "x_pitch" not in checkpoint.columns
    # Deliverable is enriched.
    enriched = pd.read_parquet(out_dir / "trajectories.parquet")
    assert "x_pitch" in enriched.columns
    assert list(result.phases["frame"]) == [0, 1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/soccer-vision/tests/test_pipeline_analyze_video.py -q`
Expected: FAIL — `ImportError: cannot import name 'analyze_video'`

- [ ] **Step 3: Add the implementation**

Add `from typing import Any` to the imports in `pipeline.py`, then append:

```python
def analyze_video(
    video_path: Path,
    out_dir: Path,
    *,
    backend: Any | None = None,
    **assemble_opts: object,
) -> PipelineResult:
    """Run the full pipeline on a video and write checkpoints + deliverables.

    Stage 1 (GPU): run the backend's pitch-aware tracking and checkpoint the raw
    px trajectories + keypoints. Stage 2 (pure): assemble and write deliverables.
    fps and total_frames are derived from the tracker output so this is testable
    with a stub backend and needs no second video read.
    """
    if backend is None:
        from soccer_vision.tracking.roboflow import RoboflowBackend  # lazy: avoids roboflow extra at import
        backend = RoboflowBackend(detect_pitch=True)

    trajectories_px, keypoints = backend.process_with_pitch(video_path)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trajectories_px.to_parquet(out / "trajectories_px.parquet", index=False)
    keypoints.to_parquet(out / "keypoints.parquet", index=False)

    resolved_fps = _infer_fps(trajectories_px)
    total_frames = int(trajectories_px["frame"].max()) + 1 if not trajectories_px.empty else 0
    result = assemble_phases(
        trajectories_px, keypoints, fps=resolved_fps, total_frames=total_frames, **assemble_opts  # type: ignore[arg-type]
    )
    _write_deliverables(result, out)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/soccer-vision/tests/test_pipeline_analyze_video.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Full verification, then commit**

Run: `uv run ruff check packages/soccer-vision/ && uv run mypy packages/soccer-vision/src && uv run pytest packages/soccer-vision/tests/ -q`
Expected: All checks pass / Success / all tests pass (no failures).

```bash
git add packages/soccer-vision/src/soccer_vision/pipeline.py packages/soccer-vision/tests/test_pipeline_analyze_video.py
git commit -m "feat(pipeline): analyze_video wrapper + Stage-1 checkpoint"
```

---

## Self-Review (completed during planning)

**Spec coverage:**
- §2 architecture (assemble_phases / analyze_video / assemble_from_parquet) → Tasks 3, 5, 4.
- §2.1 four-file output layout → Task 5 writes checkpoints; Tasks 3–5 write deliverables.
- §3 data-flow order (map → filter → modal-team → possession → smooth → phase) → Task 3 body.
- §4.1 `pitch/landmarks.py` (PITCH_LANDMARKS + build_frame_homographies) → Task 1.
- §4.2 `smooth_possession` (preserving contested) → Task 2.
- §5 public API (PipelineResult + three functions) → Tasks 3–5.
- §6 parquet schemas (enriched trajectories + per-frame phases over full range) → Tasks 3–4, asserted in tests.
- §7 failure handling (no-homography degrade, full-range fill) → Task 3 tests `test_assemble_phases_no_homography_degrades_to_unknown`, `test_assemble_phases_fills_full_frame_range`.
- §8 testing (round-trip H, smoothing, integration, from-parquet, stub-backend) → Tasks 1–5 tests.

**Placeholder scan:** none — every code step has complete code; every command has expected output.

**Type consistency:** `PipelineResult` fields, `assemble_phases` signature, `_infer_fps`, `_write_deliverables`, and `build_frame_homographies`/`smooth_possession` signatures are used identically across Tasks 3–5.

**Known implementation-time check (from spec §4.1):** after a real Colab run, verify the length/width axis orientation in `PITCH_LANDMARKS` and the `_CLUSTER_TEAM = {0: own, 1: opp}` mapping against the bake-off clip; "own"/"opp" half correctness depends on both. A flip is a one-line fix. This is verification, not a code task.
