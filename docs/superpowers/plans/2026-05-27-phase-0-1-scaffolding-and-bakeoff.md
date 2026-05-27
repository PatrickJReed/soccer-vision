# soccer-vision — Phase 0 + 1 Implementation Plan (Scaffolding + Bake-off)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `soccer-vision` Python package with CI, then run a structured bake-off of three soccer-CV repos against a 2-minute Trace clip and commit to a tracking backend.

**Architecture:** uv workspace monorepo (rarecell-style) with one package `soccer-vision`. A `TrackingBackend` Protocol defines the contract every detection backend must satisfy. Phase 1 runs each candidate in its own Colab notebook (avoids dependency conflicts), produces a standardized trajectories parquet per candidate, then a synthesis notebook scores them side-by-side.

**Tech Stack:** Python 3.11 (Colab Pro default), uv, pandas, pyarrow, pydantic, pytest, ruff, mypy, ffmpeg, OpenCV, GitHub Actions. Bake-off uses each candidate's native stack (Ultralytics YOLOv8, supervision, etc.) inside its own notebook.

**Scope of this plan:** Phase 0 + Phase 1 only. Plan B (Phases 2–3) is written *after* this plan completes — it depends on bake-off outcomes.

**Spec reference:** [`docs/superpowers/specs/2026-05-27-soccer-vision-design.md`](../specs/2026-05-27-soccer-vision-design.md)

---

## File Structure

Files created by this plan:

```
soccer-vision/
├── pyproject.toml                                          # uv workspace root
├── uv.lock                                                 # generated
├── .gitignore
├── .github/workflows/ci.yml                                # CI pipeline
├── README.md                                               # (already exists, expanded)
├── packages/soccer-vision/
│   ├── pyproject.toml
│   ├── src/soccer_vision/
│   │   ├── __init__.py
│   │   ├── py.typed                                        # PEP 561 marker
│   │   ├── tracking/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                                     # TrackingBackend Protocol
│   │   │   └── <winner>.py                                 # Filled by Task 18
│   │   ├── pitch/__init__.py                               # stub
│   │   ├── phase/__init__.py                               # stub
│   │   ├── metrics/__init__.py                             # stub
│   │   ├── io/
│   │   │   ├── __init__.py
│   │   │   └── schema.py                                   # trajectories parquet schema
│   │   └── viz/__init__.py                                 # stub
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_schema.py
│       ├── test_tracking_protocol.py
│       └── test_adapter_<winner>.py                        # Filled by Task 19
├── examples/
│   ├── bakeoff_roboflow.ipynb                              # Phase 1 candidate runner
│   ├── bakeoff_atfa.ipynb                                  # Phase 1 candidate runner
│   ├── bakeoff_tz.ipynb                                    # Phase 1 candidate runner
│   └── colab_bakeoff.ipynb                                 # Phase 1 synthesis
├── data/
│   ├── bakeoff_clip.mp4                                    # 2-minute canonical clip
│   ├── bakeoff_clip_corners.json                           # manual 4-corner annotation
│   └── bakeoff_outputs/
│       ├── roboflow/trajectories.parquet                   # generated
│       ├── atfa/trajectories.parquet                       # generated
│       ├── tz/trajectories.parquet                         # generated
│       ├── roboflow/annotated.mp4
│       ├── atfa/annotated.mp4
│       └── tz/annotated.mp4
└── docs/
    └── superpowers/
        ├── specs/2026-05-27-soccer-vision-design.md        # (already exists)
        ├── plans/2026-05-27-phase-0-1-scaffolding-and-bakeoff.md  # THIS FILE
        └── bakeoff-results.md                              # Phase 1 deliverable
```

**Design responsibilities per file:**

- `tracking/base.py` — pure typing module: `TrackingBackend` Protocol, no implementation. Single responsibility: define the contract.
- `io/schema.py` — pydantic models that validate the trajectories DataFrame columns and dtypes. Single responsibility: schema enforcement.
- `tracking/<winner>.py` — concrete adapter wrapping one upstream repo, conforming to `TrackingBackend`. Filled after bake-off chooses the winner.
- Each bake-off notebook runs ONE candidate in its own Colab runtime to avoid cross-repo dependency conflicts.
- `colab_bakeoff.ipynb` is the synthesis — reads the three parquets/videos and renders side-by-side panels for visual scoring.

---

## Phase 0: Scaffolding

### Task 1: Create uv workspace and root `pyproject.toml`

**Files:**
- Create: `pyproject.toml` (workspace root)
- Create: `.python-version`

- [ ] **Step 1: Create the root `pyproject.toml`**

```toml
[project]
name = "soccer-vision-workspace"
version = "0.0.0"
description = "Workspace root for soccer-vision"
requires-python = ">=3.11,<3.13"

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.6",
    "mypy>=1.11",
    "pandas-stubs",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "RUF"]
ignore = ["E501"]  # line length handled by formatter

[tool.mypy]
strict = true
python_version = "3.11"
files = ["packages/soccer-vision/src", "packages/soccer-vision/tests"]

[tool.pytest.ini_options]
testpaths = ["packages/soccer-vision/tests"]
addopts = "--cov=soccer_vision --cov-report=term-missing"
```

- [ ] **Step 2: Pin Python version**

Write `.python-version`:

```
3.11
```

- [ ] **Step 3: Verify uv recognizes the workspace**

Run: `cd ~/Sandbox/soccer-vision && uv sync 2>&1 | head -20`

Expected: uv installs Python 3.11 if needed, creates `.venv/`, exits with status 0. No `uv.lock` yet because no members exist.

- [ ] **Step 4: Commit**

```bash
cd ~/Sandbox/soccer-vision
git add pyproject.toml .python-version
git commit -m "chore: initialize uv workspace with Python 3.11"
```

---

### Task 2: Create the `soccer-vision` package skeleton

**Files:**
- Create: `packages/soccer-vision/pyproject.toml`
- Create: `packages/soccer-vision/src/soccer_vision/__init__.py`
- Create: `packages/soccer-vision/src/soccer_vision/py.typed`
- Create: `packages/soccer-vision/src/soccer_vision/tracking/__init__.py`
- Create: `packages/soccer-vision/src/soccer_vision/pitch/__init__.py`
- Create: `packages/soccer-vision/src/soccer_vision/phase/__init__.py`
- Create: `packages/soccer-vision/src/soccer_vision/metrics/__init__.py`
- Create: `packages/soccer-vision/src/soccer_vision/io/__init__.py`
- Create: `packages/soccer-vision/src/soccer_vision/viz/__init__.py`
- Create: `packages/soccer-vision/tests/__init__.py`
- Create: `packages/soccer-vision/tests/conftest.py`

- [ ] **Step 1: Create the package `pyproject.toml`**

```toml
[project]
name = "soccer-vision"
version = "0.1.0"
description = "Team-level positional analytics for 9v9 youth soccer from Trace camera footage"
requires-python = ">=3.11,<3.13"
dependencies = [
    "pandas>=2.2",
    "pyarrow>=17.0",
    "numpy>=1.26",
    "pydantic>=2.7",
    "scipy>=1.13",
    "shapely>=2.0",
    "matplotlib>=3.9",
    "opencv-python>=4.9",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/soccer_vision"]
```

- [ ] **Step 2: Create the package `__init__.py`**

Write `packages/soccer-vision/src/soccer_vision/__init__.py`:

```python
"""soccer-vision: team-level positional analytics for 9v9 youth soccer."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create PEP 561 marker**

Write `packages/soccer-vision/src/soccer_vision/py.typed` as an empty file:

```bash
touch packages/soccer-vision/src/soccer_vision/py.typed
```

- [ ] **Step 4: Stub the sub-packages**

For each of `tracking`, `pitch`, `phase`, `metrics`, `io`, `viz`, write `packages/soccer-vision/src/soccer_vision/<name>/__init__.py`:

```python
"""soccer_vision.<name> — see soccer_vision package README."""
```

(Replace `<name>` with each sub-package name. They start empty.)

- [ ] **Step 5: Stub the test package**

Write `packages/soccer-vision/tests/__init__.py` as empty.

Write `packages/soccer-vision/tests/conftest.py`:

```python
"""Shared pytest fixtures for soccer_vision tests."""
```

- [ ] **Step 6: Sync uv and verify import**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv sync
uv run python -c "import soccer_vision; print(soccer_vision.__version__)"
```

Expected: prints `0.1.0`.

- [ ] **Step 7: Commit**

```bash
git add packages/
git commit -m "feat: create soccer-vision package skeleton with sub-package stubs"
```

---

### Task 3: Wire ruff and mypy

**Files:**
- Verify: root `pyproject.toml` already has ruff + mypy config (from Task 1)
- No new files

- [ ] **Step 1: Run ruff on the empty tree**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 2: Run mypy on the empty tree**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run mypy
```

Expected: `Success: no issues found in N source files.`

- [ ] **Step 3: Commit nothing (verification only)**

No commit — just confirming the lint/type setup is functional.

---

### Task 4: Configure pytest with a smoke test

**Files:**
- Create: `packages/soccer-vision/tests/test_smoke.py`

- [ ] **Step 1: Write a smoke test**

Write `packages/soccer-vision/tests/test_smoke.py`:

```python
"""Smoke tests — package imports and version is set."""

import soccer_vision


def test_package_imports() -> None:
    assert soccer_vision is not None


def test_version_is_set() -> None:
    assert soccer_vision.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run pytest
```

Expected: 2 passed, coverage report printed.

- [ ] **Step 3: Commit**

```bash
git add packages/soccer-vision/tests/test_smoke.py
git commit -m "test: add smoke tests for package import and version"
```

---

### Task 5: Configure `.gitignore`

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Write `.gitignore`**

Write `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
.python-version-local
*.egg-info/

# uv
.uv-cache/

# Tests / coverage
.pytest_cache/
.coverage
htmlcov/
.ruff_cache/
.mypy_cache/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store

# Editor
.vscode/
.idea/

# Data — NOT gitignored: data/bakeoff_clip.mp4, data/bakeoff_clip_corners.json
data/games/
data/labeled/
data/bakeoff_outputs/

# Models (large binaries) — fetched via Colab
packages/soccer-vision/src/soccer_vision/models/*.pt

# OS extras
*.swp
```

- [ ] **Step 2: Verify intentional commits aren't ignored**

Run:

```bash
cd ~/Sandbox/soccer-vision
git check-ignore -v data/bakeoff_clip.mp4 || echo "OK: bakeoff_clip.mp4 will be tracked"
git check-ignore -v data/games/ || echo "FAIL: data/games should be ignored"
```

Expected: first command says "OK"; second prints the ignore rule path.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: configure .gitignore"
```

---

### Task 6: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the CI workflow**

Write `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Sync deps
        run: uv sync

      - name: Run ruff
        run: uv run ruff check .

      - name: Run mypy
        run: uv run mypy

      - name: Run pytest
        run: uv run pytest -v
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for lint and test"
```

- [ ] **Step 3: (User action) Push to GitHub when remote is ready**

This task does not push to a remote yet (none configured). When a GitHub remote is added in a later step, the CI will run automatically on push. Locally, all checks already pass from earlier tasks.

---

### Task 7: Expand the README with a quickstart

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md` content**

Write `README.md`:

```markdown
# soccer-vision

Team-level positional analytics for 9v9 youth soccer from Trace camera footage.

**Status:** Phase 0 complete (scaffolding). Phase 1 (structured bake-off) next.

## What this is

A notebook-driven Python toolkit that ingests Trace footage and emits
team-level positional analytics: shape, space and pitch ownership, gaps,
ball-relative positioning, dynamics, zonal occupation, and youth-specific
metrics like a swarm index. Output is parquet tables, static plots, a per-game
HTML report, video overlays, and season-level analyses.

Scope is deliberately narrow: positional (no passing yet), team-level (no
player-specific work), post-game (no realtime), notebook-driven (no web app).

## Quickstart

Local development:

```bash
git clone <repo-url>
cd soccer-vision
uv sync
uv run pytest
```

Colab usage (per notebook):

```python
!pip install -q git+https://github.com/<owner>/soccer-vision.git
import soccer_vision
```

## Layout

- `packages/soccer-vision/` — core Python package (uv workspace member)
- `examples/` — Colab notebooks (bake-off, pipeline demo, season analysis)
- `data/` — canonical bake-off clip + manual annotations
- `docs/superpowers/specs/` — design specifications
- `docs/superpowers/plans/` — implementation plans

## Design

See [`docs/superpowers/specs/2026-05-27-soccer-vision-design.md`](docs/superpowers/specs/2026-05-27-soccer-vision-design.md).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: expand README with quickstart and layout"
```

---

## Phase 1: Bake-off

### Task 8: Install ffmpeg and extract the 2-minute bake-off clip

**Files:**
- Create: `data/bakeoff_clip.mp4`

- [ ] **Step 1: Install ffmpeg**

Run:

```bash
brew install ffmpeg
ffmpeg -version | head -1
```

Expected: prints an ffmpeg version line.

- [ ] **Step 2: Unzip the Trace source file to a working location**

Run:

```bash
mkdir -p ~/Sandbox/soccer-vision/data/_source
cd ~/Sandbox/soccer-vision/data/_source
unzip -o ~/Desktop/PatrickReed_Trace.mov.zip
ls -la
```

Expected: a `.mov` file appears in `data/_source/`. Note the filename.

- [ ] **Step 3: Inspect the source to choose a 2-minute segment**

Run:

```bash
ffprobe -v error -show_format -show_streams ~/Sandbox/soccer-vision/data/_source/*.mov | grep -E "duration|width|height|r_frame_rate"
```

Expected: prints duration (seconds), width, height, frame rate. Confirm resolution is sensible (≥1080p) and duration is full-game length.

- [ ] **Step 4: Choose a 2-minute segment**

Pick a start timestamp in the source that contains the bake-off clip criteria from the spec §4.3:
- First 60 s: build-up play with all players visible, moderate pace
- Second 60 s: a transition or counterattack with fast motion and player occlusion

Note the chosen start time as `START_TS` (format `HH:MM:SS`). Document the choice in a comment in Task 11's annotation file.

- [ ] **Step 5: Extract the clip**

Replace `START_TS` and `SOURCE` below with the chosen values:

```bash
SOURCE=~/Sandbox/soccer-vision/data/_source/PatrickReed_Trace.mov
START_TS=00:15:00  # ← REPLACE with chosen timestamp
ffmpeg -ss "$START_TS" -i "$SOURCE" -t 120 \
    -c:v libx264 -preset slow -crf 18 \
    -an \
    ~/Sandbox/soccer-vision/data/bakeoff_clip.mp4
```

Notes:
- `-c:v libx264 -preset slow -crf 18` re-encodes for consistent decode across the three candidate notebooks. Crf 18 keeps near-lossless quality.
- `-an` strips audio (irrelevant to detection).

- [ ] **Step 6: Verify the clip**

Run:

```bash
ffprobe -v error -show_format ~/Sandbox/soccer-vision/data/bakeoff_clip.mp4 | grep duration
ls -lh ~/Sandbox/soccer-vision/data/bakeoff_clip.mp4
```

Expected: duration ≈ 120 seconds; file size 50–300 MB depending on motion.

- [ ] **Step 7: Commit the clip**

```bash
cd ~/Sandbox/soccer-vision
git add data/bakeoff_clip.mp4
git commit -m "data: add 2-minute canonical bake-off clip from Trace footage"
```

(The clip is tracked because it's the canonical evaluation fixture. Source `.mov` is in `data/_source/` which is *not* gitignored — see Step 8.)

- [ ] **Step 8: Add `_source/` to `.gitignore`**

Edit `.gitignore` to add a new line:

```gitignore
data/_source/
```

Then:

```bash
git add .gitignore
git commit -m "chore: gitignore Trace source under data/_source/"
```

---

### Task 9: Hand-annotate the four pitch corners

**Files:**
- Create: `data/bakeoff_clip_corners.json`
- Create: `scripts/annotate_corners.py`

- [ ] **Step 1: Write the corner-annotation helper script**

Write `scripts/annotate_corners.py`:

```python
"""Display first frame of bakeoff_clip.mp4; user clicks 4 pitch corners.

Outputs pixel coordinates of (top-left, top-right, bottom-left, bottom-right)
pitch corners as JSON. Pitch corners = visible field-marking corners, not
camera image corners.

Usage:
    uv run python scripts/annotate_corners.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

CLIP = Path(__file__).parent.parent / "data" / "bakeoff_clip.mp4"
OUT = Path(__file__).parent.parent / "data" / "bakeoff_clip_corners.json"
LABELS = ["top_left", "top_right", "bottom_left", "bottom_right"]


def main() -> None:
    cap = cv2.VideoCapture(str(CLIP))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame of {CLIP}")

    clicks: list[tuple[int, int]] = []

    def on_click(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            cv2.circle(frame, (x, y), 8, (0, 255, 0), 2)
            cv2.putText(
                frame, LABELS[len(clicks) - 1], (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            cv2.imshow("Click 4 pitch corners (TL, TR, BL, BR)", frame)

    cv2.imshow("Click 4 pitch corners (TL, TR, BL, BR)", frame)
    cv2.setMouseCallback("Click 4 pitch corners (TL, TR, BL, BR)", on_click)

    while len(clicks) < 4:
        if cv2.waitKey(20) & 0xFF == 27:  # Esc
            break
    cv2.destroyAllWindows()

    if len(clicks) != 4:
        raise SystemExit("Did not click 4 corners. Aborting.")

    out = {label: {"x": x, "y": y} for label, (x, y) in zip(LABELS, clicks, strict=True)}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {OUT}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the annotation script**

Run:

```bash
cd ~/Sandbox/soccer-vision
mkdir -p scripts
uv run python scripts/annotate_corners.py
```

A window opens with the first frame of the bake-off clip. Click in order:
1. Top-left pitch corner (back-left from camera view)
2. Top-right pitch corner
3. Bottom-left pitch corner (nearest to camera, left)
4. Bottom-right pitch corner

After 4 clicks the window closes and `data/bakeoff_clip_corners.json` is written.

- [ ] **Step 3: Sanity-check the output**

Run:

```bash
cat ~/Sandbox/soccer-vision/data/bakeoff_clip_corners.json
```

Expected: a JSON object with `top_left`, `top_right`, `bottom_left`, `bottom_right`, each with `x` and `y` integer pixel coords. Top corners should have smaller `y` values than bottom corners.

- [ ] **Step 4: Commit**

```bash
cd ~/Sandbox/soccer-vision
git add scripts/annotate_corners.py data/bakeoff_clip_corners.json
git commit -m "data: annotate 4 pitch corners on bake-off clip for shared homography"
```

---

### Task 10: Define the trajectories DataFrame schema

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/io/schema.py`
- Create: `packages/soccer-vision/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

Write `packages/soccer-vision/tests/test_schema.py`:

```python
"""Tests for the trajectories DataFrame schema."""

from __future__ import annotations

import pandas as pd
import pytest

from soccer_vision.io.schema import (
    REQUIRED_COLUMNS,
    TrajectorySchemaError,
    validate_trajectories,
)


def _good_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "t_seconds": [0.0, 0.0, 0.033],
            "track_id": [1, 2, 1],
            "x_px": [100.0, 200.0, 102.0],
            "y_px": [300.0, 350.0, 303.0],
            "bbox_x1": [90.0, 190.0, 92.0],
            "bbox_y1": [280.0, 330.0, 283.0],
            "bbox_x2": [110.0, 210.0, 112.0],
            "bbox_y2": [320.0, 370.0, 323.0],
            "class": ["player", "player", "player"],
            "team": ["own", "opp", "own"],
            "conf": [0.95, 0.92, 0.94],
        }
    )


def test_required_columns_listed_exhaustively() -> None:
    assert set(REQUIRED_COLUMNS) == {
        "frame", "t_seconds", "track_id",
        "x_px", "y_px",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "class", "team", "conf",
    }


def test_validate_accepts_good_df() -> None:
    validate_trajectories(_good_df())  # should not raise


def test_validate_rejects_missing_column() -> None:
    df = _good_df().drop(columns=["conf"])
    with pytest.raises(TrajectorySchemaError, match="missing.*conf"):
        validate_trajectories(df)


def test_validate_rejects_unknown_class() -> None:
    df = _good_df()
    df.loc[0, "class"] = "horse"
    with pytest.raises(TrajectorySchemaError, match="class.*horse"):
        validate_trajectories(df)


def test_validate_rejects_unknown_team() -> None:
    df = _good_df()
    df.loc[0, "team"] = "blue"
    with pytest.raises(TrajectorySchemaError, match="team.*blue"):
        validate_trajectories(df)


def test_validate_rejects_negative_frame() -> None:
    df = _good_df()
    df.loc[0, "frame"] = -1
    with pytest.raises(TrajectorySchemaError, match="frame"):
        validate_trajectories(df)


def test_validate_rejects_bad_conf() -> None:
    df = _good_df()
    df.loc[0, "conf"] = 1.5
    with pytest.raises(TrajectorySchemaError, match="conf"):
        validate_trajectories(df)
```

- [ ] **Step 2: Run and verify test fails**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run pytest packages/soccer-vision/tests/test_schema.py -v
```

Expected: 7 errors with `ModuleNotFoundError: No module named 'soccer_vision.io.schema'`.

- [ ] **Step 3: Write `io/schema.py`**

Write `packages/soccer-vision/src/soccer_vision/io/schema.py`:

```python
"""Schema for the canonical trajectories DataFrame.

Every TrackingBackend.process() return value MUST validate against this schema
before downstream code consumes it. Validation is fail-fast and explicit.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

REQUIRED_COLUMNS: Final = (
    "frame",
    "t_seconds",
    "track_id",
    "x_px",
    "y_px",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "class",
    "team",
    "conf",
)

ALLOWED_CLASSES: Final = frozenset({"player", "goalkeeper", "referee", "ball"})
ALLOWED_TEAMS: Final = frozenset({"own", "opp", "ref", "unknown"})


class TrajectorySchemaError(ValueError):
    """Raised when a DataFrame fails trajectories-schema validation."""


def validate_trajectories(df: pd.DataFrame) -> None:
    """Validate that df conforms to the trajectories schema.

    Raises TrajectorySchemaError on the first violation found.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise TrajectorySchemaError(f"trajectories DataFrame missing columns: {sorted(missing)}")

    if (df["frame"] < 0).any():
        raise TrajectorySchemaError("frame column contains negative values")

    if (df["t_seconds"] < 0).any():
        raise TrajectorySchemaError("t_seconds column contains negative values")

    bad_classes = set(df["class"].unique()) - ALLOWED_CLASSES
    if bad_classes:
        raise TrajectorySchemaError(
            f"class column contains unknown values: {sorted(bad_classes)}; "
            f"allowed: {sorted(ALLOWED_CLASSES)}"
        )

    bad_teams = set(df["team"].unique()) - ALLOWED_TEAMS
    if bad_teams:
        raise TrajectorySchemaError(
            f"team column contains unknown values: {sorted(bad_teams)}; "
            f"allowed: {sorted(ALLOWED_TEAMS)}"
        )

    if ((df["conf"] < 0) | (df["conf"] > 1)).any():
        raise TrajectorySchemaError("conf column contains values outside [0, 1]")
```

- [ ] **Step 4: Run and verify tests pass**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run pytest packages/soccer-vision/tests/test_schema.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Run lint + type checks**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run ruff check .
uv run mypy
```

Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/io/schema.py packages/soccer-vision/tests/test_schema.py
git commit -m "feat(io): add trajectories DataFrame schema with validation"
```

---

### Task 11: Define the `TrackingBackend` Protocol

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/tracking/base.py`
- Create: `packages/soccer-vision/tests/test_tracking_protocol.py`

- [ ] **Step 1: Write the failing test**

Write `packages/soccer-vision/tests/test_tracking_protocol.py`:

```python
"""Tests for the TrackingBackend protocol contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from soccer_vision.io.schema import validate_trajectories
from soccer_vision.tracking.base import TrackingBackend


class _MockBackend:
    """A minimal backend that satisfies the protocol structurally."""

    name = "mock"
    version = "0.0.0"

    def process(self, video_path: Path) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame": [0],
                "t_seconds": [0.0],
                "track_id": [1],
                "x_px": [100.0],
                "y_px": [100.0],
                "bbox_x1": [90.0],
                "bbox_y1": [90.0],
                "bbox_x2": [110.0],
                "bbox_y2": [110.0],
                "class": ["player"],
                "team": ["own"],
                "conf": [0.9],
            }
        )


def test_mock_backend_is_a_tracking_backend() -> None:
    backend: TrackingBackend = _MockBackend()
    assert backend.name == "mock"
    assert backend.version == "0.0.0"


def test_mock_backend_output_passes_schema() -> None:
    backend = _MockBackend()
    df = backend.process(Path("/dev/null"))
    validate_trajectories(df)  # should not raise


def test_protocol_is_runtime_checkable() -> None:
    """The protocol should be runtime-checkable for adapter conformance."""
    backend = _MockBackend()
    assert isinstance(backend, TrackingBackend)
```

- [ ] **Step 2: Run and verify test fails**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run pytest packages/soccer-vision/tests/test_tracking_protocol.py -v
```

Expected: 3 errors with `ModuleNotFoundError: No module named 'soccer_vision.tracking.base'`.

- [ ] **Step 3: Write `tracking/base.py`**

Write `packages/soccer-vision/src/soccer_vision/tracking/base.py`:

```python
"""TrackingBackend Protocol: the contract every detection backend must satisfy."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class TrackingBackend(Protocol):
    """A detection + tracking pipeline that consumes a video and returns trajectories.

    Implementations wrap an upstream repo (roboflow/sports, abdullahtarek, etc.)
    or a local model. The returned DataFrame must validate against
    soccer_vision.io.schema.validate_trajectories.
    """

    name: str
    """Stable identifier for the backend, e.g. 'roboflow-sports'."""

    version: str
    """Version string, e.g. the upstream commit SHA or release tag."""

    def process(self, video_path: Path) -> pd.DataFrame:
        """Run detection + tracking on the video at `video_path`.

        Returns a DataFrame with columns defined in
        soccer_vision.io.schema.REQUIRED_COLUMNS.
        """
        ...
```

- [ ] **Step 4: Run and verify tests pass**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run pytest packages/soccer-vision/tests/test_tracking_protocol.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run full test suite + lint + type**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run pytest
uv run ruff check .
uv run mypy
```

Expected: all green. Coverage on schema.py and base.py should be >90%.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/tracking/base.py packages/soccer-vision/tests/test_tracking_protocol.py
git commit -m "feat(tracking): define TrackingBackend Protocol with schema-conformant contract"
```

---

### Task 12: Bake-off candidate notebook — `roboflow/sports`

**Files:**
- Create: `examples/bakeoff_roboflow.ipynb`

This task produces a Colab-runnable notebook that clones `roboflow/sports`, runs its soccer pipeline on `bakeoff_clip.mp4`, and writes a trajectories parquet + annotated video to `data/bakeoff_outputs/roboflow/`.

The notebook is exploratory by design (per the spec, bake-off notebooks are not TDD'd).

- [ ] **Step 1: Create the notebook scaffold**

Create `examples/bakeoff_roboflow.ipynb` with the following cells (use `jupytext` or write JSON directly; structure shown as markdown for readability):

**Cell 1 (markdown):**
```markdown
# Bake-off candidate: roboflow/sports

Runs the soccer example from https://github.com/roboflow/sports on the canonical
bake-off clip. Outputs trajectories parquet + annotated video.
```

**Cell 2 (code) — environment setup:**
```python
!pip install -q "ultralytics>=8.2" "supervision>=0.22" "git+https://github.com/roboflow/sports.git"
!pip install -q git+https://github.com/PatrickJReed/soccer-vision.git
import os
from pathlib import Path
DATA = Path("data")
DATA.mkdir(exist_ok=True)
```

**Cell 3 (code) — fetch inputs from GitHub:**
```python
# In Colab, clone the soccer-vision repo for the bake-off clip + corners
!git clone https://github.com/PatrickJReed/soccer-vision.git /content/sv
INPUT_CLIP = Path("/content/sv/data/bakeoff_clip.mp4")
CORNERS = Path("/content/sv/data/bakeoff_clip_corners.json")
OUT = Path("/content/output/roboflow")
OUT.mkdir(parents=True, exist_ok=True)
assert INPUT_CLIP.exists(), INPUT_CLIP
```

**Cell 4 (code) — run roboflow/sports soccer detection:**
```python
# Use the SoccerPitchDetectionModel + SoccerPlayerDetectionModel from sports/examples/soccer
# The exact import paths depend on the repo layout at clone time; check sports/examples/soccer/main.py.
from sports.common.soccer import SoccerPitchDetectionModel, SoccerPlayerDetectionModel
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv
import cv2

CONFIG = SoccerPitchConfiguration()
player_model = SoccerPlayerDetectionModel()  # uses default weights
pitch_model = SoccerPitchDetectionModel()
byte_tracker = sv.ByteTrack()

cap = cv2.VideoCapture(str(INPUT_CLIP))
fps = cap.get(cv2.CAP_PROP_FPS)
records: list[dict] = []
frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Player detections
    player_dets = player_model.infer(frame)
    tracked = byte_tracker.update_with_detections(player_dets)

    # Each detection becomes a row
    for box, tid, cls, conf in zip(
        tracked.xyxy, tracked.tracker_id, tracked.class_id, tracked.confidence, strict=False
    ):
        if tid is None:
            continue
        x1, y1, x2, y2 = box.tolist()
        cx, cy = (x1 + x2) / 2, y2  # foot position
        cls_name = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}.get(int(cls), "player")
        records.append({
            "frame": frame_idx,
            "t_seconds": frame_idx / fps,
            "track_id": int(tid),
            "x_px": cx,
            "y_px": cy,
            "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
            "class": cls_name,
            "team": "unknown",   # team classification done in Cell 5
            "conf": float(conf),
        })
    frame_idx += 1

cap.release()
print(f"Processed {frame_idx} frames; {len(records)} detection rows.")
```

**Cell 5 (code) — team classification via SigLIP + KMeans:**
```python
# Sketch — actual code follows roboflow/sports/examples/soccer/team.py
# This cell crops each player box, embeds with SigLIP, KMeans into 2 clusters,
# assigns 'own' (cluster 0) vs 'opp' (cluster 1) — arbitrary which is which for now.
import pandas as pd
df = pd.DataFrame(records)
# ... apply SigLIP team classifier from roboflow.sports.examples.soccer.team
# ... update df["team"] with "own"/"opp" assignments
# For the bake-off scoring, we only need 2 stable clusters; which is "own" is fixed later by inspection.
```

**Cell 6 (code) — validate and write trajectories parquet:**
```python
from soccer_vision.io.schema import validate_trajectories
df = df.astype({"frame": "int64", "track_id": "int64"})
validate_trajectories(df)
df.to_parquet(OUT / "trajectories.parquet")
print(f"Wrote {OUT / 'trajectories.parquet'}: {len(df)} rows")
```

**Cell 7 (code) — produce annotated video:**
```python
# Render boxes + IDs onto the clip for visual review in the synthesis notebook
import supervision as sv

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(str(INPUT_CLIP))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(str(OUT / "annotated.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Group by frame, render each
for frame_idx, group in df.groupby("frame"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        break
    # build detections from rows and annotate (full code: use sv.Detections)
    # ... render boxes + labels ...
    writer.write(frame)

writer.release()
cap.release()
print(f"Wrote {OUT / 'annotated.mp4'}")
```

**Cell 8 (code) — upload outputs back to repo (manual step):**
```python
# Download these files from Colab and commit them at:
#   data/bakeoff_outputs/roboflow/trajectories.parquet
#   data/bakeoff_outputs/roboflow/annotated.mp4
# Or push to a Google Drive / Roboflow workspace and reference from the synthesis notebook.
from google.colab import files
files.download(str(OUT / "trajectories.parquet"))
files.download(str(OUT / "annotated.mp4"))
```

- [ ] **Step 2: Add the notebook to git**

Notebook cells with placeholder code (Cell 5) are intentionally sketchy — the actual SigLIP team-classification code comes from `roboflow/sports` and is filled in during the Colab run. The notebook commits as-is for the structural template.

```bash
cd ~/Sandbox/soccer-vision
git add examples/bakeoff_roboflow.ipynb
git commit -m "feat(bakeoff): add Colab runner notebook for roboflow/sports candidate"
```

- [ ] **Step 3: (User action) Run the notebook in Colab Pro**

- Open `examples/bakeoff_roboflow.ipynb` in Colab
- Connect to a T4 or L4 GPU runtime
- Run all cells
- Fill in Cell 5 with the team-classification code from `roboflow/sports/examples/soccer/team.py` as needed
- Download the two output files
- Commit them locally:

```bash
mkdir -p ~/Sandbox/soccer-vision/data/bakeoff_outputs/roboflow
# (copy downloaded files into that dir)
cd ~/Sandbox/soccer-vision
git add data/bakeoff_outputs/roboflow/
git commit -m "data(bakeoff): roboflow/sports candidate outputs"
```

---

### Task 13: Bake-off candidate notebook — `abdullahtarek/football_analysis`

**Files:**
- Create: `examples/bakeoff_atfa.ipynb`

Mirrors Task 12 structurally. Differences:
- Clones `https://github.com/abdullahtarek/football_analysis`
- Uses ATFA's `Tracker` class (`tracker/tracker.py`) which wraps YOLOv8 + supervision ByteTrack
- Team classification via ATFA's KMeans-on-shirt-pixel module (`team_assigner/team_assigner.py`)
- ATFA emits a per-frame dict; map to schema columns same way as Task 12

- [ ] **Step 1: Create the notebook scaffold**

Create `examples/bakeoff_atfa.ipynb` with cells following the same 8-cell pattern as Task 12. Cell 2 environment setup becomes:

```python
!pip install -q "ultralytics>=8.2" "supervision>=0.22" "scikit-learn>=1.5"
!git clone https://github.com/abdullahtarek/football_analysis /content/atfa
!pip install -q git+https://github.com/PatrickJReed/soccer-vision.git
import sys; sys.path.insert(0, "/content/atfa")
from pathlib import Path
```

Cell 4 (detection + tracking) imports ATFA's `Tracker`:

```python
from trackers.tracker import Tracker  # ATFA layout
tracker = Tracker("yolov8x.pt")  # or whichever weights file is referenced in ATFA README
tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)
# tracks is a dict with keys 'players', 'referees', 'ball' -> list-per-frame of {track_id: bbox}
```

Map ATFA's `tracks` dict into the schema-conformant DataFrame structure (frame, t_seconds, track_id, x_px, y_px, bbox_*, class, team, conf).

Cell 5 (team classification) calls ATFA's `TeamAssigner`:

```python
from team_assigner.team_assigner import TeamAssigner
team_assigner = TeamAssigner()
team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
# Loop over frames + players to assign team for each row in df
```

Cells 6, 7, 8 are identical to Task 12 (validate, write parquet, render annotated video, download).

- [ ] **Step 2: Commit the scaffold**

```bash
cd ~/Sandbox/soccer-vision
git add examples/bakeoff_atfa.ipynb
git commit -m "feat(bakeoff): add Colab runner notebook for abdullahtarek/football_analysis"
```

- [ ] **Step 3: (User action) Run in Colab and commit outputs**

Same workflow as Task 12 Step 3 but writing to `data/bakeoff_outputs/atfa/`.

---

### Task 14: Bake-off candidate notebook — `AbdelrahmanAtef01/Tactic_Zone`

**Files:**
- Create: `examples/bakeoff_tz.ipynb`

Same pattern. Differences:
- Clones `https://github.com/AbdelrahmanAtef01/Tactic_Zone`
- Uses TZ's YOLOv8x detector with the GK + ball + ref class breakdown
- TZ has its own tracking + KMeans team-color module — use those for the bake-off, ignore TZ's event-detection layers (out of scope)

- [ ] **Step 1: Create the notebook scaffold**

Create `examples/bakeoff_tz.ipynb` with the 8-cell pattern. Cell 2:

```python
!pip install -q "ultralytics>=8.2" "supervision>=0.22"
!git clone https://github.com/AbdelrahmanAtef01/Tactic_Zone /content/tz
!pip install -q git+https://github.com/PatrickJReed/soccer-vision.git
import sys; sys.path.insert(0, "/content/tz")
```

Cell 4 invokes TZ's detector and tracker. The exact import paths are determined at clone time — inspect `/content/tz/` and import the equivalent of `Tracker`. Map TZ outputs to schema columns same way as Tasks 12-13.

Cells 5-8 same pattern.

- [ ] **Step 2: Commit the scaffold**

```bash
cd ~/Sandbox/soccer-vision
git add examples/bakeoff_tz.ipynb
git commit -m "feat(bakeoff): add Colab runner notebook for Tactic_Zone"
```

- [ ] **Step 3: (User action) Run in Colab and commit outputs**

Same workflow, writing to `data/bakeoff_outputs/tz/`.

---

### Task 15: Synthesis notebook — `colab_bakeoff.ipynb`

**Files:**
- Create: `examples/colab_bakeoff.ipynb`

Loads the three candidates' trajectories parquets + annotated videos, renders side-by-side comparison panels, computes quantitative summaries, and provides a scoring template.

- [ ] **Step 1: Create the synthesis notebook**

Create `examples/colab_bakeoff.ipynb` with these cells:

**Cell 1 (markdown):**
```markdown
# Bake-off synthesis: side-by-side comparison of three candidates

Reads outputs from `bakeoff_{roboflow,atfa,tz}.ipynb` and produces:
- Side-by-side annotated-video panel
- Per-candidate quantitative summaries (detection rate, track IDs, team consistency)
- A scoring template per the spec §4.4 rubric

Run locally (not Colab): all three candidate outputs must be on disk under
`data/bakeoff_outputs/`.
```

**Cell 2 (code) — load all three:**
```python
import pandas as pd
from pathlib import Path

ROOT = Path("../data/bakeoff_outputs")
candidates = {
    "roboflow": pd.read_parquet(ROOT / "roboflow" / "trajectories.parquet"),
    "atfa":     pd.read_parquet(ROOT / "atfa" / "trajectories.parquet"),
    "tz":       pd.read_parquet(ROOT / "tz" / "trajectories.parquet"),
}
for name, df in candidates.items():
    print(f"{name}: {len(df)} rows, {df['frame'].nunique()} frames, {df['track_id'].nunique()} unique tracks")
```

**Cell 3 (code) — quantitative summaries:**
```python
def summarize(df: pd.DataFrame) -> dict:
    return {
        "n_frames": int(df["frame"].nunique()),
        "n_rows": int(len(df)),
        "unique_tracks": int(df["track_id"].nunique()),
        "mean_detections_per_frame": float(df.groupby("frame").size().mean()),
        "ball_detection_rate": float((df["class"] == "ball").groupby(df["frame"]).any().mean()),
        "mean_conf": float(df["conf"].mean()),
        "team_consistency_per_track": float(
            df.groupby("track_id")["team"].apply(lambda s: s.value_counts(normalize=True).max()).mean()
        ),
    }

import json
for name, df in candidates.items():
    print(f"\n{name}:")
    print(json.dumps(summarize(df), indent=2))
```

**Cell 4 (code) — side-by-side frame composite:**
```python
import cv2
import numpy as np

def grab_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else np.zeros((720, 1280, 3), dtype=np.uint8)

def side_by_side(frame_idx: int) -> np.ndarray:
    frames = [grab_frame(ROOT / name / "annotated.mp4", frame_idx) for name in ["roboflow", "atfa", "tz"]]
    target_h = min(f.shape[0] for f in frames)
    resized = [cv2.resize(f, (int(f.shape[1] * target_h / f.shape[0]), target_h)) for f in frames]
    return np.concatenate(resized, axis=1)

# Sample 8 stratified frames across the clip for visual review
import matplotlib.pyplot as plt
total_frames = int(min(cv2.VideoCapture(str(ROOT / "roboflow" / "annotated.mp4")).get(cv2.CAP_PROP_FRAME_COUNT) for _ in [0]))
sample_idxs = np.linspace(0, total_frames - 1, 8).astype(int)
fig, axes = plt.subplots(8, 1, figsize=(18, 24))
for ax, idx in zip(axes, sample_idxs):
    composite = side_by_side(idx)
    ax.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Frame {idx} — left: roboflow, mid: atfa, right: tz")
    ax.axis("off")
plt.tight_layout()
plt.savefig("../data/bakeoff_outputs/side_by_side.png", dpi=100)
plt.show()
```

**Cell 5 (code) — scoring template:**
```python
# Manual scoring; record values 1-5 per axis after visual review
scores = {
    "roboflow": {"player_recall": None, "ball_recall": None, "track_stability": None,
                 "team_classification": None, "homography_fidelity": None, "nine_v_nine_handling": None},
    "atfa":     {"player_recall": None, "ball_recall": None, "track_stability": None,
                 "team_classification": None, "homography_fidelity": None, "nine_v_nine_handling": None},
    "tz":       {"player_recall": None, "ball_recall": None, "track_stability": None,
                 "team_classification": None, "homography_fidelity": None, "nine_v_nine_handling": None},
}
# Fill in after watching annotated.mp4 for each and inspecting side_by_side.png
# Then:
import pandas as pd
score_df = pd.DataFrame(scores).T
score_df["total"] = score_df.sum(axis=1)
print(score_df)
score_df.to_csv("../data/bakeoff_outputs/scores.csv")
```

- [ ] **Step 2: Commit**

```bash
cd ~/Sandbox/soccer-vision
git add examples/colab_bakeoff.ipynb
git commit -m "feat(bakeoff): add synthesis notebook for side-by-side scoring"
```

- [ ] **Step 3: (User action) Run after all three candidate outputs are committed**

Once `data/bakeoff_outputs/{roboflow,atfa,tz}/` all have trajectories.parquet + annotated.mp4:

```bash
cd ~/Sandbox/soccer-vision
uv run jupyter notebook examples/colab_bakeoff.ipynb
```

Run all cells. The `side_by_side.png` and quantitative summaries inform the scoring step in Task 16.

---

### Task 16: Score the candidates and write `bakeoff-results.md`

**Files:**
- Create: `docs/superpowers/bakeoff-results.md`
- Modify: scoring cells of `examples/colab_bakeoff.ipynb` with filled-in scores

- [ ] **Step 1: Watch each annotated.mp4 with the rubric in hand**

For each of `data/bakeoff_outputs/{roboflow,atfa,tz}/annotated.mp4`, score 1–5 on:
- Player detection recall
- Ball detection rate
- Track stability (visible ID switches)
- Team classification consistency
- Homography fidelity (compare the top-down radar — if rendered — against the manual 4-corner reference)
- 9v9 handling (does it gracefully tolerate 9-per-team, or assume 11?)

- [ ] **Step 2: Fill scoring into `colab_bakeoff.ipynb` Cell 5**

Replace `None` values with integers 1–5. Re-run Cell 5 to write `scores.csv`.

- [ ] **Step 3: Write `bakeoff-results.md`**

Write `docs/superpowers/bakeoff-results.md`:

```markdown
# Bake-off Results — 2026-MM-DD

Source clip: `data/bakeoff_clip.mp4` (2 minutes, extracted from `PatrickReed_Trace.mov`).

Manual homography reference: `data/bakeoff_clip_corners.json` (4 pitch corners).

## Scoring matrix

| Axis                     | RFS | ATFA | TZ  |
|--------------------------|-----|------|-----|
| Player detection recall  | ?   | ?    | ?   |
| Ball detection rate      | ?   | ?    | ?   |
| Track stability          | ?   | ?    | ?   |
| Team classification      | ?   | ?    | ?   |
| Homography fidelity      | ?   | ?    | ?   |
| 9v9 handling             | ?   | ?    | ?   |
| **Total**                | ?   | ?    | ?   |

Runtime per minute of input video (Colab T4):
- RFS: ?
- ATFA: ?
- TZ: ?

## Winner

**Chosen backend:** ?

**Why:** [1-2 paragraphs explaining the choice. Reference specific observations
from the annotated videos and side_by_side.png.]

## Identified weakest stage (informs Phase 2)

**Weakest:** [ball detection / player detection / team-ID / homography / track stability]

**Severity:** [≤2 in all candidates triggers Phase 2 fine-tune; ≥3 in any candidate
defers fine-tune to backlog.]

**Phase 2 decision:** [proceed with fine-tune / defer]

## Notes for downstream phases

- [Any observations relevant to Phase 3+: e.g., "homography unstable when ball
  exits frame", "TZ team-ID swaps when own GK comes off line", etc.]
```

Fill in the `?` cells from `scores.csv`, write the winner paragraph, and the Phase 2 decision.

- [ ] **Step 4: Commit**

```bash
cd ~/Sandbox/soccer-vision
git add docs/superpowers/bakeoff-results.md examples/colab_bakeoff.ipynb data/bakeoff_outputs/scores.csv
git commit -m "docs(bakeoff): record scoring results, winner, and Phase 2 trigger decision"
```

---

### Task 17: Implement the winning `TrackingBackend` adapter

**Files (placeholder — exact name determined by bake-off winner):**
- Create: `packages/soccer-vision/src/soccer_vision/tracking/<winner>.py`
- Create: `packages/soccer-vision/tests/test_tracking_<winner>.py`

The adapter wraps the winning upstream repo's tracking pipeline behind the `TrackingBackend` Protocol from Task 11. Below uses `<winner>` as a placeholder — substitute the actual name (`roboflow`, `atfa`, or `tactic_zone`).

- [ ] **Step 1: Write the failing test**

Write `packages/soccer-vision/tests/test_tracking_<winner>.py`:

```python
"""Tests for the <winner> TrackingBackend adapter.

These tests use a tiny synthetic video so they run in CI without the bake-off
clip or GPU. The adapter's heavy lifting is tested empirically by Phase 1's
bake-off run.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from soccer_vision.io.schema import validate_trajectories
from soccer_vision.tracking.base import TrackingBackend
from soccer_vision.tracking.<winner> import <Winner>Backend


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    """A 30-frame, 320x240, single-color video so tests don't depend on real footage."""
    out = tmp_path / "tiny.mp4"
    fps = 30
    w, h = 320, 240
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(30):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return out


def test_adapter_satisfies_protocol() -> None:
    backend = <Winner>Backend()
    assert isinstance(backend, TrackingBackend)
    assert backend.name
    assert backend.version


def test_adapter_returns_schema_conformant_df(tiny_video: Path) -> None:
    backend = <Winner>Backend()
    df = backend.process(tiny_video)
    # Empty result is acceptable on a blank video; schema must still validate
    validate_trajectories(df)
```

- [ ] **Step 2: Run and verify test fails**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run pytest packages/soccer-vision/tests/test_tracking_<winner>.py -v
```

Expected: 2 errors with `ModuleNotFoundError: No module named 'soccer_vision.tracking.<winner>'`.

- [ ] **Step 3: Implement the adapter**

Write `packages/soccer-vision/src/soccer_vision/tracking/<winner>.py`. The body of `process()` is copy-adapted from the corresponding `bakeoff_<winner>.ipynb` cells 4–6, but wrapped in a class:

```python
"""Adapter for <winner> upstream pipeline.

This class wraps the bake-off-winning detection + tracking pipeline behind the
TrackingBackend Protocol. The implementation is a direct port of the cells in
examples/bakeoff_<winner>.ipynb, organized as importable code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from soccer_vision.io.schema import validate_trajectories


class <Winner>Backend:
    """Tracking backend wrapping the <winner> upstream pipeline."""

    name: Final = "<winner>"
    version: Final = "<commit-sha-or-tag>"  # update to the SHA used during bake-off

    def __init__(self) -> None:
        # Lazy-import upstream — keep top-level import cheap.
        # Actual upstream imports happen in process() to avoid CI failures
        # when GPU-only deps aren't available.
        pass

    def process(self, video_path: Path) -> pd.DataFrame:
        """Run detection + tracking; return a schema-conformant DataFrame."""
        # [Body adapted from bakeoff_<winner>.ipynb cells 4-6]
        # ...
        df = pd.DataFrame(columns=[
            "frame", "t_seconds", "track_id",
            "x_px", "y_px",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "class", "team", "conf",
        ])
        validate_trajectories(df)
        return df
```

**IMPORTANT:** the implementation body must be copy-adapted from the winning candidate's notebook. The placeholder `# ...` is filled with the actual upstream calls. Do NOT leave it as a placeholder in the final commit.

- [ ] **Step 4: Run and verify tests pass**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run pytest packages/soccer-vision/tests/test_tracking_<winner>.py -v
```

Expected: 2 passed (empty-but-schema-conformant DataFrame on tiny video).

- [ ] **Step 5: Run full test suite, lint, type**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run pytest
uv run ruff check .
uv run mypy
```

Expected: all green.

- [ ] **Step 6: Verify against the bake-off clip (manual, GPU required)**

Run a quick local smoke check or in Colab:

```python
from pathlib import Path
from soccer_vision.tracking.<winner> import <Winner>Backend

backend = <Winner>Backend()
df = backend.process(Path("data/bakeoff_clip.mp4"))
print(f"Got {len(df)} rows, {df['frame'].nunique()} frames")
df.to_parquet("data/bakeoff_outputs/<winner>/trajectories_from_adapter.parquet")
```

Compare row counts against `data/bakeoff_outputs/<winner>/trajectories.parquet` (from the notebook run) — they should match within a small tolerance. Numeric differences are acceptable (different RNG seeds, model versions); structural differences are not.

- [ ] **Step 7: Commit**

```bash
cd ~/Sandbox/soccer-vision
git add packages/soccer-vision/src/soccer_vision/tracking/<winner>.py packages/soccer-vision/tests/test_tracking_<winner>.py
git commit -m "feat(tracking): add <Winner>Backend adapter conforming to TrackingBackend protocol"
```

---

### Task 18: Mock backend to verify swap-later promise

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/tracking/mock.py`
- Create: `packages/soccer-vision/tests/test_tracking_mock.py`

Per the spec §9 phase-gate criteria, a second `TrackingBackend` (real or mock) is added at the end of Phase 4 to verify the Protocol's swap-later promise. We bring it forward to the end of Phase 1 because it's cheap and proves the Protocol works.

- [ ] **Step 1: Write the failing test**

Write `packages/soccer-vision/tests/test_tracking_mock.py`:

```python
"""Tests for the MockBackend — exists to prove the Protocol allows swapping."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from soccer_vision.io.schema import validate_trajectories
from soccer_vision.tracking.base import TrackingBackend
from soccer_vision.tracking.mock import MockBackend


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    out = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(10):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    return out


def test_mock_satisfies_protocol() -> None:
    backend: TrackingBackend = MockBackend()
    assert backend.name == "mock"
    assert backend.version == "0.1.0"


def test_mock_returns_schema_conformant_df(tiny_video: Path) -> None:
    backend = MockBackend()
    df = backend.process(tiny_video)
    validate_trajectories(df)


def test_mock_emits_two_teams_with_eight_players_each(tiny_video: Path) -> None:
    backend = MockBackend()
    df = backend.process(tiny_video)
    frame0 = df[df["frame"] == 0]
    assert (frame0["team"] == "own").sum() == 8
    assert (frame0["team"] == "opp").sum() == 8
```

- [ ] **Step 2: Run and verify test fails**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run pytest packages/soccer-vision/tests/test_tracking_mock.py -v
```

Expected: 3 errors with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `MockBackend`**

Write `packages/soccer-vision/src/soccer_vision/tracking/mock.py`:

```python
"""MockBackend: a deterministic stand-in TrackingBackend used in tests.

Emits 16 outfield players (8 own + 8 opp) in fixed grid positions for every
frame of the video. No detection model required. Used to verify that
downstream code is decoupled from any specific upstream tracker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import cv2
import pandas as pd

from soccer_vision.io.schema import validate_trajectories


class MockBackend:
    """Deterministic fixture backend; produces 16 well-spaced detections per frame."""

    name: Final = "mock"
    version: Final = "0.1.0"

    def process(self, video_path: Path) -> pd.DataFrame:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Two 4x2 grids of players, left half = own, right half = opp
        rows: list[dict[str, float | int | str]] = []
        for frame in range(n_frames):
            for team, x_offset, track_id_offset in [("own", w * 0.25, 0), ("opp", w * 0.75, 100)]:
                for i in range(8):
                    col, row = i % 4, i // 4
                    x = x_offset + (col - 1.5) * (w * 0.05)
                    y = h * 0.3 + row * (h * 0.3)
                    rows.append({
                        "frame": frame,
                        "t_seconds": frame / fps,
                        "track_id": track_id_offset + i,
                        "x_px": float(x),
                        "y_px": float(y),
                        "bbox_x1": float(x - 10),
                        "bbox_y1": float(y - 20),
                        "bbox_x2": float(x + 10),
                        "bbox_y2": float(y + 20),
                        "class": "player",
                        "team": team,
                        "conf": 1.0,
                    })

        df = pd.DataFrame(rows)
        df = df.astype({"frame": "int64", "track_id": "int64"})
        validate_trajectories(df)
        return df
```

- [ ] **Step 4: Run and verify tests pass**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run pytest packages/soccer-vision/tests/test_tracking_mock.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run full suite, lint, type**

Run:

```bash
cd ~/Sandbox/soccer-vision
uv run pytest
uv run ruff check .
uv run mypy
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
cd ~/Sandbox/soccer-vision
git add packages/soccer-vision/src/soccer_vision/tracking/mock.py packages/soccer-vision/tests/test_tracking_mock.py
git commit -m "feat(tracking): add MockBackend to verify TrackingBackend swap-later contract"
```

---

## Phase 0 + Phase 1 Done

At this point:

- `soccer-vision` package compiles, tests pass, CI is green
- Canonical bake-off clip + manual corner annotation are in the repo
- Three candidate notebooks ran in Colab and produced standardized outputs
- Synthesis notebook scored them; winner is documented in `bakeoff-results.md`
- The winner's adapter is implemented, tested, and conforms to the Protocol
- A `MockBackend` proves the Protocol allows backend swapping

**Ready for Plan B (Phase 2 + Phase 3):** the bake-off output drives Phase 2's
fine-tune trigger, and the chosen adapter is the input to Phase 3's pitch +
phase splitter work.

---

## Self-Review

**1. Spec coverage** — Walking spec sections against tasks:

| Spec section | Plan coverage |
|---|---|
| §2.1 Layout | Tasks 1, 2 |
| §2.2 Data flow | Schema (Task 10) + Protocol (Task 11) lay the foundation; downstream pipes in later plans |
| §3.2 PitchSpec | Deferred to Plan B Phase 3 |
| §4.1 Protocol | Task 11 |
| §4.2 Bake-off candidates | Tasks 12, 13, 14 |
| §4.3 Canonical clip | Task 8 |
| §4.3 Manual 4-corner reference | Task 9 |
| §4.4 Scoring rubric | Task 16, with 9v9-handling axis included |
| §4.5 Failure modes | Documented in `bakeoff-results.md` template (Task 16) |
| §4.6 Phase 1 deliverables | Tasks 14, 15, 16, 17 |
| §8.1 Unit tests | Tasks 10, 11, 17, 18 |
| §8.4 CI | Task 6 |

**Gap intentionally deferred:** the end-to-end regression test on the bake-off clip (§8.2) is deferred to Plan B because it needs the pitch coords (Phase 3) before the snapshot stats are meaningful.

**2. Placeholder scan**

- `<winner>` and `<Winner>` placeholders in Task 17 are intentional — they get substituted with the bake-off winner's name. The task explicitly calls this out.
- "[Body adapted from bakeoff_<winner>.ipynb cells 4-6]" in Task 17 Step 3 — this is a clear instruction with a referenced source, not a TBD. Engineer copies code from the notebook into the adapter.
- The scoring template in `bakeoff-results.md` (Task 16 Step 3) has `?` placeholders for scores — these are filled in by the engineer running the scoring step. The task explicitly directs the engineer to fill them.
- Cell 5 of `bakeoff_roboflow.ipynb` is a "sketch" — the actual code is supplied by `roboflow/sports`; the task instructs the engineer to import the team-classifier code from the cloned repo. Reasonable for a research notebook.

**3. Type consistency**

- `TrackingBackend` Protocol fields: `name: str`, `version: str`, `process(video_path: Path) -> pd.DataFrame`. Used consistently in `MockBackend`, `<Winner>Backend`, and tests.
- `validate_trajectories(df: pd.DataFrame) -> None`. Raises `TrajectorySchemaError`. Used consistently in `MockBackend.process()`, `<Winner>Backend.process()`, and all bake-off notebooks.
- `REQUIRED_COLUMNS` tuple matches the test's exhaustive-columns assertion (Task 10 Step 1).
- `ALLOWED_CLASSES` (`player`, `goalkeeper`, `referee`, `ball`) and `ALLOWED_TEAMS` (`own`, `opp`, `ref`, `unknown`) — used in tests, validated against in schema, emitted by MockBackend.
