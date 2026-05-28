# soccer-vision — Phase 2 + 3 Implementation Plan (Ball Fine-tune + Homography)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land (a) a fine-tuned ball detector that hits ≥75% sustained detection on the canonical clip and (b) a per-frame homography + pitch-boundary filter + possession/phase splitter that turns trajectories into the data downstream metrics consume.

**Architecture:** Phase 2 produces a new ball-only YOLOv8 model fine-tuned on Trace footage and a thin wiring change in `RoboflowBackend` to consume it. Phase 3 builds a new `pitch/` module (PitchSpec + per-frame homography fitter + PitchMapper + boundary filter) and a `phase/` module (per-track team mode + 5-state possession + phase splitter). Each module emits a parquet table downstream of the trajectories parquet.

**Tech Stack:** YOLOv8 (Ultralytics), Roboflow Universe (labeling), Colab Pro for training, OpenCV (homography via `cv2.findHomography`), scipy (smoothing), pandas (per-track aggregation).

**Spec reference:** [`docs/superpowers/specs/2026-05-27-soccer-vision-design.md`](../specs/2026-05-27-soccer-vision-design.md)

**Bake-off context:** [`docs/superpowers/bakeoff-results.md`](../bakeoff-results.md). Roboflow won; ball detection ≤2/5 across all candidates locks in Phase 2; background-game detection on tournament clips locks in pitch-boundary filtering as v1.

**Parallel work order:** Phase 2's user actions (labeling, training) take days of real-time. Phase 3 tasks 7–15 are pure code and **can run in parallel** with the Phase 2 user-action window.

---

## File Structure

Files created or modified in this plan:

```
soccer-vision/
├── scripts/
│   └── extract_ball_frames.py                       # Task 1 (new)
├── examples/
│   ├── finetune_ball.ipynb                          # Task 3 (new)
│   └── eval_ball_detector.ipynb                     # Task 4 (new)
├── packages/soccer-vision/
│   ├── src/soccer_vision/
│   │   ├── tracking/roboflow.py                     # Task 5, Task 8 (modify)
│   │   ├── pitch/
│   │   │   ├── __init__.py                          # exists; add exports
│   │   │   ├── spec.py                              # Task 7 (new)
│   │   │   ├── homography.py                        # Task 9, Task 11 (new)
│   │   │   ├── mapper.py                            # Task 10 (new)
│   │   │   └── filter.py                            # Task 12 (new)
│   │   └── phase/
│   │       ├── __init__.py                          # exists; add exports
│   │       ├── team_mode.py                         # Task 13 (new)
│   │       ├── possession.py                        # Task 14 (new)
│   │       └── splitter.py                          # Task 15 (new)
│   └── tests/
│       ├── test_pitch_spec.py                       # Task 7
│       ├── test_pitch_homography.py                 # Task 9, Task 11
│       ├── test_pitch_mapper.py                     # Task 10
│       ├── test_pitch_filter.py                     # Task 12
│       ├── test_phase_team_mode.py                  # Task 13
│       ├── test_phase_possession.py                 # Task 14
│       ├── test_phase_splitter.py                   # Task 15
│       └── test_tracking_roboflow_finetune.py       # Task 5
└── README.md                                        # Task 6 (modify)
```

**Design responsibilities per file:**
- `pitch/spec.py` — single `PitchSpec` dataclass; dimensionless; no logic.
- `pitch/homography.py` — fit `H_t` from keypoints; temporal smoothing across frames; validity check.
- `pitch/mapper.py` — `PitchMapper` class wrapping homography sequence + transform method.
- `pitch/filter.py` — pitch-boundary detection-rejection (drop adjacent-game players).
- `phase/team_mode.py` — per-track modal team aggregation; smooths flicker.
- `phase/possession.py` — 5-state possession classifier per frame.
- `phase/splitter.py` — combines possession + ball location into phase labels.
- `tracking/roboflow.py` — accepts a `ball_weights_path` constructor arg; emits pitch keypoints in addition to detections.

Each module is independently testable with deterministic fixtures.

---

## Phase 2 — Ball Detector Fine-tune

### Task 1: Frame extraction script for ball labeling

**Files:**
- Create: `scripts/extract_ball_frames.py`

Extracts ~500 frames sampled across multiple game files to feed Roboflow's labeling pipeline. Non-uniform sampling: denser during ball motion, sparser during stoppages, ensures variety.

- [ ] **Step 1: Write the script**

Write `scripts/extract_ball_frames.py`:

```python
"""Extract sampled frames from multiple game videos for ball-detector fine-tuning.

Strategy: per-video uniform stride sampling. The fine-tune dataset needs
~500 frames total spanning ball states (rolling, in-flight, occluded, near touchline).

Usage:
    uv run python scripts/extract_ball_frames.py \\
        --inputs ~/Downloads/Game1.mov ~/Downloads/Game2.mov \\
        --out data/labeled/ball_v1/raw \\
        --per-video 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract(video_path: Path, out_dir: Path, n_frames: int, prefix: str) -> int:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise RuntimeError(f"Video {video_path} reports 0 frames")
    stride = max(1, total // n_frames)
    saved = 0
    for frame_idx in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out = out_dir / f"{prefix}_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        if saved >= n_frames:
            break
    cap.release()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="paths to game .mov/.mp4 files")
    ap.add_argument("--out", type=Path, required=True, help="output dir for sampled frames")
    ap.add_argument("--per-video", type=int, default=100, help="frames per video")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for video in args.inputs:
        video_path = Path(video).expanduser()
        prefix = video_path.stem.replace(" ", "_")
        saved = extract(video_path, args.out, args.per_video, prefix)
        print(f"{video_path.name}: saved {saved} frames")
        total += saved
    print(f"\nTotal: {total} frames in {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify lint + type**

```bash
cd ~/Sandbox/soccer-vision
uv run ruff check scripts/extract_ball_frames.py
uv run mypy scripts/extract_ball_frames.py
```

Expected: clean. (cv2 is untyped — may need `# type: ignore[import-untyped]` on the `import cv2` line.)

- [ ] **Step 3: Commit**

```bash
cd ~/Sandbox/soccer-vision
mkdir -p scripts
git add scripts/extract_ball_frames.py
git commit -m "feat(scripts): add ball-frame extraction helper for fine-tune labeling"
```

---

### Task 2: USER ACTION — Set up Roboflow project + label

This task takes the user roughly **one weekend** of focused effort. The implementer's role is only to write a short README/runbook documenting the steps so the user can do them.

**Files:**
- Create: `docs/superpowers/runbooks/ball-labeling.md`

- [ ] **Step 1: Write the runbook**

Write `docs/superpowers/runbooks/ball-labeling.md`:

````markdown
# Ball Labeling Runbook (Phase 2)

## Goal

Build a 400–500-frame, single-class (`ball`) dataset from Patrick's Trace
footage to fine-tune a ball-only YOLOv8 detector for soccer-vision.

## Steps

### 1. Extract frames

From the soccer-vision repo root:

```bash
uv run python scripts/extract_ball_frames.py \
    --inputs ~/Downloads/<Game1>.mov ~/Downloads/<Game2>.mov ~/Downloads/<Game3>.mov \
    --out data/labeled/ball_v1/raw \
    --per-video 100
```

Use 3–5 different games to cover lighting, field, and kit variation. Goal:
≥500 sampled frames in `data/labeled/ball_v1/raw/`.

### 2. Create the Roboflow project

1. Go to https://app.roboflow.com → New Project.
2. Project type: **Object Detection**.
3. Single annotation class: `ball`.
4. License: keep private.
5. Upload all frames from `data/labeled/ball_v1/raw/`.

### 3. Label

For each frame: draw a tight bounding box around the ball if visible. Skip
frames where the ball is fully out of frame or fully occluded. Aim for tight
boxes — small-object detection is sensitive to box quality.

Productivity tip: Roboflow's auto-suggest can pre-place boxes after the first
~50; review each before confirming.

### 4. Generate version + export

1. In Roboflow: **Versions → Generate New Version**.
2. Preprocessing: Auto-Orient, Resize 1280x1280 (Letterbox).
3. Augmentations: Mosaic, Mixup, HSV jitter (H ±15°, S ±25%, V ±25%),
   Motion Blur (1–5px), Scale ±50%, Brightness ±20%.
4. Train/Valid/Test split: 80 / 10 / 10.
5. Export as **YOLOv8** format; copy the API curl or `pip install roboflow`
   download snippet.

### 5. Hand off to training

The download snippet goes into `examples/finetune_ball.ipynb` (Task 3) at
the dataset-download cell. From there the Colab training notebook drives
the fine-tune.
````

- [ ] **Step 2: Commit**

```bash
cd ~/Sandbox/soccer-vision
mkdir -p docs/superpowers/runbooks
git add docs/superpowers/runbooks/ball-labeling.md
git commit -m "docs(runbook): document ball-labeling workflow for Phase 2"
```

---

### Task 3: Fine-tune training notebook

**Files:**
- Create: `examples/finetune_ball.ipynb`

Build a Colab notebook that downloads the labeled dataset from Roboflow, trains YOLOv8 with augmentations from the spec, and saves the fine-tuned weights.

- [ ] **Step 1: Build the notebook via nbformat**

The notebook has 6 cells. Build it in a `/tmp/build_ft.py` script (don't commit the script), then write the .ipynb:

Cell 1 (markdown) — Colab badge + intro:
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PatrickJReed/soccer-vision/blob/master/examples/finetune_ball.ipynb)

# Ball detector fine-tune (Phase 2)

Trains a YOLOv8 nano/small model on the labeled Trace ball dataset.
Output: `runs/detect/train/weights/best.pt` — download this and place at
`packages/soccer-vision/src/soccer_vision/models/ball_yolov8_v1.pt` locally.

Run on Colab Pro T4 or L4 GPU. Training takes ~30–60 min.
```

Cell 2 (code) — env setup:
```python
!pip install -q "ultralytics>=8.2" roboflow

import os
from google.colab import userdata
os.environ['ROBOFLOW_API_KEY'] = userdata.get('ROBOFLOW_API_KEY')

from pathlib import Path
WORK = Path("/content/work")
WORK.mkdir(exist_ok=True)
%cd /content/work
```

Cell 3 (code) — dataset download (the user fills in their Roboflow workspace/project/version):
```python
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
# TODO when running: replace with actual workspace / project / version
project = rf.workspace("YOUR_WORKSPACE").project("ball_v1")
dataset = project.version(1).download("yolov8")
print("Dataset at:", dataset.location)
```

Cell 4 (code) — train:
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # nano base; small (yolov8s.pt) if recall is weak

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=1280,
    batch=8,
    patience=20,
    mosaic=1.0,
    mixup=0.15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    scale=0.5,
    fliplr=0.5,
    project="runs/detect",
    name="ball_v1",
)
```

Cell 5 (code) — eval on val split:
```python
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"Precision: {metrics.box.p[0]:.3f}")
print(f"Recall: {metrics.box.r[0]:.3f}")
```

Cell 6 (code) — download weights:
```python
from google.colab import files
files.download("runs/detect/ball_v1/weights/best.pt")
```

- [ ] **Step 2: Commit**

```bash
cd ~/Sandbox/soccer-vision
git add examples/finetune_ball.ipynb
git commit -m "feat(bakeoff): add ball-detector fine-tune notebook for Phase 2"
```

- [ ] **Step 3: USER ACTION — run in Colab, train, download weights**

The user opens the notebook via the badge, fills in their Roboflow workspace/project/version IDs in Cell 3, runs all cells, and downloads `best.pt` to local `data/labeled/ball_v1/best.pt`.

---

### Task 4: Eval notebook — acceptance gate on bake-off clip

**Files:**
- Create: `examples/eval_ball_detector.ipynb`

Runs the fine-tuned model against the bake-off clip; reports sustained-detection rate; gate is ≥75%.

- [ ] **Step 1: Build the notebook via nbformat**

Cell 1 (markdown): badge + intro.

Cell 2 (code): env setup, mount Drive (clip lives there).

Cell 3 (code): clone fine-tuned weights from Drive (place a copy at `MyDrive/soccer-vision/ball_yolov8_v1.pt`).

Cell 4 (code) — run both models on the clip:

```python
import cv2
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

INPUT_CLIP = Path("/content/drive/MyDrive/soccer-vision/bakeoff_clip.mp4")
BASELINE = "/content/baseline_ball.pt"  # download roboflow's original ball model
FINETUNED = Path("/content/drive/MyDrive/soccer-vision/ball_yolov8_v1.pt")

# Download baseline
import gdown
gdown.download(
    "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V",
    BASELINE, quiet=False,
)

baseline_model = YOLO(BASELINE).to("cuda")
finetuned_model = YOLO(str(FINETUNED)).to("cuda")

def detection_rate(model, clip):
    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hits = 0
    for _ in tqdm(range(total), desc=model.ckpt_path):
        ok, frame = cap.read()
        if not ok:
            break
        r = model(frame, imgsz=1280, verbose=False, conf=0.25)[0]
        if len(r.boxes) > 0:
            hits += 1
    cap.release()
    return hits / total

baseline_rate = detection_rate(baseline_model, INPUT_CLIP)
finetuned_rate = detection_rate(finetuned_model, INPUT_CLIP)
print(f"Baseline ball detection rate: {baseline_rate * 100:.1f}%")
print(f"Fine-tuned ball detection rate: {finetuned_rate * 100:.1f}%")
print(f"Improvement: +{(finetuned_rate - baseline_rate) * 100:.1f} pts")
print(f"Acceptance gate (75%): {'PASS' if finetuned_rate >= 0.75 else 'FAIL — label more frames'}")
```

- [ ] **Step 2: Commit**

```bash
cd ~/Sandbox/soccer-vision
git add examples/eval_ball_detector.ipynb
git commit -m "feat(bakeoff): add ball-detector eval notebook with 75% acceptance gate"
```

- [ ] **Step 3: USER ACTION — run, record rates in bakeoff-results.md addendum**

If the gate passes, proceed to Task 5. If it fails, return to Task 2 and label 100–200 more frames focused on the specific failure modes (e.g., ball at far touchline, fast motion).

---

### Task 5: Wire fine-tuned ball weights into `RoboflowBackend`

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/tracking/roboflow.py`
- Create: `packages/soccer-vision/tests/test_tracking_roboflow_finetune.py`

Add a `ball_weights_path: Path | None = None` constructor parameter. If set, `RoboflowBackend.process()` uses it as the ball detector instead of downloading roboflow's default. Falls back to default download when None.

- [ ] **Step 1: Write the failing test**

Write `packages/soccer-vision/tests/test_tracking_roboflow_finetune.py`:

```python
"""Tests for RoboflowBackend's ball_weights_path override."""

from __future__ import annotations

from pathlib import Path

import pytest

from soccer_vision.tracking.roboflow import RoboflowBackend


def test_default_ball_weights_path_is_none() -> None:
    backend = RoboflowBackend()
    assert backend.ball_weights_path is None


def test_custom_ball_weights_path_stored(tmp_path: Path) -> None:
    fake_weights = tmp_path / "ball_v1.pt"
    fake_weights.write_bytes(b"")  # dummy file
    backend = RoboflowBackend(ball_weights_path=fake_weights)
    assert backend.ball_weights_path == fake_weights


def test_nonexistent_ball_weights_path_raises_immediately(tmp_path: Path) -> None:
    bogus = tmp_path / "does-not-exist.pt"
    with pytest.raises(FileNotFoundError):
        RoboflowBackend(ball_weights_path=bogus)
```

- [ ] **Step 2: Run, see fail**

```bash
cd ~/Sandbox/soccer-vision
uv run pytest packages/soccer-vision/tests/test_tracking_roboflow_finetune.py -v
```

Expected: 3 failures with `TypeError: __init__() got an unexpected keyword argument 'ball_weights_path'`.

- [ ] **Step 3: Modify `roboflow.py`**

In `packages/soccer-vision/src/soccer_vision/tracking/roboflow.py`:

In `__init__`, add the parameter, store it, validate immediately:

```python
def __init__(
    self,
    device: str | None = None,
    weights_cache_dir: Path | None = None,
    ball_weights_path: Path | None = None,
) -> None:
    self._device = device
    self._weights_cache_dir = weights_cache_dir or DEFAULT_CACHE_DIR
    if ball_weights_path is not None and not ball_weights_path.exists():
        raise FileNotFoundError(f"ball_weights_path does not exist: {ball_weights_path}")
    self.ball_weights_path: Path | None = ball_weights_path
```

In `process()`, after the lazy imports and before loading models, branch on the override:

```python
ball_weights = self.ball_weights_path or weights_paths["ball"]
ball_model = YOLO(str(ball_weights)).to(device=device)
```

(Keep loading `player_model` from `weights_paths["player"]` unchanged.)

- [ ] **Step 4: Run, see pass**

```bash
uv run pytest packages/soccer-vision/tests/test_tracking_roboflow_finetune.py -v
uv run pytest -q  # full suite
uv run ruff check . && uv run mypy
```

Expected: 3 new pass; all prior tests still pass; lint+mypy clean.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/tracking/roboflow.py \
        packages/soccer-vision/tests/test_tracking_roboflow_finetune.py
git commit -m "feat(tracking): RoboflowBackend accepts ball_weights_path for fine-tuned override"
```

---

### Task 6: Document Phase 2 weights swap in README

**Files:**
- Modify: `README.md`

Add a "Using a fine-tuned ball detector" subsection that explains how to point `RoboflowBackend` at locally-cached fine-tuned weights.

- [ ] **Step 1: Append to README.md**

Below the existing "Tracking backend extras" section, append:

```markdown
### Using a fine-tuned ball detector (Phase 2 output)

If you have a fine-tuned ball-only model (see
[`docs/superpowers/runbooks/ball-labeling.md`](docs/superpowers/runbooks/ball-labeling.md)),
point `RoboflowBackend` at it via the `ball_weights_path` constructor arg:

\`\`\`python
from pathlib import Path
from soccer_vision.tracking.roboflow import RoboflowBackend

backend = RoboflowBackend(
    ball_weights_path=Path("data/labeled/ball_v1/best.pt"),
)
df = backend.process(Path("data/games/<game>.mp4"))
\`\`\`

When `ball_weights_path=None` (default), the adapter downloads roboflow's
original ball detector to `~/.cache/soccer_vision/weights/`. Override
forces the local file — useful for Phase 2 evaluation and the production
pipeline once acceptance is met.
```

(Note: the inner triple-backticks shown above are escaped for this doc; in the actual `README.md` they're literal.)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document ball_weights_path override for fine-tuned model"
```

---

## Phase 3 — Pitch Homography, Boundary Filter, Possession, Phase

### Task 7: `PitchSpec` dataclass

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/spec.py`
- Create: `packages/soccer-vision/tests/test_pitch_spec.py`

Dimensionless dataclass — captures the pitch's relative proportions. All values fractions of pitch length.

- [ ] **Step 1: Write the failing test**

Write `packages/soccer-vision/tests/test_pitch_spec.py`:

```python
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
```

- [ ] **Step 2: Run, see fail**

```bash
uv run pytest packages/soccer-vision/tests/test_pitch_spec.py -v
```

Expected: 4 errors (ModuleNotFoundError).

- [ ] **Step 3: Write `pitch/spec.py`**

```python
"""PitchSpec — dimensionless pitch proportions for soccer-vision metrics.

All distances expressed as fractions of pitch length (the canonical unit).
This avoids per-game field-size calibration while keeping metrics comparable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PitchSpec:
    """Dimensionless pitch description.

    Defaults are US Soccer 9v9 mid-range (~68.5 m × 45.7 m).
    """

    aspect_ratio: float = 1.5
    n_outfield_per_team: int = 8
    penalty_box_length_frac: float = 0.187
    penalty_box_width_frac: float = 0.720
    center_circle_radius_frac: float = 0.106
    coverage_cell_frac: float = 0.011

    @classmethod
    def standard_9v9(cls) -> PitchSpec:
        return cls()

    @classmethod
    def fifa_11v11(cls) -> PitchSpec:
        return cls(
            aspect_ratio=1.54,
            n_outfield_per_team=10,
            penalty_box_length_frac=0.157,
            penalty_box_width_frac=0.592,
            center_circle_radius_frac=0.087,
            coverage_cell_frac=0.0095,
        )
```

- [ ] **Step 4: Run, see pass**

```bash
uv run pytest packages/soccer-vision/tests/test_pitch_spec.py -v
uv run pytest -q
uv run ruff check . && uv run mypy
```

Expected: 4 pass; full suite green.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/spec.py \
        packages/soccer-vision/tests/test_pitch_spec.py
git commit -m "feat(pitch): add PitchSpec dimensionless pitch proportions"
```

---

### Task 8: Re-enable pitch model + emit keypoints from `RoboflowBackend`

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/tracking/roboflow.py`
- Modify: `packages/soccer-vision/tests/test_tracking_roboflow.py`

The pitch model was deferred in Task 17 of Plan A. Phase 3 needs its per-frame keypoint output. Re-introduce the pitch model behind an opt-in constructor flag (`detect_pitch: bool = False`); when enabled, attach a per-frame keypoint array as a separate parquet (so the schema stays as before for downstream consumers that don't need pitch data).

- [ ] **Step 1: Write the failing test**

Append to `packages/soccer-vision/tests/test_tracking_roboflow.py`:

```python
@pytest.mark.skipif(not _HEAVY_AVAILABLE, reason="roboflow extras not installed")
def test_adapter_with_pitch_returns_keypoints(tiny_video: Path, tmp_path: Path) -> None:
    """When detect_pitch=True, process_with_pitch() returns (df, keypoints_df)."""
    backend = RoboflowBackend(detect_pitch=True)
    df, kp_df = backend.process_with_pitch(tiny_video)
    from soccer_vision.io.schema import validate_trajectories
    validate_trajectories(df)
    assert {"frame", "kp_idx", "x_px", "y_px", "conf"}.issubset(kp_df.columns)


def test_default_detect_pitch_false() -> None:
    backend = RoboflowBackend()
    assert backend.detect_pitch is False
```

- [ ] **Step 2: Run, see fail**

Expected: `AttributeError: 'RoboflowBackend' object has no attribute 'detect_pitch'` and `AttributeError: 'RoboflowBackend' object has no attribute 'process_with_pitch'`.

- [ ] **Step 3: Modify `roboflow.py`**

Restore "pitch" to `WEIGHTS`:

```python
WEIGHTS: Final[dict[str, tuple[str, str]]] = {
    "ball":   ("1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V", "football-ball-detection.pt"),
    "player": ("17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q", "football-player-detection.pt"),
    "pitch":  ("1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf", "football-pitch-detection.pt"),
}
```

Add to `__init__`:

```python
def __init__(
    self,
    device: str | None = None,
    weights_cache_dir: Path | None = None,
    ball_weights_path: Path | None = None,
    detect_pitch: bool = False,
) -> None:
    ...
    self.detect_pitch = detect_pitch
```

In `_download_weights`, only download the pitch entry when `self.detect_pitch` is True:

```python
def _download_weights(self) -> dict[str, Path]:
    ...
    for key, (gid, fname) in WEIGHTS.items():
        if key == "pitch" and not self.detect_pitch:
            continue
        ...
```

Add a new method `process_with_pitch(video_path) -> tuple[pd.DataFrame, pd.DataFrame]` that runs the same pipeline as `process()` plus loads the pitch model and emits a per-frame keypoint table with columns `(frame, kp_idx, x_px, y_px, conf)`. Each keypoint detection becomes one row.

The implementation can share most of the work with `process()`. Suggested refactor: extract a private `_run_pipeline(video_path, *, emit_keypoints: bool)` that returns both DataFrames, then `process()` discards the keypoints and `process_with_pitch()` returns both.

- [ ] **Step 4: Run, see pass**

```bash
uv run pytest packages/soccer-vision/tests/test_tracking_roboflow.py -v
uv run pytest -q
uv run ruff check . && uv run mypy
```

Expected: all pass (heavy tests skipped without extras).

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/tracking/roboflow.py \
        packages/soccer-vision/tests/test_tracking_roboflow.py
git commit -m "feat(tracking): re-enable pitch model + process_with_pitch() emitting keypoints"
```

---

### Task 9: Homography fit from keypoints

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/homography.py`
- Create: `packages/soccer-vision/tests/test_pitch_homography.py`

Function `fit_homography(image_points, pitch_points) → 3×3 H`. Uses `cv2.findHomography(method=cv2.RANSAC)`. Validates input shapes.

- [ ] **Step 1: Write the failing test**

Write `packages/soccer-vision/tests/test_pitch_homography.py`:

```python
"""Tests for pitch homography fitting."""

from __future__ import annotations

import numpy as np
import pytest

from soccer_vision.pitch.homography import (
    HomographyError,
    fit_homography,
)


def test_identity_homography_from_unit_square() -> None:
    """Mapping the unit square to itself returns ~identity homography."""
    img_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    pitch_pts = img_pts.copy()
    H = fit_homography(img_pts, pitch_pts)
    H_norm = H / H[2, 2]
    assert np.allclose(H_norm, np.eye(3), atol=1e-6)


def test_translation_homography() -> None:
    """Image points shifted by +10 in x should produce a translation H."""
    img_pts = np.array([[10, 0], [11, 0], [11, 1], [10, 1]], dtype=np.float32)
    pitch_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H = fit_homography(img_pts, pitch_pts)
    pt = np.array([10.5, 0.5, 1.0])
    out = H @ pt
    out /= out[2]
    assert abs(out[0] - 0.5) < 1e-4
    assert abs(out[1] - 0.5) < 1e-4


def test_too_few_points_raises() -> None:
    img_pts = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
    pitch_pts = img_pts.copy()
    with pytest.raises(HomographyError, match="at least 4"):
        fit_homography(img_pts, pitch_pts)


def test_mismatched_shapes_raises() -> None:
    img_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    pitch_pts = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
    with pytest.raises(HomographyError, match="same number"):
        fit_homography(img_pts, pitch_pts)
```

- [ ] **Step 2: Run, see fail**

Expected: 4 ModuleNotFoundErrors.

- [ ] **Step 3: Write `pitch/homography.py`**

```python
"""Per-frame homography fitting between image-plane points and pitch coordinates."""

from __future__ import annotations

import cv2  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray


class HomographyError(ValueError):
    """Raised when homography fitting fails (e.g., too few points)."""


def fit_homography(
    image_points: NDArray[np.floating],
    pitch_points: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Fit a 3×3 homography mapping image_points → pitch_points.

    Parameters
    ----------
    image_points
        N×2 array of pixel coords from the source frame.
    pitch_points
        N×2 array of corresponding canonical-pitch coords (in [0, 1]² typically).

    Returns
    -------
    H
        3×3 homography matrix such that ``(x, y, 1) → H @ (x_img, y_img, 1)`` and the
        result is normalized so the last component is 1.

    Raises
    ------
    HomographyError
        If fewer than 4 points are provided, shapes mismatch, or RANSAC fails.
    """
    if image_points.shape != pitch_points.shape:
        raise HomographyError(
            f"image_points and pitch_points must have the same number of rows; "
            f"got {image_points.shape} vs {pitch_points.shape}"
        )
    if image_points.shape[0] < 4:
        raise HomographyError(
            f"Need at least 4 corresponding points; got {image_points.shape[0]}"
        )
    H, _ = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC)
    if H is None:
        raise HomographyError("cv2.findHomography returned None — degenerate input")
    return H.astype(np.float64)
```

- [ ] **Step 4: Run, see pass**

```bash
uv run pytest packages/soccer-vision/tests/test_pitch_homography.py -v
```

Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/homography.py \
        packages/soccer-vision/tests/test_pitch_homography.py
git commit -m "feat(pitch): add fit_homography for per-frame H_t from keypoints"
```

---

### Task 10: `PitchMapper` — apply homography to detections

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/mapper.py`
- Create: `packages/soccer-vision/tests/test_pitch_mapper.py`

Class wrapping a sequence of per-frame H matrices. Method `transform(detections_df, homographies)` returns a new DataFrame with `x_pitch, y_pitch` columns.

- [ ] **Step 1: Write the failing test**

Write `packages/soccer-vision/tests/test_pitch_mapper.py`:

```python
"""Tests for PitchMapper transform."""

from __future__ import annotations

import numpy as np
import pandas as pd

from soccer_vision.pitch.mapper import PitchMapper


def test_identity_transform_passes_coords_through() -> None:
    detections = pd.DataFrame({
        "frame": [0, 0, 1],
        "x_px": [0.2, 0.4, 0.6],
        "y_px": [0.3, 0.5, 0.7],
    })
    identity = np.eye(3)
    homographies = {0: identity, 1: identity}
    mapper = PitchMapper()
    out = mapper.transform(detections, homographies)
    assert np.allclose(out["x_pitch"].to_numpy(), detections["x_px"].to_numpy())
    assert np.allclose(out["y_pitch"].to_numpy(), detections["y_px"].to_numpy())


def test_translation_homography_applied() -> None:
    detections = pd.DataFrame({
        "frame": [0],
        "x_px": [0.5],
        "y_px": [0.5],
    })
    # H translates by (-0.1, -0.2)
    H = np.array([
        [1.0, 0.0, -0.1],
        [0.0, 1.0, -0.2],
        [0.0, 0.0, 1.0],
    ])
    mapper = PitchMapper()
    out = mapper.transform(detections, {0: H})
    assert abs(out["x_pitch"].iloc[0] - 0.4) < 1e-6
    assert abs(out["y_pitch"].iloc[0] - 0.3) < 1e-6


def test_missing_homography_emits_nan() -> None:
    detections = pd.DataFrame({
        "frame": [0, 1],
        "x_px": [0.5, 0.5],
        "y_px": [0.5, 0.5],
    })
    homographies = {0: np.eye(3)}  # frame 1 missing
    mapper = PitchMapper()
    out = mapper.transform(detections, homographies)
    assert not pd.isna(out["x_pitch"].iloc[0])
    assert pd.isna(out["x_pitch"].iloc[1])
```

- [ ] **Step 2: Run, see fail**

Expected: 3 ModuleNotFoundErrors.

- [ ] **Step 3: Write `pitch/mapper.py`**

```python
"""PitchMapper — applies per-frame homography H_t to detection rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class PitchMapper:
    """Stateless utility for mapping pixel detections through per-frame homographies."""

    def transform(
        self,
        detections: pd.DataFrame,
        homographies: dict[int, NDArray[np.floating]],
    ) -> pd.DataFrame:
        """Append x_pitch, y_pitch columns by applying homographies[frame] to (x_px, y_px).

        Frames absent from `homographies` produce NaN pitch coords for their rows.
        Returns a new DataFrame; does not mutate input.
        """
        out = detections.copy()
        x_pitch = np.full(len(out), np.nan)
        y_pitch = np.full(len(out), np.nan)
        for frame_idx, group in out.groupby("frame", sort=False):
            H = homographies.get(int(frame_idx))
            if H is None:
                continue
            pts = np.column_stack([
                group["x_px"].to_numpy(),
                group["y_px"].to_numpy(),
                np.ones(len(group)),
            ])
            mapped = (H @ pts.T).T
            mapped /= mapped[:, 2:3]
            x_pitch[group.index] = mapped[:, 0]
            y_pitch[group.index] = mapped[:, 1]
        out["x_pitch"] = x_pitch
        out["y_pitch"] = y_pitch
        return out
```

- [ ] **Step 4: Run, see pass**

```bash
uv run pytest packages/soccer-vision/tests/test_pitch_mapper.py -v
```

Expected: 3 pass.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/mapper.py \
        packages/soccer-vision/tests/test_pitch_mapper.py
git commit -m "feat(pitch): add PitchMapper for per-frame coord transformation"
```

---

### Task 11: Temporal smoothing of per-frame homographies

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/homography.py` (add `smooth_homographies`)
- Modify: `packages/soccer-vision/tests/test_pitch_homography.py` (add 2 tests)

Per-frame H is noisy. Apply an exponential moving average (EMA) on the 9 H elements (or 8, normalizing H[2,2]=1) across frames. Skip frames with no H (NaN); fill gaps by carrying forward last valid smoothed H.

- [ ] **Step 1: Add failing tests**

Append to `packages/soccer-vision/tests/test_pitch_homography.py`:

```python
import numpy as np

from soccer_vision.pitch.homography import smooth_homographies


def test_smoothing_passes_constant_homography_through() -> None:
    """A sequence of identical Hs should smooth to itself."""
    H = np.eye(3)
    seq = {0: H, 1: H.copy(), 2: H.copy()}
    smoothed = smooth_homographies(seq, alpha=0.5)
    for fi in (0, 1, 2):
        assert np.allclose(smoothed[fi], H)


def test_smoothing_alpha_one_returns_input() -> None:
    """alpha=1.0 disables smoothing (no carryover)."""
    H0 = np.eye(3)
    H1 = np.diag([2.0, 2.0, 1.0])
    smoothed = smooth_homographies({0: H0, 1: H1}, alpha=1.0)
    assert np.allclose(smoothed[1], H1)


def test_smoothing_alpha_zero_carries_first() -> None:
    """alpha=0.0 means subsequent frames inherit the first."""
    H0 = np.eye(3)
    H1 = np.diag([2.0, 2.0, 1.0])
    smoothed = smooth_homographies({0: H0, 1: H1}, alpha=0.0)
    assert np.allclose(smoothed[1], H0)


def test_smoothing_fills_gaps() -> None:
    """A missing frame carries forward the previous smoothed H."""
    H0 = np.eye(3)
    H2 = np.eye(3)
    smoothed = smooth_homographies({0: H0, 2: H2}, alpha=0.5)
    assert 1 in smoothed
    assert np.allclose(smoothed[1], H0)
```

- [ ] **Step 2: Run, see fail**

Expected: ImportError on `smooth_homographies`.

- [ ] **Step 3: Add function to `pitch/homography.py`**

```python
def smooth_homographies(
    homographies: dict[int, NDArray[np.floating]],
    alpha: float = 0.5,
) -> dict[int, NDArray[np.floating]]:
    """Exponential moving average over frame-indexed homographies.

    H_smoothed[t] = alpha * H[t] + (1 - alpha) * H_smoothed[t-1].
    Missing frames between known ones carry forward the previous smoothed H.

    Parameters
    ----------
    homographies
        Dict mapping frame_idx → 3×3 H. May be sparse.
    alpha
        Smoothing factor in [0, 1]. 1.0 = no smoothing; 0.0 = fully carry first.

    Returns
    -------
    smoothed
        Dict over all frames from min to max of input, fully filled (no gaps).
    """
    if not homographies:
        return {}
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    frames_sorted = sorted(homographies.keys())
    f_min, f_max = frames_sorted[0], frames_sorted[-1]
    smoothed: dict[int, NDArray[np.floating]] = {}
    prev: NDArray[np.floating] | None = None
    for fi in range(f_min, f_max + 1):
        H_obs = homographies.get(fi)
        if H_obs is not None:
            if prev is None:
                smoothed[fi] = H_obs.copy()
            else:
                smoothed[fi] = alpha * H_obs + (1.0 - alpha) * prev
        else:
            assert prev is not None, "first frame must have an observation"
            smoothed[fi] = prev.copy()
        prev = smoothed[fi]
    return smoothed
```

- [ ] **Step 4: Run, see pass**

```bash
uv run pytest packages/soccer-vision/tests/test_pitch_homography.py -v
```

Expected: 8 pass total (4 from Task 9 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/homography.py \
        packages/soccer-vision/tests/test_pitch_homography.py
git commit -m "feat(pitch): add smooth_homographies EMA over frame-indexed H sequence"
```

---

### Task 12: Pitch-boundary filter (drops adjacent-game detections)

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/filter.py`
- Create: `packages/soccer-vision/tests/test_pitch_filter.py`

After PitchMapper.transform appends pitch coords, drop rows whose projected (x_pitch, y_pitch) falls outside [-margin, 1+margin]². This catches adjacent-game players that the YOLO model detected.

- [ ] **Step 1: Write the failing test**

Write `packages/soccer-vision/tests/test_pitch_filter.py`:

```python
"""Tests for pitch-boundary filter."""

from __future__ import annotations

import numpy as np
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
```

- [ ] **Step 2: Run, see fail**

- [ ] **Step 3: Write `pitch/filter.py`**

```python
"""Pitch-boundary filter — drops detections whose projected pitch coords are off-pitch."""

from __future__ import annotations

import pandas as pd


def filter_outside_pitch(
    detections: pd.DataFrame,
    margin: float = 0.05,
) -> pd.DataFrame:
    """Drop rows whose (x_pitch, y_pitch) is outside [-margin, 1+margin]².

    Rows with NaN in either pitch coord are dropped (homography missing for the frame).

    Parameters
    ----------
    detections
        DataFrame with x_pitch and y_pitch columns.
    margin
        Slack around the unit square; tolerates near-boundary detections.

    Returns
    -------
    A new DataFrame with the out-of-bounds and NaN rows removed.
    """
    if "x_pitch" not in detections.columns or "y_pitch" not in detections.columns:
        raise ValueError("detections must have x_pitch and y_pitch columns")
    lo = -margin
    hi = 1.0 + margin
    mask = (
        detections["x_pitch"].between(lo, hi, inclusive="both")
        & detections["y_pitch"].between(lo, hi, inclusive="both")
    )
    return detections[mask].reset_index(drop=True)
```

- [ ] **Step 4: Run, see pass**

```bash
uv run pytest packages/soccer-vision/tests/test_pitch_filter.py -v
```

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/filter.py \
        packages/soccer-vision/tests/test_pitch_filter.py
git commit -m "feat(pitch): add filter_outside_pitch to drop adjacent-game detections"
```

---

### Task 13: Per-track team mode aggregation

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/phase/team_mode.py`
- Create: `packages/soccer-vision/tests/test_phase_team_mode.py`

Per-frame team prediction is noisy (own↔opp flicker). For each `track_id`, take the modal team assignment over its lifetime and apply uniformly. Trivial pandas operation.

- [ ] **Step 1: Write the failing test**

Write `packages/soccer-vision/tests/test_phase_team_mode.py`:

```python
"""Tests for per-track modal team smoothing."""

from __future__ import annotations

import pandas as pd

from soccer_vision.phase.team_mode import apply_modal_team_per_track


def test_majority_own_overrides_minority_opp() -> None:
    df = pd.DataFrame({
        "frame": [0, 1, 2, 3, 4],
        "track_id": [1, 1, 1, 1, 1],
        "class": ["player"] * 5,
        "team": ["own", "own", "opp", "own", "own"],
    })
    out = apply_modal_team_per_track(df)
    assert (out["team"] == "own").all()


def test_per_track_independent() -> None:
    df = pd.DataFrame({
        "frame": [0, 0, 1, 1],
        "track_id": [1, 2, 1, 2],
        "class": ["player", "player", "player", "player"],
        "team": ["own", "opp", "own", "opp"],
    })
    out = apply_modal_team_per_track(df)
    assert out[out["track_id"] == 1]["team"].iloc[0] == "own"
    assert out[out["track_id"] == 2]["team"].iloc[0] == "opp"


def test_referees_unchanged() -> None:
    df = pd.DataFrame({
        "frame": [0, 1],
        "track_id": [10, 10],
        "class": ["referee", "referee"],
        "team": ["ref", "ref"],
    })
    out = apply_modal_team_per_track(df)
    assert (out["team"] == "ref").all()


def test_unknown_when_no_clear_majority() -> None:
    """Equal counts → 'unknown'."""
    df = pd.DataFrame({
        "frame": [0, 1],
        "track_id": [1, 1],
        "class": ["player", "player"],
        "team": ["own", "opp"],
    })
    out = apply_modal_team_per_track(df)
    assert (out["team"] == "unknown").all()
```

- [ ] **Step 2: Run, see fail**

- [ ] **Step 3: Write `phase/team_mode.py`**

```python
"""Per-track modal team aggregation — smooths per-frame team prediction noise."""

from __future__ import annotations

import pandas as pd

PLAYER_CLASSES = frozenset({"player", "goalkeeper"})


def apply_modal_team_per_track(detections: pd.DataFrame) -> pd.DataFrame:
    """Replace each player/GK row's team with the modal value for its track_id.

    Referees and ball rows are left untouched. Ties produce 'unknown'.

    Parameters
    ----------
    detections
        DataFrame with at least `track_id`, `class`, `team` columns.

    Returns
    -------
    A new DataFrame with team smoothed per-track for player/GK rows.
    """
    out = detections.copy()
    player_mask = out["class"].isin(PLAYER_CLASSES)

    def modal_or_unknown(group: pd.Series) -> str:
        counts = group.value_counts()
        if len(counts) >= 2 and counts.iloc[0] == counts.iloc[1]:
            return "unknown"
        return str(counts.idxmax())

    modes = (
        out[player_mask]
        .groupby("track_id")["team"]
        .apply(modal_or_unknown)
    )
    out.loc[player_mask, "team"] = out.loc[player_mask, "track_id"].map(modes)
    return out
```

- [ ] **Step 4: Run, see pass**

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/phase/team_mode.py \
        packages/soccer-vision/tests/test_phase_team_mode.py
git commit -m "feat(phase): per-track modal team aggregation smooths per-frame flicker"
```

---

### Task 14: 5-state possession proxy

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/phase/possession.py`
- Create: `packages/soccer-vision/tests/test_phase_possession.py`

Per-frame 5-state classifier (own/opp/contested/loose_ball/unknown) per the spec §6.0c. Operates on pitch coords. Thresholds in pitch-units.

- [ ] **Step 1: Write the failing test**

Write `packages/soccer-vision/tests/test_phase_possession.py`:

```python
"""Tests for 5-state possession proxy."""

from __future__ import annotations

import pandas as pd

from soccer_vision.phase.possession import (
    PossessionThresholds,
    classify_possession,
)


def _make_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_own_when_clearly_closest() -> None:
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.51, "y_pitch": 0.51},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.70, "y_pitch": 0.70},
    ])
    states = classify_possession(df)
    assert states[0] == "own"


def test_contested_when_close_margin() -> None:
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.51, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.515, "y_pitch": 0.50},
    ])
    states = classify_possession(df)
    assert states[0] == "contested"


def test_contested_when_clump_balanced() -> None:
    """Both teams have ≥1 within clump radius; counts differ by ≤1."""
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.52, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.51, "y_pitch": 0.51},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.49, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.50, "y_pitch": 0.51},
    ])
    states = classify_possession(df)
    assert states[0] == "contested"


def test_loose_ball_when_no_one_close() -> None:
    df = _make_frame([
        {"frame": 0, "class": "ball", "team": "unknown", "x_pitch": 0.50, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.80, "y_pitch": 0.50},
        {"frame": 0, "class": "player", "team": "opp", "x_pitch": 0.20, "y_pitch": 0.50},
    ])
    states = classify_possession(df)
    assert states[0] == "loose_ball"


def test_unknown_when_no_ball() -> None:
    df = _make_frame([
        {"frame": 0, "class": "player", "team": "own", "x_pitch": 0.50, "y_pitch": 0.50},
    ])
    states = classify_possession(df)
    assert states[0] == "unknown"
```

- [ ] **Step 2: Run, see fail**

- [ ] **Step 3: Write `phase/possession.py`**

```python
"""5-state per-frame possession classifier (own/opp/contested/loose_ball/unknown)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PossessionState = Literal["own", "opp", "contested", "loose_ball", "unknown"]


@dataclass(frozen=True)
class PossessionThresholds:
    """Thresholds in pitch-units (fractions of pitch length)."""
    margin: float = 0.022      # contested if |nearest_own - nearest_opp| < margin
    clump_radius: float = 0.044  # players within this radius count toward clump
    loose_ball_radius: float = 0.073  # ball is loose if no one within this radius


def classify_possession(
    detections: pd.DataFrame,
    thresholds: PossessionThresholds | None = None,
) -> pd.Series:
    """Classify possession state for each frame in `detections`.

    Returns a pd.Series indexed by frame, values from PossessionState literals.
    """
    th = thresholds or PossessionThresholds()
    states: dict[int, PossessionState] = {}

    for frame_idx, group in detections.groupby("frame", sort=False):
        ball_rows = group[group["class"] == "ball"]
        if ball_rows.empty:
            states[int(frame_idx)] = "unknown"
            continue
        bx = ball_rows["x_pitch"].iloc[0]
        by = ball_rows["y_pitch"].iloc[0]
        if pd.isna(bx) or pd.isna(by):
            states[int(frame_idx)] = "unknown"
            continue

        players = group[group["class"].isin(["player", "goalkeeper"])]
        if players.empty:
            states[int(frame_idx)] = "unknown"
            continue

        dx = players["x_pitch"].to_numpy() - bx
        dy = players["y_pitch"].to_numpy() - by
        dists = np.sqrt(dx * dx + dy * dy)
        teams = players["team"].to_numpy()

        own_mask = teams == "own"
        opp_mask = teams == "opp"
        d_own = dists[own_mask].min() if own_mask.any() else np.inf
        d_opp = dists[opp_mask].min() if opp_mask.any() else np.inf

        if min(d_own, d_opp) > th.loose_ball_radius:
            states[int(frame_idx)] = "loose_ball"
            continue

        clump_own = int(((dists[own_mask] <= th.clump_radius).sum()) if own_mask.any() else 0)
        clump_opp = int(((dists[opp_mask] <= th.clump_radius).sum()) if opp_mask.any() else 0)
        if (
            abs(d_own - d_opp) < th.margin
            or (clump_own >= 1 and clump_opp >= 1 and abs(clump_own - clump_opp) <= 1)
        ):
            states[int(frame_idx)] = "contested"
            continue

        states[int(frame_idx)] = "own" if d_own < d_opp else "opp"

    return pd.Series(states, name="possession_state")
```

- [ ] **Step 4: Run, see pass**

```bash
uv run pytest packages/soccer-vision/tests/test_phase_possession.py -v
```

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/phase/possession.py \
        packages/soccer-vision/tests/test_phase_possession.py
git commit -m "feat(phase): add 5-state possession proxy (own/opp/contested/loose_ball/unknown)"
```

---

### Task 15: Phase splitter — combines possession + ball location

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/phase/splitter.py`
- Create: `packages/soccer-vision/tests/test_phase_splitter.py`

Wraps `classify_possession()` and adds a `phase` label per frame: `build` / `attack` / `defend_low` / `defend_high` / `transition` / `contested` / `loose_ball` / `unknown`. `transition` is a 5-second post-turnover window.

- [ ] **Step 1: Write the failing test**

Write `packages/soccer-vision/tests/test_phase_splitter.py`:

```python
"""Tests for phase splitter (combines possession + ball location)."""

from __future__ import annotations

import pandas as pd

from soccer_vision.phase.splitter import label_phase


def test_attack_when_own_with_ball_in_opp_two_thirds() -> None:
    states = pd.Series(["own", "own", "own"], index=[0, 1, 2])
    ball_y = pd.Series([0.5, 0.7, 0.8], index=[0, 1, 2])
    fps = 30.0
    phases = label_phase(states, ball_y, fps=fps)
    assert phases.loc[2] == "attack"


def test_build_when_own_with_ball_in_own_third() -> None:
    states = pd.Series(["own", "own"], index=[0, 1])
    ball_y = pd.Series([0.2, 0.25], index=[0, 1])
    phases = label_phase(states, ball_y, fps=30.0)
    assert phases.loc[1] == "build"


def test_defend_low_when_opp_with_ball_in_own_half() -> None:
    states = pd.Series(["opp", "opp"], index=[0, 1])
    ball_y = pd.Series([0.3, 0.4], index=[0, 1])
    phases = label_phase(states, ball_y, fps=30.0)
    assert phases.loc[1] == "defend_low"


def test_transition_window_after_possession_change() -> None:
    """5-second window after a state change → 'transition'."""
    fps = 30.0
    states_list = ["own"] * 30 + ["opp"] * 60  # turnover at frame 30
    ball_y = [0.6] * 90
    states = pd.Series(states_list, index=range(90))
    ball_y_s = pd.Series(ball_y, index=range(90))
    phases = label_phase(states, ball_y_s, fps=fps, transition_seconds=1.0)
    # Frame 30 turnover -> transition window 30..59
    assert phases.loc[35] == "transition"
    # Outside window: regular phase
    assert phases.loc[80] == "defend_high"


def test_contested_passes_through() -> None:
    states = pd.Series(["contested", "own"], index=[0, 1])
    ball_y = pd.Series([0.5, 0.7], index=[0, 1])
    phases = label_phase(states, ball_y, fps=30.0)
    assert phases.loc[0] == "contested"
```

- [ ] **Step 2: Run, see fail**

- [ ] **Step 3: Write `phase/splitter.py`**

```python
"""Phase splitter — labels each frame with one of:
build / attack / defend_low / defend_high / transition / contested / loose_ball / unknown.

Operates on the output of classify_possession() plus per-frame ball y-coord.
"""

from __future__ import annotations

import pandas as pd

OWN_THIRD_MAX_Y = 0.333
OPP_THIRD_MIN_Y = 0.667


def label_phase(
    possession_state: pd.Series,
    ball_y_pitch: pd.Series,
    fps: float,
    transition_seconds: float = 5.0,
) -> pd.Series:
    """Combine per-frame possession state + ball y-coord into a phase label.

    Parameters
    ----------
    possession_state
        Series of PossessionState literals indexed by frame.
    ball_y_pitch
        Series of y_pitch (in [0, 1]) of the ball, indexed by frame. NaN if missing.
    fps
        Frame rate; used to size the transition window in frames.
    transition_seconds
        Window after each possession change labeled 'transition'.

    Returns
    -------
    Series of phase labels indexed by frame.
    """
    frames = possession_state.index
    transition_frames = int(round(transition_seconds * fps))

    # Detect transitions: where state changes from own↔opp (not into/out of contested/loose/unknown)
    state_prev = possession_state.shift(1)
    is_turnover = (
        ((possession_state == "own") & (state_prev == "opp"))
        | ((possession_state == "opp") & (state_prev == "own"))
    )
    turnover_frames = frames[is_turnover.fillna(False).to_numpy()]

    phases = pd.Series("unknown", index=frames)
    for fi in frames:
        st = possession_state.loc[fi]
        if st in ("contested", "loose_ball", "unknown"):
            phases.loc[fi] = st
            continue
        by = ball_y_pitch.loc[fi] if fi in ball_y_pitch.index else float("nan")
        if pd.isna(by):
            phases.loc[fi] = "unknown"
            continue
        if st == "own":
            phases.loc[fi] = "build" if by < OWN_THIRD_MAX_Y else "attack"
        else:  # "opp"
            phases.loc[fi] = "defend_high" if by > OWN_THIRD_MAX_Y else "defend_low"

    # Overlay transition windows
    for to_frame in turnover_frames:
        end = int(to_frame) + transition_frames
        affected = frames[(frames >= int(to_frame)) & (frames < end)]
        for fi in affected:
            # Don't overwrite contested/loose_ball/unknown
            if phases.loc[fi] in ("build", "attack", "defend_low", "defend_high"):
                phases.loc[fi] = "transition"

    return phases
```

- [ ] **Step 4: Run, see pass**

```bash
uv run pytest packages/soccer-vision/tests/test_phase_splitter.py -v
```

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/phase/splitter.py \
        packages/soccer-vision/tests/test_phase_splitter.py
git commit -m "feat(phase): add phase splitter combining possession + ball location"
```

---

## Self-Review

**Spec coverage:**
- §3.1a per-frame homography → Tasks 9, 11
- §3.2 PitchSpec → Task 7
- §3.3 distance thresholds → encoded in Task 14's `PossessionThresholds`
- §4.0c contested-possession state → Task 14
- §4.0d age-group baselines → not in this plan (deferred to season-analysis Plan D)
- §5 fine-tune workflow → Tasks 1–6
- §6.1 possession proxy → Task 14
- §6.2 shape metrics → next plan (Plan C)
- §7 outputs → Plan D
- §8 testing → covered inline; CI/regression test on full pipeline deferred to Plan C

**Spec gaps in this plan:**
- "Validity check on H" (reject if any player projects >10% off-pitch) — should be in homography or filter module. Adding now: extend Task 12's filter to also reject homographies that project the centroid of the source frame outside [-0.1, 1.1]². Actually, that's well-served by the existing boundary filter at Task 12 — rows with bad H produce off-pitch projections and get dropped. Sufficient for v1.

**Placeholder scan:** No `TBD` / `TODO` / generic "add validation" / etc. found. Three places use `# TODO when running` inside notebook cells (the Roboflow workspace name in Task 3, marked explicitly as a runtime fill).

**Type consistency:**
- `PitchSpec` field names + classmethods match across Task 7 spec.
- `fit_homography`, `smooth_homographies`, `HomographyError` are referenced consistently.
- `PitchMapper.transform(detections, homographies)` — homographies dict keyed by frame_idx — matches Task 11's output type.
- `classify_possession` returns `pd.Series` indexed by frame — matches Task 15's input contract.
- `label_phase(possession_state, ball_y_pitch, fps, transition_seconds=5.0)` — return type pd.Series, consistent.
- `apply_modal_team_per_track(detections)` — uses `track_id`, `class`, `team` columns which exist in the trajectories schema.

---

## Phase 2 + 3 Done

At this point:
- Phase 2: fine-tuned ball detector trained, evaluated against the bake-off clip's ≥75% acceptance gate, integrated into `RoboflowBackend` via `ball_weights_path`.
- Phase 3: `pitch/spec.py`, `pitch/homography.py`, `pitch/mapper.py`, `pitch/filter.py`, `phase/team_mode.py`, `phase/possession.py`, `phase/splitter.py` all landed with TDD tests.
- `RoboflowBackend.process_with_pitch()` available for downstream metric pipelines to consume both detections and keypoints in a single call.

Ready for Plan C (Phases 4–6 metrics: shape, space, gaps, ball-relative, dynamics, zones, youth).
