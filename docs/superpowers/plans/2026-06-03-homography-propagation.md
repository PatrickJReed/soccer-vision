# Homography Propagation Implementation Plan (Phase 3.5a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the ~84% of Trace frames that lack pitch keypoints by registering them to nearby landmark anchors (bidirectional bounded frame-to-frame chaining), lifting pitch-homography coverage without labeling, with per-frame provenance + runtime confidence.

**Architecture:** A new CPU module `pitch/propagation.py` does masked ORB registration + bidirectional chaining + distance-weighted homography blend + a disagreement-based confidence. A new pipeline stage (`build_homographies`) runs it between detection and assembly, emitting a dense `homographies.parquet` checkpoint; `assemble_phases` gains an optional precomputed-homographies argument and stays pure.

**Tech Stack:** OpenCV (ORB, `findHomography`, `warpPerspective`), numpy, pandas, pyarrow. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-03-homography-propagation-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `packages/soccer-vision/src/soccer_vision/pitch/propagation.py` | **New.** `HomographyEntry`, `register`, `blend_homographies`, `disagreement_confidence`, `propagate_homographies`. The testable core. |
| `packages/soccer-vision/src/soccer_vision/pipeline.py` | **Modify.** `build_homographies`, `homographies` arg on `assemble_phases`, provenance columns, `PipelineResult` fields, `assemble_from_homographies`, `analyze_video` wiring, homographies parquet I/O. |
| `packages/soccer-vision/tests/test_pitch_propagation.py` | **New.** Tasks 1–2. |
| `packages/soccer-vision/tests/test_pipeline_homographies.py` | **New.** Tasks 3–5. |
| `packages/soccer-vision/tests/test_pipeline_assemble.py` | **Modify.** Task 4 (PipelineResult fields). |
| `examples/colab_homography_propagation.ipynb` | **New.** Task 6 — calibration + acceptance gate. |

Commands: `uv run pytest <path> -q`, `uv run ruff check packages/soccer-vision/`, `uv run mypy packages/soccer-vision/src` (strict; rules E,F,I,B,UP,RUF; line-length 100).

---

## Task 1: Registration + homography blend + confidence (leaf helpers)

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/propagation.py`
- Test: `packages/soccer-vision/tests/test_pitch_propagation.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/soccer-vision/tests/test_pitch_propagation.py`:

```python
"""Tests for the homography-propagation leaf helpers."""

from __future__ import annotations

import cv2
import numpy as np
from soccer_vision.pitch.propagation import (
    HomographyEntry,
    blend_homographies,
    disagreement_confidence,
    register,
)


def _textured_image(seed: int = 0) -> np.ndarray:
    """A BGR image with strong, repeatable corner features for ORB."""
    rng = np.random.default_rng(seed)
    gray = (rng.random((400, 600)) * 60).astype(np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(20, 560)), int(rng.integers(20, 360))
        cv2.rectangle(gray, (x, y), (x + 18, y + 18), int(rng.integers(80, 255)), -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_register_recovers_translation() -> None:
    base = _textured_image()
    M = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, 15.0], [0.0, 0.0, 1.0]])
    warped = cv2.warpPerspective(base, M, (600, 400))
    full = np.full((400, 600), 255, np.uint8)

    G = register(base, warped, full, full)   # maps base pixels -> warped pixels
    assert G is not None
    p = np.array([300.0, 200.0, 1.0])
    q = G @ p
    q /= q[2]
    assert abs(q[0] - 325.0) < 3.0 and abs(q[1] - 215.0) < 3.0


def test_register_returns_none_on_blank_frames() -> None:
    blank = np.zeros((400, 600, 3), np.uint8)
    full = np.full((400, 600), 255, np.uint8)
    assert register(blank, blank, full, full) is None


def test_blend_is_weighted_average_normalized() -> None:
    h1 = np.eye(3)
    h2 = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # w1=1 -> exactly h1; w1=0 -> exactly h2; w1=0.5 -> midpoint translation 5
    assert np.allclose(blend_homographies(h1, h2, 1.0), h1)
    assert np.allclose(blend_homographies(h1, h2, 0.0), h2)
    mid = blend_homographies(h1, h2, 0.5)
    assert np.isclose(mid[2, 2], 1.0)
    assert np.isclose(mid[0, 2], 5.0)


def test_disagreement_confidence_monotonic_and_clamped() -> None:
    # identical Hs -> zero disagreement -> confidence 1.0
    h = np.eye(3)
    assert disagreement_confidence(h, h, tau=0.1) == 1.0
    # a large shift -> disagreement >> tau -> clamped to 0.0
    far = np.array([[1.0, 0.0, 500.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    c = disagreement_confidence(h, far, tau=0.1)
    assert 0.0 <= c < 0.5


def test_homography_entry_fields() -> None:
    e = HomographyEntry(np.eye(3), "anchor", 1.0)
    assert e.source == "anchor" and e.confidence == 1.0 and e.H.shape == (3, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_propagation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'soccer_vision.pitch.propagation'`

- [ ] **Step 3: Write the implementation**

Create `packages/soccer-vision/src/soccer_vision/pitch/propagation.py`:

```python
"""Homography propagation: bridge no-landmark frames by registering them to anchors.

Trace is a no-parallax virtual-PTZ crop of a fixed camera, so consecutive frames
are related by a global homography recoverable from static background features.
We chain those inter-frame homographies from the nearest landmark anchors on both
sides of a gap and blend them, lifting pitch-homography coverage without labeling.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Image-space reference grid (normalized 0..1) used to measure forward/backward
# disagreement; scaled to the frame size at use.
_GRID = np.array([(x, y) for x in (0.2, 0.5, 0.8) for y in (0.2, 0.5, 0.8)], dtype=np.float64)


@dataclass(frozen=True)
class HomographyEntry:
    """A frame's homography plus where it came from."""

    H: NDArray[np.floating]
    source: str           # "anchor" | "propagated"
    confidence: float     # 1.0 for anchors; runtime estimate for propagated


def register(
    img_src: NDArray[np.uint8],
    img_dst: NDArray[np.uint8],
    mask_src: NDArray[np.uint8],
    mask_dst: NDArray[np.uint8],
    *,
    n_features: int = 3000,
    min_inliers: int = 12,
) -> NDArray[np.floating] | None:
    """Homography mapping img_src pixels -> img_dst pixels from masked ORB features.

    Returns None when too few features/matches are found (e.g. blurred or blank
    frames). Masks are uint8 (255 = use, 0 = ignore — e.g. over moving players).
    """
    orb = cv2.ORB_create(n_features)
    gs = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gd = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    ks, ds = orb.detectAndCompute(gs, mask_src)
    kd, dd = orb.detectAndCompute(gd, mask_dst)
    if ds is None or dd is None or len(ds) < min_inliers or len(dd) < min_inliers:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(ds, dd)
    if len(matches) < min_inliers:
        return None
    src = np.float32([ks[m.queryIdx].pt for m in matches])
    dst = np.float32([kd[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return None if H is None else H.astype(np.float64)


def blend_homographies(
    h_a: NDArray[np.floating], h_b: NDArray[np.floating], w_a: float
) -> NDArray[np.floating]:
    """Weighted element-wise blend of two homographies, normalized so H[2,2]=1."""
    blended = w_a * h_a + (1.0 - w_a) * h_b
    if blended[2, 2] != 0:
        blended = blended / blended[2, 2]
    return blended


def _map_points(H: NDArray[np.floating], pts: NDArray[np.floating]) -> NDArray[np.floating]:
    homog = np.column_stack([pts, np.ones(len(pts))])
    out = (H @ homog.T).T
    return out[:, :2] / out[:, 2:3]


def disagreement_confidence(
    h_fwd: NDArray[np.floating],
    h_bwd: NDArray[np.floating],
    *,
    tau: float = 0.10,
    frame_size: tuple[int, int] = (1920, 1080),
) -> float:
    """Confidence in [0,1] from how much the two chains disagree (pitch-units).

    Both Hs map the same reference grid into pitch space; their mean separation is
    a runtime estimate of propagation error. confidence = clamp(1 - disagree/tau).
    """
    grid_px = _GRID * np.array(frame_size, dtype=np.float64)
    disagree = float(np.linalg.norm(_map_points(h_fwd, grid_px) - _map_points(h_bwd, grid_px), axis=1).mean())
    return float(np.clip(1.0 - disagree / tau, 0.0, 1.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_propagation.py -q`
Expected: PASS (5 passed). If `test_register_recovers_translation` is flaky, the tolerance (3.0 px) or feature count can be nudged — registration is RANSAC-based, but a clean translation on a textured image recovers well within 3 px.

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check packages/soccer-vision/src/soccer_vision/pitch/propagation.py packages/soccer-vision/tests/test_pitch_propagation.py && uv run mypy packages/soccer-vision/src/soccer_vision/pitch/propagation.py`
Expected: clean / Success.

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/propagation.py packages/soccer-vision/tests/test_pitch_propagation.py
git commit -m "feat(pitch): registration + homography blend + disagreement confidence"
```

---

## Task 2: `propagate_homographies` — bidirectional bounded chaining

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/propagation.py`
- Test: `packages/soccer-vision/tests/test_pitch_propagation.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `packages/soccer-vision/tests/test_pitch_propagation.py`:

```python
from soccer_vision.pitch.propagation import propagate_homographies


def _shift_frame(base: np.ndarray, dx: int) -> np.ndarray:
    M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return cv2.warpPerspective(base, M, (base.shape[1], base.shape[0]))


def _scene(n_frames: int, pan_per_frame: int = 4):
    """A panning clip: frame f is base shifted by f*pan. read_frame + anchors map
    each frame's pixels back to a fixed 'pitch' = base-frame pixel coords / 1000."""
    base = _textured_image(1)
    frames = {f: _shift_frame(base, f * pan_per_frame) for f in range(n_frames)}

    def read_frame(f: int):
        return frames.get(f)

    # Anchor pitch homography for a frame: undo its pan, then scale to pitch units.
    # frame f pixels -> base pixels: shift by -f*pan ; base pixels -> pitch: /1000.
    def anchor_H(f: int) -> np.ndarray:
        undo = np.array([[1.0, 0.0, -f * pan_per_frame], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        scale = np.diag([1 / 1000.0, 1 / 1000.0, 1.0])
        return scale @ undo

    return read_frame, anchor_H, frames


def test_propagation_bridges_gap_within_window() -> None:
    read_frame, anchor_H, _ = _scene(11)
    anchors = {0: anchor_H(0), 10: anchor_H(10)}     # gap of 9 frames between
    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    out = propagate_homographies(anchors, read_frame, boxes, max_gap=15)

    assert out[0].source == "anchor" and out[10].source == "anchor"
    assert 5 in out and out[5].source == "propagated"
    # propagated H for frame 5 should match its true pitch homography closely.
    truth = anchor_H(5)
    p = np.array([300.0, 200.0])
    got = (out[5].H @ np.array([p[0], p[1], 1.0]))
    got = got[:2] / got[2]
    exp = (truth @ np.array([p[0], p[1], 1.0]))
    exp = exp[:2] / exp[2]
    assert np.linalg.norm(got - exp) < 0.02   # < 0.02 pitch-units


def test_gap_beyond_max_is_not_bridged() -> None:
    read_frame, anchor_H, _ = _scene(11)
    anchors = {0: anchor_H(0), 10: anchor_H(10)}
    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    out = propagate_homographies(anchors, read_frame, boxes, max_gap=4)   # gap 9 > 4
    assert set(out) == {0, 10}                # only anchors, nothing bridged


def test_unbridged_frames_absent_and_anchors_confident() -> None:
    read_frame, anchor_H, _ = _scene(11)
    anchors = {0: anchor_H(0), 10: anchor_H(10)}
    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    out = propagate_homographies(anchors, read_frame, boxes, max_gap=15)
    assert out[0].confidence == 1.0
    assert 0.0 <= out[5].confidence <= 1.0


def test_empty_anchors_returns_empty() -> None:
    read_frame, _, _ = _scene(3)
    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    assert propagate_homographies({}, read_frame, boxes, max_gap=15) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_propagation.py -q`
Expected: FAIL — `ImportError: cannot import name 'propagate_homographies'`

- [ ] **Step 3: Write the implementation**

Append to `packages/soccer-vision/src/soccer_vision/pitch/propagation.py`:

```python
_PLAYER_CLASSES = ("player", "goalkeeper", "referee")


def _frame_mask(boxes: pd.DataFrame, frame: int, shape: tuple[int, int]) -> NDArray[np.uint8]:
    """255 on static background, 0 over player/ref boxes (dilated)."""
    mask = np.full(shape, 255, np.uint8)
    sel = boxes[(boxes["frame"] == frame) & boxes["class"].isin(_PLAYER_CLASSES)]
    for _, r in sel.iterrows():
        cv2.rectangle(
            mask,
            (int(r["bbox_x1"]) - 12, int(r["bbox_y1"]) - 12),
            (int(r["bbox_x2"]) + 12, int(r["bbox_y2"]) + 12),
            0, -1,
        )
    return mask


def _chain(
    anchor: int,
    targets: list[int],
    read_frame: Callable[[int], NDArray[np.uint8] | None],
    boxes: pd.DataFrame,
    H_anchor: NDArray[np.floating],
    n_features: int,
    min_inliers: int,
) -> dict[int, NDArray[np.floating]]:
    """Chain consecutive registrations from `anchor` over `targets` (ordered, adjacent).

    Returns {frame: pitch_H} for each reached frame; stops at the first failure.
    """
    out: dict[int, NDArray[np.floating]] = {}
    prev_img = read_frame(anchor)
    if prev_img is None:
        return out
    shape = prev_img.shape[:2]
    W = np.eye(3)                                  # maps anchor pixels -> current pixels
    prev_frame = anchor
    for f in targets:
        cur = read_frame(f)
        if cur is None:
            break
        G = register(prev_img, cur, _frame_mask(boxes, prev_frame, shape),
                     _frame_mask(boxes, f, shape), n_features=n_features, min_inliers=min_inliers)
        if G is None:
            break
        W = G @ W
        out[f] = H_anchor @ np.linalg.inv(W)       # pixel_f -> pitch
        prev_img, prev_frame = cur, f
    return out


def propagate_homographies(
    anchors: Mapping[int, NDArray[np.floating]],
    read_frame: Callable[[int], NDArray[np.uint8] | None],
    player_boxes: pd.DataFrame,
    *,
    max_gap: int = 25,
    disagreement_tau: float = 0.10,
    n_features: int = 3000,
    min_inliers: int = 12,
) -> dict[int, HomographyEntry]:
    """Bridge no-landmark gaps between anchors via bidirectional chaining.

    Each anchor keeps source='anchor', confidence=1.0. For each gap <= max_gap,
    chain forward from the left anchor and backward from the right anchor, blend by
    distance, and set confidence from forward/backward disagreement. Frames reached
    by neither chain are absent. Edge gaps (before first / after last anchor) are
    not bridged in v1.
    """
    out: dict[int, HomographyEntry] = {
        f: HomographyEntry(np.asarray(H, dtype=np.float64), "anchor", 1.0)
        for f, H in anchors.items()
    }
    keys = sorted(anchors)
    for a, b in zip(keys, keys[1:], strict=False):
        gap = b - a - 1
        if gap < 1 or gap > max_gap:
            continue
        inner = list(range(a + 1, b))
        fwd = _chain(a, inner, read_frame, player_boxes, np.asarray(anchors[a], np.float64),
                     n_features, min_inliers)
        bwd = _chain(b, inner[::-1], read_frame, player_boxes, np.asarray(anchors[b], np.float64),
                     n_features, min_inliers)
        for t in inner:
            hf, hb = fwd.get(t), bwd.get(t)
            if hf is not None and hb is not None:
                w_f = (b - t) / (b - a)
                out[t] = HomographyEntry(
                    blend_homographies(hf, hb, w_f), "propagated",
                    disagreement_confidence(hf, hb, tau=disagreement_tau),
                )
            elif hf is not None:
                out[t] = HomographyEntry(hf, "propagated", max(0.0, 1.0 - (t - a) / (max_gap + 1)))
            elif hb is not None:
                out[t] = HomographyEntry(hb, "propagated", max(0.0, 1.0 - (b - t) / (max_gap + 1)))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_propagation.py -q`
Expected: PASS (9 passed total).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check packages/soccer-vision/src/soccer_vision/pitch/propagation.py packages/soccer-vision/tests/test_pitch_propagation.py && uv run mypy packages/soccer-vision/src/soccer_vision/pitch/propagation.py`
Expected: clean / Success.

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/propagation.py packages/soccer-vision/tests/test_pitch_propagation.py
git commit -m "feat(pitch): propagate_homographies bidirectional bounded chaining"
```

---

## Task 3: Homographies checkpoint I/O + `build_homographies`

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pipeline.py`
- Test: `packages/soccer-vision/tests/test_pipeline_homographies.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/soccer-vision/tests/test_pipeline_homographies.py`:

```python
"""Tests for homographies checkpoint I/O."""

from __future__ import annotations

import numpy as np
from soccer_vision.pipeline import homographies_from_parquet, homographies_to_parquet
from soccer_vision.pitch.propagation import HomographyEntry


def test_homographies_parquet_roundtrip(tmp_path) -> None:
    entries = {
        3: HomographyEntry(np.eye(3), "anchor", 1.0),
        4: HomographyEntry(np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]]),
                           "propagated", 0.7),
    }
    path = tmp_path / "homographies.parquet"
    homographies_to_parquet(entries, path)
    back = homographies_from_parquet(path)

    assert set(back) == {3, 4}
    assert back[3].source == "anchor" and back[3].confidence == 1.0
    assert back[4].source == "propagated" and abs(back[4].confidence - 0.7) < 1e-9
    assert np.allclose(back[4].H, entries[4].H)


def test_homographies_to_parquet_columns(tmp_path) -> None:
    import pandas as pd
    homographies_to_parquet({3: HomographyEntry(np.eye(3), "anchor", 1.0)},
                            tmp_path / "h.parquet")
    df = pd.read_parquet(tmp_path / "h.parquet")
    assert list(df.columns) == [
        "frame", *[f"h{i}{j}" for i in range(3) for j in range(3)], "source", "confidence",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/soccer-vision/tests/test_pipeline_homographies.py -q`
Expected: FAIL — `ImportError: cannot import name 'homographies_to_parquet'`

- [ ] **Step 3: Add the implementation**

In `pipeline.py`, add the import near the other pitch imports:
```python
from soccer_vision.pitch.propagation import HomographyEntry, propagate_homographies
```

Add the import for cv2 lazily inside `build_homographies` (it is a base dep but keep pipeline import light is unnecessary here — cv2 is already required by pitch modules, so a top-level import is fine). Add `import numpy as np` to the top-level imports.

Then append these functions to `pipeline.py`:

```python
_H_COLS = [f"h{i}{j}" for i in range(3) for j in range(3)]


def homographies_to_parquet(entries: dict[int, HomographyEntry], path: Path) -> None:
    """Serialize {frame: HomographyEntry} to a flat parquet (frame, h00..h22, source, conf)."""
    rows = []
    for frame, e in sorted(entries.items()):
        flat = np.asarray(e.H, dtype=np.float64).reshape(9)
        rows.append({"frame": frame, **dict(zip(_H_COLS, flat, strict=True)),
                     "source": e.source, "confidence": e.confidence})
    df = pd.DataFrame(rows, columns=["frame", *_H_COLS, "source", "confidence"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def homographies_from_parquet(path: Path) -> dict[int, HomographyEntry]:
    df = pd.read_parquet(path)
    out: dict[int, HomographyEntry] = {}
    for _, r in df.iterrows():
        H = np.array([r[c] for c in _H_COLS], dtype=np.float64).reshape(3, 3)
        out[int(r["frame"])] = HomographyEntry(H, str(r["source"]), float(r["confidence"]))
    return out


def build_homographies(
    keypoints: pd.DataFrame,
    video_path: Path,
    trajectories_px: pd.DataFrame,
    *,
    kp_conf_threshold: float = 0.5,
    max_gap: int = 25,
    disagreement_tau: float = 0.10,
) -> dict[int, HomographyEntry]:
    """Anchors from keypoints + propagation into the gaps. Reads frames from the video."""
    import cv2

    anchors = build_frame_homographies(keypoints, conf_threshold=kp_conf_threshold)
    cap = cv2.VideoCapture(str(video_path))

    def read_frame(idx: int) -> "np.ndarray | None":
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return frame if ok else None

    try:
        return propagate_homographies(
            anchors, read_frame, trajectories_px,
            max_gap=max_gap, disagreement_tau=disagreement_tau,
        )
    finally:
        cap.release()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/soccer-vision/tests/test_pipeline_homographies.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check packages/soccer-vision/src/soccer_vision/pipeline.py packages/soccer-vision/tests/test_pipeline_homographies.py && uv run mypy packages/soccer-vision/src/soccer_vision/pipeline.py`
Expected: clean / Success.

```bash
git add packages/soccer-vision/src/soccer_vision/pipeline.py packages/soccer-vision/tests/test_pipeline_homographies.py
git commit -m "feat(pipeline): homographies checkpoint I/O + build_homographies"
```

---

## Task 4: `assemble_phases` consumes homographies + provenance + coverage fields

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pipeline.py`
- Test: `packages/soccer-vision/tests/test_pipeline_assemble.py` (modify)

- [ ] **Step 1: Update PipelineResult and write the failing tests**

In `pipeline.py`, replace the `PipelineResult` dataclass with:

```python
@dataclass(frozen=True)
class PipelineResult:
    """Enriched outputs of the pipeline plus coverage diagnostics."""

    trajectories: pd.DataFrame   # per-detection, +x_pitch/+y_pitch, team modal-cleaned
    phases: pd.DataFrame         # per-frame over [0, total_frames); +homography_source/conf
    homography_coverage: float   # fraction of frames with a homography (anchor + propagated)
    ball_coverage: float         # fraction of frames with a non-NaN ball pitch coord
    anchor_coverage: float       # fraction from detected landmarks
    propagated_coverage: float   # fraction filled by propagation
```

Append to `packages/soccer-vision/tests/test_pipeline_assemble.py`:

```python
def test_assemble_phases_accepts_precomputed_homographies() -> None:
    from soccer_vision.pitch.propagation import HomographyEntry

    traj = _scene()
    kp = _identity_keypoints(3)
    # identity-H entries: frame 0 anchor, frames 1-2 propagated (lower conf)
    homs = {
        0: HomographyEntry(np.eye(3), "anchor", 1.0),
        1: HomographyEntry(np.eye(3), "propagated", 0.6),
        2: HomographyEntry(np.eye(3), "propagated", 0.6),
    }
    result = assemble_phases(traj, kp, fps=FPS, total_frames=3, homographies=homs)

    ph = result.phases.set_index("frame")
    assert ph.loc[0, "homography_source"] == "anchor"
    assert ph.loc[1, "homography_source"] == "propagated"
    assert abs(ph.loc[1, "homography_conf"] - 0.6) < 1e-9
    assert result.anchor_coverage == 1 / 3
    assert abs(result.propagated_coverage - 2 / 3) < 1e-9
    assert abs(result.homography_coverage - 1.0) < 1e-9


def test_assemble_phases_legacy_path_marks_anchors() -> None:
    traj = _scene()
    kp = _identity_keypoints(3)
    result = assemble_phases(traj, kp, fps=FPS, total_frames=3)  # no homographies arg
    ph = result.phases.set_index("frame")
    assert set(ph["homography_source"]) <= {"anchor", "none"}
    assert result.propagated_coverage == 0.0
    assert result.anchor_coverage == 1.0   # all 3 frames are landmark anchors in the fixture
```

Also update the existing `test_assemble_phases_no_homography_degrades_to_unknown` to additionally assert provenance:
```python
    assert set(result.phases["homography_source"]) == {"none"}
    assert result.anchor_coverage == 0.0 and result.propagated_coverage == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/soccer-vision/tests/test_pipeline_assemble.py -q`
Expected: FAIL — `assemble_phases() got an unexpected keyword argument 'homographies'` (and PipelineResult field errors).

- [ ] **Step 3: Rewrite `assemble_phases`**

Replace the body of `assemble_phases` (signature + implementation) with:

```python
def assemble_phases(
    trajectories_px: pd.DataFrame,
    keypoints: pd.DataFrame,
    fps: float,
    total_frames: int,
    *,
    homographies: dict[int, HomographyEntry] | None = None,
    kp_conf_threshold: float = 0.5,
    homography_alpha: float = 0.5,
    filter_margin: float = 0.05,
    possession_thresholds: PossessionThresholds | None = None,
    transition_seconds: float = 5.0,
) -> PipelineResult:
    """Run the full pitch + phase chain on tracker output. Pure; no I/O.

    When `homographies` (a precomputed {frame: HomographyEntry} from the propagation
    stage) is given, it is used directly; otherwise homographies are computed from
    keypoints (landmark anchors + carry-forward smoothing) exactly as before.
    """
    if homographies is not None:
        h_entries = homographies
    else:
        raw_h = build_frame_homographies(keypoints, conf_threshold=kp_conf_threshold)
        smoothed = smooth_homographies(raw_h, alpha=homography_alpha)
        # Only landmark frames are provenance-bearing anchors; carry-forward frames
        # are mapped but not counted as coverage (preserves the raw-anchor metric).
        h_entries = {f: HomographyEntry(smoothed[f], "anchor", 1.0) for f in raw_h}
        h_map_extra = {f: H for f, H in smoothed.items() if f not in raw_h}
    if not homographies and not h_entries:
        logger.warning("No homographies fitted; pitch coords NaN, phases all 'unknown'.")

    h_map = {f: e.H for f, e in h_entries.items()}
    if homographies is None:
        h_map.update(h_map_extra)   # legacy: still map carry-forward frames

    enriched = PitchMapper().transform(trajectories_px, h_map)
    enriched = filter_outside_pitch(enriched, margin=filter_margin)
    enriched = apply_modal_team_per_track(enriched)
    validate_trajectories(enriched)

    poss = classify_possession(enriched, possession_thresholds).sort_index()
    window = max(1, round(fps))
    poss_smoothed = smooth_possession(poss, window_frames=window)

    ball = enriched[enriched["class"] == "ball"]
    ball_by_frame = ball.sort_values("conf").groupby("frame")[["x_pitch", "y_pitch"]].last()

    full_index = pd.RangeIndex(0, total_frames, name="frame")
    poss_full = poss_smoothed.reindex(full_index, fill_value="unknown")
    ball_x_full = ball_by_frame["x_pitch"].reindex(full_index)
    ball_y_full = ball_by_frame["y_pitch"].reindex(full_index)

    phase_series = label_phase(poss_full, ball_y_full, fps, transition_seconds=transition_seconds)

    src = pd.Series("none", index=full_index)
    conf = pd.Series(0.0, index=full_index)
    for f, e in h_entries.items():
        if 0 <= f < total_frames:
            src.loc[f] = e.source
            conf.loc[f] = e.confidence

    phases = pd.DataFrame({
        "frame": full_index,
        "t_seconds": full_index.to_numpy() / fps,
        "possession_state": poss_full.to_numpy(),
        "phase": phase_series.to_numpy(),
        "ball_x_pitch": ball_x_full.to_numpy(),
        "ball_y_pitch": ball_y_full.to_numpy(),
        "homography_source": src.to_numpy(),
        "homography_conf": conf.to_numpy(),
    }).astype({
        "frame": "int64", "t_seconds": "float64",
        "possession_state": "object", "phase": "object",
        "ball_x_pitch": "float64", "ball_y_pitch": "float64",
        "homography_source": "object", "homography_conf": "float64",
    })

    n_anchor = sum(1 for e in h_entries.values() if e.source == "anchor")
    n_prop = sum(1 for e in h_entries.values() if e.source == "propagated")
    anchor_cov = n_anchor / total_frames if total_frames else 0.0
    prop_cov = n_prop / total_frames if total_frames else 0.0
    ball_cov = float(ball_y_full.notna().sum()) / total_frames if total_frames else 0.0

    return PipelineResult(
        trajectories=enriched,
        phases=phases,
        homography_coverage=anchor_cov + prop_cov,
        ball_coverage=ball_cov,
        anchor_coverage=anchor_cov,
        propagated_coverage=prop_cov,
    )
```

Add `import numpy as np` to the top of `pipeline.py` if not already present (Task 3 added it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/soccer-vision/tests/test_pipeline_assemble.py -q`
Expected: PASS (all assemble tests, including the two new ones).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check packages/soccer-vision/ && uv run mypy packages/soccer-vision/src`
Expected: clean / Success.

```bash
git add packages/soccer-vision/src/soccer_vision/pipeline.py packages/soccer-vision/tests/test_pipeline_assemble.py
git commit -m "feat(pipeline): assemble_phases consumes homographies + provenance + coverage split"
```

---

## Task 5: `assemble_from_homographies` + wire `analyze_video`

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pipeline.py`
- Test: `packages/soccer-vision/tests/test_pipeline_homographies.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `packages/soccer-vision/tests/test_pipeline_homographies.py`:

```python
import pandas as pd
from soccer_vision.pipeline import PipelineResult, assemble_from_homographies
from soccer_vision.pitch.propagation import HomographyEntry

FPS = 1.0


def _traj() -> pd.DataFrame:
    rows = []
    for f in range(3):
        rows.append({"frame": f, "t_seconds": f / FPS, "track_id": 1,
                     "x_px": 0.5, "y_px": 0.25, "bbox_x1": 0.49, "bbox_y1": 0.24,
                     "bbox_x2": 0.51, "bbox_y2": 0.26, "class": "player", "team": "own", "conf": 0.9})
        rows.append({"frame": f, "t_seconds": f / FPS, "track_id": -1 - f,
                     "x_px": 0.5, "y_px": 0.27, "bbox_x1": 0.49, "bbox_y1": 0.26,
                     "bbox_x2": 0.51, "bbox_y2": 0.28, "class": "ball", "team": "unknown", "conf": 0.9})
    return pd.DataFrame(rows).astype({"frame": "int64", "track_id": "int64"})


def test_assemble_from_homographies_roundtrip(tmp_path) -> None:
    from soccer_vision.pipeline import homographies_to_parquet

    traj = _traj()
    traj_path = tmp_path / "trajectories_px.parquet"
    h_path = tmp_path / "homographies.parquet"
    traj.to_parquet(traj_path, index=False)
    homographies_to_parquet({f: HomographyEntry(np.eye(3), "anchor", 1.0) for f in range(3)}, h_path)
    out_dir = tmp_path / "out"

    result = assemble_from_homographies(traj_path, h_path, out_dir)
    assert isinstance(result, PipelineResult)
    assert (out_dir / "trajectories.parquet").exists()
    assert (out_dir / "phases.parquet").exists()
    assert result.anchor_coverage == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/soccer-vision/tests/test_pipeline_homographies.py::test_assemble_from_homographies_roundtrip -q`
Expected: FAIL — `ImportError: cannot import name 'assemble_from_homographies'`

- [ ] **Step 3: Add `assemble_from_homographies` and wire `analyze_video`**

Append to `pipeline.py`:

```python
def assemble_from_homographies(
    trajectories_px_path: Path,
    homographies_path: Path,
    out_dir: Path,
    *,
    fps: float | None = None,
    **assemble_opts: object,
) -> PipelineResult:
    """Stage-3 re-run from precomputed homographies (instant; no video, no GPU)."""
    trajectories_px = pd.read_parquet(trajectories_px_path)
    homographies = homographies_from_parquet(homographies_path)
    resolved_fps, total_frames = _resolve_fps_and_frames(trajectories_px, fps)
    # keypoints are unused when homographies are supplied; pass an empty frame.
    empty_kp = pd.DataFrame(columns=["frame", "kp_idx", "x_px", "y_px", "conf"])
    result = assemble_phases(
        trajectories_px, empty_kp, fps=resolved_fps, total_frames=total_frames,
        homographies=homographies, **assemble_opts,  # type: ignore[arg-type]
    )
    _write_deliverables(result, Path(out_dir))
    return result
```

Then update `analyze_video` to compute and checkpoint homographies. Replace its body after the checkpoint writes with:

```python
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trajectories_px.to_parquet(out / "trajectories_px.parquet", index=False)
    keypoints.to_parquet(out / "keypoints.parquet", index=False)

    homographies = build_homographies(keypoints, video_path, trajectories_px)
    homographies_to_parquet(homographies, out / "homographies.parquet")

    resolved_fps, total_frames = _resolve_fps_and_frames(trajectories_px)
    result = assemble_phases(
        trajectories_px, keypoints, fps=resolved_fps, total_frames=total_frames,
        homographies=homographies, **assemble_opts,  # type: ignore[arg-type]
    )
    _write_deliverables(result, out)
    return result
```

Update the `analyze_video` docstring's "Stage 2 (pure)" line to read: "Stage 2 (CPU): propagate homographies from anchors. Stage 3 (pure): assemble."

Note: the existing `analyze_video` stub-backend test (`test_pipeline_analyze_video.py`) passes a dummy video path; `build_homographies` will open it via `cv2.VideoCapture`, which on a non-video path yields a cap that reads no frames — `read_frame` returns None, so propagation bridges nothing and only the (zero, in that fixture's identity-keypoint case) anchors remain. Update that test to also assert `(out_dir / "homographies.parquet").exists()`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest packages/soccer-vision/tests/ -q`
Expected: all pass (the stub-backend `analyze_video` test now also writes homographies.parquet). If the stub test fails because its fixture frames *do* register, relax its assertion to check file existence + `PipelineResult` type only.

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check packages/soccer-vision/ && uv run mypy packages/soccer-vision/src && uv run pytest packages/soccer-vision/tests/ -q`
Expected: clean / Success / all pass.

```bash
git add packages/soccer-vision/src/soccer_vision/pipeline.py packages/soccer-vision/tests/
git commit -m "feat(pipeline): assemble_from_homographies + analyze_video propagation stage"
```

---

## Task 6: Acceptance & calibration notebook

**Files:**
- Create: `examples/colab_homography_propagation.ipynb`

This notebook is the acceptance gate (run on Colab; CPU). It loads the Stage-1 checkpoint, sweeps `max_gap`, computes the held-out reprojection error (a propagated anchor's landmark error), and reports coverage gain.

- [ ] **Step 1: Build the notebook**

Create the notebook with these cells (build it programmatically with a small Python script to keep JSON valid, mirroring `examples/colab_homography_probe_v2.ipynb`). Cells:

1. **Markdown** — purpose + the gate (accuracy: held-out median ≤ 0.05; coverage: report the lift from 16%).
2. **Install** — `!rm -rf /content/soccer-vision && git clone -q ... && pip install -q "/content/soccer-vision/packages/soccer-vision[roboflow]"` then mount Drive.
3. **Calibration sweep:**
```python
import cv2, numpy as np, pandas as pd
from pathlib import Path
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS, build_frame_homographies
from soccer_vision.pitch.propagation import propagate_homographies

OUT = Path("/content/out")
if not (OUT / "keypoints.parquet").exists():
    OUT = Path("/content/drive/MyDrive/soccer-vision/out")
CLIP = "/content/drive/MyDrive/soccer-vision/bakeoff_clip.mp4"
kp = pd.read_parquet(OUT / "keypoints.parquet")
traj = pd.read_parquet(OUT / "trajectories_px.parquet")
anchors = build_frame_homographies(kp, conf_threshold=0.5)
total = int(traj["frame"].max()) + 1

cap = cv2.VideoCapture(CLIP)
def read_frame(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read(); return f if ok else None

# Held-out validation: drop every Kth anchor, propagate, compare to its landmarks.
held = sorted(anchors)[::5]
def reproj_error(entry_H, frame):
    kb = kp[(kp.frame == frame) & (kp.conf >= 0.5) & (kp.kp_idx < len(PITCH_LANDMARKS))]
    if len(kb) < 4: return None
    pts = np.column_stack([kb.x_px, kb.y_px, np.ones(len(kb))])
    m = (entry_H @ pts.T).T; m /= m[:, 2:3]
    return float(np.linalg.norm(m[:, :2] - PITCH_LANDMARKS[kb.kp_idx.to_numpy()], axis=1).mean())

for mg in [10, 20, 30, 45, 60]:
    train = {f: H for f, H in anchors.items() if f not in held}
    out = propagate_homographies(train, read_frame, traj, max_gap=mg)
    errs = [reproj_error(out[f].H, f) for f in held if f in out]
    errs = [e for e in errs if e is not None]
    cov = len(out) / total
    med = sorted(errs)[len(errs)//2] if errs else float("nan")
    print(f"max_gap={mg:3d}  coverage={cov:5.1%}  held-out median err={med:.3f}  (n={len(errs)})")
cap.release()
```
4. **Markdown** — interpretation: pick the largest `max_gap` whose held-out median ≤ 0.05; that is the calibrated default. Coverage at that `max_gap` is the lift over the 16% baseline.

- [ ] **Step 2: Validate notebook JSON**

Run: `uv run python -c "import nbformat; nbformat.validate(nbformat.read('examples/colab_homography_propagation.ipynb', as_version=4)); print('valid')"`
Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add examples/colab_homography_propagation.ipynb
git commit -m "docs(examples): homography propagation acceptance + max_gap calibration notebook"
```

- [ ] **Step 4: USER ACTION — run on Colab**

Run the notebook on the bake-off clip. Acceptance: a `max_gap` exists where held-out median error ≤ 0.05 with coverage materially above 16%. Paste the sweep table back; the chosen `max_gap` becomes the `propagate_homographies` / `build_homographies` default (one-line change + commit if it differs from 25).

---

## Self-Review

**Spec coverage:**
- §2 3-stage architecture + `homographies.parquet` checkpoint → Tasks 3, 5.
- §3.1 bidirectional bounded chaining + masking + blend → Tasks 1–2.
- §3.2 disagreement confidence → Task 1 (`disagreement_confidence`), Task 2 (wired).
- §3.3 `max_gap` window + calibration → Task 2 (cap), Task 6 (calibration).
- §3.4 robustness (either-chain reach, edge gaps) → Task 2 (fwd/bwd fallback branches).
- §4 API (`propagate_homographies`, `build_homographies`, `assemble_phases` arg, `assemble_from_homographies`) → Tasks 2, 3, 4, 5.
- §5 outputs (`homographies.parquet`, phases provenance columns, PipelineResult fields) → Tasks 3, 4.
- §6 failure handling (None registration, gap>max, zero anchors) → Task 2 tests.
- §7 streaming (gap-by-gap; `read_frame` callable) → Task 2/3 design.
- §8 testing (pure helpers, synthetic-frame register, acceptance notebook) → Tasks 1, 2, 6.

**Placeholder scan:** none — every code step has complete code; commands have expected output.

**Type consistency:** `HomographyEntry(H, source, confidence)` used identically across Tasks 1–5; `propagate_homographies` / `build_homographies` / `assemble_phases(homographies=...)` / `homographies_to_parquet` / `homographies_from_parquet` signatures match their call sites; `_H_COLS` ordering is shared by writer and reader.

**Known tuning points (flagged, not placeholders):** the ORB recovery tolerance in Task 1's `register` test (3 px) and the stub-backend `analyze_video` test assertion in Task 5 may need a one-line relax during TDD — both are noted at their steps.
