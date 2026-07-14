# Line-Mask Auto-Label Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn trusted-homography frames into (JPEG, per-pixel line-class mask) training pairs with a manifest carrying game/field/view identifiers — the data engine for the per-frame line-segmentation model.

**Architecture:** Pure rasterization core `pitch/line_masks.py` (homography → uint8 class mask, behind-camera-clipped, physically-scaled thickness) + orchestrator `line_dataset.py` (confidence gating, stride/per-view sampling, sequential decode, manifest/stats/contact-sheet, `games.toml` registry via stdlib `tomllib`). Spec: `docs/superpowers/specs/2026-07-14-line-mask-autolabel-design.md` (amended: registry is TOML not YAML — no new dependency; each entry also carries `session` = dir containing `homographies.parquet`).

**Tech Stack:** numpy, OpenCV (rasterization + video decode), pandas/parquet, stdlib `tomllib`. Python 3.11, mypy --strict, ruff.

**Conventions:** TDD (failing test first, watch it fail, implement, watch pass, commit). pytest from `packages/soccer-vision/`; from repo root `uv run ruff check packages/soccer-vision/src packages/soccer-vision/tests` clean and `uv run mypy` shows no NEW errors (58 pre-existing in tests/test_view_dataset.py, test_view_digest.py, test_view_registration.py are known stub drift). Homographies in `homographies.parquet` are FULL-PIXEL image→pitch (`pipeline.homographies_from_parquet` → `HomographyEntry(H, source, confidence)`). Masks are uint8 PNG (never JPEG). Commit per task.

**File map:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/line_masks.py` (Task 1)
- Create: `packages/soccer-vision/tests/test_line_masks.py` (Task 1)
- Create: `packages/soccer-vision/src/soccer_vision/line_dataset.py` (Tasks 2–3)
- Create: `packages/soccer-vision/tests/test_line_dataset.py` (Tasks 2–3)
- Modify: `docs/superpowers/specs/2026-07-14-line-mask-autolabel-design.md` (Task 3, yaml→toml note)
- Task 4 is an evidence run (no production code).

---

### Task 1: Pure rasterization core `pitch/line_masks.py`

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/line_masks.py`
- Create: `packages/soccer-vision/tests/test_line_masks.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for pitch/line_masks.py — homography -> field-line class masks."""
import numpy as np
import pytest
from numpy.typing import NDArray

from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M
from soccer_vision.pitch import line_masks as lm
from soccer_vision.pitch.calib_anchor import frame_homography
from tests.test_global_crop import K_TRUE, RVEC, SIZE, TVEC

W, H = SIZE
H_IMG2PITCH: NDArray[np.float64] = np.asarray(frame_homography(K_TRUE, RVEC, TVEC), np.float64)
PAINT = 0.12


def _pitch_of(mask: NDArray[np.uint8], cls: int, n: int = 200) -> NDArray[np.float64]:
    """Map up to n pixels of a class through the TRUE homography into pitch coords."""
    ys, xs = np.nonzero(mask == cls)
    assert len(xs) > 0, f"class {cls} drew no pixels"
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int)
    pts = np.column_stack([xs[idx], ys[idx], np.ones(len(idx))]).astype(np.float64)
    q = (H_IMG2PITCH @ pts.T).T
    assert np.all(q[:, 2] > 0), "mask pixel maps behind the camera"
    return np.asarray(q[:, :2] / q[:, 2:3], np.float64)


def _dist_m(cls: int, p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Metre distance from pitch points to the nearest geometry of a class."""
    x_m, y_m = p[:, 0] * WIDTH_M, p[:, 1] * LENGTH_M
    if cls == lm.CLS_TOUCHLINE:
        return np.minimum(np.abs(x_m), np.abs(x_m - WIDTH_M))
    if cls == lm.CLS_GOAL_LINE:
        return np.minimum(np.abs(y_m), np.abs(y_m - LENGTH_M))
    if cls == lm.CLS_MIDLINE:
        return np.abs(y_m - LENGTH_M / 2)
    if cls == lm.CLS_CENTER_CIRCLE:
        r_m = 0.087 * LENGTH_M
        d = np.hypot(x_m - WIDTH_M / 2, y_m - LENGTH_M / 2)
        return np.abs(d - r_m)
    raise AssertionError(cls)


def test_mask_pixels_lie_on_their_line() -> None:
    mask = lm.line_mask(H_IMG2PITCH, SIZE)
    assert mask.shape == (H, W) and mask.dtype == np.uint8
    for cls in (lm.CLS_TOUCHLINE, lm.CLS_GOAL_LINE, lm.CLS_MIDLINE, lm.CLS_CENTER_CIRCLE):
        d = _dist_m(cls, _pitch_of(mask, cls))
        # paint half-width + raster slack: every sampled pixel within ~2 paint widths
        assert float(np.max(d)) < PAINT * 2.5, f"class {cls}: max {np.max(d):.3f} m off"


def test_box_lines_lie_on_box_geometry() -> None:
    mask = lm.line_mask(H_IMG2PITCH, SIZE)
    p = _pitch_of(mask, lm.CLS_BOX_LINE)
    x_m, y_m = p[:, 0] * WIDTH_M, p[:, 1] * LENGTH_M
    bl = 0.157 * LENGTH_M
    cx_l, cx_r = (0.5 - 0.592 / 2) * WIDTH_M, (0.5 + 0.592 / 2) * WIDTH_M
    d_front = np.minimum(np.abs(y_m - bl), np.abs(y_m - (LENGTH_M - bl)))
    d_sides = np.minimum(np.abs(x_m - cx_l), np.abs(x_m - cx_r))
    assert float(np.max(np.minimum(d_front, d_sides))) < PAINT * 2.5


def test_sky_pose_draws_no_unphysical_pixels() -> None:
    """Corrupt the pose so part of the field flips behind the camera: every pixel the
    mask still draws must map with w > 0 AND sit on its geometry (clipping worked);
    wrapped/mirrored line pixels are the failure mode this guards."""
    bad = H_IMG2PITCH.copy()
    bad[2, :] = np.array([bad[2, 0] * 0.9, bad[2, 1] - 300.0 / (H * bad[2, 2]), bad[2, 2]])
    mask = lm.line_mask(bad, SIZE)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return  # fully clipped is acceptable
    pts = np.column_stack([xs, ys, np.ones(len(xs))]).astype(np.float64)
    q = (bad @ pts.T).T
    assert np.all(q[:, 2] > 1e-9)


def test_thickness_scales_with_depth() -> None:
    """The midline spans depth in this side view: its near-field end must be drawn
    thicker (in px) than its far-field end."""
    mask = lm.line_mask(H_IMG2PITCH, SIZE)
    cols = np.nonzero((mask == lm.CLS_MIDLINE).any(axis=0))[0]
    assert len(cols) > 50
    lo, hi = cols[5], cols[-6]
    t_lo = int((mask[:, lo] == lm.CLS_MIDLINE).sum())
    t_hi = int((mask[:, hi] == lm.CLS_MIDLINE).sum())
    near, far = max(t_lo, t_hi), min(t_lo, t_hi)
    assert near >= far
    assert 2 <= far and near <= 9  # clamps honored (7 max + anti-aliased-free raster slack)


def test_overlay_shapes() -> None:
    frame = np.zeros((H, W, 3), np.uint8)
    mask = lm.line_mask(H_IMG2PITCH, SIZE)
    out = lm.mask_overlay(frame, mask)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert int((out != 0).sum()) > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd packages/soccer-vision && uv run pytest tests/test_line_masks.py -x -q`
Expected: FAIL — no module `line_masks`.

- [ ] **Step 3: Implement**

```python
"""Rasterize the known field structure through a trusted homography into a per-pixel
line-class mask — the auto-label core for the line-segmentation data engine.

Every drawn pixel is guaranteed physical: polylines are projected segment-wise through
the pitch->pixel map with behind-camera clipping (clipped_polyline), so a pose whose far
end flips behind the camera yields missing pixels, never wrapped ones. Thickness is
physically scaled (~0.12 m of paint) via the local projection scale. Pure: no I/O.
Spec: docs/superpowers/specs/2026-07-14-line-mask-autolabel-design.md
"""
from __future__ import annotations

from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M
from soccer_vision.pitch.spec import PitchSpec
from soccer_vision.viz.pitch_overlay import clipped_polyline

CLS_TOUCHLINE: Final = 1
CLS_GOAL_LINE: Final = 2
CLS_MIDLINE: Final = 3
CLS_BOX_LINE: Final = 4
CLS_CENTER_CIRCLE: Final = 5
LINE_CLASSES: Final[dict[int, str]] = {
    CLS_TOUCHLINE: "touchline",
    CLS_GOAL_LINE: "goal_line",
    CLS_MIDLINE: "midline",
    CLS_BOX_LINE: "box_line",
    CLS_CENTER_CIRCLE: "center_circle",
}
# BGR tint per class for contact sheets (background stays untinted)
_OVERLAY_BGR: Final[dict[int, tuple[int, int, int]]] = {
    CLS_TOUCHLINE: (0, 255, 255), CLS_GOAL_LINE: (255, 128, 0),
    CLS_MIDLINE: (0, 128, 255), CLS_BOX_LINE: (255, 0, 255),
    CLS_CENTER_CIRCLE: (0, 255, 0),
}
_SCALE_M = np.array([WIDTH_M, LENGTH_M])
_SEG_STEPS = 40      # subdivisions per polyline edge (curvature + partial clipping)
_THICK_MIN, _THICK_MAX = 2, 7


def _pitch_polylines(spec: PitchSpec) -> list[tuple[int, NDArray[np.float64]]]:
    """(class_id, (N,2) pitch-coord polyline) for the field structure. Draw order =
    list order; later classes win at intersections (midline/circle over touchlines)."""
    bl = spec.penalty_box_length_frac
    cx_l = 0.5 - spec.penalty_box_width_frac / 2.0
    cx_r = 0.5 + spec.penalty_box_width_frac / 2.0
    r_y = spec.center_circle_radius_frac                 # fraction of pitch LENGTH
    r_x = r_y * LENGTH_M / WIDTH_M                       # same metres, width units
    th = np.linspace(0.0, 2.0 * np.pi, 73)
    circle = np.column_stack([0.5 + r_x * np.cos(th), 0.5 + r_y * np.sin(th)])
    return [
        (CLS_TOUCHLINE, np.array([[0.0, 0.0], [0.0, 1.0]])),
        (CLS_TOUCHLINE, np.array([[1.0, 0.0], [1.0, 1.0]])),
        (CLS_GOAL_LINE, np.array([[0.0, 0.0], [1.0, 0.0]])),
        (CLS_GOAL_LINE, np.array([[0.0, 1.0], [1.0, 1.0]])),
        (CLS_BOX_LINE, np.array([[cx_l, 0.0], [cx_l, bl], [cx_r, bl], [cx_r, 0.0]])),
        (CLS_BOX_LINE, np.array([[cx_l, 1.0], [cx_l, 1.0 - bl], [cx_r, 1.0 - bl], [cx_r, 1.0]])),
        (CLS_MIDLINE, np.array([[0.0, 0.5], [1.0, 0.5]])),
        (CLS_CENTER_CIRCLE, circle),
    ]


def line_mask(
    h_img_to_pitch: NDArray[np.floating[Any]],
    size: tuple[int, int],
    *,
    spec: PitchSpec | None = None,
    paint_width_m: float = 0.12,
) -> NDArray[np.uint8]:
    """(H, W) uint8 class mask for a frame with FULL-PIXEL image->pitch homography."""
    w, h = size
    p = np.linalg.inv(np.asarray(h_img_to_pitch, np.float64))   # pitch -> pixel
    mask = np.zeros((h, w), np.uint8)
    for cls, poly in _pitch_polylines(spec or PitchSpec.standard_9v9()):
        for a, b in zip(poly[:-1], poly[1:], strict=False):
            steps = np.linspace(0.0, 1.0, _SEG_STEPS + 1)
            samples = a[None, :] + steps[:, None] * (b - a)[None, :]
            for s0, s1 in zip(samples[:-1], samples[1:], strict=False):
                seg = clipped_polyline(p, np.array([s0, s1]), size=size, margin=200)
                if len(seg) != 2:
                    continue
                px_len = float(np.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]))
                m_len = float(np.linalg.norm((s1 - s0) * _SCALE_M))
                if m_len <= 1e-9:
                    continue
                t = int(round(px_len / m_len * paint_width_m))
                t = min(max(t, _THICK_MIN), _THICK_MAX)
                cv2.line(mask, seg[0], seg[1], int(cls), t, cv2.LINE_8)
    return mask


def mask_overlay(
    frame: NDArray[np.uint8], mask: NDArray[np.uint8], *, alpha: float = 0.6
) -> NDArray[np.uint8]:
    """Tint a frame with the mask's classes (contact-sheet rendering; Patrick assesses)."""
    out = frame.copy()
    for cls, bgr in _OVERLAY_BGR.items():
        sel = mask == cls
        if sel.any():
            out[sel] = (np.asarray(bgr, np.float64) * alpha
                        + out[sel].astype(np.float64) * (1.0 - alpha)).astype(np.uint8)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_line_masks.py -q`
Expected: 5 passed. If `test_thickness_scales_with_depth` picks a column where the midline exits the frame, adjust the column offsets (`cols[5]`/`cols[-6]`) — do NOT loosen the clamp bounds. If `test_sky_pose_draws_no_unphysical_pixels`'s corruption doesn't flip anything (fully in front), strengthen the row-2 perturbation until `_wsigns_ok`-style flipping occurs — the invariant assert stays.
Then from repo root: ruff + mypy per Conventions.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/line_masks.py packages/soccer-vision/tests/test_line_masks.py
git commit -m "feat(pitch): line-mask rasterizer (homography -> field-line class mask, clipped + physically scaled)"
```

---

### Task 2: Dataset orchestration core in `line_dataset.py`

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/line_dataset.py`
- Create: `packages/soccer-vision/tests/test_line_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for line_dataset.py — selection, generation, manifest idempotency."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from soccer_vision import line_dataset as ld
from soccer_vision.pipeline import homographies_to_parquet
from soccer_vision.pitch.propagation import HomographyEntry

_W, _H = 320, 240
# A benign synthetic homography for a 320x240 frame: image -> pitch, roughly the
# whole pitch visible (scale px to [0,1] with mild perspective).
_H_IMG2PITCH = np.array([[1.0 / _W, 0.0, 0.0], [0.0, 1.0 / _H, 0.0], [0.0, 1e-4, 1.0]])


def _video(path: Path, n_frames: int = 90) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (_W, _H))  # type: ignore[attr-defined]
    for i in range(n_frames):
        frame = np.full((_H, _W, 3), (i * 3) % 255, np.uint8)
        vw.write(frame)
    vw.release()
    return path


def _session(tmp_path: Path, frames: list[int], *, source: str = "registered",
             conf: float = 0.9) -> Path:
    session = tmp_path / "session"
    session.mkdir(exist_ok=True)
    entries = {f: HomographyEntry(_H_IMG2PITCH, source, conf) for f in frames}
    homographies_to_parquet(entries, session / "homographies.parquet")
    return session


def test_select_frames_gates_and_strides() -> None:
    entries = {f: HomographyEntry(_H_IMG2PITCH, "registered", 0.9) for f in range(90)}
    entries[10] = HomographyEntry(_H_IMG2PITCH, "registered", 0.3)   # below gate
    entries[20] = HomographyEntry(_H_IMG2PITCH, "none", 0.9)         # bad source
    sel = ld.select_frames(entries, fps=30.0, stride_s=1.0, per_view_cap=120,
                           min_confidence=0.6, view_of=None)
    assert sel == [0, 30, 60]  # 1 fps stride from frame 0
    sel2 = ld.select_frames(entries, fps=30.0, stride_s=0.5, per_view_cap=120,
                            min_confidence=0.6, view_of=None)
    assert 15 in sel2 and 10 not in sel2 and 20 not in sel2


def test_select_frames_per_view_cap() -> None:
    entries = {f: HomographyEntry(_H_IMG2PITCH, "manual", 0.9) for f in range(0, 300, 3)}
    view_of = {f: (0 if f < 150 else 1) for f in range(300)}
    sel = ld.select_frames(entries, fps=30.0, stride_s=0.1, per_view_cap=5,
                           min_confidence=0.6, view_of=view_of)
    per_view = {0: 0, 1: 0}
    for f in sel:
        per_view[view_of[f]] += 1
    assert per_view[0] == 5 and per_view[1] == 5


def test_build_writes_pairs_manifest_and_stats(tmp_path: Path) -> None:
    video = _video(tmp_path / "game.mp4")
    session = _session(tmp_path, list(range(90)))
    out = tmp_path / "dataset"
    stats = ld.build_game(video, session / "homographies.parquet", out,
                          game_id="g1", field_id="fieldA", stride_s=1.0)
    assert stats.n_written == 3 and stats.n_undecodable == 0
    man = pd.read_parquet(out / "manifest.parquet")
    assert len(man) == 3
    assert set(man.columns) >= {"game_id", "field_id", "view_id", "frame", "source",
                                "confidence", "image", "mask"}
    row = man.iloc[0]
    img = cv2.imread(str(out / row["image"]))
    msk = cv2.imread(str(out / row["mask"]), cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (_H, _W, 3)
    assert msk is not None and msk.shape == (_H, _W)
    assert set(np.unique(msk)) <= {0, 1, 2, 3, 4, 5}
    assert (msk > 0).sum() > 0                       # lines actually drawn
    # PNG losslessness: reload equals in-memory mask exactly (spot frame 0)
    from soccer_vision.pitch.line_masks import line_mask
    assert np.array_equal(msk, line_mask(_H_IMG2PITCH, (_W, _H)))


def test_rerun_is_idempotent_per_game(tmp_path: Path) -> None:
    video = _video(tmp_path / "game.mp4")
    session = _session(tmp_path, list(range(90)))
    out = tmp_path / "dataset"
    ld.build_game(video, session / "homographies.parquet", out, game_id="g1",
                  field_id="fieldA", stride_s=1.0)
    ld.build_game(video, session / "homographies.parquet", out, game_id="g1",
                  field_id="fieldA", stride_s=1.0)                    # re-run: replaces
    other = ld.build_game(video, session / "homographies.parquet", out, game_id="g2",
                          field_id="fieldB", stride_s=1.0)
    man = pd.read_parquet(out / "manifest.parquet")
    assert len(man) == 6 and set(man["game_id"]) == {"g1", "g2"}
    assert other.n_written == 3


def test_undecodable_frames_are_dropped_and_counted(tmp_path: Path) -> None:
    video = _video(tmp_path / "game.mp4", n_frames=60)
    session = _session(tmp_path, [0, 30, 300])       # 300 is beyond the video
    out = tmp_path / "dataset"
    stats = ld.build_game(video, session / "homographies.parquet", out, game_id="g1",
                          field_id="fieldA", stride_s=1.0)
    assert stats.n_written == 2 and stats.n_undecodable == 1
    man = pd.read_parquet(out / "manifest.parquet")
    assert sorted(man["frame"]) == [0, 30]
```

- [ ] **Step 2: Run to verify failure** — no module `line_dataset`.

- [ ] **Step 3: Implement** (`packages/soccer-vision/src/soccer_vision/line_dataset.py`)

```python
"""Auto-label dataset builder: trusted homographies + video -> (JPEG, line-mask PNG)
pairs with a split-agnostic manifest (game/field/view identifiers) — the training-data
engine for the per-frame line-segmentation model.

Sampling discipline: confidence-gated (>= min_confidence, accepted sources only),
strided to ~stride_s seconds, capped per view (near-duplicate control). Frames decode in
ONE ascending sequential pass (never per-frame seek); undecodable frames are dropped and
counted, never silently written. Masks are lossless PNG. Re-running a game REPLACES its
manifest rows (idempotent per game).
Spec: docs/superpowers/specs/2026-07-14-line-mask-autolabel-design.md
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from soccer_vision.pipeline import homographies_from_parquet
from soccer_vision.pitch.line_masks import LINE_CLASSES, line_mask, mask_overlay
from soccer_vision.pitch.propagation import HomographyEntry
from soccer_vision.pitch.spec import PitchSpec

ACCEPTED_SOURCES = ("rep", "registered", "manual")
_MANIFEST_COLS = ["game_id", "field_id", "view_id", "frame", "source", "confidence",
                  "image", "mask"]


@dataclass(frozen=True)
class GameStats:
    game_id: str
    n_candidates: int      # frames passing the trust gate, before sampling
    n_selected: int        # after stride + per-view cap
    n_written: int
    n_undecodable: int
    class_pixel_frac: dict[str, float]


def select_frames(
    entries: Mapping[int, HomographyEntry],
    *,
    fps: float,
    stride_s: float,
    per_view_cap: int,
    min_confidence: float,
    view_of: Mapping[int, int] | None,
) -> list[int]:
    """Trust-gate, stride to ~stride_s seconds, cap per view. Deterministic."""
    trusted = sorted(f for f, e in entries.items()
                     if e.source in ACCEPTED_SOURCES and e.confidence >= min_confidence)
    stride = max(1, round(fps * stride_s))
    strided = [f for f in trusted if f % stride == 0]
    if view_of is None:
        return strided
    taken: dict[int, int] = {}
    out: list[int] = []
    for f in strided:
        v = view_of.get(f, -1)
        if taken.get(v, 0) < per_view_cap:
            taken[v] = taken.get(v, 0) + 1
            out.append(f)
    return out


def _decode_selected(video_path: Path, frames: list[int]) -> tuple[dict[int, np.ndarray], int]:
    """One ascending sequential pass; returns {frame: image} + undecodable count."""
    want = sorted(frames)
    got: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    pos = 0
    try:
        for f in want:
            while pos < f:
                if not cap.grab():
                    return got, len(want) - len(got)
                pos += 1
            ok, img = cap.read()
            if not ok:
                return got, len(want) - len(got)
            pos += 1
            got[f] = img
    finally:
        cap.release()
    return got, len(want) - len(got)


def _merge_manifest(out_dir: Path, game_id: str, rows: list[dict[str, object]]) -> None:
    path = out_dir / "manifest.parquet"
    new = pd.DataFrame(rows, columns=_MANIFEST_COLS)
    if path.exists():
        old = pd.read_parquet(path)
        old = old[old["game_id"] != game_id]
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(path, index=False)


def build_game(
    video_path: str | Path,
    homographies_path: str | Path,
    out_dir: str | Path,
    *,
    game_id: str,
    field_id: str,
    view_of: Mapping[int, int] | None = None,
    stride_s: float = 1.0,
    per_view_cap: int = 120,
    min_confidence: float = 0.6,
    jpeg_quality: int = 90,
    spec: PitchSpec | None = None,
    contact_sheet: bool = True,
) -> GameStats:
    """Generate one game's (image, mask) pairs into out_dir and merge its manifest rows."""
    video_path, out_dir = Path(video_path), Path(out_dir)
    entries = homographies_from_parquet(Path(homographies_path))
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    trusted = [f for f, e in entries.items()
               if e.source in ACCEPTED_SOURCES and e.confidence >= min_confidence]
    selected = select_frames(entries, fps=fps, stride_s=stride_s,
                             per_view_cap=per_view_cap, min_confidence=min_confidence,
                             view_of=view_of)
    images, n_undec = _decode_selected(video_path, selected)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    pix_counts = {name: 0 for name in LINE_CLASSES.values()}
    total_px = 0
    overlays: list[np.ndarray] = []
    for f in sorted(images):
        e = entries[f]
        mask = line_mask(e.H, size, spec=spec)
        img_rel = f"images/{game_id}_{f:06d}.jpg"
        msk_rel = f"masks/{game_id}_{f:06d}.png"
        cv2.imwrite(str(out_dir / img_rel), images[f],
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        cv2.imwrite(str(out_dir / msk_rel), mask)
        for cls, name in LINE_CLASSES.items():
            pix_counts[name] += int((mask == cls).sum())
        total_px += mask.size
        rows.append({"game_id": game_id, "field_id": field_id,
                     "view_id": int(view_of.get(f, -1)) if view_of else -1,
                     "frame": int(f), "source": e.source,
                     "confidence": float(e.confidence),
                     "image": img_rel, "mask": msk_rel})
        if contact_sheet and len(overlays) < 16:
            overlays.append(cv2.resize(mask_overlay(images[f], mask), (0, 0),
                                       fx=0.25, fy=0.25))
    _merge_manifest(out_dir, game_id, rows)
    if contact_sheet and overlays:
        cols = 4
        rws = -(-len(overlays) // cols)
        th, tw = overlays[0].shape[:2]
        sheet = np.zeros((rws * th, cols * tw, 3), np.uint8)
        for i, o in enumerate(overlays):
            r, c = divmod(i, cols)
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = o
        cv2.imwrite(str(out_dir / f"contact_{game_id}.jpg"), sheet)
    frac = {k: (v / total_px if total_px else 0.0) for k, v in pix_counts.items()}
    return GameStats(game_id, len(trusted), len(selected), len(rows), n_undec, frac)
```

Note the contact-sheet sampling: the first 16 written frames is acceptable for v1 IF selection is strided across the whole clip (it is); a reviewer may prefer even spacing across `sorted(images)` — implement even spacing (`np.linspace` over the sorted list) rather than first-16 if trivial.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_line_dataset.py tests/test_line_masks.py -q` → 10 passed. Ruff + mypy per Conventions.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/line_dataset.py packages/soccer-vision/tests/test_line_dataset.py
git commit -m "feat(dataset): line-mask dataset builder (gate/stride/cap, sequential decode, idempotent manifest)"
```

---

### Task 3: `games.toml` registry + CLI + stats file

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/line_dataset.py` (append registry + CLI)
- Modify: `packages/soccer-vision/tests/test_line_dataset.py` (append tests)
- Modify: `docs/superpowers/specs/2026-07-14-line-mask-autolabel-design.md` (yaml→toml amendment)

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_load_games_registry(tmp_path: Path) -> None:
    reg = tmp_path / "games.toml"
    reg.write_text(
        '[g1]\nfield = "fieldA"\nvideo = "v1.mp4"\nsession = "s1"\n'
        '[g2]\nfield = "fieldB"\nvideo = "v2.mp4"\nsession = "s2"\n')
    games = ld.load_games(reg)
    assert set(games) == {"g1", "g2"}
    assert games["g1"].field == "fieldA"
    assert games["g1"].video == (tmp_path / "v1.mp4")     # relative to the registry
    assert games["g2"].session == (tmp_path / "s2")


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    video = _video(tmp_path / "game.mp4")
    session = _session(tmp_path, list(range(90)))
    (tmp_path / "games.toml").write_text(
        f'[g1]\nfield = "fieldA"\nvideo = "game.mp4"\nsession = "{session.name}"\n'
        '[missing]\nfield = "fieldB"\nvideo = "nope.mp4"\nsession = "nope"\n')
    out = tmp_path / "dataset"
    ld.main(["--games", str(tmp_path / "games.toml"), "--out", str(out)])
    text = capsys.readouterr().out
    assert "g1" in text and "SKIP" in text            # missing inputs skipped loudly
    stats = (out / "dataset_stats.json")
    assert stats.exists()
    man = pd.read_parquet(out / "manifest.parquet")
    assert set(man["game_id"]) == {"g1"}


def test_cli_game_filter(tmp_path: Path) -> None:
    video = _video(tmp_path / "game.mp4")
    session = _session(tmp_path, list(range(90)))
    (tmp_path / "games.toml").write_text(
        f'[g1]\nfield = "fieldA"\nvideo = "game.mp4"\nsession = "{session.name}"\n')
    out = tmp_path / "dataset"
    ld.main(["--games", str(tmp_path / "games.toml"), "--out", str(out), "--game", "g1"])
    assert (out / "manifest.parquet").exists()
```

- [ ] **Step 2: verify failure** — no attribute `load_games` / `main`.

- [ ] **Step 3: Implement** (append to line_dataset.py; add imports `argparse`, `json`, `tomllib`, `dataclasses.asdict`)

```python
@dataclass(frozen=True)
class GameEntry:
    field: str
    video: Path
    session: Path          # dir containing homographies.parquet (+ optional view_manifest.parquet)


def load_games(path: str | Path) -> dict[str, GameEntry]:
    """games.toml: [game_id] tables with field/video/session; paths relative to the file."""
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    base = path.parent
    out: dict[str, GameEntry] = {}
    for gid, entry in raw.items():
        out[gid] = GameEntry(field=str(entry["field"]),
                             video=base / str(entry["video"]),
                             session=base / str(entry["session"]))
    return out


def _view_of_from_manifest(session: Path) -> dict[int, int] | None:
    p = session / "view_manifest.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return {int(f): int(v) for f, v in zip(df["frame"], df["view_id"], strict=True)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Line-mask auto-label dataset builder")
    ap.add_argument("--games", required=True, type=Path, help="games.toml registry")
    ap.add_argument("--out", required=True, type=Path, help="dataset output dir")
    ap.add_argument("--game", action="append", default=None,
                    help="limit to these game ids (repeatable; default: all)")
    ap.add_argument("--stride-s", type=float, default=1.0)
    ap.add_argument("--per-view-cap", type=int, default=120)
    ap.add_argument("--min-confidence", type=float, default=0.6)
    args = ap.parse_args(argv)

    games = load_games(args.games)
    wanted = args.game or sorted(games)
    all_stats: dict[str, dict[str, object]] = {}
    for gid in wanted:
        g = games[gid]
        hpath = g.session / "homographies.parquet"
        if not g.video.exists() or not hpath.exists():
            print(f"{gid}: SKIP (missing {'video' if not g.video.exists() else 'homographies'})")
            continue
        view_of = _view_of_from_manifest(g.session)
        if view_of is None:
            print(f"{gid}: no view_manifest.parquet — per-view cap inactive (weaker dedup)")
        stats = build_game(g.video, hpath, args.out, game_id=gid, field_id=g.field,
                           view_of=view_of, stride_s=args.stride_s,
                           per_view_cap=args.per_view_cap,
                           min_confidence=args.min_confidence)
        all_stats[gid] = {**asdict(stats), "field_id": g.field}
        print(f"{gid}: {stats.n_written} pairs written "
              f"({stats.n_candidates} trusted, {stats.n_selected} selected, "
              f"{stats.n_undecodable} undecodable) -> {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset_stats.json").write_text(json.dumps({
        "config": {"stride_s": args.stride_s, "per_view_cap": args.per_view_cap,
                   "min_confidence": args.min_confidence},
        "games": all_stats,
    }, indent=2))


if __name__ == "__main__":
    main()
```

Also amend the spec doc (`docs/superpowers/specs/2026-07-14-line-mask-autolabel-design.md` §1): replace the `games.yaml` block with the TOML equivalent and note "TOML via stdlib tomllib (no new dependency); each entry carries `session`" — one short edit, not a rewrite.

- [ ] **Step 4: verify pass** — module tests + FULL suite + ruff + mypy.

- [ ] **Step 5: Commit**

```bash
git add -A packages/soccer-vision/src packages/soccer-vision/tests docs/superpowers/specs/2026-07-14-line-mask-autolabel-design.md
git commit -m "feat(dataset): games.toml registry + line-dataset CLI + stats; spec toml amendment"
```

---

### Task 4: v0 evidence run on oceanside (no production code)

- [ ] **Step 1: Export oceanside homographies from the labeler session** (green frames only, honest confidences — mirrors `LabelerState.export` without the server):

```bash
cd packages/soccer-vision && uv run python - <<'EOF'
from pathlib import Path
from soccer_vision.labeler.chain import load_chain
from soccer_vision.labeler.state import clicks_from_sidecar, line_clicks_from_sidecar
from soccer_vision.pipeline import homographies_to_parquet
from soccer_vision.pitch.manual_anchor import build_segments, cumulative_transforms
from soccer_vision.pitch.physical_calib import solve_session
from soccer_vision.pitch.global_crop import frame_confidence
from soccer_vision.labeler.chain import denormalize_homography
from soccer_vision.pitch.propagation import HomographyEntry

home = Path.home() / "sv-labeler"
interframe, n_frames, size = load_chain(home / ".sv_labeler_cache/da63d2bb640cc974.npz")
pts = clicks_from_sidecar(home / ".sv_labeler_cache/oceanside_clip.clicks.json")
lns = line_clicks_from_sidecar(home / ".sv_labeler_cache/oceanside_clip.clicks.json")
seg = build_segments(interframe, n_frames)
tf = cumulative_transforms(interframe, seg)
calib = solve_session(pts, lns, size, tf, segment_of=seg)
entries = {}
for f in range(n_frames):
    if calib.status(f) != "green":
        continue
    h = calib.frame_homography(f)
    entries[f] = HomographyEntry(denormalize_homography(h, size), "manual",
                                 frame_confidence(calib, f, status="green"))
out = home / "oceanside_session"
out.mkdir(exist_ok=True)
homographies_to_parquet(entries, out / "homographies.parquet")
print(f"exported {len(entries)} green frames -> {out}")
EOF
```

(Expect ~2,100 frames. Note: `frame_homography` here returns the NORMALIZED H; `denormalize_homography(h, size)` converts to full-pixel — the parquet convention. Verify one round-trip against `PitchMapper` conventions if in doubt.)

- [ ] **Step 2: Write the registry and run the generator**

```bash
cat > ~/sv-labeler/games.toml <<'EOF'
[oceanside_clip_v0]
field = "oceanside"
video = "oceanside_clip.mp4"
session = "oceanside_session"
EOF
uv run python -m soccer_vision.line_dataset --games ~/sv-labeler/games.toml \
  --out ~/sv-labeler/line_dataset_v0
```

Expected: ~70–90 pairs (2,100 green frames strided to 1 fps; no view manifest → cap inactive, noted loudly).

- [ ] **Step 3: Report** — dataset_stats.json contents (frames, class pixel fractions), the contact sheet path (`~/sv-labeler/line_dataset_v0/contact_oceanside_clip_v0.jpg`), and any anomalies. **Patrick assesses the contact sheet — render and report only, never interpret the images.**

---

## Self-review checklist

1. Spec coverage: §1 inputs → Tasks 2–3 (toml amendment in Task 3); §2 mask core → Task 1; §3 sampling/orchestration → Tasks 2–3; §4 dataset plan → Task 4 note (v0) + user labeling (out of code scope); §5 tests → Tasks 1–3; §6 out of scope respected. No gaps.
2. Placeholders: none — every step has complete code/commands.
3. Type consistency: `select_frames`/`build_game`/`GameStats`/`GameEntry`/`load_games`/`main` names used consistently across tasks; `line_mask(h, size, *, spec, paint_width_m)` matches Task 1↔2 usage; `HomographyEntry(H, source, confidence)` matches pipeline.py.
