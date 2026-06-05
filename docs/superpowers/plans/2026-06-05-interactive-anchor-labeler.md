# Interactive Anchor Labeler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local browser tool to manually anchor field landmarks on a fixed-camera video — each landmark clicked once, registration propagates it across frames — producing per-frame homographies with a live coverage/confidence readout, exported into the existing analytics pipeline.

**Architecture:** A pure point-propagation core (`pitch/manual_anchor.py`) computes per-frame homographies from clicks + the reused inter-frame registration chain, with the fit's reprojection residual as confidence. A stdlib-HTTP `labeler/` app wraps it: a precompute+cache layer, a testable `LabelerState`, an HTTP server, and a canvas frontend. Output is `homographies.parquet` + `keypoints.parquet` feeding the existing `assemble_from_homographies`.

**Tech Stack:** Python 3, numpy, OpenCV, pandas, stdlib `http.server`, vanilla HTML/JS canvas. pytest, mypy strict, ruff. Package at `packages/soccer-vision/`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/soccer_vision/pitch/manual_anchor.py` | Pure core: segments, cumulative transforms, point mapping, per-frame fit + residual, status/coverage, export helpers | Create |
| `src/soccer_vision/labeler/__init__.py` | Package marker | Create |
| `src/soccer_vision/labeler/chain.py` | Registration precompute over a video + on-disk cache | Create |
| `src/soccer_vision/labeler/state.py` | `LabelerState`: holds transforms + clicks, recomputes fits, exports parquets (no HTTP) | Create |
| `src/soccer_vision/labeler/server.py` | stdlib HTTP handler + `run()` wiring video → state → endpoints | Create |
| `src/soccer_vision/labeler/static/index.html` | Canvas UI shell | Create |
| `src/soccer_vision/labeler/static/app.js` | Scrub/arm/click/overlay/timeline logic | Create |
| `src/soccer_vision/labeler/__main__.py` | `python -m soccer_vision.labeler --video …` launcher | Create |
| `tests/test_pitch_manual_anchor.py` | Pure-core unit tests | Create |
| `tests/test_labeler_state.py` | `LabelerState` + export tests | Create |
| `tests/test_labeler_chain.py` | Chain cache round-trip test | Create |

Reused unchanged: `pitch/propagation.py::compute_interframe_homographies`,
`pitch/homography.py::fit_homography`/`HomographyError`,
`pitch/landmarks.py::PITCH_LANDMARKS`, `pitch/propagation.py::HomographyEntry`,
`pipeline.py::homographies_to_parquet`/`assemble_from_homographies`.

**Convention:** all homographies map **image pixels → pitch [0,1]²** (as
`fit_homography` produces). Inter-frame `G[i]` maps frame *i* pixels → frame *i+1*
pixels. `PITCH_LANDMARKS` is the 21-row `(x, y)` table.

---

## Task 1: Core dataclasses + segment detection

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_manual_anchor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pitch_manual_anchor.py`:

```python
"""Tests for the manual-anchor point-propagation core."""

from __future__ import annotations

import numpy as np
from soccer_vision.pitch.manual_anchor import Click, FrameFit, build_segments


def test_click_and_framefit_fields() -> None:
    c = Click(frame=3, kp_idx=0, x=10.0, y=20.0)
    assert (c.frame, c.kp_idx, c.x, c.y) == (3, 0, 10.0, 20.0)
    f = FrameFit(H=np.eye(3), residual=0.01, n_points=5)
    assert f.n_points == 5 and f.residual == 0.01


def test_segments_single_connected_run() -> None:
    interframe = {0: np.eye(3), 1: np.eye(3), 2: np.eye(3)}  # links 0-1-2-3
    seg = build_segments(interframe, n_frames=4)
    assert seg == {0: 0, 1: 0, 2: 0, 3: 0}


def test_segments_split_on_missing_link() -> None:
    interframe = {0: np.eye(3), 2: np.eye(3)}  # link 0-1, gap at 1-2, link 2-3
    seg = build_segments(interframe, n_frames=4)
    assert seg == {0: 0, 1: 0, 2: 1, 3: 1}


def test_segments_all_isolated() -> None:
    seg = build_segments({}, n_frames=3)
    assert seg == {0: 0, 1: 1, 2: 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -v`
Expected: FAIL — `ModuleNotFoundError: soccer_vision.pitch.manual_anchor`

- [ ] **Step 3: Create the module with dataclasses + build_segments**

Create `src/soccer_vision/pitch/manual_anchor.py`:

```python
"""Manual-anchor point propagation: turn sparse landmark clicks into per-frame
homographies using the fixed camera's frame-to-frame registration chain.

Each click is one (frame, kp_idx, pixel) observation. Because the Trace camera
has no parallax, a click in one frame can be mapped into any other frame in the
same registration-connected segment via cumulative inter-frame transforms. A
frame that accumulates >=4 distinct landmarks (clicked there or propagated in)
gets a homography; the fit's reprojection residual is its confidence.

All homographies map image pixels -> pitch [0,1]^2. Pure: no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Click:
    """One landmark observation: pixel (x, y) of keypoint kp_idx in a frame."""

    frame: int
    kp_idx: int
    x: float
    y: float


@dataclass(frozen=True)
class FrameFit:
    """A fitted per-frame homography plus its quality."""

    H: NDArray[np.floating]
    residual: float        # mean reprojection error in pitch units
    n_points: int          # distinct landmarks used


def build_segments(
    interframe: Mapping[int, NDArray[np.floating]], n_frames: int
) -> dict[int, int]:
    """Assign each frame 0..n_frames-1 a registration-segment id.

    interframe[i] present means frames i and i+1 are linked. A run of consecutive
    linked frames shares a segment id; a missing link starts a new segment.
    """
    seg: dict[int, int] = {}
    cur = 0
    for f in range(n_frames):
        if f == 0:
            seg[f] = 0
        elif (f - 1) in interframe:
            seg[f] = seg[f - 1]
        else:
            cur += 1
            seg[f] = cur
    return seg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py packages/soccer-vision/tests/test_pitch_manual_anchor.py
git commit -m "feat(manual_anchor): Click/FrameFit dataclasses + segment detection"
```

---

## Task 2: Cumulative transforms + point mapping

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_manual_anchor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pitch_manual_anchor.py`:

```python
from soccer_vision.pitch.manual_anchor import cumulative_transforms, map_point


def test_cumulative_identity_chain() -> None:
    interframe = {0: np.eye(3), 1: np.eye(3)}
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    for f in range(3):
        assert np.allclose(M[f], np.eye(3))


def test_cumulative_translation_chain() -> None:
    # each frame shifts +10px in x relative to the previous (i -> i+1).
    g = np.eye(3); g[0, 2] = 10.0
    interframe = {0: g, 1: g}
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    # M[f] maps frame f -> reference(0). frame 2 is +20px from frame 0, so
    # mapping a point back to ref subtracts 20.
    assert np.allclose(M[2] @ np.array([20.0, 0.0, 1.0]), [0.0, 0.0, 1.0])


def test_cumulative_resets_per_segment() -> None:
    g = np.eye(3); g[0, 2] = 10.0
    interframe = {0: g}  # link 0-1 only; frame 2 is a new segment
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    assert np.allclose(M[2], np.eye(3))  # segment start -> identity


def test_map_point_through_translation() -> None:
    g = np.eye(3); g[0, 2] = 10.0
    interframe = {0: g, 1: g}
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    # a point at x=5 in frame 0 appears at x=25 in frame 2 (camera moved +20).
    x, y = map_point(M[0], M[2], 5.0, 0.0)
    assert np.isclose(x, 25.0) and np.isclose(y, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -k "cumulative or map_point" -v`
Expected: FAIL — `ImportError: cumulative_transforms`

- [ ] **Step 3: Add the functions**

Append to `manual_anchor.py`:

```python
def cumulative_transforms(
    interframe: Mapping[int, NDArray[np.floating]], segment_of: Mapping[int, int]
) -> dict[int, NDArray[np.float64]]:
    """M[f] maps frame f pixels -> its segment's reference (first) frame pixels.

    M[start] = I; M[f] = M[f-1] @ inv(interframe[f-1]) within a segment, since
    interframe[f-1] maps f-1 -> f so its inverse maps f -> f-1.
    """
    transforms: dict[int, NDArray[np.float64]] = {}
    for f in sorted(segment_of):
        prev_same = (f - 1) in segment_of and segment_of[f] == segment_of[f - 1]
        if not prev_same:
            transforms[f] = np.eye(3)
        else:
            g = np.asarray(interframe[f - 1], dtype=np.float64)
            transforms[f] = transforms[f - 1] @ np.linalg.inv(g)
    return transforms


def map_point(
    m_src: NDArray[np.floating], m_dst: NDArray[np.floating], x: float, y: float
) -> tuple[float, float]:
    """Map pixel (x, y) from the source frame into the destination frame.

    src -> reference via m_src, reference -> dst via inv(m_dst). Both must be in
    the same segment (caller ensures this).
    """
    ref = np.asarray(m_src, dtype=np.float64) @ np.array([x, y, 1.0])
    dst = np.linalg.inv(np.asarray(m_dst, dtype=np.float64)) @ ref
    return float(dst[0] / dst[2]), float(dst[1] / dst[2])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py packages/soccer-vision/tests/test_pitch_manual_anchor.py
git commit -m "feat(manual_anchor): cumulative transforms + cross-frame point mapping"
```

---

## Task 3: Per-frame homography fit from propagated clicks

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_manual_anchor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pitch_manual_anchor.py`:

```python
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import fit_frame_homographies

# A consistent synthetic scene: image pixels = pitch coords * 1000 (+offset),
# identity inter-frame chain (no camera motion) over 5 frames.
_SCALE = 1000.0
_FIT_IDXS = [0, 3, 6, 11, 16, 19]


def _identity_chain(n: int) -> dict[int, np.ndarray]:
    return {i: np.eye(3) for i in range(n - 1)}


def _clicks_one_per_frame() -> list[Click]:
    # spread the 6 landmarks across frames 0..5, one landmark per frame.
    clicks = []
    for f, idx in enumerate(_FIT_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        clicks.append(Click(frame=f, kp_idx=idx, x=float(px), y=float(py)))
    return clicks


def test_fit_recovers_homography_from_spread_clicks() -> None:
    n = 6
    interframe = _identity_chain(n)
    seg = build_segments(interframe, n)
    M = cumulative_transforms(interframe, seg)
    fits = fit_frame_homographies(
        _clicks_one_per_frame(), M, seg, PITCH_LANDMARKS, window=10
    )
    # every frame sees all 6 landmarks (identity chain, window covers all)
    assert set(fits) == set(range(n))
    f3 = fits[3]
    assert f3.n_points == 6
    assert f3.residual < 1e-6
    # the recovered H maps image (pitch*1000) back to pitch
    pt = np.array([PITCH_LANDMARKS[0, 0] * _SCALE, PITCH_LANDMARKS[0, 1] * _SCALE, 1.0])
    mapped = f3.H @ pt
    mapped = mapped[:2] / mapped[2]
    assert np.allclose(mapped, PITCH_LANDMARKS[0], atol=1e-6)


def test_window_excludes_distant_clicks() -> None:
    # put all 6 landmark clicks at frames 0..5, but window=1 at frame 5 only sees
    # frames 4,5 -> 2 landmarks -> no fit.
    n = 6
    interframe = _identity_chain(n)
    seg = build_segments(interframe, n)
    M = cumulative_transforms(interframe, seg)
    fits = fit_frame_homographies(
        _clicks_one_per_frame(), M, seg, PITCH_LANDMARKS, window=1
    )
    assert 5 not in fits  # only 2 landmarks within window -> uncovered


def test_fewer_than_four_landmarks_uncovered() -> None:
    n = 3
    interframe = _identity_chain(n)
    seg = build_segments(interframe, n)
    M = cumulative_transforms(interframe, seg)
    clicks = [
        Click(0, _FIT_IDXS[0], *(PITCH_LANDMARKS[_FIT_IDXS[0]] * _SCALE)),
        Click(0, _FIT_IDXS[1], *(PITCH_LANDMARKS[_FIT_IDXS[1]] * _SCALE)),
        Click(0, _FIT_IDXS[2], *(PITCH_LANDMARKS[_FIT_IDXS[2]] * _SCALE)),
    ]
    fits = fit_frame_homographies(clicks, M, seg, PITCH_LANDMARKS, window=10)
    assert fits == {}


def test_clicks_do_not_cross_segments() -> None:
    # 4 landmarks clicked in segment 0 (frames 0,1), none in segment 1 (frame 2).
    interframe = {0: np.eye(3)}  # links 0-1; frame 2 isolated
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    clicks = [Click(0 if i < 2 else 1, idx, *(PITCH_LANDMARKS[idx] * _SCALE))
              for i, idx in enumerate(_FIT_IDXS[:4])]
    fits = fit_frame_homographies(clicks, M, seg, PITCH_LANDMARKS, window=10)
    assert 2 not in fits  # different segment, no clicks propagate in
    assert 0 in fits and 1 in fits
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -k fit -v`
Expected: FAIL — `ImportError: fit_frame_homographies`

- [ ] **Step 3: Implement the fit**

Append to `manual_anchor.py` (add `from soccer_vision.pitch.homography import HomographyError, fit_homography` to the imports at the top of the file):

```python
def _apply(H: NDArray[np.floating], pts: NDArray[np.floating]) -> NDArray[np.float64]:
    """Apply a 3x3 homography to (N, 2) points -> (N, 2)."""
    homog = np.column_stack([pts, np.ones(len(pts))])
    out = (np.asarray(H, dtype=np.float64) @ homog.T).T
    return out[:, :2] / out[:, 2:3]


def fit_frame_homographies(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    landmarks: NDArray[np.floating],
    *,
    window: int,
    min_points: int = 4,
) -> dict[int, FrameFit]:
    """Fit each frame's image->pitch homography from clicks propagated into it.

    For target frame g, gather clicks in the same segment with |click.frame - g|
    <= window, mapped into g's pixels; keep one observation per landmark (nearest
    click frame wins). With >= min_points distinct landmarks, fit a homography and
    record its mean reprojection residual (pitch units).
    """
    fits: dict[int, FrameFit] = {}
    for g in sorted(transforms):
        seg_g = segment_of[g]
        # nearest click (by frame distance) per landmark, within window + segment
        best: dict[int, tuple[int, float, float]] = {}  # kp_idx -> (dist, x, y)
        for c in clicks:
            if segment_of.get(c.frame) != seg_g:
                continue
            dist = abs(c.frame - g)
            if dist > window:
                continue
            x, y = map_point(transforms[c.frame], transforms[g], c.x, c.y)
            if c.kp_idx not in best or dist < best[c.kp_idx][0]:
                best[c.kp_idx] = (dist, x, y)
        if len(best) < min_points:
            continue
        idxs = sorted(best)
        image_pts = np.array([[best[i][1], best[i][2]] for i in idxs], dtype=np.float64)
        pitch_pts = np.asarray(landmarks, dtype=np.float64)[idxs]
        try:
            H = fit_homography(image_pts, pitch_pts)
        except HomographyError:
            continue
        residual = float(np.linalg.norm(_apply(H, image_pts) - pitch_pts, axis=1).mean())
        fits[g] = FrameFit(H=H, residual=residual, n_points=len(idxs))
    return fits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py packages/soccer-vision/tests/test_pitch_manual_anchor.py
git commit -m "feat(manual_anchor): per-frame homography fit from propagated clicks"
```

---

## Task 4: Status, coverage, and export helpers

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_manual_anchor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pitch_manual_anchor.py`:

```python
import pandas as pd
from soccer_vision.pitch.manual_anchor import (
    clicks_to_keypoints_df,
    coverage_fraction,
    frame_status,
    to_homography_entries,
)


def _fits() -> dict[int, FrameFit]:
    return {
        0: FrameFit(np.eye(3), residual=0.01, n_points=6),   # green
        1: FrameFit(np.eye(3), residual=0.09, n_points=5),   # yellow (> 0.05)
        # frame 2 missing -> red
    }


def test_frame_status_green_yellow_red() -> None:
    status = frame_status(_fits(), n_frames=3, residual_threshold=0.05)
    assert status == {0: "green", 1: "yellow", 2: "red"}


def test_coverage_fraction_counts_green_only() -> None:
    assert coverage_fraction(_fits(), n_frames=3, residual_threshold=0.05) == 1 / 3


def test_to_homography_entries_keeps_green_with_source_manual() -> None:
    entries = to_homography_entries(_fits(), residual_threshold=0.05)
    assert set(entries) == {0}  # only the green frame
    assert entries[0].source == "manual"
    assert 0.0 <= entries[0].confidence <= 1.0


def test_clicks_to_keypoints_df_schema() -> None:
    clicks = [Click(2, 0, 10.0, 20.0), Click(5, 3, 30.0, 40.0)]
    df = clicks_to_keypoints_df(clicks)
    assert list(df.columns) == ["frame", "kp_idx", "x_px", "y_px", "conf"]
    assert (df["conf"] == 1.0).all()
    assert df.iloc[0].to_dict() == {
        "frame": 2, "kp_idx": 0, "x_px": 10.0, "y_px": 20.0, "conf": 1.0
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -k "status or coverage or entries or keypoints_df" -v`
Expected: FAIL — `ImportError: frame_status`

- [ ] **Step 3: Implement the helpers**

Append to `manual_anchor.py` (add `import pandas as pd` and
`from soccer_vision.pitch.propagation import HomographyEntry` to the top imports):

```python
def frame_status(
    fits: Mapping[int, FrameFit], n_frames: int, *, residual_threshold: float = 0.05
) -> dict[int, str]:
    """Per-frame label: green (fit & residual<=thr), yellow (fit & residual>thr),
    red (no fit / <4 landmarks)."""
    status: dict[int, str] = {}
    for f in range(n_frames):
        fit = fits.get(f)
        if fit is None:
            status[f] = "red"
        elif fit.residual <= residual_threshold:
            status[f] = "green"
        else:
            status[f] = "yellow"
    return status


def coverage_fraction(
    fits: Mapping[int, FrameFit], n_frames: int, *, residual_threshold: float = 0.05
) -> float:
    """Fraction of frames that are green (covered & low residual)."""
    if n_frames == 0:
        return 0.0
    green = sum(1 for fit in fits.values() if fit.residual <= residual_threshold)
    return green / n_frames


def to_homography_entries(
    fits: Mapping[int, FrameFit], *, residual_threshold: float = 0.05
) -> dict[int, HomographyEntry]:
    """Green frames -> HomographyEntry(source='manual', confidence from residual)."""
    out: dict[int, HomographyEntry] = {}
    for f, fit in fits.items():
        if fit.residual > residual_threshold:
            continue
        conf = float(np.clip(1.0 - fit.residual / residual_threshold, 0.0, 1.0))
        out[f] = HomographyEntry(np.asarray(fit.H, dtype=np.float64), "manual", conf)
    return out


def clicks_to_keypoints_df(clicks: Sequence[Click]) -> pd.DataFrame:
    """Clicks -> keypoints DataFrame (frame, kp_idx, x_px, y_px, conf=1.0)."""
    df = pd.DataFrame(
        [{"frame": c.frame, "kp_idx": c.kp_idx, "x_px": c.x, "y_px": c.y, "conf": 1.0}
         for c in clicks],
        columns=["frame", "kp_idx", "x_px", "y_px", "conf"],
    )
    return df.astype({"frame": "int64", "kp_idx": "int64", "x_px": "float64",
                      "y_px": "float64", "conf": "float64"})
```

- [ ] **Step 4: Run test to verify it passes + typecheck the module**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -v && uv run mypy src/soccer_vision/pitch/manual_anchor.py && uv run ruff check src/soccer_vision/pitch/manual_anchor.py`
Expected: PASS (16 tests), no type/lint errors.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py packages/soccer-vision/tests/test_pitch_manual_anchor.py
git commit -m "feat(manual_anchor): status/coverage + homography & keypoints export helpers"
```

---

## Task 5: LabelerState (clicks → fits → export, no HTTP)

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/labeler/__init__.py`
- Create: `packages/soccer-vision/src/soccer_vision/labeler/state.py`
- Test: `packages/soccer-vision/tests/test_labeler_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_labeler_state.py`:

```python
"""Tests for LabelerState: click handling, coverage, and parquet export."""

from __future__ import annotations

import numpy as np
import pandas as pd
from soccer_vision.labeler.state import LabelerState
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

_SCALE = 1000.0
_IDXS = [0, 3, 6, 11, 16, 19]


def _state(n: int = 6) -> LabelerState:
    interframe = {i: np.eye(3) for i in range(n - 1)}
    return LabelerState(interframe=interframe, n_frames=n, window=10)


def test_add_click_updates_coverage() -> None:
    st = _state()
    assert st.coverage() == 0.0
    for f, idx in enumerate(_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        st.add_click(frame=f, kp_idx=idx, x=float(px), y=float(py))
    assert st.coverage() > 0.0
    assert st.frame_homography(3) is not None  # frame 3 now fittable


def test_remove_last_click() -> None:
    st = _state()
    st.add_click(0, 0, 1.0, 2.0)
    assert len(st.clicks) == 1
    st.remove_last()
    assert len(st.clicks) == 0


def test_status_list_length_matches_frames() -> None:
    st = _state(5)
    assert len(st.status_list()) == 5
    assert set(st.status_list()) <= {"green", "yellow", "red"}


def test_export_writes_both_parquets(tmp_path) -> None:
    st = _state()
    for f, idx in enumerate(_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        st.add_click(frame=f, kp_idx=idx, x=float(px), y=float(py))
    st.export(tmp_path)
    kp = pd.read_parquet(tmp_path / "keypoints.parquet")
    hom = pd.read_parquet(tmp_path / "homographies.parquet")
    assert list(kp.columns) == ["frame", "kp_idx", "x_px", "y_px", "conf"]
    assert len(kp) == len(_IDXS)
    assert "source" in hom.columns and (hom["source"] == "manual").all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -v`
Expected: FAIL — `ModuleNotFoundError: soccer_vision.labeler.state`

- [ ] **Step 3: Create the package + state**

Create `src/soccer_vision/labeler/__init__.py`:

```python
"""Interactive manual anchor labeler (local browser app)."""
```

Create `src/soccer_vision/labeler/state.py`:

```python
"""LabelerState: hold the registration chain + clicks, recompute per-frame
homographies on demand, and export the keypoints/homographies parquets.

Separated from the HTTP server so it is testable without a socket or a video.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from soccer_vision.pipeline import homographies_to_parquet
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import (
    Click,
    FrameFit,
    build_segments,
    clicks_to_keypoints_df,
    coverage_fraction,
    cumulative_transforms,
    fit_frame_homographies,
    frame_status,
    to_homography_entries,
)


class LabelerState:
    """Mutable session: clicks in, per-frame homographies + coverage out."""

    def __init__(
        self,
        interframe: Mapping[int, NDArray[np.floating]],
        n_frames: int,
        *,
        window: int = 60,
        residual_threshold: float = 0.05,
    ) -> None:
        self.n_frames = n_frames
        self.window = window
        self.residual_threshold = residual_threshold
        self._segment_of = build_segments(interframe, n_frames)
        self._transforms = cumulative_transforms(interframe, self._segment_of)
        self.clicks: list[Click] = []
        self._fits: dict[int, FrameFit] = {}

    def _recompute(self) -> None:
        self._fits = fit_frame_homographies(
            self.clicks, self._transforms, self._segment_of,
            PITCH_LANDMARKS, window=self.window,
        )

    def add_click(self, frame: int, kp_idx: int, x: float, y: float) -> None:
        self.clicks.append(Click(frame=frame, kp_idx=kp_idx, x=x, y=y))
        self._recompute()

    def remove_last(self) -> None:
        if self.clicks:
            self.clicks.pop()
            self._recompute()

    def coverage(self) -> float:
        return coverage_fraction(
            self._fits, self.n_frames, residual_threshold=self.residual_threshold
        )

    def status_list(self) -> list[str]:
        status = frame_status(
            self._fits, self.n_frames, residual_threshold=self.residual_threshold
        )
        return [status[f] for f in range(self.n_frames)]

    def frame_homography(self, frame: int) -> FrameFit | None:
        return self._fits.get(frame)

    def export(self, out_dir: Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        clicks_to_keypoints_df(self.clicks).to_parquet(
            out / "keypoints.parquet", index=False
        )
        entries = to_homography_entries(
            self._fits, residual_threshold=self.residual_threshold
        )
        homographies_to_parquet(entries, out / "homographies.parquet")
```

- [ ] **Step 4: Run test to verify it passes + typecheck**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -v && uv run mypy src/soccer_vision/labeler/state.py`
Expected: PASS (4 tests), no type errors.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/__init__.py packages/soccer-vision/src/soccer_vision/labeler/state.py packages/soccer-vision/tests/test_labeler_state.py
git commit -m "feat(labeler): LabelerState — clicks to per-frame homographies + parquet export"
```

---

## Task 6: Registration precompute + on-disk cache

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/labeler/chain.py`
- Test: `packages/soccer-vision/tests/test_labeler_chain.py`

The ORB registration itself is already covered by `pitch/propagation.py` tests;
this task adds the video-reading wrapper and a deterministic on-disk cache.

- [ ] **Step 1: Write the failing test**

Create `tests/test_labeler_chain.py`:

```python
"""Tests for the registration-chain cache round-trip + normalization."""

from __future__ import annotations

import numpy as np
from soccer_vision.labeler.chain import load_chain, normalize_homography, save_chain


def test_normalize_homography_translation() -> None:
    # a full-res px translation of +10 in x on a width-100 frame is +0.1 normalized.
    g = np.eye(3); g[0, 2] = 10.0
    gn = normalize_homography(g, (100, 50))
    out = gn @ np.array([0.2, 0.0, 1.0])
    out = out[:2] / out[2]
    assert np.allclose(out, [0.3, 0.0])


def test_chain_cache_round_trip(tmp_path) -> None:
    interframe = {0: np.eye(3), 2: np.diag([1.0, 2.0, 1.0])}
    path = tmp_path / "chain.npz"
    save_chain(path, interframe, n_frames=4, size=(1920, 1080))
    loaded_if, n_frames, size = load_chain(path)
    assert n_frames == 4
    assert size == (1920, 1080)
    assert set(loaded_if) == {0, 2}
    assert np.allclose(loaded_if[2], np.diag([1.0, 2.0, 1.0]))


def test_load_missing_returns_none(tmp_path) -> None:
    assert load_chain(tmp_path / "nope.npz") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_chain.py -v`
Expected: FAIL — `ModuleNotFoundError: soccer_vision.labeler.chain` (and `ImportError: normalize_homography`)

- [ ] **Step 3: Implement chain.py**

Create `src/soccer_vision/labeler/chain.py`:

```python
"""Compute the inter-frame registration chain over a video, with an on-disk cache.

Thin wrapper over pitch.propagation.compute_interframe_homographies (which the
propagation tests already cover) plus a deterministic .npz cache so reopening a
video is instant.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.pitch.propagation import compute_interframe_homographies


def normalize_homography(
    g: NDArray[np.floating], size: tuple[int, int]
) -> NDArray[np.float64]:
    """Rescale a full-res px->px homography to normalized [0,1] image coords.

    G_norm = S @ G @ inv(S), S = diag(1/W, 1/H, 1). This lets clicks (sent as
    normalized canvas fractions) compose with the inter-frame chain consistently.
    """
    w, h = size
    s = np.diag([1.0 / w, 1.0 / h, 1.0])
    s_inv = np.diag([float(w), float(h), 1.0])
    return (s @ np.asarray(g, dtype=np.float64) @ s_inv).astype(np.float64)


def save_chain(
    path: Path,
    interframe: dict[int, NDArray[np.floating]],
    n_frames: int,
    size: tuple[int, int],
) -> None:
    """Persist {i: 3x3} + n_frames + (w, h) to a single .npz."""
    flat = {f"H{i}": np.asarray(H, dtype=np.float64) for i, H in interframe.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        keys=np.array(sorted(interframe), dtype=np.int64),
        n_frames=np.array(n_frames),
        size=np.array(size, dtype=np.int64),
        **flat,
    )


def load_chain(
    path: Path,
) -> tuple[dict[int, NDArray[np.float64]], int, tuple[int, int]] | None:
    """Inverse of save_chain; None if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p)
    interframe = {int(k): np.asarray(data[f"H{int(k)}"], dtype=np.float64)
                  for k in data["keys"]}
    size = (int(data["size"][0]), int(data["size"][1]))
    return interframe, int(data["n_frames"]), size


def _video_hash(video_path: Path) -> str:
    """Stable hash of a video from its path, size, and mtime (cheap, no full read)."""
    st = Path(video_path).stat()
    key = f"{Path(video_path).resolve()}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def compute_chain(
    video_path: Path,
    *,
    cache_dir: Path | None = None,
    downscale: float = 1.0,
    player_boxes: pd.DataFrame | None = None,
) -> tuple[dict[int, NDArray[np.float64]], int, tuple[int, int]]:
    """Inter-frame chain for the whole video (cached). Returns (interframe, n, (w,h))."""
    cache_dir = Path(cache_dir or (Path(video_path).parent / ".sv_labeler_cache"))
    cache_path = cache_dir / f"{_video_hash(video_path)}.npz"
    cached = load_chain(cache_path)
    if cached is not None:
        return cached

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    pos = 0

    def read_frame(idx: int) -> NDArray[np.uint8] | None:
        nonlocal pos
        if idx < pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            pos = idx
        while pos < idx:
            if not cap.grab():
                return None
            pos += 1
        ok, frame = cap.read()
        pos += 1
        return frame if ok else None

    boxes = player_boxes if player_boxes is not None else pd.DataFrame(
        columns=["frame", "class", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    )
    needed = set(range(n_frames - 1))
    try:
        interframe_px = compute_interframe_homographies(
            read_frame, needed, boxes, downscale=downscale
        )
    finally:
        cap.release()

    # Normalize to [0,1] image coords so clicks (normalized canvas fractions)
    # compose with the chain consistently.
    interframe = {
        i: normalize_homography(g, (width, height)) for i, g in interframe_px.items()
    }
    save_chain(cache_path, interframe, n_frames, (width, height))
    return interframe, n_frames, (width, height)
```

- [ ] **Step 4: Run test to verify it passes + typecheck/lint**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_chain.py -v && uv run mypy src/soccer_vision/labeler/chain.py && uv run ruff check src/soccer_vision/labeler/chain.py`
Expected: PASS (2 tests), no errors.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/chain.py packages/soccer-vision/tests/test_labeler_chain.py
git commit -m "feat(labeler): registration-chain precompute + .npz cache"
```

---

## Task 7: HTTP server

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/labeler/server.py`
- Test: `packages/soccer-vision/tests/test_labeler_server.py`

The server's routing/state logic is tested by driving a `LabelerHandler` against
an in-memory state with a stub frame source (no real video, no live socket).

- [ ] **Step 1: Write the failing test**

Create `tests/test_labeler_server.py`:

```python
"""Tests for the labeler HTTP app via a threaded server on a loopback port."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import HTTPServer

import numpy as np
from soccer_vision.labeler.server import make_handler
from soccer_vision.labeler.state import LabelerState
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS


def _serve():
    interframe = {i: np.eye(3) for i in range(5)}
    state = LabelerState(interframe=interframe, n_frames=6, window=10)

    def frame_jpeg(idx: int) -> bytes:
        return b"\xff\xd8stub-jpeg"  # not a real image; endpoint just returns bytes

    handler = make_handler(state, frame_jpeg, landmark_names=["pitch"] * 21)
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, state


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def test_click_then_state_reports_coverage() -> None:
    httpd, _ = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        from soccer_vision.pitch.landmarks import PITCH_LANDMARKS as L
        for f, idx in enumerate([0, 3, 6, 11, 16, 19]):
            px, py = L[idx] * 1000.0
            _post(f"{base}/api/click",
                  {"frame": f, "kp_idx": int(idx), "x": float(px), "y": float(py)})
        state = _get(f"{base}/api/state")
        assert state["coverage"] > 0.0
        assert len(state["status"]) == 6
    finally:
        httpd.shutdown()


def test_frame_endpoint_returns_bytes() -> None:
    httpd, _ = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/frame/2") as r:
            assert r.read().startswith(b"\xff\xd8")
    finally:
        httpd.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_server.py -v`
Expected: FAIL — `ModuleNotFoundError: soccer_vision.labeler.server`

- [ ] **Step 3: Implement server.py**

Create `src/soccer_vision/labeler/server.py`:

```python
"""Stdlib HTTP app for the manual anchor labeler.

make_handler() builds a BaseHTTPRequestHandler bound to a LabelerState and a
frame-bytes provider, so it is testable without a real video. run() wires a real
video (chain precompute + JPEG frames) and serves the static UI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from soccer_vision.labeler.chain import compute_chain
from soccer_vision.labeler.state import LabelerState

_STATIC = Path(__file__).parent / "static"


def make_handler(
    state: LabelerState,
    frame_jpeg: Callable[[int], bytes],
    landmark_names: list[str],
    *,
    landmark_xy: list[list[float]] | None = None,
    export_dir: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class closed over the session state.

    landmark_xy is the canonical [x, y] of each keypoint in pitch [0,1]^2, sent to
    the frontend so it can draw the reprojected pitch overlay.
    """
    xy = landmark_xy or []

    class LabelerHandler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: dict[str, Any], code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _state_payload(self) -> dict[str, Any]:
            return {
                "n_frames": state.n_frames,
                "coverage": state.coverage(),
                "status": state.status_list(),
                "n_clicks": len(state.clicks),
                "landmark_names": landmark_names,
                "landmark_xy": xy,
            }

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, (_STATIC / "index.html").read_bytes(), "text/html")
            elif self.path == "/app.js":
                self._send(200, (_STATIC / "app.js").read_bytes(),
                           "application/javascript")
            elif self.path == "/api/state":
                self._json(self._state_payload())
            elif self.path.startswith("/api/frame_h/"):
                idx = int(self.path.rsplit("/", 1)[1])
                fit = state.frame_homography(idx)
                self._json({"h": None if fit is None
                            else [float(v) for v in np.asarray(fit.H).reshape(9)]})
            elif self.path.startswith("/api/frame/"):
                idx = int(self.path.rsplit("/", 1)[1])
                self._send(200, frame_jpeg(idx), "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/click":
                state.add_click(int(payload["frame"]), int(payload["kp_idx"]),
                                float(payload["x"]), float(payload["y"]))
                self._json(self._state_payload())
            elif self.path == "/api/undo":
                state.remove_last()
                self._json(self._state_payload())
            elif self.path == "/api/export":
                state.export(export_dir or Path.cwd())
                self._json({"exported_to": str(export_dir or Path.cwd())})
            else:
                self._send(404, b"not found", "text/plain")

    return LabelerHandler


def run(
    video_path: Path,
    *,
    port: int = 8000,
    downscale_display: float = 0.5,
    export_dir: Path | None = None,
) -> None:  # pragma: no cover - launches a blocking server
    """Precompute the chain, open the video, and serve the labeler UI."""
    interframe, n_frames, _ = compute_chain(video_path)
    state = LabelerState(interframe=interframe, n_frames=n_frames)
    cap = cv2.VideoCapture(str(video_path))

    def frame_jpeg(idx: int) -> bytes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return b""
        small = cv2.resize(frame, None, fx=downscale_display, fy=downscale_display)
        ok2, buf = cv2.imencode(".jpg", small)
        return buf.tobytes() if ok2 else b""

    from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
    names = [f"kp{i}" for i in range(len(PITCH_LANDMARKS))]
    xy = [[float(x), float(y)] for x, y in PITCH_LANDMARKS]
    handler = make_handler(state, frame_jpeg, names, landmark_xy=xy, export_dir=export_dir)
    httpd = HTTPServer(("127.0.0.1", port), handler)
    print(f"Labeler running at http://127.0.0.1:{port}  (video: {video_path})")
    httpd.serve_forever()
```

- [ ] **Step 4: Run test to verify it passes + typecheck/lint**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_server.py -v && uv run mypy src/soccer_vision/labeler/server.py && uv run ruff check src/soccer_vision/labeler/server.py`
Expected: PASS (2 tests), no errors.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/server.py packages/soccer-vision/tests/test_labeler_server.py
git commit -m "feat(labeler): stdlib HTTP server (click/state/frame/undo/export endpoints)"
```

---

## Task 8: Browser frontend (canvas UI)

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/labeler/static/index.html`
- Create: `packages/soccer-vision/src/soccer_vision/labeler/static/app.js`

No unit tests (canvas UI; verified manually in Task 9). Provide the full files.

- [ ] **Step 1: Create index.html**

Create `src/soccer_vision/labeler/static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Anchor Labeler</title>
<style>
  body { background:#0f1115; color:#e6e6e6; font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; padding:12px 16px; }
  #app { display:flex; gap:14px; }
  #stage { flex:1; }
  #frame { position:relative; border:1px solid #2a2f3a; border-radius:6px; }
  #canvas { display:block; width:100%; background:#16331f; border-radius:6px; cursor:crosshair; }
  #palette { width:210px; background:#1a1e26; border:1px solid #2a2f3a; border-radius:6px; padding:8px; max-height:560px; overflow:auto; }
  .kp { padding:3px 6px; border-radius:4px; font-size:12px; cursor:pointer; }
  .kp.armed { background:#2a3550; outline:1px solid #4aa8ff; }
  .kp.placed { color:#7ee0a6; }
  #timeline { height:24px; display:flex; border:1px solid #2a2f3a; border-radius:4px; overflow:hidden; margin-top:10px; }
  #timeline div { height:100%; }
  .stats { font-size:13px; margin-top:8px; display:flex; gap:18px; }
  .btn { background:#222a36; border:1px solid #313a48; color:#dfe7ee; padding:5px 10px; border-radius:5px; font-size:12px; cursor:pointer; }
  .btn.primary { background:#1f6f43; border-color:#2a8a55; }
  #controls { margin-top:8px; display:flex; gap:6px; align-items:center; }
  input[type=range] { flex:1; }
</style>
</head>
<body>
  <div id="app">
    <div id="stage">
      <div id="frame"><canvas id="canvas" width="960" height="540"></canvas></div>
      <div id="controls">
        <button class="btn" id="prevRed">◀ red</button>
        <input type="range" id="scrub" min="0" value="0">
        <button class="btn" id="nextRed">red ▶</button>
        <span id="frameNum">0</span>
      </div>
      <div id="timeline"></div>
      <div class="stats">
        <span>coverage <b id="cov">0%</b></span>
        <span>residual <b id="res">—</b></span>
        <span>clicks <b id="nclicks">0</b></span>
      </div>
      <div style="margin-top:8px;">
        <button class="btn" id="undo">undo</button>
        <button class="btn" id="grid">toggle grid</button>
        <button class="btn primary" id="export">Export</button>
      </div>
    </div>
    <div id="palette"></div>
  </div>
  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create app.js**

Create `src/soccer_vision/labeler/static/app.js`:

All coordinates are **normalized [0,1] image space** (the registration chain is
normalized in `compute_chain`, so clicks, transforms, and the homography all live
in the same unit square — no frame-size bookkeeping in the UI).

```javascript
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const NAMES = []; let LXY = []; let N = 0; let armed = 0; let cur = 0; let showGrid = true;
let status = []; let placed = new Set(); let clicks = []; let curH = null;
const img = new Image();

// canonical pitch edges (landmark index pairs) for the reprojected overlay
const EDGES = [[0,1],[1,3],[3,2],[2,0],[4,6],[9,10],[11,12],[9,11],[10,12],
               [13,14],[15,16],[13,15],[14,16],[17,18],[19,20]];

async function api(path, opts){ const r = await fetch(path, opts); return r.json(); }
function postJSON(path, body){
  return api(path, {method:"POST", headers:{"Content-Type":"application/json"},
                    body: JSON.stringify(body)}); }
function colorFor(s){return s==="green"?"#39d98a":s==="yellow"?"#ffb454":"#e0524d";}

function inv3(m){ // invert a flat 9-array 3x3
  const a=m[0],b=m[1],c=m[2],d=m[3],e=m[4],f=m[5],g=m[6],h=m[7],i=m[8];
  const A=e*i-f*h, B=-(d*i-f*g), C=d*h-e*g;
  const det=a*A+b*B+c*C; if(Math.abs(det)<1e-12) return null;
  const id=1/det;
  return [A*id,(c*h-b*i)*id,(b*f-c*e)*id, B*id,(a*i-c*g)*id,(c*d-a*f)*id,
          C*id,(b*g-a*h)*id,(a*e-b*d)*id];
}
function applyH(m,x,y){ const w=m[6]*x+m[7]*y+m[8];
  return [(m[0]*x+m[1]*y+m[2])/w, (m[3]*x+m[4]*y+m[5])/w]; }

function renderPalette(){
  const p=document.getElementById("palette");
  p.innerHTML="<h3 style='font-size:12px;color:#9aa4b2'>LANDMARK</h3>";
  for(let i=0;i<N;i++){ if(i===5) continue;
    const d=document.createElement("div");
    d.className="kp"+(i===armed?" armed":"")+(placed.has(i)?" placed":"");
    d.textContent=`${i} ${NAMES[i]||""}`+(placed.has(i)?" ✓":"");
    d.onclick=()=>{armed=i; renderPalette();}; p.appendChild(d); }
}
function renderTimeline(){
  const t=document.getElementById("timeline"); t.innerHTML="";
  for(const s of status){const d=document.createElement("div");
    d.style.flex="1"; d.style.background=colorFor(s); t.appendChild(d);}
}

function drawOverlay(){
  if(!showGrid || !curH || !LXY.length) return;
  const hi=inv3(curH); if(!hi) return;            // pitch -> normalized image
  ctx.strokeStyle="#39d98a"; ctx.lineWidth=1.5; ctx.globalAlpha=0.85;
  for(const [a,b] of EDGES){
    const pa=applyH(hi, LXY[a][0], LXY[a][1]), pb=applyH(hi, LXY[b][0], LXY[b][1]);
    ctx.beginPath(); ctx.moveTo(pa[0]*canvas.width, pa[1]*canvas.height);
    ctx.lineTo(pb[0]*canvas.width, pb[1]*canvas.height); ctx.stroke();
  }
  ctx.globalAlpha=1.0;
}

function drawFrame(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(img.complete && img.naturalWidth) ctx.drawImage(img,0,0,canvas.width,canvas.height);
  drawOverlay();
  for(const c of clicks) if(c.frame===cur){
    ctx.fillStyle="#39d98a";
    ctx.beginPath(); ctx.arc(c.x*canvas.width, c.y*canvas.height,6,0,7); ctx.fill();
    ctx.fillStyle="#0f1115"; ctx.font="10px sans-serif";
    ctx.fillText(c.kp_idx, c.x*canvas.width-3, c.y*canvas.height+3);
  }
}

async function loadFrame(i){
  cur=i; document.getElementById("frameNum").textContent=i;
  const fh=await api(`/api/frame_h/${i}`); curH=fh.h;
  document.getElementById("res").textContent = status[i] || "—";
  img.onload=drawFrame; img.src=`/api/frame/${i}?t=${Date.now()}`;
}

function applyState(st){
  N=st.landmark_names.length; for(let i=0;i<N;i++) NAMES[i]=st.landmark_names[i];
  LXY=st.landmark_xy; status=st.status;
  document.getElementById("cov").textContent=Math.round(st.coverage*100)+"%";
  document.getElementById("nclicks").textContent=st.n_clicks;
  document.getElementById("scrub").max=st.n_frames-1;
  renderPalette(); renderTimeline(); drawFrame();
}

canvas.onclick=async(e)=>{
  const r=canvas.getBoundingClientRect();
  const x=(e.clientX-r.left)/r.width, y=(e.clientY-r.top)/r.height;
  clicks.push({frame:cur, kp_idx:armed, x, y}); placed.add(armed);
  applyState(await postJSON("/api/click",{frame:cur,kp_idx:armed,x,y}));
  const fh=await api(`/api/frame_h/${cur}`); curH=fh.h; drawFrame();
};

document.getElementById("scrub").oninput=(e)=>loadFrame(+e.target.value);
document.getElementById("undo").onclick=async()=>{clicks.pop();
  applyState(await postJSON("/api/undo",{})); const fh=await api(`/api/frame_h/${cur}`);
  curH=fh.h; drawFrame();};
document.getElementById("grid").onclick=()=>{showGrid=!showGrid; drawFrame();};
document.getElementById("export").onclick=async()=>{
  const r=await postJSON("/api/export",{}); alert("Exported to "+r.exported_to);};
function jumpRed(dir){let i=cur+dir;
  while(i>=0&&i<status.length){if(status[i]==="red"){loadFrame(i);return;} i+=dir;}}
document.getElementById("nextRed").onclick=()=>jumpRed(1);
document.getElementById("prevRed").onclick=()=>jumpRed(-1);
window.onkeydown=(e)=>{ if(e.key>="0"&&e.key<="9"){armed=+e.key; renderPalette();} };

(async()=>{applyState(await api("/api/state")); loadFrame(0);})();
```

- [ ] **Step 3: Validate the files exist and are non-empty**

Run: `cd packages/soccer-vision && test -s src/soccer_vision/labeler/static/index.html && test -s src/soccer_vision/labeler/static/app.js && echo OK`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/static/
git commit -m "feat(labeler): canvas frontend (scrub/arm/click/timeline/export)"
```

---

## Task 9: Launcher + end-to-end smoke

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/labeler/__main__.py`

- [ ] **Step 1: Create the launcher**

Create `src/soccer_vision/labeler/__main__.py`:

```python
"""CLI entry: python -m soccer_vision.labeler --video game.mp4 [--port 8000]."""

from __future__ import annotations

import argparse
from pathlib import Path

from soccer_vision.labeler.server import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive pitch anchor labeler")
    ap.add_argument("--video", required=True, type=Path, help="path to an H.264 mp4")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--export-dir", type=Path, default=None,
                    help="where Export writes parquets (default: cwd)")
    args = ap.parse_args()
    run(args.video, port=args.port, export_dir=args.export_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI parses (no server launch)**

Run: `cd packages/soccer-vision && uv run python -c "import soccer_vision.labeler.__main__ as m; import argparse; print('import ok')"`
Expected: `import ok`

- [ ] **Step 3: Full suite + typecheck + lint**

Run: `cd packages/soccer-vision && uv run pytest && uv run mypy src/ && uv run ruff check src/ tests/`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/__main__.py
git commit -m "feat(labeler): CLI launcher (python -m soccer_vision.labeler)"
```

- [ ] **Step 5: Manual end-to-end smoke (documented, run by Patrick)**

Run locally: `cd packages/soccer-vision && uv run python -m soccer_vision.labeler --video /path/to/training.mp4 --export-dir /tmp/anchor_out`
Then open `http://127.0.0.1:8000`, place a handful of anchors across the clip,
confirm: coverage % climbs as you click, the timeline greens up, clicked dots
render on the frame, and Export writes `keypoints.parquet` + `homographies.parquet`.
Finally confirm the export feeds the pipeline:
`uv run python -c "from soccer_vision.pipeline import assemble_from_homographies; from pathlib import Path; print('ok')"`
(Full assemble needs a `trajectories_px.parquet` from detection; this step just
confirms the export parquets load and the import path is wired.)

---

## Final verification

- [ ] **Run the full gate**

Run: `cd packages/soccer-vision && uv run pytest && uv run mypy src/ && uv run ruff check src/ tests/`
Expected: all green (existing suite + new manual_anchor/state/chain/server tests).

---

## Notes for the implementer

- **Coordinate space is normalized [0,1] everywhere.** `compute_chain` rescales
  the inter-frame chain to the unit square (`normalize_homography`), so clicks
  (normalized canvas fractions), cumulative transforms, and the fitted homography
  all agree. Residual is still in pitch units (`PITCH_LANDMARKS` is [0,1]²).
- **Homography direction is image→pitch everywhere.** `fit_homography(image_pts,
  pitch_pts)`; the overlay uses `inv(H)` (pitch→image) via `/api/frame_h/<idx>`,
  drawn behind the "toggle grid" button — this is the primary correctness check.
- **Window default (60)** mirrors the propagation probe's tested reach; it is a
  `LabelerState` constructor arg, tunable once we see real registration reach.

**Explicitly deferred (out of this plan's scope; spec calls them optional):**
- **Session autosave/resume** — clicks are held in memory and written only on
  Export. Crash-resume (autosave clicks to a sidecar JSON) is an additive
  follow-up, not implemented here.
- **`--player-boxes` masking** — `compute_chain` accepts a `player_boxes`
  DataFrame, but the CLI/`run()` don't wire a flag yet; masking stays off by
  default (RANSAC handles moving players). Add the flag when needed.
- **Point nudge** — the UI supports place + undo; dragging a placed point to
  fine-tune is a later enhancement.
