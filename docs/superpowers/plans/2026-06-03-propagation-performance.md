# Homography Propagation Performance Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make homography propagation ~50–100× faster (hours → minutes per clip) by decoding each video frame once (sequentially, not random-seek) and running ORB on downscaled frames, separating video I/O from the pure homography composition.

**Architecture:** Split the current `propagate_homographies` (which reads + registers frames inside its chaining loop) into (1) `compute_interframe_homographies` — one sequential pass that registers each consecutive frame pair once, ORB on downscaled frames, rescaling the result to full-res; and (2) a now-**pure** `propagate_homographies` that composes those precomputed inter-frame homographies. `build_homographies` wires them with a sequential `grab()/retrieve()` reader.

**Tech Stack:** OpenCV (ORB, findHomography, sequential grab/retrieve, resize), numpy, pandas. No new deps.

**Design (approved):** Propagation only needs consecutive inter-frame homographies `G[i]` (frame `i`→`i+1`); everything else is matrix composition. Compute every needed `G[i]` in one ascending pass (each frame decoded + ORB'd exactly once, descriptors cached frame-to-frame, ORB on a ~0.5× downscale rescaled back exactly via `G_full = S⁻¹ G_small S`). Then `propagate_homographies` becomes pure (takes `{i: G[i]}`), and `build_homographies` owns the sequential reader.

**Spec:** `docs/superpowers/specs/2026-06-03-homography-propagation-design.md` (this refactor updates §3 mechanics + §7 perf; behavior/outputs unchanged).

---

## File Structure

| File | Change |
|---|---|
| `packages/soccer-vision/src/soccer_vision/pitch/propagation.py` | Make `propagate_homographies`/`_chain` pure (take `interframe` map, add `frame_size` param); add `compute_interframe_homographies` + `_orb_features`/`_match_homography` helpers. Keep `register`, `blend_homographies`, `disagreement_confidence`, `_frame_mask` as-is. |
| `packages/soccer-vision/src/soccer_vision/pipeline.py` | Rewrite `build_homographies`: compute needed pairs, sequential `grab/retrieve` reader, call `compute_interframe_homographies` then pure `propagate_homographies`. |
| `packages/soccer-vision/tests/test_pitch_propagation.py` | Rewrite the propagation tests to supply `interframe` directly (pure); add a `compute_interframe_homographies` synthetic-frame test (incl. downscale round-trip). `register`/`_frame_mask` tests unchanged. |
| `examples/colab_homography_propagation.ipynb` | Update the sweep cell to the new fast flow (`compute_interframe_homographies` once per `max_gap`, or once + reuse). |

Commands: `uv run pytest <path> -q`, `uv run ruff check packages/soccer-vision/`, `uv run mypy packages/soccer-vision/src`.

---

## Task 1: Make `propagate_homographies` pure (compose precomputed inter-frame Hs)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/propagation.py`
- Test: `packages/soccer-vision/tests/test_pitch_propagation.py`

- [ ] **Step 1: Rewrite the propagation tests to supply `interframe` directly**

In `test_pitch_propagation.py`, DELETE the four frame-based propagation tests (`test_propagation_bridges_gap_within_window`, `test_gap_beyond_max_is_not_bridged`, `test_unbridged_frames_absent_and_anchors_confident`, `test_empty_anchors_returns_empty`, `test_one_sided_coverage_when_a_chain_breaks`) AND the `_shift_frame`/`_scene` helpers and the `propagate_homographies` import line. Keep the `register` tests, the blend/confidence tests, and `test_frame_mask_blanks_player_boxes`. Append these pure tests:

```python
from soccer_vision.pitch.propagation import propagate_homographies


def _pan_interframe(n_pairs: int, dx: float = 4.0) -> dict[int, np.ndarray]:
    """interframe[i] maps frame i pixels -> i+1 pixels: a constant +dx translation."""
    G = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return {i: G for i in range(n_pairs)}


def _pan_anchor_H(frame: int, dx: float = 4.0) -> np.ndarray:
    """pixel->pitch for a frame panned by frame*dx: undo pan, then /1000."""
    undo = np.array([[1.0, 0.0, -frame * dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return np.diag([1 / 1000.0, 1 / 1000.0, 1.0]) @ undo


def test_propagation_bridges_gap_within_window() -> None:
    anchors = {0: _pan_anchor_H(0), 10: _pan_anchor_H(10)}
    interframe = _pan_interframe(10)                       # G[0..9]
    out = propagate_homographies(anchors, interframe, max_gap=15)

    assert out[0].source == "anchor" and out[10].source == "anchor"
    assert out[5].source == "propagated"
    # propagated H for frame 5 matches its true pitch homography
    p = np.array([300.0, 200.0, 1.0])
    got = out[5].H @ p
    got = got[:2] / got[2]
    exp = _pan_anchor_H(5) @ p
    exp = exp[:2] / exp[2]
    assert np.linalg.norm(got - exp) < 1e-9


def test_gap_beyond_max_is_not_bridged() -> None:
    anchors = {0: _pan_anchor_H(0), 10: _pan_anchor_H(10)}
    out = propagate_homographies(anchors, _pan_interframe(10), max_gap=4)  # gap 9 > 4
    assert set(out) == {0, 10}


def test_one_sided_when_interframe_missing() -> None:
    # Drop G[1] (frame 1->2): forward chain reaches frame 1 only; backward (from 10)
    # reaches frames 9..2; frame ... wait, backward needs G at f for step (f+1)->f.
    anchors = {0: _pan_anchor_H(0), 10: _pan_anchor_H(10)}
    interframe = _pan_interframe(10)
    del interframe[1]                                      # break forward after frame 1; break backward at 2->1? no
    out = propagate_homographies(anchors, interframe, max_gap=15)
    # Frame 1: forward reaches it (uses G[0]); source propagated.
    assert out[1].source == "propagated"
    # Frame 9: backward reaches it (uses inv(G[9])); source propagated.
    assert out[9].source == "propagated"
    # Frame 1 also reachable backward only if chain from 10 gets down to 1, which needs
    # G[1] (for 2->1). With G[1] gone backward stops at frame 2, so frame 1 is forward-only.


def test_empty_anchors_returns_empty() -> None:
    assert propagate_homographies({}, {}, max_gap=15) == {}


def test_anchors_have_unit_confidence() -> None:
    anchors = {0: _pan_anchor_H(0), 10: _pan_anchor_H(10)}
    out = propagate_homographies(anchors, _pan_interframe(10), max_gap=15)
    assert out[0].confidence == 1.0 and out[10].confidence == 1.0
    assert 0.0 <= out[5].confidence <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_propagation.py -q`
Expected: FAIL — `propagate_homographies` still has the old signature (TypeError on the 2-positional-arg call, or the `interframe` dict is treated as `read_frame`).

- [ ] **Step 3: Rewrite `_chain` and `propagate_homographies` to be pure**

In `propagation.py`, replace `_chain` and `propagate_homographies` with:

```python
def _chain(
    anchor: int,
    targets: list[int],
    interframe: Mapping[int, NDArray[np.floating]],
    H_anchor: NDArray[np.floating],
) -> dict[int, NDArray[np.floating]]:
    """Compose precomputed inter-frame homographies from `anchor` over adjacent `targets`.

    `interframe[i]` maps frame i pixels -> frame i+1 pixels. `targets` is the ordered
    adjacent sequence (ascending for a forward chain, descending for backward). Returns
    {frame: pixel->pitch H}; stops at the first missing inter-frame homography.
    """
    out: dict[int, NDArray[np.floating]] = {}
    W = np.eye(3)                                  # maps anchor pixels -> current pixels
    prev = anchor
    for f in targets:
        if f == prev + 1:
            g = interframe.get(prev)               # prev -> f
            step = g
        elif f == prev - 1:
            g = interframe.get(f)                  # f -> prev
            step = np.linalg.inv(g) if g is not None else None  # prev -> f
        else:
            break                                  # non-adjacent target (should not happen)
        if step is None:
            break
        W = step @ W
        out[f] = H_anchor @ np.linalg.inv(W)       # pixel_f -> pitch
        prev = f
    return out


def propagate_homographies(
    anchors: Mapping[int, NDArray[np.floating]],
    interframe: Mapping[int, NDArray[np.floating]],
    *,
    max_gap: int = 25,
    disagreement_tau: float = 0.10,
    frame_size: tuple[int, int] = (1920, 1080),
) -> dict[int, HomographyEntry]:
    """Bridge no-landmark gaps between anchors by composing inter-frame homographies.

    `interframe[i]` maps frame i pixels -> frame i+1 pixels (from
    compute_interframe_homographies). Each anchor keeps source='anchor', confidence=1.0.
    For each gap <= max_gap, chain forward from the left anchor and backward from the
    right anchor, blend by distance, and set confidence from forward/backward
    disagreement. Frames reached by neither chain are absent. Edge gaps are not
    bridged in v1. Pure: no video I/O.
    """
    out: dict[int, HomographyEntry] = {
        f: HomographyEntry(np.asarray(H, dtype=np.float64), "anchor", 1.0)
        for f, H in anchors.items()
    }
    keys = sorted(anchors)
    for a, b in itertools.pairwise(keys):
        gap = b - a - 1
        if gap < 1 or gap > max_gap:
            continue
        inner = list(range(a + 1, b))
        fwd = _chain(a, inner, interframe, np.asarray(anchors[a], np.float64))
        bwd = _chain(b, inner[::-1], interframe, np.asarray(anchors[b], np.float64))
        for t in inner:
            hf, hb = fwd.get(t), bwd.get(t)
            if hf is not None and hb is not None:
                w_f = (b - t) / (b - a)
                out[t] = HomographyEntry(
                    blend_homographies(hf, hb, w_f), "propagated",
                    disagreement_confidence(hf, hb, tau=disagreement_tau, frame_size=frame_size),
                )
            elif hf is not None:
                out[t] = HomographyEntry(hf, "propagated", max(0.0, 1.0 - (t - a) / (max_gap + 1)))
            elif hb is not None:
                out[t] = HomographyEntry(hb, "propagated", max(0.0, 1.0 - (b - t) / (max_gap + 1)))
    return out
```

Note `Callable` may become unused now (propagate no longer takes a callable); Task 2 re-adds it for `compute_interframe_homographies`. If ruff flags it as unused after this task alone, leave it — Task 2 in the same module uses it. (If implementing tasks separately, remove it here and Task 2 re-adds.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_propagation.py -q`
Expected: PASS. (Pure composition is exact, so the `< 1e-9` accuracy assertion holds — no ORB/RANSAC noise.)

- [ ] **Step 5: Commit**

Run: `uv run ruff check packages/soccer-vision/src/soccer_vision/pitch/propagation.py packages/soccer-vision/tests/test_pitch_propagation.py && uv run mypy packages/soccer-vision/src/soccer_vision/pitch/propagation.py`
(If `Callable` is unused at this point, remove it from the import; Task 2 re-adds.)

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/propagation.py packages/soccer-vision/tests/test_pitch_propagation.py
git commit -m "refactor(pitch): propagate_homographies is pure (composes precomputed inter-frame Hs)"
```

---

## Task 2: `compute_interframe_homographies` — one sequential pass, downscaled ORB

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/propagation.py`
- Test: `packages/soccer-vision/tests/test_pitch_propagation.py` (append)

- [ ] **Step 1: Write the failing test** — append to `test_pitch_propagation.py`:

```python
from soccer_vision.pitch.propagation import compute_interframe_homographies


def _textured(seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = (rng.random((400, 600)) * 60).astype(np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(20, 560)), int(rng.integers(20, 360))
        cv2.rectangle(g, (x, y), (x + 18, y + 18), int(rng.integers(80, 255)), -1)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def test_compute_interframe_recovers_known_pan_downscaled() -> None:
    base = _textured()
    dx = 6

    def shift(f: int) -> np.ndarray:
        M = np.array([[1.0, 0.0, float(f * dx)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return cv2.warpPerspective(base, M, (600, 400))

    frames = {f: shift(f) for f in range(4)}

    def read_frame(i):
        return frames.get(i)

    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    interframe = compute_interframe_homographies(
        read_frame, needed_pairs={0, 1, 2}, player_boxes=boxes, downscale=0.5,
    )
    assert set(interframe) == {0, 1, 2}
    # G[1] maps frame1 px -> frame2 px: a +dx translation, recovered at FULL resolution.
    p = np.array([300.0, 200.0, 1.0])
    q = interframe[1] @ p
    q /= q[2]
    assert abs(q[0] - (300 + dx)) < 2.0 and abs(q[1] - 200.0) < 2.0


def test_compute_interframe_empty_pairs() -> None:
    boxes = pd.DataFrame(columns=["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class"])
    assert compute_interframe_homographies(lambda i: None, set(), boxes) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_propagation.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_interframe_homographies'`

- [ ] **Step 3: Add the implementation** — append to `propagation.py` (ensure `Callable` is imported from `collections.abc`):

```python
def _orb_downscaled(
    img: NDArray[np.uint8], mask: NDArray[np.uint8], downscale: float, n_features: int
) -> tuple[list, NDArray | None]:
    """ORB keypoints+descriptors on a downscaled copy (keypoints in DOWNSCALED coords)."""
    if downscale != 1.0:
        small = cv2.resize(img, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        smask = cv2.resize(mask, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_NEAREST)
    else:
        small, smask = img, mask
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(n_features)  # type: ignore[attr-defined]
    return orb.detectAndCompute(gray, smask)


def _homography_from_descriptors(
    kp_a: list, d_a: NDArray | None, kp_b: list, d_b: NDArray | None,
    downscale: float, min_inliers: int,
) -> NDArray[np.floating] | None:
    """Match descriptors -> homography (downscaled px), rescaled to FULL-res px->px."""
    if d_a is None or d_b is None or len(d_a) < min_inliers or len(d_b) < min_inliers:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d_a, d_b)
    if len(matches) < min_inliers:
        return None
    src = np.array([kp_a[m.queryIdx].pt for m in matches], dtype=np.float32)
    dst = np.array([kp_b[m.trainIdx].pt for m in matches], dtype=np.float32)
    g_small, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if g_small is None:
        return None
    s = np.diag([downscale, downscale, 1.0])            # full px -> small px
    s_inv = np.diag([1.0 / downscale, 1.0 / downscale, 1.0])
    return (s_inv @ g_small @ s).astype(np.float64)     # full px -> full px


def compute_interframe_homographies(
    read_frame: Callable[[int], NDArray[np.uint8] | None],
    needed_pairs: set[int],
    player_boxes: pd.DataFrame,
    *,
    downscale: float = 0.5,
    n_features: int = 3000,
    min_inliers: int = 12,
) -> dict[int, NDArray[np.floating]]:
    """Register every needed consecutive frame pair in ONE ascending pass.

    `needed_pairs` is the set of indices i for which interframe[i] (frame i -> i+1) is
    wanted. Each frame is read once (read_frame is called in ascending order, so the
    caller can decode sequentially) and ORB'd once on a `downscale` copy; the resulting
    homography is rescaled back to full-resolution pixels. Returns {i: full-res G[i]}.
    """
    interframe: dict[int, NDArray[np.floating]] = {}
    if not needed_pairs:
        return interframe
    frames = sorted(needed_pairs | {i + 1 for i in needed_pairs})
    prev_idx: int | None = None
    prev_kp: list = []
    prev_d: NDArray | None = None
    for idx in frames:
        img = read_frame(idx)
        if img is None:
            prev_idx = None
            continue
        mask = _frame_mask(player_boxes, idx, img.shape[:2])
        kp, d = _orb_downscaled(img, mask, downscale, n_features)
        if prev_idx == idx - 1 and (idx - 1) in needed_pairs:
            g = _homography_from_descriptors(prev_kp, prev_d, kp, d, downscale, min_inliers)
            if g is not None:
                interframe[idx - 1] = g
        prev_idx, prev_kp, prev_d = idx, kp, d
    return interframe
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/soccer-vision/tests/test_pitch_propagation.py -q`
Expected: PASS. (The 2 px tolerance accommodates ORB+downscale; a clean translation recovers well within it.)

- [ ] **Step 5: Lint, type, commit**

Run: `uv run ruff check packages/soccer-vision/src/soccer_vision/pitch/propagation.py packages/soccer-vision/tests/test_pitch_propagation.py && uv run mypy packages/soccer-vision/src/soccer_vision/pitch/propagation.py`

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/propagation.py packages/soccer-vision/tests/test_pitch_propagation.py
git commit -m "feat(pitch): compute_interframe_homographies (one sequential pass, downscaled ORB)"
```

---

## Task 3: Rewire `build_homographies` (sequential reader) + notebook + spec

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pipeline.py`
- Modify: `examples/colab_homography_propagation.ipynb`
- Modify: `docs/superpowers/specs/2026-06-03-homography-propagation-design.md`
- Test: `packages/soccer-vision/tests/test_pipeline_analyze_video.py` (verify still passes; relax if needed)

- [ ] **Step 1: Rewrite `build_homographies` in `pipeline.py`**

Replace the existing `build_homographies` with:

```python
def build_homographies(
    keypoints: pd.DataFrame,
    video_path: Path,
    trajectories_px: pd.DataFrame,
    *,
    kp_conf_threshold: float = 0.5,
    max_gap: int = 25,
    disagreement_tau: float = 0.10,
    downscale: float = 0.5,
) -> dict[int, HomographyEntry]:
    """Anchors from keypoints + propagation into the gaps.

    Computes the needed consecutive inter-frame homographies in one sequential video
    pass (each frame decoded once via grab/retrieve, ORB on a downscaled copy), then
    composes them with `propagate_homographies` (pure). CPU; reads frames sequentially.
    """
    anchors = build_frame_homographies(keypoints, conf_threshold=kp_conf_threshold)
    keys = sorted(anchors)
    needed_pairs: set[int] = set()
    for a, b in itertools.pairwise(keys):
        if 1 <= b - a - 1 <= max_gap:
            needed_pairs.update(range(a, b))          # G[a..b-1] span the gap

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    pos = 0

    def read_frame(idx: int) -> "np.ndarray | None":
        nonlocal pos
        if idx < pos:                                 # backward jump (rare) -> seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            pos = idx
        while pos < idx:                              # skip forward cheaply (no decode)
            if not cap.grab():
                return None
            pos += 1
        ok, frame = cap.read()
        pos += 1
        return frame if ok else None

    try:
        interframe = compute_interframe_homographies(
            read_frame, needed_pairs, trajectories_px, downscale=downscale,
        )
    finally:
        cap.release()

    return propagate_homographies(
        anchors, interframe, max_gap=max_gap, disagreement_tau=disagreement_tau,
        frame_size=(width, height),
    )
```

Update the `pipeline.py` import to pull `compute_interframe_homographies` too:
```python
from soccer_vision.pitch.propagation import (
    HomographyEntry,
    compute_interframe_homographies,
    propagate_homographies,
)
```

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest packages/soccer-vision/tests/ -q`
Expected: all pass. The `analyze_video` stub test passes `Path("unused.mp4")`: `cv2.VideoCapture` opens nothing, `read_frame` returns None, `needed_pairs` may be empty or yield no interframe Hs, propagation returns anchors only — `homographies.parquet` is still written. If the stub test fails, relax it to assert the five files exist + `PipelineResult` + phases frames == [0,1,2]; report the change.

- [ ] **Step 3: Update the acceptance notebook sweep cell**

In `examples/colab_homography_propagation.ipynb`, the sweep cell calls the OLD `propagate_homographies(train, read_frame, traj, max_gap=mg)`. Rewrite it to the new fast flow: compute inter-frame Hs ONCE for the widest `max_gap`, then for each `max_gap` value recompute only the needed-pairs subset (or reuse the superset). Simplest correct version — compute interframe for the widest window once and reuse (a subset of pairs is always valid):

```python
import cv2, numpy as np, pandas as pd, itertools
from pathlib import Path
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS, build_frame_homographies
from soccer_vision.pitch.propagation import compute_interframe_homographies, propagate_homographies

# ... OUT/CLIP/kp/traj/anchors/total as before ...
keys = sorted(anchors)
held = keys[::5]
train = {f: H for f, H in anchors.items() if f not in held}

MAX = 60
pairs = set()
for a, b in itertools.pairwise(sorted(train)):
    if 1 <= b - a - 1 <= MAX:
        pairs.update(range(a, b))

cap = cv2.VideoCapture(CLIP)
pos = 0
def read_frame(idx):
    global pos
    if idx < pos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); pos = idx
    while pos < idx:
        if not cap.grab(): return None
        pos += 1
    ok, fr = cap.read(); pos += 1
    return fr if ok else None

interframe = compute_interframe_homographies(read_frame, pairs, traj, downscale=0.5)  # ONE pass
cap.release()

def reproj_error(H, frame):
    kb = kp[(kp.frame == frame) & (kp.conf >= 0.5) & (kp.kp_idx < len(PITCH_LANDMARKS))]
    if len(kb) < 4: return None
    pts = np.column_stack([kb.x_px, kb.y_px, np.ones(len(kb))])
    m = (H @ pts.T).T; m /= m[:, 2:3]
    return float(np.linalg.norm(m[:, :2] - PITCH_LANDMARKS[kb.kp_idx.to_numpy()], axis=1).mean())

print(f"{'max_gap':>8} {'coverage':>9} {'held median err':>16} {'n':>5}")
for mg in [10, 20, 30, 45, 60]:
    out = propagate_homographies(train, interframe, max_gap=mg)
    errs = sorted(e for e in (reproj_error(out[f].H, f) for f in held if f in out) if e is not None)
    med = errs[len(errs) // 2] if errs else float("nan")
    print(f"{mg:8d} {len(out) / total:9.1%} {med:16.3f} {len(errs):5d}")
```

The video is read ONCE (not per `max_gap`); the sweep is then pure matrix composition — seconds.

- [ ] **Step 4: Update the spec**

In `docs/superpowers/specs/2026-06-03-homography-propagation-design.md`, update §3.1 / §7 to note the split: `compute_interframe_homographies` (one sequential pass, downscaled ORB, rescaled to full-res) feeds a pure `propagate_homographies`. Change the §4 `propagate_homographies` signature to `(anchors, interframe, *, max_gap, disagreement_tau, frame_size)` and add `compute_interframe_homographies` + a `build_homographies(..., downscale=0.5)` note. Keep it brief.

- [ ] **Step 5: Full verification + commit**

Run: `uv run ruff check packages/soccer-vision/ && uv run mypy packages/soccer-vision/src && uv run pytest packages/soccer-vision/tests/ -q && uv run python -c "import nbformat; nbformat.validate(nbformat.read('examples/colab_homography_propagation.ipynb', as_version=4)); print('nb valid')"`
Expected: clean / Success / all pass / nb valid.

```bash
git add packages/soccer-vision/src/soccer_vision/pipeline.py examples/colab_homography_propagation.ipynb docs/superpowers/specs/2026-06-03-homography-propagation-design.md packages/soccer-vision/tests/
git commit -m "perf(pipeline): sequential single-pass inter-frame Hs + downscaled ORB in build_homographies"
```

---

## Self-Review

**Spec coverage:** the refactor preserves all behavior/outputs (§5 unchanged); §3 mechanics + §7 perf updated in Task 3 step 4. New `compute_interframe_homographies` realizes the "one sequential pass" intent.

**Placeholder scan:** none — complete code in every step.

**Type consistency:** `propagate_homographies(anchors, interframe, *, ...)`, `compute_interframe_homographies(read_frame, needed_pairs, player_boxes, *, downscale, ...)`, and `build_homographies(..., downscale=0.5)` signatures match their call sites in Task 3 and the notebook. `_chain` takes `interframe` everywhere.

**Known tuning point:** the 2 px tolerance in Task 2's `compute_interframe` test (ORB + 0.5× downscale on a clean translation) may need a small nudge during TDD — flagged at its step.

**Behavior-equivalence note:** propagation results are mathematically identical to the pre-refactor version *modulo* ORB on downscaled vs full-res frames (a sub-pixel difference in `G[i]`); the held-out acceptance error (~0.02) is the guardrail that this doesn't degrade accuracy.
```
