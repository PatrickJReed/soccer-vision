# Track Hygiene & Team Re-clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A post-hoc CPU stage that filters detections to on-pitch, stitches ByteTrack fragments in pitch space, re-clusters teams from kit colors on stitched tracks, and assigns goalkeepers positionally — emitting a cleaned `trajectories_px_clean.parquet` for the existing `assemble_from_homographies`.

**Architecture:** New `soccer_vision/hygiene/` subpackage: `core.py` (pure DataFrame/numpy functions, fully TDD), `run.py` (impure driver: parquet/video I/O, crop extraction, contact sheets, report), `__main__.py` (CLI), `agreement.py` (ground-truth possession agreement CLI). Reuses `PitchMapper`, `filter_outside_pitch`, `PitchSpec`.

**Tech Stack:** Python 3.11, numpy, pandas, OpenCV (no sklearn — a small deterministic weighted k-means lives in core). pytest, mypy strict, ruff.

---

## CRITICAL conventions for every task (lessons from prior phases)

- **The real mypy gate is bare `uv run mypy` from the REPO ROOT** (checks src AND tests). Your new files must add ZERO errors. Annotate every test helper and fixture (`tmp_path: Path`), no bare `dict` generics.
- Ruff (E,F,I,B,UP,RUF): imports at the TOP of files (sorted), never two statements joined with `;`. Lint BOTH src and tests.
- Pitch coords: `x_pitch` ∈ [0,1] across the WIDTH, `y_pitch` ∈ [0,1] along the LENGTH (goal-to-goal). Isotropic distances need `x_len = x_pitch / aspect_ratio` (PitchSpec.standard_9v9().aspect_ratio = 1.5).
- `trajectories_px` schema: `frame:int64, t_seconds:float64, track_id:int64, x_px, y_px, bbox_x1, bbox_y1, bbox_x2, bbox_y2:float64, class:str, team:str, conf:float64`. Classes: player, goalkeeper, referee, ball.
- Homography parquet → `soccer_vision.pipeline.homographies_from_parquet(path) -> dict[int, HomographyEntry]` (`.H` maps full-pixel → pitch). `PitchMapper().transform(df, {frame: H}) -> df + x_pitch/y_pitch` (NaN where no homography). `soccer_vision.pitch.filter.filter_outside_pitch(df, margin)` drops NaN/out-of-bounds rows.

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/soccer_vision/hygiene/__init__.py` | Package marker, re-export core API | Create |
| `src/soccer_vision/hygiene/core.py` | Pure: fragments, stitching, k-means, team clustering, own-kit map, GK assignment, team application, balance gate, agreement math | Create |
| `src/soccer_vision/hygiene/run.py` | Impure: load artifacts, crop features from video, contact sheets, report.json, orchestration `run_hygiene()` | Create |
| `src/soccer_vision/hygiene/__main__.py` | CLI | Create |
| `src/soccer_vision/hygiene/agreement.py` | Ground-truth CSV vs phases agreement CLI | Create |
| `tests/test_hygiene_core.py` | Pure-core tests | Create |
| `tests/test_hygiene_run.py` | Driver tests (synthetic video) | Create |

---

## Task 1: Fragments + pitch-space stitching (core)

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/hygiene/__init__.py`
- Create: `packages/soccer-vision/src/soccer_vision/hygiene/core.py`
- Test: `packages/soccer-vision/tests/test_hygiene_core.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/soccer-vision/tests/test_hygiene_core.py`:

```python
"""Tests for the hygiene pure core: stitching, clustering, teams, gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from soccer_vision.hygiene.core import Fragment, extract_fragments, stitch_tracks


def _rows(
    track_id: int,
    frames: list[int],
    xy: list[tuple[float, float]],
    cls: str = "player",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": frames,
            "t_seconds": [f / 30.0 for f in frames],
            "track_id": track_id,
            "x_pitch": [p[0] for p in xy],
            "y_pitch": [p[1] for p in xy],
            "class": cls,
        }
    )


def test_extract_fragments_uses_first_last_valid_pitch_rows() -> None:
    df = _rows(7, [10, 11, 12], [(0.5, 0.2), (np.nan, np.nan), (0.5, 0.3)])
    frags = extract_fragments(df)
    assert len(frags) == 1
    f = frags[0]
    assert isinstance(f, Fragment)
    assert (f.track_id, f.start_frame, f.end_frame) == (7, 10, 12)
    # x is length-normalized: 0.5 / 1.5 aspect
    assert np.isclose(f.start_xy[0], 0.5 / 1.5)
    assert np.isclose(f.end_xy[1], 0.3)


def test_extract_fragments_skips_tracks_without_pitch_coords() -> None:
    df = _rows(7, [10, 11], [(np.nan, np.nan), (np.nan, np.nan)])
    assert extract_fragments(df) == []


def test_stitch_joins_close_fragments() -> None:
    # same player: fragment A frames 0-10 ending at (0.5,0.5); B starts frame 20
    # (0.33s later) a tiny distance away -> stitched.
    a = _rows(1, [0, 10], [(0.5, 0.5), (0.5, 0.5)])
    b = _rows(2, [20, 30], [(0.5, 0.52), (0.5, 0.6)])
    df = pd.concat([a, b], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    assert out.loc[out.orig_track_id == 2, "track_id"].unique().tolist() == [1]
    assert out.loc[out.orig_track_id == 1, "track_id"].unique().tolist() == [1]


def test_stitch_refuses_teleport() -> None:
    # B starts 0.33s later but across the pitch -> speed bound violated -> no stitch.
    a = _rows(1, [0, 10], [(0.5, 0.1), (0.5, 0.1)])
    b = _rows(2, [20, 30], [(0.5, 0.9), (0.5, 0.9)])
    df = pd.concat([a, b], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    assert out.track_id.nunique() == 2


def test_stitch_refuses_long_gap() -> None:
    # B starts 4s later (> max_gap_s=2.0) right next door -> no stitch.
    a = _rows(1, [0, 10], [(0.5, 0.5), (0.5, 0.5)])
    b = _rows(2, [130, 140], [(0.5, 0.51), (0.5, 0.51)])
    df = pd.concat([a, b], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    assert out.track_id.nunique() == 2


def test_stitch_refuses_overlapping_fragments() -> None:
    # two people visible simultaneously can't be the same track.
    a = _rows(1, [0, 20], [(0.5, 0.5), (0.5, 0.5)])
    b = _rows(2, [10, 30], [(0.5, 0.52), (0.5, 0.52)])
    df = pd.concat([a, b], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    assert out.track_id.nunique() == 2


def test_stitch_picks_nearest_candidate() -> None:
    a = _rows(1, [0, 10], [(0.5, 0.5), (0.5, 0.5)])
    near = _rows(2, [20, 30], [(0.5, 0.52), (0.5, 0.52)])
    far = _rows(3, [20, 30], [(0.5, 0.58), (0.5, 0.58)])
    df = pd.concat([a, near, far], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    by_orig = out.groupby("orig_track_id")["track_id"].first()
    assert by_orig[2] == 1          # nearest joined the chain
    assert by_orig[3] == 3          # other stays its own chain


def test_stitch_classes_do_not_mix() -> None:
    a = _rows(1, [0, 10], [(0.5, 0.5), (0.5, 0.5)], cls="player")
    b = _rows(2, [20, 30], [(0.5, 0.52), (0.5, 0.52)], cls="goalkeeper")
    df = pd.concat([a, b], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    assert out.track_id.nunique() == 2
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_hygiene_core.py -v`
Expected: FAIL — `ModuleNotFoundError: soccer_vision.hygiene`

- [ ] **Step 3: Implement**

Create `packages/soccer-vision/src/soccer_vision/hygiene/__init__.py`:

```python
"""Track hygiene: on-pitch filtering, fragment stitching, team re-clustering."""
```

Create `packages/soccer-vision/src/soccer_vision/hygiene/core.py`:

```python
"""Pure core of the track-hygiene stage.

Operates on trajectories DataFrames that already carry x_pitch/y_pitch columns
(added by PitchMapper). Distances are isotropic length-normalized pitch units:
x_len = x_pitch / aspect_ratio, y_len = y_pitch (y is the goal-to-goal axis).

No I/O. See docs/superpowers/specs/2026-06-11-track-hygiene-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.pitch.spec import PitchSpec

# Tolerances are physical, so they need a nominal pitch length to convert m/s
# into pitch-length units. US Soccer 9v9 mid-range; a tolerance, not a metric.
PITCH_LENGTH_M_DEFAULT = 68.5


@dataclass(frozen=True)
class Fragment:
    """A contiguous tracker fragment's pitch-space endpoints (length-normalized)."""

    track_id: int
    start_frame: int
    end_frame: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]


def extract_fragments(
    traj: pd.DataFrame, *, aspect_ratio: float | None = None
) -> list[Fragment]:
    """One Fragment per track_id from its first/last rows with valid pitch coords.

    Tracks with no valid pitch coordinate are omitted (they cannot stitch).
    """
    ar = aspect_ratio if aspect_ratio is not None else PitchSpec.standard_9v9().aspect_ratio
    frags: list[Fragment] = []
    valid = traj.dropna(subset=["x_pitch", "y_pitch"])
    for tid, g in valid.groupby("track_id", sort=True):
        g = g.sort_values("frame")
        first = g.iloc[0]
        last = g.iloc[-1]
        frags.append(
            Fragment(
                track_id=int(tid),
                start_frame=int(first["frame"]),
                end_frame=int(last["frame"]),
                start_xy=(float(first["x_pitch"]) / ar, float(first["y_pitch"])),
                end_xy=(float(last["x_pitch"]) / ar, float(last["y_pitch"])),
            )
        )
    return frags


def stitch_fragments(
    fragments: list[Fragment],
    *,
    fps: float,
    max_gap_s: float = 2.0,
    max_speed_ms: float = 8.0,
    pitch_length_m: float = PITCH_LENGTH_M_DEFAULT,
    slack: float = 0.02,
) -> dict[int, int]:
    """Greedy chain assembly: {orig track_id -> stitched id (min id in chain)}.

    Fragment B continues a chain if it starts 1..max_gap frames after the chain
    ends AND its start is within max_speed*gap + slack (length-normalized pitch
    units) of the chain end. Nearest-in-space eligible chain wins. Overlapping
    fragments never stitch (two people visible at once are not one person).
    """
    max_gap_frames = int(round(max_gap_s * fps))
    speed_pu = max_speed_ms / pitch_length_m  # pitch-length units per second
    # chains: list of [stitched_id, end_frame, end_xy]
    chains: list[tuple[int, int, tuple[float, float]]] = []
    mapping: dict[int, int] = {}
    for frag in sorted(fragments, key=lambda f: (f.start_frame, f.track_id)):
        best_i = -1
        best_dist = float("inf")
        for i, (_, end_frame, end_xy) in enumerate(chains):
            gap = frag.start_frame - end_frame
            if gap < 1 or gap > max_gap_frames:
                continue
            dist = float(np.hypot(frag.start_xy[0] - end_xy[0],
                                  frag.start_xy[1] - end_xy[1]))
            if dist > speed_pu * (gap / fps) + slack:
                continue
            if dist < best_dist:
                best_dist = dist
                best_i = i
        if best_i >= 0:
            sid, _, _ = chains[best_i]
            chains[best_i] = (sid, frag.end_frame, frag.end_xy)
            mapping[frag.track_id] = sid
        else:
            chains.append((frag.track_id, frag.end_frame, frag.end_xy))
            mapping[frag.track_id] = frag.track_id
    return mapping


def stitch_tracks(
    traj: pd.DataFrame,
    *,
    fps: float,
    classes: tuple[str, ...] = ("player", "goalkeeper"),
    max_gap_s: float = 2.0,
    max_speed_ms: float = 8.0,
    pitch_length_m: float = PITCH_LENGTH_M_DEFAULT,
    slack: float = 0.02,
) -> pd.DataFrame:
    """Rewrite track_id with stitched ids (per class); keep orig_track_id.

    Rows of other classes (ball, referee) pass through with track_id unchanged.
    """
    out = traj.copy()
    out["orig_track_id"] = out["track_id"]
    for cls in classes:
        sub = out[out["class"] == cls]
        if sub.empty:
            continue
        mapping = stitch_fragments(
            extract_fragments(sub),
            fps=fps, max_gap_s=max_gap_s, max_speed_ms=max_speed_ms,
            pitch_length_m=pitch_length_m, slack=slack,
        )
        mask = out["class"] == cls
        out.loc[mask, "track_id"] = out.loc[mask, "orig_track_id"].map(
            lambda t: mapping.get(int(t), int(t))
        )
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_hygiene_core.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Gate + commit**

Run: `cd /Users/patrickreed/Sandbox/soccer-vision && uv run mypy 2>&1 | grep -E "hygiene" || echo CLEAN` (expect CLEAN), then `cd packages/soccer-vision && uv run ruff check src/soccer_vision/hygiene/ tests/test_hygiene_core.py`.

```bash
git add packages/soccer-vision/src/soccer_vision/hygiene/ packages/soccer-vision/tests/test_hygiene_core.py
git commit -m "feat(hygiene): fragments + pitch-space track stitching"
```

---

## Task 2: Weighted k-means, team clustering, own-kit mapping (core)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/hygiene/core.py`
- Test: `packages/soccer-vision/tests/test_hygiene_core.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hygiene_core.py` (add imports to the TOP import block:
`from soccer_vision.hygiene.core import cluster_teams, map_own_cluster, weighted_kmeans2`):

```python
def test_weighted_kmeans2_separates_two_blobs() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal((0, 0, 0, 0, 0, 0), 0.5, size=(20, 6))
    b = rng.normal((10, 10, 10, 10, 10, 10), 0.5, size=(20, 6))
    x = np.vstack([a, b])
    w = np.ones(40)
    labels, centroids = weighted_kmeans2(x, w, seed=0)
    assert set(labels[:20].tolist()) != set(labels[20:].tolist())
    assert len(set(labels.tolist())) == 2
    assert centroids.shape == (2, 6)


def test_weighted_kmeans2_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, size=(30, 6))
    w = np.ones(30)
    l1, _ = weighted_kmeans2(x, w, seed=7)
    l2, _ = weighted_kmeans2(x, w, seed=7)
    assert np.array_equal(l1, l2)


def test_cluster_teams_boundary_tracks_are_unknown() -> None:
    # two tight blobs + one feature exactly between them -> None (unknown).
    feats = {
        1: np.zeros(6), 2: np.zeros(6) + 0.1,
        3: np.full(6, 10.0), 4: np.full(6, 10.1),
        5: np.full(6, 5.0),  # equidistant
    }
    weights = {k: 1.0 for k in feats}
    teams, centroids = cluster_teams(feats, weights, seed=0)
    assert teams[5] is None
    assert teams[1] is not None and teams[3] is not None
    assert teams[1] != teams[3]


def test_map_own_cluster_picks_nearer_shirt_color() -> None:
    # cluster 0 shirt ~ white (Lab L high, a/b neutral), cluster 1 ~ dark blue.
    centroids = np.array([
        [250.0, 128.0, 128.0, 100.0, 128.0, 128.0],   # white shirt
        [40.0, 130.0, 80.0, 180.0, 120.0, 190.0],     # dark-blue shirt
    ])
    own, warning = map_own_cluster(centroids, "white")
    assert own == 0
    assert warning is None


def test_map_own_cluster_unknown_color_word_raises() -> None:
    centroids = np.zeros((2, 6))
    try:
        map_own_cluster(centroids, "tartan")
    except ValueError as e:
        assert "tartan" in str(e)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_hygiene_core.py -k "kmeans or cluster or own" -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement**

Append to `core.py` (add `import cv2` to the top imports):

```python
def weighted_kmeans2(
    x: NDArray[np.floating],
    weights: NDArray[np.floating],
    *,
    n_iter: int = 50,
    seed: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Deterministic weighted 2-means (Lloyd's, farthest-point init).

    Returns (labels (n,), centroids (2, d)). Tiny problem sizes (hundreds of
    tracks) — no need for sklearn.
    """
    pts = np.asarray(x, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(pts)))
    d0 = np.linalg.norm(pts - pts[first], axis=1)
    second = int(d0.argmax())
    centroids = np.stack([pts[first], pts[second]])
    labels = np.zeros(len(pts), dtype=np.int64)
    for _ in range(n_iter):
        d = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = d.argmin(axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for c in (0, 1):
            mask = labels == c
            if mask.any():
                centroids[c] = np.average(pts[mask], axis=0, weights=w[mask])
    return labels, centroids


def cluster_teams(
    features: dict[int, NDArray[np.floating]],
    weights: dict[int, float],
    *,
    boundary_ratio: float = 0.8,
    seed: int = 0,
) -> tuple[dict[int, int | None], NDArray[np.float64]]:
    """Cluster per-track kit features into 2 teams.

    Returns ({track_id: 0|1|None}, centroids). None = near the decision boundary
    (d_near/d_far > boundary_ratio) -> better unknown than wrong.
    """
    ids = sorted(features)
    x = np.stack([np.asarray(features[i], dtype=np.float64) for i in ids])
    w = np.array([weights[i] for i in ids], dtype=np.float64)
    labels, centroids = weighted_kmeans2(x, w, seed=seed)
    out: dict[int, int | None] = {}
    for row, tid in enumerate(ids):
        d = np.linalg.norm(centroids - x[row], axis=1)
        near, far = int(d.argmin()), int(d.argmax())
        if d[far] > 0 and d[near] / d[far] > boundary_ratio:
            out[tid] = None
        else:
            out[tid] = int(labels[row])
    return out, centroids


# BGR anchors for --own-kit color words; converted to Lab at runtime.
KIT_ANCHORS_BGR: dict[str, tuple[int, int, int]] = {
    "white": (245, 245, 245),
    "black": (20, 20, 20),
    "blue": (200, 80, 30),
    "dark blue": (110, 40, 10),
    "navy": (80, 30, 5),
    "red": (40, 40, 200),
    "yellow": (40, 210, 230),
    "green": (60, 160, 40),
    "orange": (30, 130, 240),
    "purple": (150, 50, 110),
}


def _lab_anchor(color_word: str) -> NDArray[np.float64]:
    key = color_word.strip().lower()
    if key not in KIT_ANCHORS_BGR:
        raise ValueError(
            f"unknown kit color {color_word!r}; known: {sorted(KIT_ANCHORS_BGR)}"
        )
    bgr = np.array([[KIT_ANCHORS_BGR[key]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]
    return lab.astype(np.float64)


def map_own_cluster(
    centroids: NDArray[np.floating],
    own_kit: str,
    *,
    warn_margin: float = 1.2,
) -> tuple[int, str | None]:
    """Pick the cluster whose SHIRT Lab centroid (first 3 dims) matches own_kit.

    Returns (own_cluster_index, warning|None). Warns when the two clusters are
    nearly equidistant from the hint (ratio < warn_margin) — contact sheets
    arbitrate then.
    """
    anchor = _lab_anchor(own_kit)
    shirt = np.asarray(centroids, dtype=np.float64)[:, :3]
    d = np.linalg.norm(shirt - anchor, axis=1)
    own = int(d.argmin())
    other = 1 - own
    warning = None
    if d[own] > 0 and d[other] / d[own] < warn_margin:
        warning = (
            f"--own-kit {own_kit!r} matches both clusters similarly "
            f"(d={d[own]:.1f} vs {d[other]:.1f}); verify the contact sheets"
        )
    return own, warning
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_hygiene_core.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Gate + commit**

Same gate commands as Task 1, then:

```bash
git add packages/soccer-vision/src/soccer_vision/hygiene/core.py packages/soccer-vision/tests/test_hygiene_core.py
git commit -m "feat(hygiene): weighted 2-means team clustering + own-kit mapping"
```

---

## Task 3: GK assignment, team application, balance gate (core)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/hygiene/core.py`
- Test: `packages/soccer-vision/tests/test_hygiene_core.py`

- [ ] **Step 1: Write the failing tests**

Append (top-import `assign_goalkeepers, apply_team_labels, balance_gate`):

```python
def _scene_with_gks() -> pd.DataFrame:
    rows = []
    # own outfield players around y=0.3; opp around y=0.7 (3 frames)
    for f in range(3):
        for i, (tid, team_cluster) in enumerate([(1, 0), (2, 0), (3, 1), (4, 1)]):
            y = 0.3 if team_cluster == 0 else 0.7
            rows.append({"frame": f, "t_seconds": f / 30.0, "track_id": tid,
                         "x_pitch": 0.4 + 0.05 * i, "y_pitch": y, "class": "player"})
        rows.append({"frame": f, "t_seconds": f / 30.0, "track_id": 90,
                     "x_pitch": 0.5, "y_pitch": 0.05, "class": "goalkeeper"})
        rows.append({"frame": f, "t_seconds": f / 30.0, "track_id": 91,
                     "x_pitch": 0.5, "y_pitch": 0.95, "class": "goalkeeper"})
    return pd.DataFrame(rows)


def test_assign_goalkeepers_by_nearest_team() -> None:
    df = _scene_with_gks()
    teams = {1: "own", 2: "own", 3: "opp", 4: "opp"}
    gk = assign_goalkeepers(df, teams)
    assert gk[90] == "own"   # near the y=0.3 group
    assert gk[91] == "opp"


def test_apply_team_labels_inherits_and_passes_through() -> None:
    df = _scene_with_gks()
    df["team"] = "stale"
    teams = {1: "own", 2: "own", 3: "opp", 4: "opp", 90: "own", 91: "opp"}
    out = apply_team_labels(df, teams)
    assert (out.loc[out.track_id == 1, "team"] == "own").all()
    assert (out.loc[out.track_id == 91, "team"] == "opp").all()


def test_apply_team_labels_unknown_for_unmapped() -> None:
    df = _scene_with_gks()
    df["team"] = "stale"
    out = apply_team_labels(df, {1: "own"})
    assert (out.loc[out.track_id == 3, "team"] == "unknown").all()


def test_balance_gate() -> None:
    df = _scene_with_gks()
    df = apply_team_labels(df, {1: "own", 2: "own", 3: "opp", 4: "opp"})
    ratio, passed = balance_gate(df)
    assert np.isclose(ratio, 1.0)
    assert passed
    df_bad = apply_team_labels(df, {1: "own", 2: "opp", 3: "opp", 4: "opp"})
    ratio_bad, passed_bad = balance_gate(df_bad)
    assert not passed_bad
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_hygiene_core.py -k "goalkeeper or apply_team or balance" -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement**

Append to `core.py`:

```python
def assign_goalkeepers(
    traj: pd.DataFrame, player_team_by_track: dict[int, str]
) -> dict[int, str]:
    """Assign each goalkeeper track the team whose players are nearest on average.

    Uses frames where the GK has pitch coords; compares mean distance (length-
    normalized) to own- vs opp-assigned players in the same frames. GK tracks
    with no usable frames or no co-visible teammates -> 'unknown'.
    """
    ar = PitchSpec.standard_9v9().aspect_ratio
    players = traj[traj["class"] == "player"].dropna(subset=["x_pitch", "y_pitch"]).copy()
    players["team_label"] = players["track_id"].map(
        lambda t: player_team_by_track.get(int(t), "unknown")
    )
    players = players[players["team_label"].isin(["own", "opp"])]
    out: dict[int, str] = {}
    gks = traj[traj["class"] == "goalkeeper"].dropna(subset=["x_pitch", "y_pitch"])
    for tid, g in gks.groupby("track_id", sort=True):
        merged = g.merge(players, on="frame", suffixes=("_gk", "_pl"))
        if merged.empty:
            out[int(tid)] = "unknown"
            continue
        dx = (merged["x_pitch_gk"] - merged["x_pitch_pl"]) / ar
        dy = merged["y_pitch_gk"] - merged["y_pitch_pl"]
        merged["dist"] = np.hypot(dx, dy)
        mean_by_team = merged.groupby("team_label")["dist"].mean()
        if len(mean_by_team) == 0:
            out[int(tid)] = "unknown"
        else:
            out[int(tid)] = str(mean_by_team.idxmin())
    return out


def apply_team_labels(
    traj: pd.DataFrame, team_by_track: dict[int, str]
) -> pd.DataFrame:
    """Overwrite player/GK rows' team from {stitched track_id -> own/opp/unknown}.

    Unmapped player/GK tracks become 'unknown'. Ball/referee rows untouched.
    """
    out = traj.copy()
    mask = out["class"].isin(["player", "goalkeeper"])
    out.loc[mask, "team"] = out.loc[mask, "track_id"].map(
        lambda t: team_by_track.get(int(t), "unknown")
    )
    return out


def balance_gate(
    traj: pd.DataFrame, *, lo: float = 0.6, hi: float = 1.6
) -> tuple[float, bool]:
    """Mean per-frame own:opp player-detection ratio and whether it's in [lo, hi]."""
    players = traj[traj["class"].isin(["player", "goalkeeper"])]
    per_frame = players.groupby("frame")["team"].value_counts().unstack(fill_value=0)
    n_own = float(per_frame.get("own", pd.Series(dtype=float)).sum())
    n_opp = float(per_frame.get("opp", pd.Series(dtype=float)).sum())
    if n_opp == 0:
        return float("inf"), False
    ratio = n_own / n_opp
    return ratio, lo <= ratio <= hi
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_hygiene_core.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Gate + commit**

```bash
git add packages/soccer-vision/src/soccer_vision/hygiene/core.py packages/soccer-vision/tests/test_hygiene_core.py
git commit -m "feat(hygiene): GK assignment, team application, balance gate"
```

---

## Task 4: Possession agreement vs ground-truth CSV (core + CLI)

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/hygiene/core.py`
- Create: `packages/soccer-vision/src/soccer_vision/hygiene/agreement.py`
- Test: `packages/soccer-vision/tests/test_hygiene_core.py`

- [ ] **Step 1: Write the failing tests**

Append (top-import `expand_ground_truth, possession_agreement`):

```python
def test_expand_ground_truth_change_points() -> None:
    gt = pd.DataFrame({"t_seconds": [0.0, 4.0, 10.0],
                       "possession": ["own", "opp", "none"]})
    t = pd.Series([0.0, 2.0, 4.0, 9.9, 10.0, 12.0])
    states = expand_ground_truth(gt, t)
    assert states.tolist() == ["own", "own", "opp", "opp", "none", "none"]


def test_possession_agreement_counts_team_frames_only() -> None:
    phases = pd.DataFrame({
        "frame": range(6),
        "t_seconds": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "possession_state": ["own", "own", "opp", "loose_ball", "opp", "own"],
    })
    gt = pd.DataFrame({"t_seconds": [0.0, 2.0],
                       "possession": ["own", "opp"]})
    res = possession_agreement(gt, phases)
    # comparable frames: t=0,1 (own/own, own/own), 2,4 (opp/opp, opp/opp), 5 (opp gt vs own pred)
    # loose_ball frame excluded.
    assert res.n_compared == 5
    assert np.isclose(res.agreement, 4 / 5)
    assert len(res.disagreements) == 1
    t0, t1, gt_s, pred_s = res.disagreements[0]
    assert (gt_s, pred_s) == ("opp", "own")
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_hygiene_core.py -k agreement -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement**

Append to `core.py`:

```python
@dataclass(frozen=True)
class AgreementResult:
    """Frame-level possession agreement vs hand-labeled ground truth."""

    n_compared: int
    agreement: float
    disagreements: list[tuple[float, float, str, str]]  # (t_start, t_end, gt, pred)


def expand_ground_truth(gt: pd.DataFrame, t_seconds: pd.Series) -> pd.Series:
    """Expand change-point ground truth (t_seconds, possession) to a per-time series.

    Each row's state holds from its t_seconds until the next row. Times before
    the first row are 'none'.
    """
    gt_sorted = gt.sort_values("t_seconds")
    idx = np.searchsorted(gt_sorted["t_seconds"].to_numpy(), t_seconds.to_numpy(),
                          side="right") - 1
    states = np.where(idx >= 0,
                      gt_sorted["possession"].to_numpy()[np.clip(idx, 0, None)],
                      "none")
    return pd.Series(states, index=t_seconds.index)


def possession_agreement(gt: pd.DataFrame, phases: pd.DataFrame) -> AgreementResult:
    """Compare predicted possession_state with ground truth on team-attributed frames.

    Only frames where BOTH sides say own/opp are compared (loose/contested/
    unknown/none excluded). Disagreements are merged into contiguous spans.
    """
    pred = phases["possession_state"]
    gt_states = expand_ground_truth(gt, phases["t_seconds"])
    both = pred.isin(["own", "opp"]) & gt_states.isin(["own", "opp"])
    n = int(both.sum())
    if n == 0:
        return AgreementResult(0, 0.0, [])
    agree = (pred[both] == gt_states[both])
    spans: list[tuple[float, float, str, str]] = []
    cur: tuple[float, str, str] | None = None
    last_t = 0.0
    for i in phases.index[both]:
        t = float(phases.loc[i, "t_seconds"])
        if bool(agree.loc[i]):
            if cur is not None:
                spans.append((cur[0], last_t, cur[1], cur[2]))
                cur = None
        else:
            g, p = str(gt_states.loc[i]), str(pred.loc[i])
            if cur is None or (cur[1], cur[2]) != (g, p):
                if cur is not None:
                    spans.append((cur[0], last_t, cur[1], cur[2]))
                cur = (t, g, p)
        last_t = t
    if cur is not None:
        spans.append((cur[0], last_t, cur[1], cur[2]))
    return AgreementResult(n, float(agree.mean()), spans)
```

Create `src/soccer_vision/hygiene/agreement.py`:

```python
"""CLI: compare hand-labeled possession CSV against a phases.parquet.

CSV format (change points): header `t_seconds,possession`, one row whenever
possession changes; possession in {own, opp, none}.

Usage: python -m soccer_vision.hygiene.agreement --phases phases.parquet --gt gt.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soccer_vision.hygiene.core import possession_agreement


def main() -> None:
    ap = argparse.ArgumentParser(description="Possession ground-truth agreement")
    ap.add_argument("--phases", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    args = ap.parse_args()
    phases = pd.read_parquet(args.phases)
    gt = pd.read_csv(args.gt)
    res = possession_agreement(gt, phases)
    print(f"compared frames (both attribute a team): {res.n_compared}")
    print(f"agreement: {res.agreement * 100:.1f}%   (target >= 80%)")
    if res.disagreements:
        print("disagreement spans (t_start..t_end  gt -> pred):")
        for t0, t1, g, p in res.disagreements:
            print(f"  {t0:7.1f}..{t1:7.1f}s   {g} -> {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_hygiene_core.py -v`
Expected: PASS (19 tests)

- [ ] **Step 5: Gate + commit**

```bash
git add packages/soccer-vision/src/soccer_vision/hygiene/ packages/soccer-vision/tests/test_hygiene_core.py
git commit -m "feat(hygiene): possession ground-truth agreement (core + CLI)"
```

---

## Task 5: Driver — crops, features, contact sheets, report, orchestration

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/hygiene/run.py`
- Test: `packages/soccer-vision/tests/test_hygiene_run.py`

- [ ] **Step 1: Write the failing test (synthetic video end-to-end)**

Create `packages/soccer-vision/tests/test_hygiene_run.py`:

```python
"""End-to-end driver test on a tiny synthetic video with two kit colors."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from soccer_vision.hygiene.run import run_hygiene
from soccer_vision.pipeline import homographies_to_parquet
from soccer_vision.pitch.propagation import HomographyEntry

_W, _H, _N = 320, 240, 30
# identity-ish homography: pixels -> pitch with a simple scale so all bboxes land
# in-bounds: x_pitch = x_px / W, y_pitch = y_px / H.
_H_PX = np.diag([1.0 / _W, 1.0 / _H, 1.0])

# two "players": white shirt/blue shorts at left, dark-blue shirt/yellow shorts right
_KITS = {
    1: ((255, 255, 255), (150, 60, 20), 60),    # shirt BGR, shorts BGR, x position
    2: ((110, 40, 10), (40, 210, 230), 220),
}


def _write_video(path: Path) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (_W, _H))
    for _ in range(_N):
        frame = np.full((_H, _W, 3), (40, 120, 40), dtype=np.uint8)  # grass
        for _tid, (shirt, shorts, x) in _KITS.items():
            cv2.rectangle(frame, (x, 100), (x + 30, 130), shirt, -1)   # torso
            cv2.rectangle(frame, (x, 130), (x + 30, 150), shorts, -1)  # shorts
        vw.write(frame)
    vw.release()


def _traj() -> pd.DataFrame:
    rows = []
    for f in range(_N):
        for tid, (_, _, x) in _KITS.items():
            rows.append({
                "frame": f, "t_seconds": f / 30.0, "track_id": tid,
                "x_px": x + 15.0, "y_px": 125.0,
                "bbox_x1": float(x), "bbox_y1": 100.0,
                "bbox_x2": float(x + 30), "bbox_y2": 150.0,
                "class": "player", "team": "stale", "conf": 0.9,
            })
    return pd.DataFrame(rows).astype({"frame": "int64", "track_id": "int64"})


def test_run_hygiene_end_to_end(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    _write_video(video)
    traj_path = tmp_path / "traj.parquet"
    _traj().to_parquet(traj_path, index=False)
    hom_path = tmp_path / "hom.parquet"
    homographies_to_parquet(
        {f: HomographyEntry(_H_PX, "manual", 1.0) for f in range(_N)}, hom_path
    )
    out = tmp_path / "out"

    report = run_hygiene(
        traj_path=traj_path, homographies_path=hom_path, video_path=video,
        out_dir=out, own_kit="white",
    )

    clean = pd.read_parquet(out / "trajectories_px_clean.parquet")
    assert "orig_track_id" in clean.columns
    teams = clean.groupby("track_id")["team"].first()
    assert set(teams.values) == {"own", "opp"}
    # white-shirt track (id 1) must be own
    assert teams[1] == "own"
    assert (out / "hygiene_report.json").exists()
    saved = json.loads((out / "hygiene_report.json").read_text())
    assert saved["balance"]["passed"] is True
    sheets = list(out.glob("team_cluster_*.png"))
    assert len(sheets) == 2
    assert any("_OWN" in s.name for s in sheets)
    assert report["balance"]["passed"] is True
```

- [ ] **Step 2: Run to verify fail**

Run: `cd packages/soccer-vision && uv run pytest tests/test_hygiene_run.py -v`
Expected: FAIL — `ModuleNotFoundError: soccer_vision.hygiene.run`

- [ ] **Step 3: Implement run.py**

Create `src/soccer_vision/hygiene/run.py`:

```python
"""Impure driver for the hygiene stage: artifact I/O, crop features, orchestration.

Reads trajectories_px + homographies parquets and the local video; writes
trajectories_px_clean.parquet, contact sheets, and hygiene_report.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.hygiene.core import (
    apply_team_labels,
    assign_goalkeepers,
    balance_gate,
    cluster_teams,
    map_own_cluster,
    stitch_tracks,
)
from soccer_vision.pipeline import homographies_from_parquet
from soccer_vision.pitch.filter import filter_outside_pitch
from soccer_vision.pitch.mapper import PitchMapper

_CROPS_PER_TRACK = 10
_MIN_CROP_PX = 8
_SHEET_COLS = 10
_SHEET_CELL = 64


def _resolve_fps(traj: pd.DataFrame) -> float:
    by_frame = traj.drop_duplicates("frame").sort_values("frame")
    dt = by_frame["t_seconds"].diff().median()
    df = by_frame["frame"].diff().median()
    if not dt or pd.isna(dt) or dt <= 0:
        return 30.0
    return float(df / dt)


def _sample_frames(track_rows: pd.DataFrame) -> list[int]:
    frames = sorted(track_rows["frame"].unique().tolist())
    if len(frames) <= _CROPS_PER_TRACK:
        return [int(f) for f in frames]
    idx = np.linspace(0, len(frames) - 1, _CROPS_PER_TRACK).astype(int)
    return [int(frames[i]) for i in idx]


def _region_lab(crop: NDArray[np.uint8], y0: float, y1: float) -> NDArray[np.float64] | None:
    h, w = crop.shape[:2]
    if h < _MIN_CROP_PX or w < _MIN_CROP_PX:
        return None
    region = crop[int(h * y0):int(h * y1), int(w * 0.2):int(w * 0.8)]
    if region.size == 0:
        return None
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, axis=0).astype(np.float64)


def extract_track_features(
    video_path: Path, traj: pd.DataFrame
) -> tuple[dict[int, NDArray[np.float64]], dict[int, float], dict[int, list[NDArray[np.uint8]]]]:
    """Per stitched player track: 6-dim shirt+shorts Lab feature, weight, crops.

    Only rows flagged on_pitch contribute (adjacent-field/no-homography rows are
    excluded from features). Reads the video once, sequentially.
    """
    players = traj[(traj["class"] == "player") & traj["on_pitch"]]
    wanted: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for tid, g in players.groupby("track_id", sort=True):
        for f in _sample_frames(g):
            row = g[g["frame"] == f].iloc[0]
            box = (float(row["bbox_x1"]), float(row["bbox_y1"]),
                   float(row["bbox_x2"]), float(row["bbox_y2"]))
            wanted.setdefault(int(f), []).append((int(tid), box))

    shirt_feats: dict[int, list[NDArray[np.float64]]] = {}
    shorts_feats: dict[int, list[NDArray[np.float64]]] = {}
    crops: dict[int, list[NDArray[np.uint8]]] = {}
    cap = cv2.VideoCapture(str(video_path))
    pos = 0
    try:
        for f in sorted(wanted):
            while pos < f:
                if not cap.grab():
                    break
                pos += 1
            ok, frame = cap.read()
            pos += 1
            if not ok:
                continue
            for tid, (x1, y1, x2, y2) in wanted[f]:
                crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                shirt = _region_lab(crop, 0.15, 0.50)
                shorts = _region_lab(crop, 0.50, 0.80)
                if shirt is None or shorts is None:
                    continue
                shirt_feats.setdefault(tid, []).append(shirt)
                shorts_feats.setdefault(tid, []).append(shorts)
                crops.setdefault(tid, []).append(crop)
    finally:
        cap.release()

    features: dict[int, NDArray[np.float64]] = {}
    weights: dict[int, float] = {}
    track_len = players.groupby("track_id")["frame"].nunique()
    for tid in shirt_feats:
        features[tid] = np.concatenate([
            np.median(np.stack(shirt_feats[tid]), axis=0),
            np.median(np.stack(shorts_feats[tid]), axis=0),
        ])
        weights[tid] = float(track_len.get(tid, 1))
    return features, weights, crops


def write_contact_sheet(
    crops: list[NDArray[np.uint8]], path: Path, *, max_cells: int = 40
) -> None:
    """Grid of resized crops for at-a-glance cluster verification."""
    cells = [cv2.resize(c, (_SHEET_CELL, _SHEET_CELL)) for c in crops[:max_cells]
             if c.size > 0]
    if not cells:
        cells = [np.zeros((_SHEET_CELL, _SHEET_CELL, 3), dtype=np.uint8)]
    rows = []
    for i in range(0, len(cells), _SHEET_COLS):
        row = cells[i:i + _SHEET_COLS]
        row += [np.zeros_like(cells[0])] * (_SHEET_COLS - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows))


def run_hygiene(
    *,
    traj_path: Path,
    homographies_path: Path,
    video_path: Path,
    out_dir: Path,
    own_kit: str,
    max_gap_s: float = 2.0,
    max_speed_ms: float = 8.0,
    margin: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Full hygiene pass; returns (and writes) the report dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj = pd.read_parquet(traj_path)
    entries = homographies_from_parquet(homographies_path)
    h_map = {f: e.H for f, e in entries.items()}

    # sanity: the video must match the trajectories
    cap = cv2.VideoCapture(str(video_path))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_video < int(traj["frame"].max()):
        raise ValueError(
            f"video has {n_video} frames but trajectories reference frame "
            f"{int(traj['frame'].max())} — wrong video?"
        )

    fps = _resolve_fps(traj)
    mapped = PitchMapper().transform(traj, h_map)

    # Step 1: on-pitch filter for player/GK rows; keep-but-flag no-homography rows.
    is_person = mapped["class"].isin(["player", "goalkeeper"])
    has_pitch = mapped["x_pitch"].notna()
    on_pitch_idx = filter_outside_pitch(mapped[is_person], margin).index
    mapped["on_pitch"] = mapped.index.isin(on_pitch_idx)
    drop = is_person & has_pitch & ~mapped["on_pitch"]
    n_dropped = int(drop.sum())
    kept = mapped[~drop].copy()

    # Step 2: stitch.
    n_tracks_before = int(kept.loc[kept["class"].isin(["player", "goalkeeper"]),
                                   "track_id"].nunique())
    stitched = stitch_tracks(kept, fps=fps, max_gap_s=max_gap_s,
                             max_speed_ms=max_speed_ms)
    n_tracks_after = int(stitched.loc[stitched["class"].isin(["player", "goalkeeper"]),
                                      "track_id"].nunique())

    # Step 3: cluster teams.
    features, weights, crops = extract_track_features(video_path, stitched)
    warning: str | None = None
    if len(features) < 2:
        team_by_track: dict[int, str] = {}
        warning = "fewer than 2 tracks with features; all players unknown"
        centroid_list: list[list[float]] = []
    else:
        cluster_of, centroids = cluster_teams(features, weights, seed=seed)
        centroid_list = [[float(v) for v in c] for c in centroids]
        if float(np.linalg.norm(centroids[0] - centroids[1])) < 10.0:
            # degenerate clustering: centroids nearly identical (Lab units) ->
            # confident garbage is never emitted.
            team_by_track = {tid: "unknown" for tid in features}
            warning = "cluster centroids nearly identical; all players set unknown"
        else:
            own_cluster, warning = map_own_cluster(centroids, own_kit)
            label = {own_cluster: "own", 1 - own_cluster: "opp"}
            team_by_track = {tid: ("unknown" if c is None else label[c])
                             for tid, c in cluster_of.items()}
            for c_idx in (0, 1):
                tids = [t for t, c in cluster_of.items() if c == c_idx]
                sheet = [crop for t in tids for crop in crops.get(t, [])[:3]]
                suffix = "_OWN" if c_idx == own_cluster else ""
                write_contact_sheet(sheet, out / f"team_cluster_{c_idx}{suffix}.png")

    # Step 4: goalkeepers.
    team_by_track.update(assign_goalkeepers(stitched, team_by_track))

    clean = apply_team_labels(stitched, team_by_track)
    ratio, passed = balance_gate(clean)
    clean = clean.drop(columns=["x_pitch", "y_pitch", "on_pitch"])
    clean.to_parquet(out / "trajectories_px_clean.parquet", index=False)

    spans = (clean[clean["class"].isin(["player", "goalkeeper"])]
             .groupby("track_id")["frame"].agg(["min", "max"]))
    report: dict[str, Any] = {
        "n_dropped_off_pitch": n_dropped,
        "tracks_before": n_tracks_before,
        "tracks_after": n_tracks_after,
        "median_track_span_frames": float((spans["max"] - spans["min"] + 1).median())
        if len(spans) else 0.0,
        "unknown_fraction": float(
            (clean.loc[clean["class"] == "player", "team"] == "unknown").mean()
        ) if (clean["class"] == "player").any() else 0.0,
        "cluster_centroids_lab": centroid_list,
        "balance": {"own_opp_ratio": ratio, "passed": passed},
        "warning": warning,
        "params": {"own_kit": own_kit, "max_gap_s": max_gap_s,
                   "max_speed_ms": max_speed_ms, "margin": margin, "fps": fps},
    }
    (out / "hygiene_report.json").write_text(json.dumps(report, indent=2))
    print(f"tracks {n_tracks_before} -> {n_tracks_after}; "
          f"dropped off-pitch rows: {n_dropped}")
    print(f"balance own:opp = {ratio:.2f}  [{'PASS' if passed else 'FAIL'}]")
    if warning:
        print(f"WARNING: {warning}")
    return report
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_hygiene_run.py -v`
Expected: PASS (1 test). If the synthetic clusters land as `unknown` (boundary
rule) the test fails — fix by checking the feature separation, not by loosening
the assert.

- [ ] **Step 5: Gate + commit**

Repo-root `uv run mypy` (zero hygiene errors) + ruff both files, then:

```bash
git add packages/soccer-vision/src/soccer_vision/hygiene/run.py packages/soccer-vision/tests/test_hygiene_run.py
git commit -m "feat(hygiene): driver — crop features, contact sheets, report, orchestration"
```

---

## Task 6: CLI + full gate

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/hygiene/__main__.py`

- [ ] **Step 1: Create the CLI**

```python
"""CLI: python -m soccer_vision.hygiene --traj ... --homographies ... --video ...

See docs/superpowers/specs/2026-06-11-track-hygiene-design.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from soccer_vision.hygiene.run import run_hygiene


def main() -> None:
    ap = argparse.ArgumentParser(description="Track hygiene & team re-clustering")
    ap.add_argument("--traj", required=True, type=Path, help="trajectories_px.parquet")
    ap.add_argument("--homographies", required=True, type=Path)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--own-kit", required=True,
                    help="your shirt color (e.g. white, dark blue)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-gap-s", type=float, default=2.0)
    ap.add_argument("--max-speed", type=float, default=8.0)
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()
    run_hygiene(
        traj_path=args.traj, homographies_path=args.homographies,
        video_path=args.video, out_dir=args.out_dir, own_kit=args.own_kit,
        max_gap_s=args.max_gap_s, max_speed_ms=args.max_speed, margin=args.margin,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import + full gate**

Run: `cd packages/soccer-vision && uv run python -c "import soccer_vision.hygiene.__main__ as m; print('ok', hasattr(m, 'main'))"` → `ok True`.
Then the full gate: `cd /Users/patrickreed/Sandbox/soccer-vision && uv run mypy 2>&1 | tail -1` (Success), `cd packages/soccer-vision && uv run pytest -q 2>&1 | tail -1` (all pass), `uv run ruff check src/ tests/`.

- [ ] **Step 3: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/hygiene/__main__.py
git commit -m "feat(hygiene): CLI launcher"
```

---

## Task 7: Real-data acceptance (controller/Patrick, local; documented)

Not a subagent task. With the bake-off artifacts in `~/sv-labeler/`:

```bash
cd packages/soccer-vision
uv run python -m soccer_vision.hygiene \
  --traj ~/sv-labeler/trajectories_px.parquet \
  --homographies ~/sv-labeler/out/homographies.parquet \
  --video ~/sv-labeler/clip.mp4 \
  --own-kit white --out-dir ~/sv-labeler/hygiene_out
```

Checks: balance gate PASS; tracks ~1088 → a few dozen; contact sheets visually
clean (white-kit sheet is `_OWN`); then re-assemble:

```python
from soccer_vision.pipeline import assemble_from_homographies
assemble_from_homographies("~/sv-labeler/hygiene_out/trajectories_px_clean.parquet",
                           "~/sv-labeler/out/homographies.parquet",
                           "~/sv-labeler/analysis2")
```

Possession breakdown must now include own-possession (build/attack frames > 0).
Finally Patrick writes the ground-truth CSV and runs:

```bash
uv run python -m soccer_vision.hygiene.agreement \
  --phases ~/sv-labeler/analysis2/phases.parquet --gt ~/sv-labeler/possession_gt.csv
```

Target ≥80% agreement; below that, tune possession thresholds against the CSV
(cheap recompute) before concluding.
