# View-Dataset Exporter (Slice 1.5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export a video + its `ViewDigest` into a manifest-first training dataset (per-frame view pseudo-labels + confidence/margin) that feeds the Colab contrastive + MAE objectives from one artifact.

**Architecture:** New `labeler/view_dataset.py` beside `view_digest.py`. Pure core (rectangular ORB scoring → nearest-view assignment → temporal smoothing → splits → manifest assembly) + streaming video orchestration with an `.npz` match cache + thin persist/CLI + a Colab-side `ViewFrameReader`. One DRY refactor extracts the per-pair scorer from `view_digest.similarity_matrix` so the rectangular scorer shares the exact metric.

**Tech Stack:** Python, numpy, pandas/parquet, OpenCV (ORB/decode), scipy (already used), pytest/mypy/ruff via `uv`. Reuses `view_digest` primitives (`frame_descriptors`, `_read_frames`, `_build_masks`, `_video_hash`, `DEFAULT_*`).

**Reference:** `docs/superpowers/specs/2026-07-02-view-dataset-exporter-design.md`. Run `uv run pytest` from `packages/soccer-vision`; run `mypy`/`ruff` from the **repo root**. Commit after each task.

---

### Task 1: DRY refactor + rectangular cross-match metric

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/view_digest.py` (extract `_pair_match_fraction` from `similarity_matrix`; lines ~129-140)
- Create: `packages/soccer-vision/src/soccer_vision/labeler/view_dataset.py` (module header + `cross_match_fractions`)
- Test: `packages/soccer-vision/tests/test_view_dataset.py`

- [ ] **Step 1: Write failing tests** (`tests/test_view_dataset.py`)

```python
"""Tests for the view-dataset exporter (annotation-scaling Slice 1.5)."""
from __future__ import annotations

import numpy as np
import cv2
from numpy.typing import NDArray
from soccer_vision.labeler.view_digest import (
    frame_descriptors, similarity_matrix, _pair_match_fraction)
from soccer_vision.labeler.view_dataset import cross_match_fractions

H, W = 240, 320


def _pattern(seed: int) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 30, np.uint8)
    for _ in range(45):
        x1, x2 = sorted(rng.integers(0, W, size=2).tolist())
        y1, y2 = sorted(rng.integers(0, H, size=2).tolist())
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        else:
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
    return img


def test_pair_match_fraction_matches_similarity_matrix() -> None:
    descs, counts = frame_descriptors([_pattern(0), _pattern(1), _pattern(0)])
    s = similarity_matrix(descs, counts)
    # the extracted helper reproduces each off-diagonal similarity entry exactly
    assert abs(_pair_match_fraction(descs[0], descs[1], counts[0], counts[1]) - s[0, 1]) < 1e-12
    assert abs(_pair_match_fraction(descs[0], descs[2], counts[0], counts[2]) - s[0, 2]) < 1e-12


def test_cross_match_fractions_parity_with_similarity_matrix() -> None:
    # rectangular query-vs-reps equals the cross block of the full NxN similarity.
    frames = [_pattern(5), _pattern(1), _pattern(2), _pattern(3)]
    q, refs = frames[:1], frames[1:]
    qd, qc = frame_descriptors(q)
    rd, rc = frame_descriptors(refs)
    cross = cross_match_fractions(qd, qc, rd, rc)          # (1, 3)
    full = similarity_matrix(*frame_descriptors(q + refs))  # (4, 4)
    assert cross.shape == (1, 3)
    assert np.allclose(cross[0], full[0, 1:], atol=1e-12)


def test_cross_match_fractions_low_keypoints_zero() -> None:
    qd, qc = frame_descriptors([_pattern(0)])
    rd, rc = frame_descriptors([_pattern(1)])
    cross = cross_match_fractions(qd, qc, rd, [3], min_keypoints=10)  # ref "blind"
    assert cross[0, 0] == 0.0
```

- [ ] **Step 2: Run tests, verify they fail** (`_pair_match_fraction` / module absent)

Run: `cd packages/soccer-vision && uv run pytest tests/test_view_dataset.py -x -q`
Expected: ImportError / AttributeError.

- [ ] **Step 3: Refactor `view_digest.similarity_matrix`** — extract the per-pair scorer:

```python
def _pair_match_fraction(
    da: _Descriptor, db: _Descriptor, ca: int, cb: int, *,
    min_match_dist: int = DEFAULT_MIN_MATCH_DIST, min_keypoints: int = DEFAULT_MIN_KEYPOINTS,
) -> float:
    """Fraction of cross-checked ORB matches (dist < min_match_dist) / min(keypoints)."""
    if da is None or db is None or ca < min_keypoints or cb < min_keypoints:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    good = sum(1 for m in matcher.match(da, db) if m.distance < min_match_dist)
    return good / max(1, min(ca, cb))
```
Then rewrite the inner loop of `similarity_matrix` to call it (behavior identical; the diagonal stays 1.0). Existing `test_similarity_*` in `test_view_digest.py` must still pass. (Constructing one `BFMatcher` per call is fine; if profiling later shows it matters, hoist it — not now.)

- [ ] **Step 4: Create `view_dataset.py` header + `cross_match_fractions`:**

```python
"""Manifest-first view-dataset exporter (annotation-scaling Slice 1.5).

Turns a video + its ViewDigest into a compact per-frame training manifest (view
pseudo-label + confidence + margin + split) the Colab model consumes for both the
contrastive (use view_id) and masked-autoencoder (ignore it) objectives — writing
zero pixels by default; Colab decodes frames on demand from the same ALL-INTRA mp4.

See docs/superpowers/specs/2026-07-02-view-dataset-exporter-design.md.
"""
from __future__ import annotations

# ... imports: dataclasses, json, hashlib, pathlib, typing, numpy, pandas, cv2, NDArray;
#     from soccer_vision.labeler.view_digest import (frame_descriptors, _read_frames,
#     _build_masks, _video_hash, _pair_match_fraction, _Descriptor, DEFAULT_N_FEATURES,
#     DEFAULT_DOWNSCALE, DEFAULT_MIN_MATCH_DIST, DEFAULT_MIN_KEYPOINTS, ViewDigest)

DEFAULT_ASSIGN_STRIDE = 5
DEFAULT_AMBIGUITY_MARGIN = 0.05
DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_VAL_FRAC = 0.1
DEFAULT_CHUNK = 512
SCHEMA_VERSION = 1


def cross_match_fractions(
    query_descriptors: list, query_counts: list[int],
    ref_descriptors: list, ref_counts: list[int], *,
    min_match_dist: int = DEFAULT_MIN_MATCH_DIST, min_keypoints: int = DEFAULT_MIN_KEYPOINTS,
) -> NDArray[np.float64]:
    """(Q,R) match-fraction of each query frame vs each representative — same metric as
    similarity_matrix, rectangular (no NxN over queries)."""
    q, r = len(query_descriptors), len(ref_descriptors)
    out = np.zeros((q, r), dtype=np.float64)
    for i in range(q):
        for j in range(r):
            out[i, j] = _pair_match_fraction(
                query_descriptors[i], ref_descriptors[j], query_counts[i], ref_counts[j],
                min_match_dist=min_match_dist, min_keypoints=min_keypoints)
    return out
```

- [ ] **Step 5: Run tests to green**

Run: `cd packages/soccer-vision && uv run pytest tests/test_view_dataset.py tests/test_view_digest.py -q`
Expected: PASS (parity + existing similarity tests).

- [ ] **Step 6: mypy + ruff (repo root), then commit**

Run: `uv run mypy packages/soccer-vision/src/soccer_vision/labeler/view_dataset.py packages/soccer-vision/src/soccer_vision/labeler/view_digest.py && uv run ruff check packages/soccer-vision/src/soccer_vision/labeler/ packages/soccer-vision/tests/test_view_dataset.py`
```bash
git add -A && git commit -m "refactor(labeler): extract _pair_match_fraction; add cross_match_fractions"
```

---

### Task 2: assignment core — `assign_nearest_view`, `smooth_view_sequence`, `assign_splits`

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/view_dataset.py`
- Test: `packages/soccer-vision/tests/test_view_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
import pandas as pd
from soccer_vision.labeler.view_dataset import (
    assign_nearest_view, smooth_view_sequence, assign_splits)


def test_assign_nearest_view_fields() -> None:
    # 3 reps for views [0,1,2]; query rows pick a best + runner-up VIEW.
    match = np.array([
        [0.8, 0.3, 0.1],   # -> view 0, second view 1, conf .8, margin .5
        [0.2, 0.2, 0.9],   # -> view 2, second view 0/1 (.2), conf .9, margin .7
        [0.0, 0.0, 0.0],   # -> unassigned
    ])
    df = assign_nearest_view(match, [0, 1, 2])
    assert list(df["view_id_raw"]) == [0, 2, -1]
    assert df.loc[0, "view_second"] == 1
    assert abs(df.loc[0, "confidence"] - 0.8) < 1e-9
    assert abs(df.loc[0, "margin"] - 0.5) < 1e-9      # best - best-of-a-different-view
    assert df.loc[2, "confidence"] == 0.0 and df.loc[2, "view_second"] == -1


def test_assign_margin_is_best_minus_other_view_not_other_rep() -> None:
    # two reps share view 0; a third is view 1. margin must compare across VIEWS,
    # so two strong same-view reps do NOT produce a tiny margin.
    match = np.array([[0.8, 0.75, 0.1]])
    df = assign_nearest_view(match, [0, 0, 1])
    assert df.loc[0, "view_id_raw"] == 0
    assert abs(df.loc[0, "margin"] - (0.8 - 0.1)) < 1e-9


def test_smooth_view_sequence_removes_singleton() -> None:
    seq = [0, 0, 0, 1, 0, 0, 0]          # lone 1 is noise
    out = smooth_view_sequence(seq, window=5)
    assert list(out) == [0, 0, 0, 0, 0, 0, 0]


def test_smooth_window_one_is_identity() -> None:
    seq = [0, 1, 0, 2]
    assert list(smooth_view_sequence(seq, window=1)) == seq


def test_smooth_tie_keeps_original() -> None:
    seq = [0, 0, 1, 1]                    # window=3 at idx1: {0,0,1}->0; no ties here
    out = smooth_view_sequence(seq, window=3)
    assert list(out) == [0, 0, 1, 1]


def test_assign_splits_per_view_tail_every_view_in_both() -> None:
    df = pd.DataFrame({"frame": list(range(20)),
                       "view_id": [0]*10 + [1]*10})
    out = assign_splits(df, val_frac=0.2, policy="per_view_tail")
    for v in (0, 1):
        sub = out[out["view_id"] == v]
        assert (sub["split"] == "train").any() and (sub["split"] == "val").any()
        assert (sub["split"] == "val").sum() == 2      # last 20% of each view's 10 rows
    # val is the temporal TAIL of each view
    assert set(out[(out["view_id"] == 0) & (out["split"] == "val")]["frame"]) == {8, 9}


def test_assign_splits_holdout_views() -> None:
    df = pd.DataFrame({"frame": list(range(20)), "view_id": [0]*10 + [1]*10})
    out = assign_splits(df, policy="holdout_views", holdout_views={1})
    assert (out[out["view_id"] == 1]["split"] == "val").all()
    assert (out[out["view_id"] == 0]["split"] == "train").all()
```

- [ ] **Step 2: Run, verify fail.** `uv run pytest tests/test_view_dataset.py -x -q`

- [ ] **Step 3: Implement** the three functions:
  - `assign_nearest_view(match, ref_view_ids)`: `ref_view_ids` array aligned to columns. Best column = argmax; `confidence` = that max; `view_id_raw` = its view (or `-1` if best == 0.0). `view_second` = the view of the best column whose view != best view (or `-1`). `margin` = `confidence − max(sim over columns whose view != best view)` (0 if none). Return a `pd.DataFrame` with int32/float32 dtypes.
  - `smooth_view_sequence(view_ids, *, window=5)`: odd sliding window majority (mode) over the sequence; `window<=1` → unchanged; on a tie keep the original center label; edges use the truncated window. Return `NDArray[int]`.
  - `assign_splits(manifest, *, val_frac=0.1, policy='per_view_tail', holdout_views=None)`: copy; add `split`. `per_view_tail`: within each `view_id` sorted by `frame`, last `ceil(val_frac*n)` → `val`, rest `train`. `holdout_views`: rows whose `view_id ∈ holdout_views` → `val`, else `train`. Deterministic.

- [ ] **Step 4: Run to green.** **Step 5: mypy+ruff (repo root).** **Step 6: Commit** `feat(labeler): view assignment + temporal smoothing + splits`.

---

### Task 3: pure manifest assembly — `build_manifest` + constants/dtypes

**Files:** Modify `view_dataset.py`; Test `tests/test_view_dataset.py`.

- [ ] **Step 1: Failing tests**

```python
from soccer_vision.labeler.view_dataset import build_manifest


def test_build_manifest_schema_and_content() -> None:
    match = np.array([[0.8, 0.2], [0.15, 0.7], [0.75, 0.1]])   # frames -> views [0,1,0]
    df = build_manifest(query_frames=[0, 5, 10], match=match, ref_view_ids=[0, 1],
                        keypoint_counts=[300, 280, 310], game="oceanside", fps=30.0,
                        ambiguity_margin=0.05, smooth_window=1, val_frac=0.34)
    assert list(df.columns) == ["game", "frame", "t_seconds", "view_id", "view_id_raw",
        "view_second", "view_key", "confidence", "weight", "margin", "ambiguous",
        "n_keypoints", "n_boxes", "split"]
    assert df["view_id"].dtype == np.int32 and df["confidence"].dtype == np.float32
    assert list(df["view_id"]) == [0, 1, 0]
    assert list(df["view_key"]) == ["oceanside:0", "oceanside:1", "oceanside:0"]
    assert (df["weight"].to_numpy() == df["confidence"].to_numpy()).all()
    assert abs(df.loc[0, "t_seconds"] - 0.0) < 1e-9 and abs(df.loc[2, "t_seconds"] - 10/30) < 1e-6
    assert list(df["n_boxes"]) == [0, 0, 0]              # None -> 0
    assert df["frame"].is_monotonic_increasing


def test_build_manifest_ambiguous_flag() -> None:
    match = np.array([[0.50, 0.49]])                    # margin 0.01 < 0.05 -> ambiguous
    df = build_manifest([0], match, [0, 1], [300], game="g", fps=30.0, ambiguity_margin=0.05)
    assert bool(df.loc[0, "ambiguous"]) is True


def test_build_manifest_smoothing_preserves_raw() -> None:
    match = np.array([[0.9, 0.1]]*3 + [[0.1, 0.9]] + [[0.9, 0.1]]*3)  # lone view1 at idx3
    df = build_manifest(list(range(7)), match, [0, 1], [300]*7, game="g", fps=30.0,
                        smooth_window=5)
    assert df.loc[3, "view_id_raw"] == 1 and df.loc[3, "view_id"] == 0   # raw kept, smoothed fixed


def test_build_manifest_records_boxes() -> None:
    match = np.array([[0.8, 0.2]])
    df = build_manifest([0], match, [0, 1], [300], game="g", fps=30.0, n_boxes=[4])
    assert df.loc[0, "n_boxes"] == 4
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `build_manifest`** — pure assembly, no video:
  1. `assign_nearest_view(match, ref_view_ids)` → base frame.
  2. `view_id` = `smooth_view_sequence(view_id_raw, window=smooth_window)`.
  3. Add `game`, `frame`, `t_seconds = frame/fps`, `view_key = f"{game}:{view_id}"`, `weight = confidence`, `ambiguous = margin < ambiguity_margin`, `n_keypoints`, `n_boxes` (0 when `None`).
  4. `assign_splits(..., val_frac, policy=split_policy)`.
  5. Enforce column order + dtypes above; sort by `frame`; `reset_index(drop=True)`. Deterministic.

- [ ] **Step 4: Green. Step 5: mypy+ruff. Step 6: Commit** `feat(labeler): pure manifest assembly`.

---

### Task 4: streaming video orchestration — `build_view_assignment` (+ cache, masking)

**Files:** Modify `view_dataset.py`; Test `tests/test_view_dataset.py`.

- [ ] **Step 1: Failing tests** (synthetic mp4 helper; skip if writer unavailable)

```python
from pathlib import Path
import pytest
from soccer_vision.labeler.view_digest import compute_view_digest
from soccer_vision.labeler.view_dataset import build_view_assignment, ViewAssignment


def _write_video(path: Path, frames: list) -> bool:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    vw = cv2.VideoWriter(str(path), fourcc, 30.0, (W, H))
    if not vw.isOpened():
        return False
    for fr in frames:
        vw.write(fr)
    vw.release()
    return path.exists() and path.stat().st_size > 0


def _digest_video(tmp_path: Path):
    frames = [_pattern(v) for v in ([0]*10 + [1]*10 + [2]*10)]
    video = tmp_path / "clip.mp4"
    if not _write_video(video, frames):
        pytest.skip("platform OpenCV cannot write mp4")
    digest = compute_view_digest(video, stride=3, dist_threshold=0.5, cache_dir=tmp_path)
    return video, digest


def test_build_view_assignment_manifest(tmp_path: Path) -> None:
    video, digest = _digest_video(tmp_path)
    va = build_view_assignment(video, digest, game="synth", assign_stride=2,
                               smooth_window=1, cache_dir=tmp_path)
    assert isinstance(va, ViewAssignment)
    m = va.manifest
    assert len(m) == len(range(0, 30, 2))
    assert set(m["view_id"]).issubset(set(digest.view_of.values()))
    assert 0.0 <= va.switch_rate <= 1.0
    assert va.n_frames == len(m) and va.n_views >= 1


def test_build_view_assignment_cache_hit(tmp_path: Path) -> None:
    video, digest = _digest_video(tmp_path)
    va1 = build_view_assignment(video, digest, game="synth", assign_stride=2, cache_dir=tmp_path)
    # 2nd call: no decode — corrupt the video, expect identical assignment from cache.
    video.write_bytes(b"broken")
    va2 = build_view_assignment(video, digest, game="synth", assign_stride=2, cache_dir=tmp_path)
    assert list(va1.manifest["view_id_raw"]) == list(va2.manifest["view_id_raw"])
    assert np.allclose(va1.manifest["confidence"], va2.manifest["confidence"])


def test_build_view_assignment_masking_records_boxes(tmp_path: Path) -> None:
    video, digest = _digest_video(tmp_path)
    boxes = pd.DataFrame({"frame": [0, 0], "class": ["player", "player"],
                          "bbox_x1": [0, 100], "bbox_y1": [0, 100],
                          "bbox_x2": [50, 200], "bbox_y2": [50, 200]})
    va = build_view_assignment(video, digest, game="synth", assign_stride=2,
                               player_boxes=boxes, cache_dir=tmp_path)
    row0 = va.manifest[va.manifest["frame"] == 0].iloc[0]
    assert row0["n_boxes"] == 2
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `ViewAssignment` dataclass + `build_view_assignment`:**
  - `@dataclass(frozen=True) class ViewAssignment`: `manifest: pd.DataFrame`, `representatives: dict[int,int]`, `meta: dict[str, Any]`; properties `n_frames = len(manifest)`, `n_views = manifest["view_id"].nunique()`, `n_ambiguous = int(manifest["ambiguous"].sum())`, `switch_rate = fraction of consecutive rows (frame-sorted) where view_id changes`. (`materialize` added in Task 5.)
  - `build_view_assignment`: probe video for `n_frames, fps, width, height`. `query_frames = range(0, n_frames, assign_stride)`. Representatives: `rep_items = sorted(digest.representatives.items())`; decode rep frames via `_read_frames`; `frame_descriptors` → `rep_desc, rep_counts`; `ref_view_ids = [view for view,_ in rep_items]`. Compute a `reps_fingerprint` (hash of rep frame indices + params). **Cache**: `viewassign_<hash>.npz` keyed by video_hash+assign_stride+n_features+downscale+min_match_dist+min_keypoints+reps_fingerprint, storing `query_frames, match(Q×R), keypoint_counts, ref_view_ids`. If hit (and `player_boxes is None`), skip decode. Else stream: for each chunk of `chunk` query indices, `_read_frames` → optional `_build_masks(player_boxes, chunk_idx, (width,height))` → `frame_descriptors(masks=...)` → `cross_match_fractions(chunk_desc, chunk_counts, rep_desc, rep_counts)` → append match rows + counts + per-frame `n_boxes` (from `player_boxes.groupby('frame').size()`, 0 default) → **discard pixels/descriptors**. Save cache (only when `player_boxes is None`). Then `build_manifest(query_frames, match, ref_view_ids, keypoint_counts, game=game or Path(video).stem, fps=fps, n_boxes=..., ...)`. Build `meta` (see Task 5 sidecar keys). Return `ViewAssignment`.

- [ ] **Step 4: Green (skips gracefully w/o mp4 writer). Step 5: mypy+ruff. Step 6: Commit** `feat(labeler): streaming view-assignment with match cache + masking hook`.

---

### Task 5: persist + Colab reader — `write_export`, `load_manifest`, `ViewFrameReader`, `materialize`

**Files:** Modify `view_dataset.py`; Test `tests/test_view_dataset.py`.

- [ ] **Step 1: Failing tests**

```python
import json
from soccer_vision.labeler.view_dataset import (
    write_export, load_manifest, ViewFrameReader)


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    video, digest = _digest_video(tmp_path)
    va = build_view_assignment(video, digest, game="synth", assign_stride=2, cache_dir=tmp_path)
    out = tmp_path / "export"
    paths = write_export(va, out, video_path=video)
    assert (out / "view_dataset.parquet").exists() and (out / "view_dataset.json").exists()
    m, meta = load_manifest(out)
    assert len(m) == va.n_frames
    assert meta["schema_version"] == 1
    assert meta["video"]["video_hash"] and meta["stats"]["n_frames"] == va.n_frames
    assert set(str(k) for k in meta["digest"]["representatives"])  # view->frame present


def test_frame_reader_roundtrip(tmp_path: Path) -> None:
    video, digest = _digest_video(tmp_path)
    va = build_view_assignment(video, digest, game="synth", assign_stride=2, cache_dir=tmp_path)
    out = tmp_path / "export"; write_export(va, out, video_path=video)
    m, meta = load_manifest(out)
    reader = ViewFrameReader(video, m)
    frame, mask, row = reader.read(0)
    assert frame.shape == (H, W, 3) and mask is None
    assert int(row["frame"]) == int(m.iloc[0]["frame"])
    reader.close()


def test_frame_reader_wrong_video_raises(tmp_path: Path) -> None:
    video, digest = _digest_video(tmp_path)
    va = build_view_assignment(video, digest, game="synth", assign_stride=2, cache_dir=tmp_path)
    out = tmp_path / "export"; write_export(va, out, video_path=video)
    m, _ = load_manifest(out)
    other = tmp_path / "other.mp4"
    _write_video(other, [_pattern(9) for _ in range(30)])
    with pytest.raises(ValueError):
        ViewFrameReader(other, m)          # video_hash mismatch


def test_materialize_folders_relative_paths(tmp_path: Path) -> None:
    video, digest = _digest_video(tmp_path)
    va = build_view_assignment(video, digest, game="synth", assign_stride=2, cache_dir=tmp_path)
    out = tmp_path / "mat"
    paths = va.materialize(out, video_path=video, image_downscale=0.5)
    assert paths and all(p.exists() for p in paths)
    # every written image lives under out/frames/<split>/view_XX/ with a relative path
    for p in paths:
        rel = p.relative_to(out)
        assert rel.parts[0] == "frames" and rel.parts[1] in {"train", "val"}
        assert rel.parts[2].startswith("view_")
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**:
  - `write_export(assignment, out_dir, *, video_path)`: mkdir; `manifest.to_parquet(out/"view_dataset.parquet", index=False)`; write `out/"view_dataset.json"` from `assignment.meta` (must already carry `schema_version, dataset_fingerprint, video{path,abspath,video_hash,n_frames,fps,width,height}, params, digest{n_views,representatives}, stats{per_view_counts,n_ambiguous,n_train,n_val,confidence_median,margin_median,switch_rate}, boxes_source`). Return `[parquet, json]`. (Build `meta` in Task 4's orchestration; if a stat is cleaner to compute here, compute from the manifest — keep it deterministic, JSON-safe: cast numpy scalars to py types, view ids to str keys.)
  - `load_manifest(export_dir)`: read parquet + json → `(df, meta)`.
  - `ViewFrameReader(video_path, manifest, *, boxes=None, downscale=1.0)`: on init, `if _video_hash(video_path) != <hash from a companion? >` — the reader gets the hash by re-reading the sidecar? No: reader takes `manifest` only, so pass the expected hash via the manifest's `meta`? Simplest: `ViewFrameReader` reads the sidecar next to nothing — instead accept the manifest and **look up the video hash from `load_manifest` meta by requiring the caller** … Decision: give `ViewFrameReader` the sidecar `meta` is overkill; instead compute `_video_hash(video_path)` and compare to a `video_hash` column? No. **Final:** `ViewFrameReader.__init__` accepts an optional `expected_video_hash: str | None = None`; when provided and it != `_video_hash(video_path)`, raise `ValueError`. `load_manifest` returns meta with the hash, and the test passes `ViewFrameReader(video, m)` — so also accept the hash implicitly: store no hash → to satisfy `test_frame_reader_wrong_video_raises` **without** meta, have the reader raise when the video cannot be opened OR when a quick probe of `n_frames` is smaller than the manifest's max frame. Simpler and robust: **raise if `max(manifest["frame"]) >= probed n_frames`** (a different/short video) — covers the test (other.mp4 has 30 frames but the manifest references the same indices, so instead compare `_video_hash`). To keep it clean, thread the hash: `write_export` adds a `video_hash` attr into `manifest.attrs["video_hash"]`; `load_manifest` restores it; `ViewFrameReader` reads `manifest.attrs.get("video_hash")` and compares. If attrs don't survive parquet, fall back to `expected_video_hash` param. Implement whichever is robust and covered by the test; the **contract** is: constructing a reader against a different video raises `ValueError`. Frame decode: forward-grab fast path + `CAP_PROP_POS_FRAMES` fallback (mirror `_read_frames`); `read(i)` returns `(frame_bgr, keep_mask|None, manifest_row)`; `keep_mask` from `boxes` when given (via `_build_masks` for that frame), else `None`; `close()` releases the capture.
  - `ViewAssignment.materialize(out_dir, *, video_path, image_downscale=0.5, jpeg_quality=90, boxes=None)`: pure re-materialization off the manifest — for each row decode via a `ViewFrameReader`, optionally downscale, write `out/frames/<split>/view_{view_id:02d}/{game}_f{frame:06d}.jpg` (cv2.imwrite q=jpeg_quality); if `boxes`, also write `out/masks/<split>/view_XX/..png`. Return the list of written paths (all under `out`, relative-safe).

  > Note the hash-threading ambiguity above — pick the robust implementation and make `test_frame_reader_wrong_video_raises` pass; do not add scope beyond a `ValueError` on mismatch.

- [ ] **Step 4: Green. Step 5: mypy+ruff. Step 6: Commit** `feat(labeler): export persist + Colab ViewFrameReader + materialize`.

---

### Task 6: CLI + Colab snippet + oceanside validation

**Files:** Modify `view_dataset.py`; Test `tests/test_view_dataset.py`.

- [ ] **Step 1: Failing test** (CLI on synthetic video)

```python
def test_cli_writes_export(tmp_path: Path) -> None:
    from soccer_vision.labeler.view_dataset import main
    frames = [_pattern(v) for v in ([0]*10 + [1]*10 + [2]*10)]
    video = tmp_path / "clip.mp4"
    if not _write_video(video, frames):
        pytest.skip("no mp4 writer")
    out = tmp_path / "cli_out"
    main(["--video", str(video), "--out", str(out), "--game", "synth",
          "--assign-stride", "2", "--stride", "3", "--dist-threshold", "0.5",
          "--cache-dir", str(tmp_path)])
    assert (out / "view_dataset.parquet").exists()
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `main(argv)`**: argparse — `--video` (required), `--out`, `--game`, `--digest-json` (optional; else `compute_view_digest` with `--stride/--dist-threshold`), `--assign-stride`, `--smooth-window`, `--ambiguity-margin`, `--val-frac`, `--boxes` (parquet), `--cache-dir`, and `--materialize` (flag) `--image-downscale`/`--jpeg-quality`. Flow: obtain `ViewDigest` (load from `--digest-json` via a small reader, or compute); `player_boxes = pd.read_parquet(--boxes)` if given; `build_view_assignment(...)`; `write_export(...)`; if `--materialize`, `va.materialize(...)`. Print: `n_frames`, `n_views`, `switch_rate`, `n_ambiguous`, `n_train/n_val`, output dir. Add a `if __name__ == "__main__": main()`.

- [ ] **Step 4: Add a Colab-loader docstring** at module top (contrastive + MAE):
```python
# In Colab:
#   from soccer_vision.labeler.view_dataset import load_manifest, ViewFrameReader
#   m, meta = load_manifest("export/")
#   reader = ViewFrameReader("oceanside_clip.mp4", m, expected_video_hash=meta["video"]["video_hash"])
#   # contrastive: label = row.view_key (or view_id); positives share it; weight = row.weight
#   # MAE: frame, keep_mask, _ = reader.read(i); mask players via keep_mask when present
```

- [ ] **Step 5: Green + full suite + mypy + ruff.**

Run: `cd packages/soccer-vision && uv run pytest -q` then from repo root `uv run mypy packages/soccer-vision/src/soccer_vision/labeler/view_dataset.py && uv run ruff check packages/soccer-vision/`

- [ ] **Step 6: Oceanside validation** (report numbers as facts):
```bash
uv run python -m soccer_vision.labeler.view_dataset \
  --video ~/sv-labeler/oceanside_clip.mp4 --out ~/sv-labeler/view_dataset_out \
  --game oceanside --assign-stride 5
# expect ~530 rows, switch_rate ~5%, 2nd run hits the cache instantly
```

- [ ] **Step 7: Commit** `feat(labeler): view-dataset CLI + Colab loader; validate on oceanside`.

---

## Self-Review notes
- **Spec coverage:** cross-match parity (T1), assignment/margin-vs-view/smoothing/splits (T2), manifest schema+dtypes+ambiguous+soft-label preservation (T3), streaming+cache+masking (T4), persist+reader+hash-guard+materialize (T5), CLI+Colab snippet+oceanside (T6). Deferred items (tar-shards, cross-game vocab, PNG/normalized-boxes) intentionally absent.
- **Type consistency:** `cross_match_fractions`/`_pair_match_fraction` signatures identical across tasks; `ViewAssignment`/`ViewFrameReader`/`build_manifest` names stable throughout.
- **Known ambiguity flagged in T5** (video-hash threading) — implementer picks the robust option; contract = `ValueError` on mismatch, covered by a test.
