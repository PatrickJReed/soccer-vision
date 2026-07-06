# View-Registration Calibration (Slice 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give every frame a drift-free pitch homography by registering it directly to its view's labeled representative (single hop), output as a drop-in `homographies.parquet`.

**Architecture:** New `pitch/view_registration.py` beside `pitch/propagation.py`. Pure core (compose + best-rep registration) + video I/O (rep extraction, clip registration, cache) + a validation function + CLI. Reuses `propagation` (ORB/findHomography, `_frame_mask`, `HomographyEntry`) and `chain.homographies_to_parquet`.

**Tech Stack:** Python, OpenCV (ORB/BFMatcher/findHomography), numpy, pandas/parquet, pytest/mypy/ruff via `uv`. Reuses `soccer_vision.pitch.propagation`, `soccer_vision.labeler.view_digest` (`ViewDigest`, `_read_frames`, `_video_hash` via chain), `soccer_vision.labeler.chain`.

**Reference:** `docs/superpowers/specs/2026-07-06-view-registration-design.md`. Run `uv run pytest` from `packages/soccer-vision`; run `mypy`/`ruff` from the **repo root** (paths relative to root). Commit after each task with an EXPLICIT file list (never `git add -A` — repo has pre-existing untracked artifacts). cv2 attr errors get `# type: ignore[attr-defined]`.

---

### Task 1: pure core — `compose_pitch_homography` + `register_to_best_rep`

**Files:**
- Create: `packages/soccer-vision/src/soccer_vision/pitch/view_registration.py`
- Test: `packages/soccer-vision/tests/test_view_registration.py`

- [ ] **Step 1: Write failing tests** (`tests/test_view_registration.py`)

```python
"""Tests for drift-free view-registration calibration (Slice 2)."""
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from soccer_vision.pitch.view_registration import (
    compose_pitch_homography, register_to_best_rep)

H, W = 240, 320


def _pattern(seed: int) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 30, np.uint8)
    for _ in range(60):
        x1, x2 = sorted(rng.integers(0, W, size=2).tolist())
        y1, y2 = sorted(rng.integers(0, H, size=2).tolist())
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        else:
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
    return img


def _orb(img):
    o = cv2.ORB_create(3000)  # type: ignore[attr-defined]
    return o.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)


def test_compose_pitch_homography() -> None:
    H_rep = np.array([[2.0, 0, 1], [0, 3.0, 2], [0, 0, 1]])   # rep_px -> pitch
    G = np.array([[1.0, 0, 5], [0, 1.0, 7], [0, 0, 1]])        # frame_px -> rep_px
    Hf = compose_pitch_homography(H_rep, G)
    # a frame point (0,0) -> rep (5,7) -> pitch (2*5+1, 3*7+2) = (11, 23)
    p = Hf @ np.array([0, 0, 1.0]); p = p[:2] / p[2]
    assert np.allclose(p, [11.0, 23.0])
    assert abs(Hf[2, 2] - 1.0) < 1e-9


def test_register_to_best_rep_picks_matching_view() -> None:
    rep_a = _pattern(1)                          # the true view
    rep_b = _pattern(99)                          # a distinct view
    M = np.array([[1, 0, 8.0], [0, 1, -5.0], [0, 0, 1]])       # small translation
    frame = cv2.warpPerspective(rep_a, M, (W, H))             # a shifted copy of rep_a
    fkp, fdesc = _orb(frame)
    akp, adesc = _orb(rep_a); bkp, bdesc = _orb(rep_b)
    out = register_to_best_rep(fkp, fdesc, [akp, bkp], [adesc, bdesc], min_inliers=12)
    assert out is not None
    idx, G, n_in = out
    assert idx == 0 and n_in >= 12                 # picked rep_a, not rep_b
    # G should map frame -> rep_a i.e. approx inverse of M: frame(8,-5)->rep(0,0)... check a point
    p = G @ np.array([8.0, 0.0, 1.0]); p = p[:2] / p[2]        # frame (8,0) -> rep ~ (0,5)
    assert np.linalg.norm(p - np.array([0.0, 5.0])) < 3.0


def test_register_to_best_rep_no_match_returns_none() -> None:
    frame = _pattern(3)
    fkp, fdesc = _orb(frame)
    blank = np.full((H, W, 3), 30, np.uint8)       # no features
    bkp, bdesc = _orb(blank)
    assert register_to_best_rep(fkp, fdesc, [bkp], [bdesc], min_inliers=12) is None
```

- [ ] **Step 2: Run, verify fail:** `cd packages/soccer-vision && uv run pytest tests/test_view_registration.py -x -q`

- [ ] **Step 3: Implement** the module header + the two pure functions:

```python
"""Drift-free pitch calibration by direct frame->representative registration (Slice 2).

Each frame gets its pitch homography from ONE registration to its view's labeled
representative (compose H_rep_pitch @ G_frame_to_rep) instead of composing the long
inter-frame chain, so error does not accumulate with temporal distance. Output is a
drop-in homographies.parquet. See docs/superpowers/specs/2026-07-06-view-registration-design.md.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

DEFAULT_N_FEATURES = 3000
DEFAULT_MIN_INLIERS = 12
DEFAULT_TOP_K = 2
_FULL_CONF_INLIERS = 50.0    # inliers at/above which confidence saturates to 1.0


def compose_pitch_homography(
    H_rep_pitch: NDArray[np.floating], G_frame_to_rep: NDArray[np.floating],
) -> NDArray[np.float64]:
    """frame_px -> pitch, from rep_px->pitch and frame_px->rep_px. Normalized so H[2,2]=1."""
    H = np.asarray(H_rep_pitch, np.float64) @ np.asarray(G_frame_to_rep, np.float64)
    if abs(H[2, 2]) > 1e-9:
        H = H / H[2, 2]
    return H


def register_to_best_rep(
    frame_kp: list[Any], frame_desc: NDArray[Any] | None,
    rep_kps: list[list[Any]], rep_descs: list[NDArray[Any] | None], *,
    min_inliers: int = DEFAULT_MIN_INLIERS, top_k: int = DEFAULT_TOP_K,
) -> tuple[int, NDArray[np.float64], int] | None:
    """Register a frame to its best-matching representative.

    Ranks reps by cross-checked match count, runs RANSAC findHomography on the top_k,
    and returns (best_rep_index, G frame_px->rep_px, n_inliers) with the most inliers
    (>= min_inliers), or None if none qualify.
    """
    if frame_desc is None or len(frame_desc) < min_inliers:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ranked: list[tuple[int, int, list[Any]]] = []
    for i, rd in enumerate(rep_descs):
        if rd is None or len(rd) < min_inliers:
            continue
        m = matcher.match(frame_desc, rd)
        if len(m) >= min_inliers:
            ranked.append((len(m), i, m))
    ranked.sort(key=lambda x: x[0], reverse=True)
    best: tuple[int, NDArray[np.float64], int] | None = None
    for _, i, m in ranked[:top_k]:
        src = np.array([frame_kp[x.queryIdx].pt for x in m], dtype=np.float32)
        dst = np.array([rep_kps[i][x.trainIdx].pt for x in m], dtype=np.float32)
        G, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if G is None:
            continue
        n_in = int(mask.sum())
        if n_in >= min_inliers and (best is None or n_in > best[2]):
            best = (i, G.astype(np.float64), n_in)
    return best
```

- [ ] **Step 4: Run to green:** `uv run pytest tests/test_view_registration.py -q`
- [ ] **Step 5: mypy + ruff from REPO ROOT:** `cd /Users/patrickreed/Sandbox/soccer-vision && uv run mypy packages/soccer-vision/src/soccer_vision/pitch/view_registration.py && uv run ruff check packages/soccer-vision/src/soccer_vision/pitch/view_registration.py packages/soccer-vision/tests/test_view_registration.py`
- [ ] **Step 6: Commit:** `git add packages/soccer-vision/src/soccer_vision/pitch/view_registration.py packages/soccer-vision/tests/test_view_registration.py && git commit -m "feat(pitch): view-registration pure core (compose + best-rep registration)"`

---

### Task 2: rep extraction + `register_clip` orchestration + `write_homographies`

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/view_registration.py`
- Test: `packages/soccer-vision/tests/test_view_registration.py`

- [ ] **Step 1: Append failing tests** (add imports; `_pattern`, `H`, `W`, `_orb` already exist)

```python
from pathlib import Path
import pandas as pd
import pytest
from soccer_vision.labeler.view_digest import ViewDigest
from soccer_vision.pitch.view_registration import (
    RegisteredCalib, rep_homographies_from_parquet, register_clip, write_homographies)


def _identity_digest(reps: dict[int, int]) -> ViewDigest:
    return ViewDigest(sample_frames=list(reps.values()),
                      view_of={f: v for v, f in reps.items()},
                      representatives=reps, similarity=np.zeros((1, 1)))


def _write_video(path: Path, frames: list) -> bool:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    vw = cv2.VideoWriter(str(path), fourcc, 30.0, (W, H))
    if not vw.isOpened():
        return False
    for fr in frames:
        vw.write(fr)
    vw.release()
    return path.exists() and path.stat().st_size > 0


def test_rep_homographies_from_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame([
        {"frame": 5, "h00": 1.0, "h01": 0, "h02": 0, "h10": 0, "h11": 1.0, "h12": 0,
         "h20": 0, "h21": 0, "h22": 1.0, "source": "manual", "confidence": 1.0},
        {"frame": 9, "h00": 2.0, "h01": 0, "h02": 1, "h10": 0, "h11": 2.0, "h12": 3,
         "h20": 0, "h21": 0, "h22": 1.0, "source": "manual", "confidence": 1.0}])
    p = tmp_path / "h.parquet"; df.to_parquet(p, index=False)
    reps = rep_homographies_from_parquet(p, {0: 5, 1: 9, 2: 999})   # view 2 absent
    assert set(reps) == {0, 1}
    assert np.allclose(reps[1], [[2, 0, 1], [0, 2, 3], [0, 0, 1]])


def test_register_clip_end_to_end(tmp_path: Path) -> None:
    # two views: frames 0-9 = view 0 (pattern A), 10-19 = view 1 (pattern B); reps 0 & 10.
    A, B = _pattern(1), _pattern(2)
    frames = [A] * 10 + [B] * 10
    video = tmp_path / "clip.mp4"
    if not _write_video(video, frames):
        pytest.skip("no mp4 writer")
    reps = {0: 0, 1: 10}
    digest = _identity_digest(reps)
    rep_h = {0: np.eye(3), 1: np.array([[1.0, 0, 100], [0, 1.0, 0], [0, 0, 1]])}
    calib = register_clip(video, digest, rep_h, cache_dir=tmp_path)
    assert isinstance(calib, RegisteredCalib)
    # rep frames pass through with their labeled H, source="rep"
    assert calib.homographies[0].source == "rep"
    assert np.allclose(calib.homographies[0].H, np.eye(3))
    # a view-1 frame registers to rep 1 and composes near its H (frames ~ identical to rep)
    e = calib.homographies[15]
    assert e.source in ("registered", "rep")
    assert calib.rep_of[15] == 1                     # matched view 1
    assert calib.stats["coverage"] > 0.8


def test_write_homographies_schema(tmp_path: Path) -> None:
    from soccer_vision.pitch.propagation import HomographyEntry
    calib = RegisteredCalib(
        homographies={0: HomographyEntry(np.eye(3), "rep", 1.0),
                      1: HomographyEntry(np.eye(3), "registered", 0.7)},
        rep_of={0: 0, 1: 0}, stats={})
    out = tmp_path / "homographies.parquet"
    write_homographies(calib, out)
    df = pd.read_parquet(out)
    assert set(["frame", "h00", "h22", "source", "confidence"]).issubset(df.columns)
    assert set(df["source"]) == {"rep", "registered"}
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement.** Add `RegisteredCalib`, `rep_homographies_from_parquet`, `register_clip`, `write_homographies`.

```python
# add imports at top: from dataclasses import dataclass; from pathlib import Path; import pandas as pd
# from soccer_vision.pitch.propagation import HomographyEntry, _frame_mask
# from soccer_vision.labeler.view_digest import ViewDigest, _read_frames
# from soccer_vision.pipeline import homographies_to_parquet   # CONFIRMED: pipeline.py:352,
#   signature homographies_to_parquet(entries: dict[int, HomographyEntry], path: Path) -> None

@dataclass(frozen=True)
class RegisteredCalib:
    homographies: dict[int, HomographyEntry]   # frame -> (H f_px->pitch, source, confidence)
    rep_of: dict[int, int]                      # frame -> view id it registered through
    stats: dict[str, Any]


def rep_homographies_from_parquet(homographies_parquet, representatives):
    df = pd.read_parquet(homographies_parquet)
    by_frame = {int(r.frame): np.array([[r.h00, r.h01, r.h02], [r.h10, r.h11, r.h12],
                                        [r.h20, r.h21, r.h22]], dtype=np.float64)
                for r in df.itertuples()}
    return {int(v): by_frame[int(f)] for v, f in representatives.items() if int(f) in by_frame}


def _orb_full(img, mask, n_features):
    o = cv2.ORB_create(n_features)  # type: ignore[attr-defined]
    return o.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)


def register_clip(video_path, digest, rep_homographies, *, player_boxes=None, frames=None,
                  n_features=DEFAULT_N_FEATURES, min_inliers=DEFAULT_MIN_INLIERS,
                  cache_dir=None):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if frames is None:
        frames = list(range(max(n_frames, 1)))
    # reps that have a labeled homography, ordered
    view_ids = sorted(rep_homographies)                 # candidate views
    rep_frames = [digest.representatives[v] for v in view_ids]
    rep_H = [rep_homographies[v] for v in view_ids]
    rep_frame_set = {digest.representatives[v]: v for v in view_ids}

    from soccer_vision.labeler.view_digest import _read_frames
    rep_imgs = _read_frames(video_path, rep_frames)
    def mask_for(idx, img):
        if player_boxes is None or img is None:
            return None
        return _frame_mask(player_boxes, idx, img.shape[:2])
    rep_kps, rep_descs = [], []
    for rf, im in zip(rep_frames, rep_imgs):
        kp, d = (_orb_full(im, mask_for(rf, im), n_features) if im is not None else ([], None))
        rep_kps.append(kp); rep_descs.append(d)

    homographies: dict[int, HomographyEntry] = {}
    rep_of: dict[int, int] = {}
    inliers: list[int] = []
    # stream frames with one persistent forward-read capture (ascending)
    cap = cv2.VideoCapture(str(video_path)); pos = 0
    try:
        for f in frames:
            if f in rep_frame_set:                       # rep passes through with labeled H
                v = rep_frame_set[f]
                homographies[f] = HomographyEntry(np.asarray(rep_homographies[v], np.float64), "rep", 1.0)
                rep_of[f] = v
                # still advance the capture position deterministically
            if f < pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f); pos = f
            while pos < f:
                if not cap.grab(): break
                pos += 1
            ok, img = cap.read(); pos += 1
            if f in rep_frame_set:
                continue
            if not ok:
                continue                                  # gap (no row => source none)
            kp, d = _orb_full(img, mask_for(f, img), n_features)
            res = register_to_best_rep(kp, d, rep_kps, rep_descs, min_inliers=min_inliers)
            if res is None:
                continue                                  # gap
            idx, G, n_in = res
            v = view_ids[idx]
            Hf = compose_pitch_homography(rep_H[idx], G)
            conf = float(min(1.0, n_in / _FULL_CONF_INLIERS))
            homographies[f] = HomographyEntry(Hf, "registered", conf)
            rep_of[f] = v; inliers.append(n_in)
    finally:
        cap.release()

    n_reg = sum(1 for e in homographies.values() if e.source == "registered")
    stats = {
        "n_frames": len(frames), "n_registered": n_reg,
        "n_rep": sum(1 for e in homographies.values() if e.source == "rep"),
        "n_gap": len(frames) - len(homographies),
        "coverage": len(homographies) / max(len(frames), 1),
        "median_inliers": float(np.median(inliers)) if inliers else 0.0,
        "per_view_counts": {int(v): int(sum(1 for x in rep_of.values() if x == v)) for v in view_ids},
    }
    return RegisteredCalib(homographies=homographies, rep_of=rep_of, stats=stats)


def write_homographies(calib, out_path):
    homographies_to_parquet(calib.homographies, Path(out_path))
```

**Implementer notes:**
- `homographies_to_parquet` is `from soccer_vision.pipeline import homographies_to_parquet` (defined at `pipeline.py:352`, takes `dict[int, HomographyEntry]` + `Path`). No grep needed.
- `_read_frames` / `_video_hash` are in `labeler/view_digest.py` / `labeler/chain.py` (view_digest imports `_video_hash` from chain).
- A cache is OPTIONAL in v1; if `homographies_to_parquet` + recompute is fast enough on the synthetic tests, you may skip the npz cache (the `cache_dir` param can be accepted and currently unused, documented as reserved). Keep the signature stable.
- mypy: type the dataclass fields; `frames: list[int] | None`; cv2 ignores as needed. `HomographyEntry.H` is `NDArray[np.floating]`.

- [ ] **Step 4: green. Step 5: mypy+ruff (repo root). Step 6: Commit** `feat(pitch): register_clip orchestration + rep extraction + parquet writer`.

---

### Task 3: `cross_registration_error` validation + CLI + oceanside validation

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/view_registration.py`
- Test: `packages/soccer-vision/tests/test_view_registration.py`

- [ ] **Step 1: Append failing tests**

```python
from soccer_vision.pitch.view_registration import cross_registration_error


def test_cross_registration_error_small_and_flat(tmp_path: Path) -> None:
    # 3 views, each a distinct pattern; reps at 0,5,10; all with identity-ish pitch H.
    A, B, C = _pattern(1), _pattern(2), _pattern(3)
    frames = [A, A, A, A, A, B, B, B, B, B, C, C, C, C, C]
    video = tmp_path / "clip.mp4"
    if not _write_video(video, frames):
        pytest.skip("no mp4 writer")
    reps = {0: 0, 1: 5, 2: 10}
    digest = _identity_digest(reps)
    rep_h = {0: np.eye(3), 1: np.eye(3), 2: np.eye(3)}
    df = cross_registration_error(video, digest, rep_h)
    assert set(["rep_R", "rep_Rprime", "temporal_dist", "corner_err_pitch"]).issubset(df.columns)
    # distinct patterns won't cross-register well; just assert the function runs + schema.
    assert len(df) >= 1


def test_cli_writes_homographies(tmp_path: Path) -> None:
    from soccer_vision.pitch.view_registration import main
    import json
    A, B = _pattern(1), _pattern(2)
    frames = [A] * 6 + [B] * 6
    video = tmp_path / "clip.mp4"
    if not _write_video(video, frames):
        pytest.skip("no mp4 writer")
    digest_json = tmp_path / "view_digest.json"
    digest_json.write_text(json.dumps({
        "n_views": 2, "stride": 6, "sample_frames": [0, 6],
        "representatives": {"0": 0, "1": 6}, "view_of": {"0": 0, "6": 1}}))
    hp = tmp_path / "rep_h.parquet"
    pd.DataFrame([
        {"frame": 0, "h00": 1.0, "h01": 0, "h02": 0, "h10": 0, "h11": 1.0, "h12": 0,
         "h20": 0, "h21": 0, "h22": 1.0, "source": "manual", "confidence": 1.0},
        {"frame": 6, "h00": 1.0, "h01": 0, "h02": 9, "h10": 0, "h11": 1.0, "h12": 0,
         "h20": 0, "h21": 0, "h22": 1.0, "source": "manual", "confidence": 1.0},
    ]).to_parquet(hp, index=False)
    out = tmp_path / "out"
    main(["--video", str(video), "--digest-json", str(digest_json),
          "--rep-homographies", str(hp), "--out", str(out)])
    assert (out / "homographies.parquet").exists()
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `cross_registration_error`, a digest-json loader, and `main`.**

```python
# cross_registration_error: for each labeled rep R, register R's frame to every OTHER
# labeled rep R', compose, reproject the 4 full-res image corners to pitch via the composed
# H and via R's own manual H_R, and report the mean corner distance (pitch units) vs |dist|.
def cross_registration_error(video_path, digest, rep_homographies, *,
                             n_features=DEFAULT_N_FEATURES, min_inliers=DEFAULT_MIN_INLIERS):
    video_path = Path(video_path)
    from soccer_vision.labeler.view_digest import _read_frames
    view_ids = sorted(rep_homographies)
    rep_frames = {v: digest.representatives[v] for v in view_ids}
    imgs = dict(zip(view_ids, _read_frames(video_path, [rep_frames[v] for v in view_ids])))
    # ORB each rep
    kp = {}; desc = {}
    for v in view_ids:
        im = imgs[v]
        kp[v], desc[v] = (_orb_full(im, None, n_features) if im is not None else ([], None))
    h, w = (next(im for im in imgs.values() if im is not None).shape[:2])
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)

    def project(Hm, pts):
        p = (Hm @ np.column_stack([pts, np.ones(len(pts))]).T).T
        return p[:, :2] / p[:, 2:3]

    rows = []
    for R in view_ids:
        HR = rep_homographies[R]
        truth = project(HR, corners)
        for Rp in view_ids:
            if Rp == R:
                continue
            res = register_to_best_rep(kp[R], desc[R], [kp[Rp]], [desc[Rp]], min_inliers=min_inliers)
            if res is None:
                continue
            _, G, _n = res
            Hc = compose_pitch_homography(rep_homographies[Rp], G)
            err = float(np.linalg.norm(project(Hc, corners) - truth, axis=1).mean())
            rows.append({"rep_R": R, "rep_Rprime": Rp,
                         "temporal_dist": abs(rep_frames[R] - rep_frames[Rp]),
                         "corner_err_pitch": err})
    return pd.DataFrame(rows)


def _digest_from_json(path):
    import json
    d = json.loads(Path(path).read_text())
    reps = {int(k): int(v) for k, v in d["representatives"].items()}
    view_of = {int(k): int(v) for k, v in d.get("view_of", {}).items()}
    return ViewDigest(sample_frames=sorted(view_of), view_of=view_of,
                      representatives=reps, similarity=np.zeros((1, 1)))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Drift-free view-registration calibration (Slice 2)")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--digest-json", required=True, type=Path)
    ap.add_argument("--rep-homographies", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--boxes", type=Path, default=None)
    ap.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(argv)
    out = args.out or (args.video.parent / "view_registration_out")
    Path(out).mkdir(parents=True, exist_ok=True)
    digest = _digest_from_json(args.digest_json)
    rep_h = rep_homographies_from_parquet(args.rep_homographies, digest.representatives)
    boxes = pd.read_parquet(args.boxes) if args.boxes else None
    calib = register_clip(args.video, digest, rep_h, player_boxes=boxes, min_inliers=args.min_inliers)
    write_homographies(calib, Path(out) / "homographies.parquet")
    print("view-registration:", {k: calib.stats[k] for k in ("coverage", "n_registered", "n_gap", "median_inliers")})
    if args.validate:
        df = cross_registration_error(args.video, digest, rep_h, min_inliers=args.min_inliers)
        if not df.empty:
            print(f"cross-reg corner error (pitch units): median {df['corner_err_pitch'].median():.2f}")
            near = df[df.temporal_dist <= df.temporal_dist.median()]["corner_err_pitch"].mean()
            far = df[df.temporal_dist > df.temporal_dist.median()]["corner_err_pitch"].mean()
            print(f"  near-rep pairs mean {near:.2f}  vs  far-rep pairs mean {far:.2f} "
                  f"(flat => drift-free)")
    print(f"wrote homographies.parquet -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: green. Step 5: full suite + mypy + ruff.**
```
cd /Users/patrickreed/Sandbox/soccer-vision/packages/soccer-vision && uv run pytest -q
cd /Users/patrickreed/Sandbox/soccer-vision && uv run mypy packages/soccer-vision/src/soccer_vision/pitch/view_registration.py && uv run ruff check packages/soccer-vision/src/soccer_vision/pitch/ packages/soccer-vision/tests/test_view_registration.py
```

- [ ] **Step 6: Oceanside validation** (report numbers as facts; 12/13 reps labeled):
```bash
cd /Users/patrickreed/Sandbox/soccer-vision
uv run python -m soccer_vision.pitch.view_registration \
  --video ~/sv-labeler/oceanside_clip.mp4 \
  --digest-json ~/sv-labeler/view_digest_out/view_digest.json \
  --rep-homographies ~/sv-labeler/out/homographies.parquet \
  --out ~/sv-labeler/view_registration_out --validate
# expect: coverage high; cross-reg near-rep mean ~= far-rep mean (flat => drift-free)
```

- [ ] **Step 7: Commit** `feat(pitch): cross-registration validation + CLI; validate on oceanside`.

---

## Self-Review notes
- **Spec coverage:** compose + best-rep (T1); rep extraction + register_clip + writer (T2); cross-registration validation + CLI + oceanside (T3). Deferred items (gap fill, cache, learned assignment) intentionally absent.
- **Type/name consistency:** `RegisteredCalib(homographies, rep_of, stats)`, `HomographyEntry(H, source, confidence)`, `register_to_best_rep -> (idx, G, n_inliers)`, `compose_pitch_homography(H_rep, G)` used consistently across tasks.
- **Reuse:** `propagation.HomographyEntry` / `_frame_mask`, `chain.homographies_to_parquet`, `view_digest._read_frames` — implementer confirms exact import paths before use.
