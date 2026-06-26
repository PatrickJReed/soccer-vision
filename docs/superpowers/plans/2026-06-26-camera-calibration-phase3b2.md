# Camera Calibration Phase 3b-2 — Line-Click UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add line-clicking end-to-end (arm a line → click along it → constraint stored, chain-propagated, and fed to `refine_pose` on affected frames) to tighten the near touchline / midline where points can't reach.

**Architecture:** A `LineClick` model + `propagate_line_clicks` (chain-maps line clicks like points, but all-in-window contribute) build per-frame `line_obs`, which `LabelerState` passes into 3b-1's `poses_by_click_propagation` (it already runs `refine_pose` on frames that have line constraints). The server gains a line-click endpoint and the frontend a LINES palette with distinct drawing. Lines don't touch the focal/bootstrap/outlier preprocessing.

**Tech Stack:** Python, NumPy, OpenCV/scipy (inside `refine_pose`), stdlib HTTP, vanilla canvas JS, pytest, mypy (strict), ruff.

**Spec:** `docs/superpowers/specs/2026-06-26-camera-calibration-phase3b2-design.md`

**Conventions (from 3b-1):**
- `Click(frame, kp_idx, x, y)` (normalized x,y) and `propagate_clicks(clicks, transforms, segment_of, *, window, frames=None)` live in `pitch/manual_anchor.py`.
- `poses_by_click_propagation(clicks, transforms, segment_of, k, size, *, window, min_points=4, line_obs=None, frames=None)` (3b-1) already runs `refine_pose` per frame when `line_obs.get(f)` is non-empty; `line_obs` is PIXEL-space `{frame: [(line_id, x_px, y_px)]}`.
- `LabelerState` (3b-1) holds `clicks`, `_K`, `_calibrated`, `_outliers`, `_fits: dict[int, CalibFrame]`, `_transforms`, `_segment_of`; `_refit(frames)` / `_affected(frame)` (windowed); `_recompute_all`; `add_click`/`add_clicks`/`remove_last`/`recalibrate`; `coverage`/`status_list`/`frame_homography`/`export`; `_autosave` writes a JSON sidecar (currently a bare list of point dicts); `clicks_from_sidecar`/`clicks_from_keypoints_parquet`.
- The 5 lines: `from soccer_vision.calib.field_model import FIELD_LINES` (keys: `near_touchline, far_touchline, own_goal_line, opp_goal_line, midline`).
- Lint gate: **`uv run mypy` from repo root**; `uv run ruff check` + `uv run pytest` from `packages/soccer-vision`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py` | clicks + propagation | Add `LineClick`, `propagate_line_clicks` |
| `packages/soccer-vision/src/soccer_vision/labeler/state.py` | session | line_clicks, `add_line_click`, line_obs in refit, unified undo, persistence |
| `packages/soccer-vision/src/soccer_vision/labeler/server.py` | HTTP | `/api/line_click`, `/api/clicks` returns both, line names in state, resume |
| `packages/soccer-vision/src/soccer_vision/labeler/static/{index.html,app.js}` | UI | LINES palette, arming, place/draw/undo line clicks |
| `packages/soccer-vision/tests/test_pitch_manual_anchor.py` | propagation tests | Add `propagate_line_clicks` tests (or the calib_anchor test file) |
| `packages/soccer-vision/tests/test_labeler_state.py` | state tests | Add line-click tests |
| `packages/soccer-vision/tests/test_labeler_server.py` | server tests | Add line endpoint tests |

---

## Task 1: `LineClick` + `propagate_line_clicks`

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py`
- Test: `packages/soccer-vision/tests/test_pitch_manual_anchor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pitch_manual_anchor.py` (it already imports from `manual_anchor`; add `LineClick`, `propagate_line_clicks`, `build_segments`, `cumulative_transforms` to the imports if not present):

```python
import numpy as np
from soccer_vision.pitch.manual_anchor import (
    LineClick, build_segments, cumulative_transforms, propagate_line_clicks,
)


def test_propagate_line_clicks_carries_along_identity_chain() -> None:
    interframe = {i: np.eye(3) for i in range(5)}  # frames 0..5 linked
    seg = build_segments(interframe, 6)
    transforms = cumulative_transforms(interframe, seg)
    lcs = [LineClick(frame=0, line_id="midline", x=0.4, y=0.6)]
    prop = propagate_line_clicks(lcs, transforms, seg, window=10)
    assert prop[2] == [("midline", 0.4, 0.6)]   # unchanged under identity
    assert prop[5] == [("midline", 0.4, 0.6)]
    small = propagate_line_clicks(lcs, transforms, seg, window=1)
    assert 5 not in small                        # outside the window


def test_propagate_line_clicks_emits_all_in_window() -> None:
    # two clicks on the same line -> BOTH propagate into a frame (not nearest-wins)
    interframe = {i: np.eye(3) for i in range(5)}
    seg = build_segments(interframe, 6)
    transforms = cumulative_transforms(interframe, seg)
    lcs = [LineClick(0, "near_touchline", 0.1, 0.9),
           LineClick(3, "near_touchline", 0.2, 0.95)]
    prop = propagate_line_clicks(lcs, transforms, seg, window=10, frames=[3])
    assert set(prop) == {3}                                  # frames= restricts
    assert ("near_touchline", 0.1, 0.9) in prop[3]
    assert ("near_touchline", 0.2, 0.95) in prop[3]
    assert len(prop[3]) == 2


def test_propagate_line_clicks_respects_segments() -> None:
    interframe = {0: np.eye(3), 1: np.eye(3), 3: np.eye(3)}   # link missing at 2
    seg = build_segments(interframe, 5)
    transforms = cumulative_transforms(interframe, seg)
    lcs = [LineClick(0, "midline", 0.5, 0.5)]
    prop = propagate_line_clicks(lcs, transforms, seg, window=10)
    assert 1 in prop                                         # same segment
    assert 4 not in prop                                     # other segment
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -k propagate_line -v`
Expected: FAIL with `ImportError: cannot import name 'LineClick'`.

- [ ] **Step 3: Implement**

In `pitch/manual_anchor.py`, add `LineClick` next to `Click` (find the `@dataclass(frozen=True)\nclass Click` block and add after it):

```python
@dataclass(frozen=True)
class LineClick:
    """One line observation: pixel (x, y) asserted to lie on field line `line_id`."""

    frame: int
    line_id: str
    x: float
    y: float
```

And add `propagate_line_clicks` after `propagate_clicks`:

```python
def propagate_line_clicks(
    line_clicks: Sequence[LineClick],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    *,
    window: int,
    frames: Sequence[int] | None = None,
) -> dict[int, list[tuple[str, float, float]]]:
    """For each target frame, EVERY in-window same-segment line click chain-mapped in.

    A line click is a pixel on a known line; the chain maps it to a pixel still on
    that line in the target frame (the line is rigid; the camera has no parallax).
    Unlike `propagate_clicks` (nearest-wins per landmark), all in-window line clicks
    contribute — each is an independent constraint. Returns {frame: [(line_id, x, y)]}
    in normalized image space (same as the clicks/transforms).
    """
    out: dict[int, list[tuple[str, float, float]]] = {}
    if not line_clicks or not transforms:
        return out
    if frames is None:
        target = sorted(transforms)
    else:
        target = sorted(set(transforms) & {int(f) for f in frames})
    if not target:
        return out
    m_inv = {g: np.linalg.inv(np.asarray(transforms[g], dtype=np.float64)) for g in target}
    seg_t = {g: segment_of.get(g) for g in target}
    for lc in line_clicks:
        src_m = transforms.get(lc.frame)
        seg = segment_of.get(lc.frame)
        if src_m is None or seg is None:
            continue
        ref = np.asarray(src_m, dtype=np.float64) @ np.array([lc.x, lc.y, 1.0])
        for g in target:
            if seg_t[g] != seg or abs(g - lc.frame) > window:
                continue
            dst = m_inv[g] @ ref
            out.setdefault(g, []).append(
                (lc.line_id, float(dst[0] / dst[2]), float(dst[1] / dst[2])))
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd packages/soccer-vision && uv run pytest tests/test_pitch_manual_anchor.py -k propagate_line -v`
Expected: PASS (3). Then the full file: `uv run pytest tests/test_pitch_manual_anchor.py -q` (no regression).

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/pitch/manual_anchor.py tests/test_pitch_manual_anchor.py`
Then from REPO ROOT: `uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py packages/soccer-vision/tests/test_pitch_manual_anchor.py
git commit -m "feat(pitch): LineClick + propagate_line_clicks (all-in-window chain propagation)"
```

---

## Task 2: `LabelerState` line-click integration + persistence

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py`
- Test: `packages/soccer-vision/tests/test_labeler_state.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_labeler_state.py` (reuse the existing `_pan_session`/`_K`/`_look_at` helpers and `LabelerState` import; add `LineClick` to the manual_anchor import):

```python
from soccer_vision.pitch.manual_anchor import LineClick


def test_labeler_add_line_click_refits_and_persists(tmp_path) -> None:
    interframe, _poses, clicks = _pan_session(9)
    sidecar = tmp_path / "s.json"
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080),
                      window=360, autosave_path=sidecar)
    st.add_clicks(clicks)                     # bootstrap on points
    cf_before = st.frame_homography(4)
    st.add_line_click(4, "midline", 0.5, 0.5)
    assert len(st.line_clicks) == 1
    assert st.frame_homography(4) is not None  # still covered (refine ran)
    # sidecar carries both
    import json
    data = json.loads(sidecar.read_text())
    assert data["line_clicks"] == [{"frame": 4, "line_id": "midline", "x": 0.5, "y": 0.5}]
    assert len(data["clicks"]) == len(clicks)


def test_labeler_remove_last_pops_line_then_point() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    st.add_clicks(clicks)
    st.add_line_click(4, "near_touchline", 0.1, 0.9)
    n_pts = len(st.clicks)
    st.remove_last()                          # pops the line click (added last)
    assert len(st.line_clicks) == 0 and len(st.clicks) == n_pts
    st.remove_last()                          # now pops a point
    assert len(st.clicks) == n_pts - 1


def test_labeler_export_writes_line_clicks_parquet(tmp_path) -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)
    st.add_clicks(clicks)
    st.add_line_click(4, "midline", 0.5, 0.5)
    st.export(tmp_path)
    import pandas as pd
    df = pd.read_parquet(tmp_path / "line_clicks.parquet")
    assert list(df.columns) == ["frame", "line_id", "x_px", "y_px"]
    assert df.iloc[0]["line_id"] == "midline"
    assert abs(df.iloc[0]["x_px"] - 0.5 * 1920) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -k "line" -v`
Expected: FAIL (`add_line_click` missing).

- [ ] **Step 3: Implement the state changes**

In `labeler/state.py`:

(a) Imports — add `LineClick`, `propagate_line_clicks` to the `manual_anchor` import; add `pd` if not present (it is). 

(b) In `__init__`, after `self.clicks: list[Click] = []`, add:
```python
        self.line_clicks: list[LineClick] = []
        self._seq: list[str] = []  # insertion order across clicks ("pt") + line_clicks ("ln")
```

(c) Build line_obs in the engine calls — replace the body of `_refit` and `_recompute_all` so they pass `line_obs`:
```python
    def _line_obs(self, frames: Sequence[int] | None) -> dict[int, list[tuple[str, float, float]]]:
        w, h = self.size
        prop = propagate_line_clicks(
            self.line_clicks, self._transforms, self._segment_of,
            window=self.window, frames=frames)
        return {f: [(lid, x * w, y * h) for (lid, x, y) in lst] for f, lst in prop.items()}

    def _refit(self, frames: list[int]) -> None:
        if not self._calibrated or self._K is None:
            for f in frames:
                self._fits.pop(f, None)
            return
        sub = poses_by_click_propagation(
            self._active_clicks(), self._transforms, self._segment_of, self._K,
            self.size, window=self.window, frames=frames, line_obs=self._line_obs(frames))
        for f in frames:
            if f in sub:
                self._fits[f] = self._calib_frame(sub[f])
            else:
                self._fits.pop(f, None)

    def _recompute_all(self, chunk: int = 5000) -> None:
        self._fits = {}
        if not self._calibrated or self._K is None:
            return
        allf = sorted(self._transforms)
        active = self._active_clicks()
        for i in range(0, len(allf), chunk):
            part = allf[i:i + chunk]
            sub = poses_by_click_propagation(
                active, self._transforms, self._segment_of, self._K, self.size,
                window=self.window, frames=part, line_obs=self._line_obs(part))
            for f, pose in sub.items():
                self._fits[f] = self._calib_frame(pose)
```

(d) Track order in the point mutators + add the line mutators. Update `add_click`, `add_clicks` to append to `_seq`, and add `add_line_click` / `add_line_clicks`:
```python
    def add_click(self, frame: int, kp_idx: int, x: float, y: float) -> None:
        self.clicks.append(Click(frame=frame, kp_idx=kp_idx, x=x, y=y))
        self._seq.append("pt")
        if not self._calibrated and self._try_bootstrap():
            self._recompute_all()
        elif self._calibrated:
            self._refit(self._affected(frame))
        self._autosave()

    def add_clicks(self, clicks: Sequence[Click], *, chunk: int = 5000) -> None:
        self.clicks.extend(clicks)
        self._seq.extend("pt" for _ in clicks)
        self._try_bootstrap()
        self._recompute_all(chunk=chunk)
        self._autosave()

    def add_line_click(self, frame: int, line_id: str, x: float, y: float) -> None:
        self.line_clicks.append(LineClick(frame=frame, line_id=line_id, x=x, y=y))
        self._seq.append("ln")
        if self._calibrated:
            self._refit(self._affected(frame))
        self._autosave()

    def add_line_clicks(self, line_clicks: Sequence[LineClick], *, chunk: int = 5000) -> None:
        self.line_clicks.extend(line_clicks)
        self._seq.extend("ln" for _ in line_clicks)
        self._recompute_all(chunk=chunk)
        self._autosave()
```

(e) Unified `remove_last`:
```python
    def remove_last(self) -> None:
        # The frozen focal stays valid after an undo (constant lens); recalibrate() refreshes.
        if not self._seq:
            return
        kind = self._seq.pop()
        if kind == "ln" and self.line_clicks:
            removed_frame = self.line_clicks.pop().frame
        elif self.clicks:
            removed_frame = self.clicks.pop().frame
        else:
            return
        if self._calibrated:
            self._refit(self._affected(removed_frame))
        self._autosave()
```

(f) Persistence — `_autosave` writes a dict with both arrays; `export` adds `line_clicks.parquet`. Replace `_autosave`:
```python
    def _autosave(self) -> None:
        """Atomically persist normalized clicks + line clicks to the sidecar."""
        if self.autosave_path is None:
            return
        self.autosave_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "clicks": [{"frame": c.frame, "kp_idx": c.kp_idx, "x": c.x, "y": c.y}
                       for c in self.clicks],
            "line_clicks": [{"frame": lc.frame, "line_id": lc.line_id, "x": lc.x, "y": lc.y}
                            for lc in self.line_clicks],
        }
        tmp = self.autosave_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.autosave_path)
```
And in `export`, after the homographies write, add the line-clicks parquet:
```python
        if self.line_clicks:
            pd.DataFrame(
                [{"frame": lc.frame, "line_id": lc.line_id, "x_px": lc.x * w, "y_px": lc.y * h}
                 for lc in self.line_clicks],
                columns=["frame", "line_id", "x_px", "y_px"],
            ).to_parquet(out / "line_clicks.parquet", index=False)
```
(Place this inside `export`, after the homographies write; reuse the existing `w, h =
self.size` already in `export`. `pd` is already imported at the top of `state.py` — do
NOT re-import it.)

(g) Update the module-level loaders for back-compat + the new line loaders (replace `clicks_from_sidecar`, add two functions):
```python
def clicks_from_sidecar(path: Path) -> list[Click]:
    """Load the autosave sidecar's POINT clicks (handles the old bare-list format)."""
    data = json.loads(Path(path).read_text())
    rows = data["clicks"] if isinstance(data, dict) else data
    return [Click(frame=int(d["frame"]), kp_idx=int(d["kp_idx"]),
                  x=float(d["x"]), y=float(d["y"])) for d in rows]


def line_clicks_from_sidecar(path: Path) -> list[LineClick]:
    """Load the autosave sidecar's LINE clicks ([] for the old bare-list format)."""
    data = json.loads(Path(path).read_text())
    rows = data.get("line_clicks", []) if isinstance(data, dict) else []
    return [LineClick(frame=int(d["frame"]), line_id=str(d["line_id"]),
                      x=float(d["x"]), y=float(d["y"])) for d in rows]


def line_clicks_from_parquet(path: Path, size: tuple[int, int]) -> list[LineClick]:
    """Load an exported line_clicks.parquet (full-pixel) back into normalized LineClicks."""
    df = pd.read_parquet(path)
    w, h = size
    return [LineClick(frame=int(f), line_id=str(lid), x=float(x) / w, y=float(y) / h)
            for f, lid, x, y in zip(df["frame"].to_numpy(), df["line_id"].to_numpy(),
                                    df["x_px"].to_numpy(), df["y_px"].to_numpy(), strict=True)]
```

- [ ] **Step 4: Run to verify pass + no regression**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_state.py -v`
Expected: new line tests PASS; existing tests still pass.
Run: `cd packages/soccer-vision && uv run pytest -k labeler -q`
Expected: all pass.

- [ ] **Step 5: Lint + typecheck**

Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/labeler/state.py tests/test_labeler_state.py`
Then from REPO ROOT: `uv run mypy`
Expected: clean. (If ruff flags the inline `import pandas as pd` in `export`, hoist `import pandas as pd` to the top of the file instead.)

- [ ] **Step 6: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/state.py packages/soccer-vision/tests/test_labeler_state.py
git commit -m "feat(labeler): line clicks in LabelerState (line_obs refit, unified undo, persistence)"
```

---

## Task 3: Server — line endpoint + line state + resume

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/server.py`
- Test: `packages/soccer-vision/tests/test_labeler_server.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_labeler_server.py` (it already has a calibrated-state fixture — reuse it; if it builds a `LabelerState`, add a line click via the handler logic). Add:

```python
def test_make_handler_accepts_line_names_and_state_exposes_line_clicks() -> None:
    from soccer_vision.calib.field_model import FIELD_LINES
    from soccer_vision.labeler.state import LabelerState
    # self-contained: an empty chain is fine — add_line_click on an uncalibrated state
    # just stores the click (no refit), which is what we assert.
    st = LabelerState(interframe={}, n_frames=5, size=(1920, 1080), window=360)
    st.add_line_click(2, "midline", 0.5, 0.5)
    assert st.line_clicks[0].line_id == "midline"
    handler_cls = make_handler(st, lambda i: b"", ["kp%d" % i for i in range(21)],
                               line_names=sorted(FIELD_LINES))
    assert handler_cls is not None  # make_handler accepts the line_names kwarg
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_server.py -k line_names -v`
Expected: FAIL (`make_handler` has no `line_names` param).

- [ ] **Step 3: Implement**

In `server.py`:

(a) Add a `line_names` param to `make_handler` (after `landmark_names`):
```python
def make_handler(
    state: LabelerState,
    frame_jpeg: Callable[[int], bytes],
    landmark_names: list[str],
    *,
    landmark_xy: list[list[float]] | None = None,
    line_names: list[str] | None = None,
    export_dir: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
```
and inside, `lines: list[str] = line_names or []`, and add `"line_names": lines` to the `_state_payload()` dict.

(b) `/api/clicks` GET — return both lists:
```python
            elif path == "/api/clicks":
                self._json({
                    "clicks": [{"frame": c.frame, "kp_idx": c.kp_idx, "x": c.x, "y": c.y}
                               for c in state.clicks],
                    "line_clicks": [{"frame": lc.frame, "line_id": lc.line_id,
                                     "x": lc.x, "y": lc.y} for lc in state.line_clicks],
                })
```

(c) `do_POST` — add the line-click endpoint (validate the line id):
```python
            elif self.path == "/api/line_click":
                from soccer_vision.calib.field_model import FIELD_LINES
                line_id = str(payload["line_id"])
                if line_id not in FIELD_LINES:
                    self._json({"error": f"unknown line_id {line_id!r}"}, code=400)
                    return
                state.add_line_click(int(payload["frame"]), line_id,
                                     float(payload["x"]), float(payload["y"]))
                self._json(self._state_payload())
```

(d) In `run()`, pass `line_names` to `make_handler` and load line clicks on resume:
```python
    from soccer_vision.calib.field_model import FIELD_LINES
    from soccer_vision.labeler.state import (
        clicks_from_keypoints_parquet, clicks_from_sidecar,
        line_clicks_from_parquet, line_clicks_from_sidecar,
    )
```
After the existing resume/restore block (where it loads point clicks), add line loading:
```python
    if resume is not None:
        lc_path = Path(resume).parent / "line_clicks.parquet"
        if lc_path.exists():
            state.add_line_clicks(line_clicks_from_parquet(lc_path, size))
    elif sidecar.exists():
        state.add_line_clicks(line_clicks_from_sidecar(sidecar))
```
(Place this right after the corresponding point-clicks resume/restore lines; the `clicks_from_sidecar` now handles the dict format, so the existing point-restore still works.)
And in the `make_handler(...)` call add `line_names=sorted(FIELD_LINES)`.

- [ ] **Step 4: Run + lint + typecheck**

Run: `cd packages/soccer-vision && uv run pytest tests/test_labeler_server.py -v` (pass)
Run: `cd packages/soccer-vision && uv run ruff check src/soccer_vision/labeler/server.py tests/test_labeler_server.py`
Then from REPO ROOT: `uv run mypy`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add packages/soccer-vision/src/soccer_vision/labeler/server.py packages/soccer-vision/tests/test_labeler_server.py
git commit -m "feat(labeler): server line-click endpoint + line names + line-click resume"
```

---

## Task 4: Frontend — LINES palette, place/draw/undo line clicks

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/static/index.html`
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/static/app.js`

This task is verified manually (the canvas app has no JS unit harness — the server tests cover the API). Make the edits, then do the manual check in Step 4.

- [ ] **Step 1: `app.js` — state + arming**

In `app.js`, add line state near the top globals (after the `let ... clicks = []` line):
```javascript
let LINE_NAMES = []; let armedLine = null; let lineClicks = [];
const LINE_COLORS = {near_touchline:"#ff5ca8", far_touchline:"#5cc8ff",
  own_goal_line:"#ffd95c", opp_goal_line:"#b07cff", midline:"#5cffa8"};
```
In `applyState(st)`, capture the line names: add `LINE_NAMES = st.line_names || [];` and call a `renderPalette()` (already called).

- [ ] **Step 2: `app.js` — palette LINES section + place line clicks**

In `renderPalette()`, after the landmark loop, append a LINES section:
```javascript
  const lh=document.createElement("h3");
  lh.style.cssText="font-size:12px;color:#9aa4b2;margin-top:10px"; lh.textContent="LINES";
  p.appendChild(lh);
  for(const name of LINE_NAMES){
    const d=document.createElement("div");
    d.className="kp"+(name===armedLine?" armed":"");
    d.textContent=name; d.style.color=LINE_COLORS[name]||"#dfe7ee";
    d.onclick=()=>{armedLine=name; armed=-1; renderPalette();};  // arming a line clears the point
    p.appendChild(d);
  }
```
And guard the point palette's arm so arming a point clears the line: in the landmark `d.onclick`, change to `()=>{armed=i; armedLine=null; renderPalette();}`.

Update `canvas.onclick` to branch on what's armed:
```javascript
canvas.onclick = async (e) => {
  if (didDrag) { didDrag = false; return; }
  const [x, y] = canvasNorm(e);
  if (armedLine) {
    lineClicks.push({ frame: cur, line_id: armedLine, x, y });
    applyState(await postJSON("/api/line_click", { frame: cur, line_id: armedLine, x, y }));
  } else {
    clicks.push({ frame: cur, kp_idx: armed, x, y }); placed.add(armed);
    applyState(await postJSON("/api/click", { frame: cur, kp_idx: armed, x, y }));
  }
  const fh = await api(`/api/frame_h/${cur}`); curH = fh.h; drawFrame();
};
```

- [ ] **Step 2b: `app.js` — draw line clicks (diamonds)**

In `drawFrame()`, after the point-dots loop, draw the current frame's line clicks as diamonds:
```javascript
  for(const lc of lineClicks) if(lc.line_id && lc.frame===cur){
    const cx=lc.x*canvas.width, cy=lc.y*canvas.height, r=6;
    ctx.fillStyle=LINE_COLORS[lc.line_id]||"#5cffa8";
    ctx.beginPath();
    ctx.moveTo(cx,cy-r); ctx.lineTo(cx+r,cy); ctx.lineTo(cx,cy+r); ctx.lineTo(cx-r,cy);
    ctx.closePath(); ctx.fill();
  }
```

- [ ] **Step 3: `app.js` — undo + load both**

In the `undo` button handler, also pop the local line list (the server's `/api/undo` already removes whichever was last; the client mirrors by reloading both):
```javascript
document.getElementById("undo").onclick=async()=>{
  applyState(await postJSON("/api/undo",{}));
  const cl=await api("/api/clicks"); clicks=cl.clicks; lineClicks=cl.line_clicks||[];
  placed=new Set(clicks.map(c=>c.kp_idx));
  const fh=await api(`/api/frame_h/${cur}`); curH=fh.h; drawFrame();
};
```
And in the initial load IIFE at the bottom, capture line clicks:
```javascript
  const cl = await api("/api/clicks");
  clicks = cl.clicks; lineClicks = cl.line_clicks || [];
  placed = new Set(clicks.map(c=>c.kp_idx));
```
(No `index.html` change is required — the palette renders the LINES section dynamically. Leave `index.html` as is.)

- [ ] **Step 4: Manual verification**

Run the labeler on the bake-off clip (or any local video):
```bash
cd /Users/patrickreed/Sandbox/soccer-vision/packages/soccer-vision
uv run python -m soccer_vision.labeler --video ../../data/bakeoff_clip.mp4
```
Open `http://127.0.0.1:8000`. Verify: (a) a LINES section appears in the palette with 5 colored entries; (b) arming a line then clicking the canvas drops a colored diamond and the coverage/overlay update; (c) the diamond persists on that frame and undo removes it; (d) re-opening the page restores both points and line diamonds. Report what you observe (do not assess overlay accuracy — that's Patrick's call).

- [ ] **Step 5: Lint (JS is not linted; just sanity-check) + commit**

```bash
cd /Users/patrickreed/Sandbox/soccer-vision
git add packages/soccer-vision/src/soccer_vision/labeler/static/app.js packages/soccer-vision/src/soccer_vision/labeler/static/index.html
git commit -m "feat(labeler): LINES palette + place/draw/undo line clicks (frontend)"
```

---

## Done criteria
- `LineClick` + `propagate_line_clicks` (all-in-window chain propagation), tested.
- `LabelerState`: `add_line_click`, line_obs in `_refit`/`_recompute_all` (→ `refine_pose` on line frames), unified `remove_last`, sidecar + `line_clicks.parquet` persistence + resume, tested.
- Server: `/api/line_click` (validated), `/api/clicks` returns both, line names in state, resume loads lines, tested.
- Frontend: LINES palette, one-of arming, place/draw (diamonds) / undo line clicks; manually verified.
- Full suite + ruff + root mypy green for touched Python files.

## Real validation (Patrick, visual)
Run the labeler on a real session, click the **near touchline + midline**, and confirm the overlay snaps onto those edges where points couldn't reach. Claude renders; Patrick assesses (no self-interpretation).

## Deferred
Schema additions (goal box, circle∩midline); model-path (keypoints→pose).
