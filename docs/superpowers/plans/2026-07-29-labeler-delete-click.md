# Labeler Delete-Specific-Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Right-click deletion of specific point/line labels on any frame in the labeler, per `docs/superpowers/specs/2026-07-29-labeler-delete-click-design.md` (the contract — read it first).

**Architecture:** Two `LabelerState` methods that maintain the `_seq` undo-lockstep invariant under the lock, two thin server POSTs mirroring `/api/nudge`, and a `contextmenu` handler in app.js that hit-tests the click arrays it already holds in canvas-pixel space.

**Tech Stack:** Python 3.12 (stdlib http server, threading lock discipline), vanilla JS canvas frontend. Gates: full pytest (current baseline 510 passed / 3 skipped), `uv run ruff check src tests`, repo-root bare `uv run mypy` at exactly 58 errors. All commands from `packages/soccer-vision/`.

---

### Task 1: The whole feature (state + server + frontend), TDD on the Python parts

**Files:**
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/state.py` (add `delete_click`, `delete_line_click` next to `remove_last` ~line 328)
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/server.py` (two POSTs in `do_POST` ~line 132)
- Modify: `packages/soccer-vision/src/soccer_vision/labeler/static/app.js` (contextmenu + hit-test + hover cue)
- Test: `packages/soccer-vision/tests/test_labeler_state.py`, `packages/soccer-vision/tests/test_labeler_server.py`

- [ ] **Step 1: Read first** — spec; `state.py` (`remove_last`, `nudge_click`, `add_click`, `_affected`, `_autosave`, the `_seq` bookkeeping, `_active_clicks` locking); `server.py` `do_POST`; `app.js` (`canvasNorm`, `onmousemove`/`onmouseup` drag-nudge flow, the Undo handler's refresh sequence, `drawFrame`'s marker rendering & coordinate space); existing tests in `test_labeler_state.py` (fixtures `_pan_session` etc.) and `test_labeler_server.py` (how a server+state harness is built).

- [ ] **Step 2: Write the failing state tests** (append to `tests/test_labeler_state.py`; adapt fixture usage to the file's existing helpers):

```python
def test_delete_click_removes_all_for_landmark_and_keeps_seq() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        st.add_clicks(clicks)
        st.wait_idle(timeout=10)
        f0 = clicks[0].frame
        kp0 = clicks[0].kp_idx
        # stack a duplicate mislabel on the same landmark
        st.add_click(f0, kp0, 0.9, 0.9)
        st.wait_idle(timeout=10)
        n_before = len(st.clicks)
        removed = st.delete_click(f0, kp0)
        assert removed == 2  # original + duplicate die together
        assert len(st.clicks) == n_before - 2
        assert not any(c.frame == f0 and c.kp_idx == kp0 for c in st.clicks)
        assert len(st._seq) == len(st.clicks) + len(st.line_clicks)  # lockstep
        # undo after a targeted delete pops the correct (surviving) last click
        last = st.clicks[-1]
        st.remove_last()
        assert not any(c is last for c in st.clicks)
        assert len(st._seq) == len(st.clicks) + len(st.line_clicks)
    finally:
        st.stop_worker()


def test_delete_click_missing_returns_zero() -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        st.add_clicks(clicks)
        st.wait_idle(timeout=10)
        n = len(st.clicks)
        assert st.delete_click(0, 19) == 0 or True  # compute a truly-absent pair below
        absent_kp = next(k for k in range(21)
                         if not any(c.frame == 0 and c.kp_idx == k for c in st.clicks))
        assert st.delete_click(0, absent_kp) == 0
        assert len(st.clicks) == n
    finally:
        st.stop_worker()


def test_delete_line_click_nearest_within_eps() -> None:
    interframe, poses, clicks = _pan_session(9)
    anchors = _spread_anchors(9)
    st = LabelerState(interframe, 9, size=SIZE)
    try:
        st.add_clicks(clicks)
        st.add_line_clicks(_near_tl_clicks(poses, anchors))
        st.wait_idle(timeout=10)
        target = st.line_clicks[0]
        n = len(st.line_clicks)
        # exact stored coords (the UI path): removed
        assert st.delete_line_click(target.frame, target.line_id, target.x, target.y)
        assert len(st.line_clicks) == n - 1
        assert len(st._seq) == len(st.clicks) + len(st.line_clicks)
        # far-away coords: nothing within eps
        assert not st.delete_line_click(target.frame, target.line_id, 0.0, 0.0)
        assert len(st.line_clicks) == n - 1
    finally:
        st.stop_worker()


def test_delete_persists_to_sidecar(tmp_path: Path) -> None:
    interframe, _poses, clicks = _pan_session(9)
    st = LabelerState(interframe, 9, size=SIZE, autosave_path=tmp_path / "s.json")
    try:
        st.add_clicks(clicks)
        st.wait_idle(timeout=10)
        f0, kp0 = clicks[0].frame, clicks[0].kp_idx
        st.delete_click(f0, kp0)
        import json
        data = json.loads((tmp_path / "s.json").read_text())
        assert not any(d["frame"] == f0 and d["kp_idx"] == kp0 for d in data["clicks"])
    finally:
        st.stop_worker()
```
(Adapt the autosave-path kwarg name to `LabelerState`'s actual signature — read it; if autosave is keyed off a video path/sidecar convention instead, construct the state the way existing autosave tests do. If no autosave test pattern exists, assert via the state's own autosave trigger mechanism.)

- [ ] **Step 3: Run to verify failure:** `uv run pytest tests/test_labeler_state.py -q -k delete` → AttributeError (no `delete_click`).

- [ ] **Step 4: Implement the state methods** (next to `remove_last`; adjust to the file's actual lock/worker idioms):

```python
    def delete_click(self, frame: int, kp_idx: int) -> int:
        """Remove ALL point clicks for (frame, kp_idx); returns how many were removed.

        Targeted deletion (spec 2026-07-29): duplicates for one landmark die together.
        _seq lockstep: the i-th "pt" entry of _seq corresponds to self.clicks[i], so
        the matching _seq positions are dropped in the same locked block.
        """
        with self._lock:
            keep = [i for i, c in enumerate(self.clicks)
                    if not (c.frame == frame and c.kp_idx == kp_idx)]
            removed = len(self.clicks) - len(keep)
            if not removed:
                return 0
            pt_pos = [j for j, k in enumerate(self._seq) if k == "pt"]
            keep_set = set(keep)
            drop = {pt_pos[i] for i in range(len(self.clicks)) if i not in keep_set}
            self.clicks = [self.clicks[i] for i in keep]
            self._seq = [k for j, k in enumerate(self._seq) if j not in drop]
        if self._calibrated:
            dirty = self._affected(frame) if self._engine == "crop" else range(self.n_frames)
            self._worker.mark_dirty(dirty)
        self._autosave()
        return removed

    def delete_line_click(self, frame: int, line_id: str, x: float, y: float,
                          *, eps: float = 1e-6) -> bool:
        """Remove the nearest line click of `line_id` on `frame` within `eps`
        (normalized Euclidean). The UI sends the exact stored coordinates of the
        marker it hit (float64 round-trips exactly through JSON), so eps is only
        float paranoia; programmatic callers may pass a larger tolerance."""
        with self._lock:
            best_i, best_d = -1, eps
            for i, lc in enumerate(self.line_clicks):
                if lc.frame != frame or lc.line_id != line_id:
                    continue
                d = math.hypot(lc.x - x, lc.y - y)
                if d <= best_d:
                    best_i, best_d = i, d
            if best_i < 0:
                return False
            ln_pos = [j for j, k in enumerate(self._seq) if k == "ln"]
            del self.line_clicks[best_i]
            del self._seq[ln_pos[best_i]]
        if self._calibrated:
            self._worker.mark_dirty(self._affected(frame))
        self._autosave()
        return True
```
(Import `math` if the module doesn't already. If `self.clicks` rebinding conflicts with how other threads hold references, mutate in place with a slice assignment `self.clicks[:] = ...` — check how `_active_clicks`/the worker read it and match `remove_last`'s discipline.)

- [ ] **Step 5: Run state tests:** `uv run pytest tests/test_labeler_state.py -q` → all pass.

- [ ] **Step 6: Write the failing server tests** (append to `tests/test_labeler_server.py`, mirroring its existing harness):

```python
def test_delete_click_endpoint_roundtrip() -> None:
    # build harness exactly like the file's existing POST tests; then:
    # POST /api/click {frame:0, kp_idx:2, x:0.5, y:0.5} -> 200
    # POST /api/delete_click {frame:0, kp_idx:2} -> 200, payload has state keys
    # GET /api/clicks -> the click is gone
    # POST /api/delete_click {frame:0, kp_idx:2} again -> 404 with {"error": ...}
    ...


def test_delete_line_click_endpoint_roundtrip() -> None:
    # POST /api/line_click {frame:0, line_id:"midline", x:0.4, y:0.6} -> 200
    # POST /api/delete_line_click same coords -> 200; GET shows it gone
    # POST /api/delete_line_click again -> 404
    ...
```
Write these as REAL tests against the file's actual harness (the `...` above is a
sketch because the harness shape is file-specific — the assertions listed are the
contract; no test may be left as a stub).

- [ ] **Step 7: Implement the endpoints** in `server.py` `do_POST`, after the `nudge` branch:

```python
            elif self.path == "/api/delete_click":
                n = state.delete_click(int(payload["frame"]), int(payload["kp_idx"]))
                if n:
                    self._json(self._state_payload())
                else:
                    self._json({"error": "no click at frame/kp_idx"}, code=404)
            elif self.path == "/api/delete_line_click":
                found = state.delete_line_click(
                    int(payload["frame"]), str(payload["line_id"]),
                    float(payload["x"]), float(payload["y"]))
                if found:
                    self._json(self._state_payload())
                else:
                    self._json({"error": "no line click at x/y"}, code=404)
```

- [ ] **Step 8: Run server tests:** `uv run pytest tests/test_labeler_server.py -q` → all pass.

- [ ] **Step 9: Frontend** (`static/app.js`). Add after the existing canvas handlers:

```javascript
function nearestLabel(nx, ny){   // nearest deletable marker on cur frame, canvas-px space
  const R = 14;
  let best = null, bestD = R;
  for(const c of clicks) if(c.frame === cur){
    const d = Math.hypot((c.x - nx) * canvas.width, (c.y - ny) * canvas.height);
    if(d <= bestD){ bestD = d; best = {kind: "pt", kp_idx: c.kp_idx, x: c.x, y: c.y}; }
  }
  for(const lc of lineClicks) if(lc.frame === cur){
    const d = Math.hypot((lc.x - nx) * canvas.width, (lc.y - ny) * canvas.height);
    if(d <= bestD){ bestD = d; best = {kind: "ln", line_id: lc.line_id, x: lc.x, y: lc.y}; }
  }
  return best;
}

canvas.oncontextmenu = async (e) => {
  e.preventDefault();
  const [nx, ny] = canvasNorm(e);
  const t = nearestLabel(nx, ny);
  if(!t) return;
  const r = t.kind === "pt"
    ? await postJSON("/api/delete_click", {frame: cur, kp_idx: t.kp_idx})
    : await postJSON("/api/delete_line_click",
                     {frame: cur, line_id: t.line_id, x: t.x, y: t.y});
  if(r.error) return;
  applyState(r);
  const cl = await api("/api/clicks"); clicks = cl.clicks; lineClicks = cl.line_clicks || [];
  placed = new Set(clicks.map(c => c.kp_idx));
  const fh = await api(`/api/frame_h/${cur}`); curH = fh.h; drawFrame();
};
```
Match `canvasNorm`'s actual return shape (read it — if it returns an object, destructure accordingly). Hover cue: extend the existing `onmousemove` path to hit-test via `nearestLabel` and store the result in a module-level `hoverDelete` variable; in `drawFrame`, when `hoverDelete` is set, draw a red ring (`ctx.strokeStyle="#e0524d"`, radius ~10, lineWidth 2) at `(hoverDelete.x * canvas.width, hoverDelete.y * canvas.height)`. Make sure the mousemove path still calls `drawFrame` only as often as it already does (don't add an unconditional redraw storm; reuse its existing redraw or redraw only when the hover target CHANGES).

- [ ] **Step 10: Manual verification (required — ISOLATED from the live session):**
Patrick's labeler is RUNNING on port 8000 against `~/sv-labeler/home_g4_oceanside/rep_pack/rep_video.mp4` — do NOT kill it and do NOT launch anything against that video path (the clicks sidecar lives next to the video; two processes sharing it would corrupt his session). Instead: copy the video to the scratchpad (`cp ~/sv-labeler/home_g4_oceanside/rep_pack/rep_video.mp4 <scratchpad>/delete_test/rep_video.mp4`), launch a second labeler on it with `--port 8123 --export-dir <scratchpad>/delete_test/export --workers 1` (background), wait for port 8123, seed two point clicks and one line click via `/api/click` + `/api/line_click` curl POSTs, then exercise both delete endpoints (success + 404 cases), confirming via `/api/clicks` after each step. Kill ONLY that labeler afterward (by its PID, never `pkill -f soccer_vision.labeler` — that would kill Patrick's). Paste the curl transcript in your report.

- [ ] **Step 11: Full gates:** `uv run pytest -q 2>&1 | tail -3 && uv run ruff check src tests` and repo-root `uv run mypy 2>&1 | grep -c "error:"` → expect 510+new passed / 3 skipped, ruff clean, exactly 58.

- [ ] **Step 12: Commit:**
```bash
git add src/soccer_vision/labeler/state.py src/soccer_vision/labeler/server.py src/soccer_vision/labeler/static/app.js tests/test_labeler_state.py tests/test_labeler_server.py
git commit -m "feat(labeler): right-click deletion of specific point/line labels"
```
End with: Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

## Self-review checklist
Spec coverage: UX (right-click, 14px canvas-px hit-test, hover cue, contextmenu suppressed) / backend (delete-all-duplicates semantics, `_seq` lockstep under lock, dirty scopes, autosave, eps) / server (two POSTs + 404s) / tests (state incl. undo-after-delete + sidecar persistence; server round-trips) — all present. The sidecar autosave-path kwarg and the server-test harness are the two adapt-to-reality points; everything else is contract.
