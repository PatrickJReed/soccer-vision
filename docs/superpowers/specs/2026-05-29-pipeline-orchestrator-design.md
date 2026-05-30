---
name: pipeline-orchestrator-design
description: Design spec for the soccer-vision pipeline orchestrator — the Phase 3 integration layer that chains the existing (but unwired) pitch/ and phase/ modules into an end-to-end video → enriched-trajectories + per-frame-phases pipeline, emitting parquet artifacts downstream consumers (Phase 4 metrics) read.
status: approved
date: 2026-05-29
---

# soccer-vision — Pipeline Orchestrator Design

## 1. Problem & Context

Phases 2–3 built nine modules — `pitch/{spec,homography,mapper,filter}.py` and
`phase/{team_mode,possession,splitter}.py` — each independently implemented and
unit-tested. **None is wired into an end-to-end flow.** Every one is referenced
only by its own definition and its own test; the package exports nothing but
`__version__`, and the only code that instantiates `RoboflowBackend` or calls
`process_with_pitch` is the test suite.

The Phase 2–3 implementation plan
(`docs/superpowers/plans/2026-05-28-phase-2-3-finetune-and-homography.md`) ended
at Task 15 (the splitter) with no orchestration task, even though its framing
("each module emits a parquet table downstream of the trajectories parquet")
assumed the chain existed. This spec defines that missing capstone.

It is the realization of **Phase 3 — "Pitch + phase: Homography + 5-state
possession"** from the master spec
(`docs/superpowers/specs/2026-05-27-soccer-vision-design.md`, §9), sitting in the
exact slot §2.2's data flow places it: between `TrackingBackend.process()` and
`metrics/*`.

### 1.1 Goal

A pipeline that turns a Trace video into (a) enriched per-detection trajectories
in canonical pitch coordinates with cleaned team labels, and (b) per-frame
possession/phase labels — persisted as parquet for the Phase 4 metrics layer to
consume.

### 1.2 Non-goals (v1 scope guards)

- **No metrics.** That is Phase 4+. This stage stops at `phases.parquet`.
- **No rendering / video overlays.** The `viz/` module stays empty for now.
- **No new detection or model work.** Phase 2 closed that; this is pure
  integration of existing, tested units (plus two small additions, §4).

## 2. Architecture (Approach A — pure assembly + thin video wrapper)

Two stages, each with a parquet checkpoint, so re-tuning thresholds never
re-runs GPU tracking (master spec §2.3: "Parquet between every stage").

```
Stage 1 (GPU, thin):  analyze_video(video, out_dir, backend=None, opts)
    └─ RoboflowBackend(detect_pitch=True).process_with_pitch(video)
       └─ writes trajectories_px.parquet + keypoints.parquet   ← checkpoint
          └─ calls assemble_phases(...)

Stage 2 (pure, the unit under test):  assemble_phases(traj_px, kp, fps, total_frames, opts)
    └─ writes trajectories.parquet + phases.parquet            ← deliverables

Re-assembly (CPU only):  assemble_from_parquet(traj_px_path, kp_path, out_dir, opts)
    └─ reads the Stage-1 checkpoint, re-runs Stage 2 cheaply
```

`assemble_phases` is pure (no models, no GPU, no `ultralytics`/`sports` import —
`cv2`/`scipy`/`pandas` only, all base deps) so the integration logic — the exact
thing that was never wired up — is fully testable without a GPU. The untestable
surface shrinks to "call two model methods and write files," which a
stub-backend test still exercises.

`pipeline.py` must import `RoboflowBackend` **lazily inside `analyze_video`**
(mirroring `roboflow.py`'s own lazy-import pattern), so the module imports — and
`assemble_phases` / `assemble_from_parquet` run — without the `roboflow` optional
extra installed.

### 2.1 Output directory layout

`out_dir/` after a run contains four files:

| File | Stage | Granularity | Purpose |
|---|---|---|---|
| `trajectories_px.parquet` | 1 | per-detection (pixels) | checkpoint: verbatim `process_with_pitch` trajectories |
| `keypoints.parquet` | 1 | per-keypoint | checkpoint: verbatim pitch keypoints |
| `trajectories.parquet` | 2 | per-detection (enriched) | **deliverable** |
| `phases.parquet` | 2 | per-frame | **deliverable** |

The two deliverables are the chosen output contract; the two checkpoints satisfy
the cheap-recompute requirement.

## 3. Data flow (inside `assemble_phases`)

```
trajectories_px, keypoints
    │
    ├─ build_frame_homographies(keypoints, conf_threshold, min_points=4) → {frame: H}  (sparse)
    │     └─ smooth_homographies(alpha) → {frame: H}  (dense, carry-forward)
    │
    └─ PitchMapper.transform(trajectories_px, H)       # +x_pitch, +y_pitch
          └─ filter_outside_pitch(margin)              # drop adjacent-game / off-pitch
                └─ apply_modal_team_per_track          # smooth per-track team flicker
                      ├─ classify_possession(thresholds)        → state (per frame)
                      │     └─ smooth_possession(window_frames)  → state (mode-smoothed)
                      │
                      └─ ball y_pitch per frame ─────────────────┐
                                                                 │
                            label_phase(smoothed_state, ball_y, fps) → phase (per frame)
```

**Order is deliberate:** map → filter → modal-team → possession → smooth →
phase. Filtering runs after mapping (needs pitch coords) and before possession
(so adjacent-game players don't pollute nearest-player logic). Modal-team runs
after filtering (so dropped detections don't skew a track's modal vote).
`smooth_possession` runs before `label_phase` because the splitter detects
turnovers off the possession series — un-smoothed per-frame flicker would spray
spurious `transition` labels.

## 4. New / extended components

### 4.1 `pitch/landmarks.py` (new) — the missing link

`fit_homography(image_points, pitch_points)` needs each keypoint's canonical
pitch coordinate, but the keypoints DataFrame carries only `kp_idx, x_px, y_px`.
There is no `kp_idx → pitch-coordinate` table in the codebase.

- **`PITCH_LANDMARKS: NDArray`** — an `(N, 2)` constant of normalized `[0,1]²`
  pitch coordinates, one row per pitch-model keypoint index, vendored from
  roboflow's `SoccerPitchConfiguration` vertex layout. Vendored as a documented
  constant (not a live `sports` import) to keep the pure layer free of the heavy
  `sports`/`ultralytics` dependency. Axis convention: roboflow's length axis
  (goal-to-goal) maps to our `y` (so `y < 0.333` = own third, `y > 0.667` = opp
  third, matching the splitter); width maps to `x`.
- **`build_frame_homographies(keypoints, *, conf_threshold=0.5, min_points=4) -> dict[int, NDArray]`**
  — per frame: take keypoint rows with `conf ≥ conf_threshold` and a known
  `kp_idx`; `image_points = (x_px, y_px)`, `pitch_points = PITCH_LANDMARKS[kp_idx]`;
  if `≥ min_points`, `fit_homography(...) → H[frame]`, else skip the frame.
  Returns the sparse `{frame: H}` dict `smooth_homographies` densifies.

> **Open verification:** the length/width axis orientation and the cluster→team
> mapping (`_CLUSTER_TEAM = {0: own, 1: opp}` in `roboflow.py`) jointly determine
> whether "own" half is correct. Both must be confirmed empirically against the
> bake-off clip during implementation; flipping is a one-line fix once observed.

### 4.2 `phase/possession.py` (extend) — `smooth_possession`

`smooth_possession(possession_state: pd.Series, window_frames: int) -> pd.Series`,
implementing master spec §6.1 ("smoothed over a 30-frame window using mode,
preserving contested"), which Task 14 never built.

- Rolling mode over a centered `window_frames` window (caller passes
  `round(fps)` ≈ 30).
- **Preserving contested:** a frame whose *raw* state is `contested` stays
  `contested` regardless of the window mode; all other states are replaced by
  the window mode.
- Operates positionally on the sorted-by-frame series (row-count window, not
  strict frame distance) — acceptable since detections exist in nearly every
  frame.

## 5. Public API

```python
@dataclass(frozen=True)
class PipelineResult:
    trajectories: pd.DataFrame   # per-detection, enriched: +x_pitch, +y_pitch; team = modal-cleaned
    phases: pd.DataFrame         # per-frame, full contiguous frame range
    homography_coverage: float   # fraction of frames where the pitch model fit a homography (pre-smoothing, from raw_h)
    ball_coverage: float         # fraction of frames with a non-NaN ball pitch coord


def assemble_phases(
    trajectories_px: pd.DataFrame,
    keypoints: pd.DataFrame,
    fps: float,
    total_frames: int,
    *,
    kp_conf_threshold: float = 0.5,
    homography_alpha: float = 0.5,
    filter_margin: float = 0.05,
    possession_thresholds: PossessionThresholds | None = None,
    transition_seconds: float = 5.0,
) -> PipelineResult: ...


def analyze_video(
    video_path: Path,
    out_dir: Path,
    *,
    backend: object | None = None,   # defaults to RoboflowBackend(detect_pitch=True)
    **assemble_opts,
) -> PipelineResult: ...


def assemble_from_parquet(
    trajectories_px_path: Path,
    keypoints_path: Path,
    out_dir: Path,
    *,
    fps: float | None = None,        # inferred from t_seconds/frame if None
    **assemble_opts,
) -> PipelineResult: ...
```

`assemble_phases` returns the result without writing; `analyze_video` and
`assemble_from_parquet` write the parquet files (their respective stages) and
return the result.

## 6. Parquet schemas

- **`trajectories.parquet`** — existing trajectory schema (`frame, t_seconds,
  track_id, x_px, y_px, bbox_*, class, team, conf`) **+ `x_pitch`, `y_pitch`**
  (float64, nullable). `team` holds the modal-smoothed value.
- **`phases.parquet`** — `frame` (int64), `t_seconds` (float64),
  `possession_state` (object), `phase` (object), `ball_x_pitch`, `ball_y_pitch`
  (float64, nullable). **One row per frame across `[0, total_frames)`** —
  `unknown` where no usable data, so downstream time-series metrics get a
  gapless index.
- **`trajectories_px.parquet` / `keypoints.parquet`** — verbatim
  `process_with_pitch` output (the checkpoint).

`total_frames` is derived from the source clip in `analyze_video`, and from
`max(frame) + 1` of the checkpoint in `assemble_from_parquet`.

## 7. Failure handling (graceful degradation, never raises on data quality)

- **<4 confident keypoints** in a frame → no raw H → `smooth_homographies`
  carries forward the last good H within its range; frames before the first-ever
  H get no H.
- **No H for a frame** → `PitchMapper` yields NaN pitch coords →
  `filter_outside_pitch` drops those rows (NaN fails the bounds test) → that
  frame contributes no usable detections → reindexed to `unknown` in
  `phases.parquet`.
- **Zero homographies all clip** → all-`unknown` output, `homography_coverage =
  0.0`, a logged warning. Valid output; the caller inspects coverage.
- **Empty trajectories** → empty `trajectories`, full-`unknown` `phases` over
  `[0, total_frames)`.

The homography "validity check" from master spec §3.1a is intentionally
subsumed by `filter_outside_pitch` — per the Phase 2–3 plan Self-Review, a bad H
projects detections off-pitch and they get dropped. Sufficient for v1.

## 8. Testing

Unit-test target follows master spec §8.1 (≥80% per module). All pure functions
are covered without a GPU.

- **`build_frame_homographies`** — synthetic known-H round trip (project
  `PITCH_LANDMARKS` through H⁻¹ to make image points, recover H); `<4` points →
  frame skipped; conf-threshold filtering.
- **`smooth_possession`** — own/opp/own flicker → smooths to own; `contested`
  preserved; window edges.
- **`PITCH_LANDMARKS`** — shape `(N, 2)`, all values in `[0, 1]`, N matches the
  pitch model's keypoint count.
- **`assemble_phases` end-to-end** (the integration test that was missing) — a
  synthetic `trajectories_px` + `keypoints` fixture engineered so one frame is
  ball-near-own-player-in-own-third (→ `own`/`build`) and one is loose (→
  `loose_ball`); asserts pitch coords added, an off-pitch detection dropped, team
  flicker smoothed, and `phases.parquet` spans the full frame range.
- **`assemble_from_parquet`** — round-trips fixture parquets.
- **`analyze_video` wiring with a stub backend** (no GPU) — a fake backend whose
  `process_with_pitch` returns fixtures; asserts all four parquet files are
  written with correct schemas and `PipelineResult` is populated.

The real-`RoboflowBackend` path stays a GPU/Colab concern (skipped in CI), but
the stub-backend test covers the wiring and I/O logic.

## 9. Alignment with prior specs

- **Master spec §2.2 data flow / §9 Phase 3** — this is that stage, in that slot.
- **§2.3 "parquet between every stage"** — honored by the Stage-1 checkpoint +
  `assemble_from_parquet`.
- **§3.1a homography robustness** — temporal smoothing (have it); validity check
  subsumed by the boundary filter (per-plan decision); the spec's
  "interpolate H from nearest good frames" is simplified to carry-forward for v1.
- **§6.1 possession smoothing** — added here (`smooth_possession`).
- **Output naming divergence:** the master spec sketched `pitch_trajectories.parquet`
  + `phased.parquet`; this design uses a normalized `trajectories.parquet`
  (enriched) + per-frame `phases.parquet`. The Phase 4 metrics layer joins
  `phases` onto `trajectories` by `frame`.

## 10. Open questions (deferred)

- Axis-orientation / team-mapping empirical confirmation (§4.1) — an
  implementation-time check on the bake-off clip, not a design fork.
- Whether `smooth_possession` should use a strict-frame-distance window instead
  of a positional one — revisit only if frame gaps prove large in practice.
