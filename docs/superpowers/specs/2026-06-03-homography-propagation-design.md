---
name: homography-propagation-design
description: Design spec for Phase 3.5a — homography propagation. Fills the ~84% of Trace frames that lack pitch keypoints by registering them to nearby landmark anchors (bidirectional bounded frame-to-frame chaining), lifting pitch-homography coverage without labeling. Runs as a new CPU pipeline stage emitting a dense homography checkpoint with per-frame provenance + confidence.
status: approved
date: 2026-06-03
---

# soccer-vision — Homography Propagation Design (Phase 3.5a)

## 1. Problem & Context

The Phase 3 pipeline runs end-to-end, but on the bake-off clip only **16.3% of
frames** get a pitch homography — the broadcast-trained pitch-keypoint model
detects too few landmarks on faint, patchy youth fields. With no homography,
detections have no pitch coordinates and the frame is `unknown`, so possession/
phase is `unknown` on ~89% of frames. Pitch-homography coverage is the binding
constraint on the entire analytics layer (see
`memory/project_pitch_homography_coverage_constraint.md`).

Two feasibility probes (`examples/colab_homography_probe*.ipynb`) established:
- **Frame-to-frame registration is reliable.** Trace is a no-parallax virtual-PTZ
  crop of a fixed camera, so consecutive frames are related by a global
  homography recoverable from background features. Probe v1 chained across 44
  gaps with **0 registration failures**.
- **Naive one-sided chaining drifts** — median reprojection error 0.076
  pitch-units; short gaps tight (≤0.05), long gaps drift.
- **Direct-to-anchor does not help** (probe v2): worse than chaining on short
  gaps, and fails on hard frames. Chaining is the better primitive.

This spec covers **3.5a only**: propagation as a gap-filler. A separate future
phase (3.5b) may fine-tune the pitch detector to densify anchors; the two are
synergistic but independently buildable, and propagation is built first because
it needs no labeling and is measurable on the existing checkpoint.

### 1.1 Goal

Lift pitch-homography coverage by propagating the ~16% of anchor homographies
into the surrounding no-landmark frames, **accuracy-first** — bridge fewer frames
trustworthily rather than inflate coverage with drift — and tag every frame's
homography with its source (`anchor`/`propagated`) and a confidence so downstream
metrics can weight or filter.

### 1.2 Non-goals (v1 scope guards)

- **No pitch-detector fine-tune** (that is 3.5b).
- **No global mosaic / bundle adjustment** (a possible 3.5a-v2 if this plateaus).
- **No full-game performance tuning** — v1 targets the 2-minute bake-off clip;
  the stage is written streaming so games are *possible*, but downscaled-frame
  ORB / coarser sampling are explicitly v1+.

## 2. Architecture — a 3-stage pipeline

Propagation needs pixels (it registers video frames), so it cannot live in the
pure `assemble_phases` stage. It slots in as a new middle stage:

```
1. Detect     (GPU)        video ───────────────────► trajectories_px.parquet + keypoints.parquet
2. Homography (CPU+video)  keypoints + video ────────► homographies.parquet   ← NEW
3. Assemble   (pure)       trajectories_px + homographies ─► trajectories.parquet + phases.parquet
```

- **Stage 2 (new, CPU):** computes anchor homographies from `keypoints`
  (existing `build_frame_homographies`), propagates them into the gaps via
  registration, writes a dense `homographies.parquet` (`frame, h00..h22, source,
  confidence`). Only frames that obtain an H appear; the rest stay absent
  (→ `unknown`, unchanged behavior).
- **`assemble_phases` gains an optional `homographies` argument.** When supplied
  with the dense map it uses it directly (skips its internal
  `build_frame_homographies → smooth_homographies`); otherwise it behaves exactly
  as today. Stage 3 stays pure and video-free, backward-compatible.
- **`analyze_video` orchestrates all three** (it holds the video).
  `assemble_from_homographies(...)` runs Stage 3 alone for instant downstream
  re-tuning.

**Cheap-recompute by cost tier:** downstream params (possession thresholds,
filter margin) → re-run Stage 3 only (instant). Propagation params (`max_gap`) →
re-run Stage 2 (CPU + a video read, no GPU). Detection → re-run Stage 1 (GPU).

**Module layout:** one new file `pitch/propagation.py` (registration + chaining +
blend — the testable core); thin `build_homographies()` orchestration in
`pipeline.py`; the new `assemble_phases` argument and `assemble_from_homographies`.

## 3. The propagation algorithm (`pitch/propagation.py`)

### 3.1 Bidirectional bounded chaining

For each gap between consecutive anchors A and B with `gap ≤ max_gap`, bridge
every frame `t` in the gap:

- **Forward chain from A:** register consecutive frames A→A+1→…→t (each transform
  is near-identity — the reliable primitive), accumulating `W_fwd` (A-pixels →
  t-pixels). `H_fwd(t) = H_A · inv(W_fwd)`.
- **Backward chain from B:** symmetric. `H_bwd(t) = H_B · inv(W_bwd)`.
- **Blend** by distance: `w_fwd = (B − t) / (B − A)` (trust the nearer anchor's
  shorter chain), combining `H_fwd` and `H_bwd` via element-wise weighted average
  + renormalize (`h22 = 1`) — the same technique `smooth_homographies` already
  uses — so a single H per frame is emitted and the `{frame: H}` interface is
  preserved.

Each registration uses ORB features + `cv2.findHomography(RANSAC)` between two
frames, **masking out player/referee boxes** (from `trajectories_px`, dilated)
so matching keys on static field/goal/background, not moving people.

Inter-frame homographies are computed in **one sequential video pass**
(`compute_interframe_homographies`): frames are read in ascending order via
grab/retrieve (each decoded once), ORB'd on a downscaled copy, and the resulting
homography is rescaled back to full-resolution pixels. `propagate_homographies`
then composes them — pure matrix multiplication, no video I/O.

### 3.2 Confidence — measured at runtime

For a frame bridged by **both** chains, `H_fwd` and `H_bwd` are two independent
estimates of the same mapping. Push a fixed reference point set (an image-space
grid) through both; their mean disagreement in pitch-units is a runtime estimate
of the propagation error — the same quantity the probes measured against
landmarks, but available everywhere without landmarks.

`confidence = clamp(1 − disagreement / τ, 0, 1)`, with `τ` a calibrated scale.
Anchors: `confidence = 1.0`, `source = "anchor"`. Frames reached by only one
chain (the other failed): `source = "propagated"`, confidence from a
distance-based fallback (lower). This confidence flows to `phases.parquet`.

### 3.3 Drift control & the `max_gap` window

- **Bidirectional halving:** each chain spans at most half the gap, roughly
  halving worst-case drift versus one-sided chaining.
- **Hard `max_gap` cap:** gaps longer than the window are not bridged (stay
  `none`). Calibrated empirically (probe held-out method): sweep `max_gap`,
  measure held-out reprojection error at bridged anchors, pick the largest window
  whose median stays ≤ 0.05. Seed default ~20–30 frames (~1 s); calibration
  confirms.

### 3.4 Robustness

The two chains are independent: a frame is bridged if **either** reaches it. A
mid-gap forward-chain failure does not lose the frame when the backward chain
arrives — bidirectional gives both an accuracy win (halved drift) and a coverage
win (either-side reach). **Edge gaps** (before the first / after the last anchor)
are **not bridged in v1** — they would require a one-sided chain from the single
adjacent anchor (with distance-based confidence) and are deferred to v1+; the
between-anchor gaps are where the coverage is.

## 4. Public API

```python
# pitch/propagation.py
def compute_interframe_homographies(
    read_frame: Callable[[int], NDArray | None],  # frame index -> image (sequential; called in ascending order)
    needed_pairs: set[int],               # indices i for which G[i] (frame i -> i+1) is wanted
    player_boxes: pd.DataFrame,           # trajectories_px (for masking)
    *,
    downscale: float = 1.0,
    n_features: int = 3000,
    min_inliers: int = 12,
) -> dict[int, NDArray]: ...              # {i: full-res G[i]}

def propagate_homographies(
    anchors: Mapping[int, NDArray],       # {frame: H} from build_frame_homographies
    interframe: Mapping[int, NDArray],    # precomputed {i: G[i]} from compute_interframe_homographies
    *,
    max_gap: int = 45,
    disagreement_tau: float = 0.10,
    frame_size: tuple[int, int] = (1920, 1080),
) -> dict[int, HomographyEntry]: ...      # frame -> (H, source, confidence)
# Note: propagate_homographies is pure (no video I/O); compute_interframe_homographies
# owns the sequential video read. build_homographies orchestrates both.

# pipeline.py
def build_homographies(
    keypoints: pd.DataFrame,
    video_path: Path,
    trajectories_px: pd.DataFrame,
    *,
    kp_conf_threshold: float = 0.5,
    max_gap: int = 45,
    disagreement_tau: float = 0.10,
    downscale: float = 1.0,
) -> dict[int, HomographyEntry]: ...

def assemble_phases(
    trajectories_px, keypoints, fps, total_frames, *,
    homographies: dict[int, HomographyEntry] | None = None,   # NEW; if given, use directly
    ...
) -> PipelineResult: ...

def assemble_from_homographies(
    trajectories_px_path: Path, homographies_path: Path, out_dir: Path, **opts,
) -> PipelineResult: ...
```

`HomographyEntry` is a small dataclass `(H: NDArray, source: str, confidence: float)`.

## 5. Outputs

- **`homographies.parquet`** (new checkpoint): `frame` (int64), `h00..h22`
  (float64 ×9), `source` (object: `anchor`/`propagated`), `confidence` (float64).
- **`phases.parquet`**: adds `homography_source` (object) + `homography_conf`
  (float64) per frame; `unknown` frames get `source="none"`, `conf=0.0`.
  Metrics join provenance by `frame`.
- **`trajectories.parquet`**: unchanged (still `+x_pitch/+y_pitch`); provenance is
  joinable from phases.
- **`PipelineResult`**: `anchor_coverage` + `propagated_coverage` added;
  `homography_coverage` becomes the combined total (anchor + propagated).

## 6. Failure handling (graceful, never raises on data quality)

- Mid-chain registration failure → other chain may reach the frame; neither →
  `none` → NaN pitch → `unknown`.
- Gap > `max_gap` → not bridged (`none`).
- Zero anchors → propagation no-ops; all `none`; same warning as today's
  zero-homography degradation.
- Unreadable frame / `findHomography` returns None → treated as a registration
  failure for that step.

## 7. Performance & scale

Stage 2 reads gap frames in **one sequential pass** (grab/retrieve, each frame
decoded once); the sequential read is the dominant speedup (minutes per clip,
not hours). ORB runs at full resolution by default (`downscale=1.0`): on Trace
1080p, `downscale=0.5` dropped too many ORB features on the masked low-texture
grass and propagation bridged almost nothing — full-res is required for
registration recall here. `propagate_homographies` then composes the precomputed
inter-frame map — pure matrix multiplication, no further video I/O.

**Calibration result (bake-off clip, downscale=1.0):** held-out reprojection
error is a flat **0.021** (well under the 0.05 gate) from `max_gap` 10 → 60, with
held-out coverage climbing 18% → 27% (true full-anchor coverage ~30–35%). No
accuracy knee; `max_gap=45` is the default (balances coverage against
less-validated long bridges, which the runtime disagreement-confidence flags).
Propagation roughly doubles homography coverage but leaves the long landmark-free
stretches unbridged — denser anchors (a 3.5b pitch-detector fine-tune) remain the
path to high coverage, and compound with propagation.
Written **streaming** (process gap-by-gap, never holding the whole video) so
full games are possible.

## 8. Testing & acceptance

**Pure unit tests (no video):**
- Homography blend: element-wise weighted average + renormalize (synthetic Hs).
- Chaining composition: `H_fwd`/`H_bwd` recovered exactly under synthetic known
  inter-frame transforms (e.g. identity and a known shift/scale).
- `max_gap` window + provenance assignment (bridged vs `none`, anchor vs
  propagated).
- Disagreement → confidence mapping (monotonic, clamped to [0,1]).
- Bidirectional coverage: with one chain forced to fail, the frame is still
  bridged by the other.

**Synthetic-frame test:** warp a textured image by a known homography; confirm
`register()` recovers it within tolerance — exercises ORB + `findHomography`
without real video.

**Acceptance gate (bake-off clip, notebook like the ball eval):**
1. **Accuracy:** held-out reprojection error (probe metric) **median ≤ 0.05** for
   propagated frames.
2. **Coverage:** `homography_coverage` materially above 16% — reported,
   accuracy-bounded (the achievable number falls out of `max_gap` calibration),
   not a hard threshold.
3. Provenance + confidence emitted so soft frames are flagged, not hidden.

Accuracy-first by design: bridge fewer frames trustworthily over inflating
coverage with drift.

## 9. Alignment with prior work

- Realizes the §3.1a "robustness measures" intent of the master spec
  (`2026-05-27-soccer-vision-design.md`) — temporal handling of the virtual-PTZ —
  beyond the v1 carry-forward, which propagation replaces for gap frames.
- Reuses the checkpoint / cheap-recompute architecture from the pipeline
  orchestrator spec (`2026-05-29-pipeline-orchestrator-design.md`): a new
  per-stage checkpoint (`homographies.parquet`) and a pure Stage-3 re-run.
- `pitch/landmarks.build_frame_homographies` supplies the anchors unchanged.

## 10. Open questions (deferred)

- Optional light EMA over consecutive anchors before propagation (reduce anchor
  jitter before it spreads into a gap) — measure whether it helps; off for v1.
- Whether `phases.parquet` provenance should also be mirrored onto
  `trajectories.parquet` for per-detection weighting — deferred until a metric
  needs it (YAGNI; join by frame for now).
- 3.5b pitch-detector fine-tune (denser anchors → shorter gaps) — separate phase,
  informed by this stage's measured coverage gain.
