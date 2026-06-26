---
name: tracking-own-opp-grounding
description: Ground own/opp team labels in the default analyze_video path (the kit-grounded mapping currently lives only in the separate hygiene CLI, so a whole game's own/opp can silently invert), make the TrackingBackend protocol real enough to swap, test the untested _run_pipeline core, and clear the remaining tracking + labeler-infra lows from the 2026-06-26 audit.
status: approved
date: 2026-06-26
---

# Tracking — Own/Opp Grounding + Backend/Pipeline Correctness

Sub-project 2 of the post-audit program (see `docs/superpowers/2026-06-26-codebase-audit.md`).
This sub-project removes the correctness landmines in the **detection/tracking → trajectory**
path (chiefly: own/opp can be globally inverted in the default pipeline) and clears the
backend-contract, test-coverage, and labeler-infrastructure findings. It is independent of the
calibration rework (SP1) except where grounding consumes the homographies SP1 produces; the two
proceed in parallel.

## 1. Problem

The default trajectory pipeline (`pipeline.analyze_video`) emits `team` labels that are **not
grounded in the user's actual kit**. The tracker assigns team from an *arbitrary* KMeans cluster
index:

- `tracking/roboflow.py:96` — `_CLUSTER_TEAM: Final[dict[int, str]] = {0: "own", 1: "opp"}` —
  is applied verbatim at `roboflow.py:581` (`_classify_teams_per_track(..., _CLUSTER_TEAM)`).
  Which physical team the `sports` `TeamClassifier` calls cluster 0 vs 1 is not stable, so a
  whole game's own/opp can come out **silently inverted**.
- The only step `analyze_video` runs downstream is `apply_modal_team_per_track`
  (`pipeline.py:91`), which only *smooths* per-track team noise — it never *flips* or *grounds*
  the labels.
- The **only kit-grounded path** is `hygiene.core.map_own_cluster` (`core.py:237-260`), reached
  through `hygiene.run.run_hygiene` (`run.py:223-224`) and gated on a **required** `--own-kit`
  hint (`hygiene/__main__.py:19`). `analyze_video` never calls it. The pipeline does not run
  hygiene at all.

Every own-team metric (possession %, shape, space) is meaningless if own/opp is inverted, so this
is the headline finding. Around it sit a cluster of smaller tracking/backend/labeler issues the
audit verified: a decorative backend protocol, an untested `_run_pipeline` core, GK kit
mis-classification in the default path, an untuned tracker, a redundant video decode, a latent
ball track-id collision, stale metadata, and four labeler-infra papercuts.

Findings closed by this sub-project: `own-opp-ungrounded-default-path` (high),
`tracking-backend-protocol-decorative` (med), `run-pipeline-core-untested` (med),
`gk-kit-misclassified-default-path` (low), `bytetrack-bare-defaults` (low),
`tracker-two-video-decodes` (low), `ball-trackid-collision-within-frame` (low),
`tracking-metadata-staleness` (low), `labeler-chain-pool-macos-spawn-hang` (low),
`labeler-framejpeg-reseek` (low), `labeler-state-payload-double-status` (low),
`labeler-bulk-add-unlocked` (low), and the `bakeoff-homography-axis` documentation note (med,
doc-only).

## 2. Goals & non-goals

**Goals**
- The default `analyze_video` output is **own/opp-grounded** whenever the caller supplies the kit
  hint, and is **loudly flagged as ungrounded** when they do not — never silently inverted.
- The `TrackingBackend` protocol describes what the pipeline actually calls, and `MockBackend`
  can drive `analyze_video` end-to-end.
- `_run_pipeline`'s detection→track→team→rows→keypoint core is exercised on canned detections
  (not just a blank video that returns early).
- The tracking lows (GK, ByteTrack, double decode, ball-id collision, metadata) and labeler-infra
  lows (chain pool, frame cache, double status, unlocked bulk add) are fixed or explicitly
  documented as the canonical path.

**Non-goals (YAGNI)**
- No new team-classification model and no re-implementation of kit matching inside the tracker.
  Grounding reuses the already-verified `hygiene` Lab-kit path; the tracker keeps the opaque
  `sports.TeamClassifier`.
- No ByteTrack hyperparameter sweep (no per-session ground truth to tune against; the audit notes
  hygiene stitching already addressed track shredding). We document + make params injectable, not
  tuned.
- No change to the parquet schemas, the labeler HTTP/UI transport, or the calibration geometry
  (SP1 owns that).
- The bake-off is **not re-run** here; finding #10 is a written note only.

## 3. Design (per finding)

### 3.1 Ground own/opp in `analyze_video` [HIGH]

**Where:** `tracking/roboflow.py:96`, `:581`; `pipeline.py:91`, `:205-241`. Kit-grounded path:
`hygiene/run.py:147-267` (`run_hygiene`), reusing `hygiene/core.py` `cluster_teams` (`:183`),
`map_own_cluster` (`:237`), `assign_goalkeepers` (`:263`), `apply_team_labels` (`:296`).

**Fix (recommended): thread an `own_kit` hint through `analyze_video` and, when present, chain the
existing hygiene grounding; when absent, keep current behavior but warn loudly.**

Add `own_kit: str | None = None` to `analyze_video`. Flow:

1. As today: `backend.process_with_pitch` → `trajectories_px`, `keypoints`; write
   `trajectories_px.parquet`, `keypoints.parquet`; `build_homographies` → `homographies.parquet`.
2. **If `own_kit` is set:** lazily import and call `run_hygiene(traj_path=out/"trajectories_px.parquet",
   homographies_path=out/"homographies.parquet", video_path=video_path, out_dir=out,
   own_kit=own_kit)`. This runs the verified grounded path (pitch-space stitch → kit cluster →
   `map_own_cluster` against the kit anchor → positional GK assignment → balance gate), writing
   `trajectories_px_clean.parquet` + `hygiene_report.json` + contact sheets. Read the cleaned
   trajectories back and feed **those** to `assemble_phases`, so possession/phase are computed on
   grounded labels. The lazy import is required because `hygiene.run` already imports from
   `pipeline` (`homographies_from_parquet`) — a top-level import would be circular; this matches
   the existing lazy `RoboflowBackend` import at `pipeline.py:218-223`.
3. **If `own_kit` is None:** behavior is unchanged (cluster-index convention), but emit a single
   prominent `logger.warning` that own/opp is ungrounded and may be globally inverted, and that
   `own_kit=` (or the hygiene CLI) is required for trustworthy own-team analytics.

**Why this over threading the hint into the backend:** the in-tracker labels come from `sports`'
opaque embedding clusters (SigLIP/UMAP/KMeans), which expose no Lab centroid to match a colour
word against; grounding there would mean re-deriving kit colours and duplicating
`hygiene.map_own_cluster`'s Lab logic with no cleaner result. Chaining hygiene reuses code the
audit explicitly endorsed ("genuinely solid; keep") at the cost of one extra sequential video read
that hygiene already performs.

**Test:** (a) `analyze_video(..., own_kit="white", backend=stub)` with `run_hygiene`
monkeypatched to return a known grounded `team_by_track` that *inverts* the stub's raw labels —
assert the written `trajectories.parquet` `team` column matches the grounded map, not the raw
cluster map (the anti-inversion guarantee). (b) `analyze_video(..., own_kit=None)` — assert labels
pass through unchanged and `caplog` captured the ungrounded warning. (c) a focused
permutation-invariance test reusing the pure `cluster_teams` + `map_own_cluster`: feed the same
kit features with the two cluster indices swapped and assert `map_own_cluster` returns the same
physical own team both times (locks in that grounding is invariant to KMeans label order).

### 3.2 Make the `TrackingBackend` protocol real [MED]

**Where:** `tracking/base.py:26` declares only `process`; production calls
`backend.process_with_pitch` (`pipeline.py:225`), which is **not** on the protocol; `backend` is
typed `Any | None` (`pipeline.py:209`); `MockBackend` implements only `process` (`mock.py:25`) so
it cannot drive `analyze_video`.

**Fix (recommended): promote `process_with_pitch` onto the protocol and give `MockBackend` an
implementation.** Add to `TrackingBackend`:

```python
def process_with_pitch(self, video_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run detection + tracking AND pitch-keypoint detection in ONE video pass.
    Returns (trajectories, keypoints); keypoints may be empty but schema-conformant."""
    ...
```

`MockBackend.process_with_pitch` returns its existing mock trajectories plus an empty,
schema-conformant keypoints frame (reuse `RoboflowBackend._empty_kp_df`'s columns). Narrow
`analyze_video`'s `backend: Any | None` to `TrackingBackend | None`.

**Why not split a separate `PitchDetector`:** keypoint inference is co-located with detection in a
single decode for performance (`roboflow.py:_run_pipeline`, `emit_keypoints=True`); splitting it
into its own component would force a second full video decode. The one-pass coupling is real and
worth keeping, so the protocol should describe the one-pass method rather than pull it apart.

**Test:** extend `test_tracking_protocol.py` / `test_tracking_mock.py` to assert `MockBackend`
satisfies the extended `runtime_checkable` protocol and that `process_with_pitch` returns a
schema-conformant `(df, kp_df)`; add a test that `analyze_video(backend=MockBackend())` runs
end-to-end and writes all five parquets (currently impossible — `MockBackend` lacks the method).

### 3.3 Test the `_run_pipeline` core [MED]

**Where:** `tracking/roboflow.py:368-622`. The existing heavy-path test
(`test_tracking_roboflow.py:54-72`) runs only on a 320×240 all-zeros clip, which yields zero
detections and returns the empty-df early — the detection→ByteTrack→team→rows→keypoint-reshape
core is never exercised.

**Fix:** add a unit test that monkeypatches the lazily-imported heavy modules so the core runs
without GPU/weights. Because the imports live *inside* `_run_pipeline` (`roboflow.py:393-404`),
inject fakes via `sys.modules` for `supervision`, `torch`, `sports.common.team`, and
`ultralytics` before calling `_run_pipeline` on a tiny real video (real `cv2.VideoCapture` for
metadata + frame generator; fake models return canned detections regardless of frame content).
Assert the produced rows encode:
- player/GK **foot point**: `y_px == y2`, `x_px == (x1+x2)/2` (`roboflow.py:484-485`);
- **ball center**: `y_px == (y1+y2)/2` (`roboflow.py:533`);
- class mapping via `_CLS_NAME` (`roboflow.py:481`);
- per-track team-map applied to player/GK rows (`roboflow.py:582-584`);
- synthetic ball ids `== _BALL_TRACK_ID_BASE - frame_idx` (`roboflow.py:537`);
- keypoint reshape: `N_instances × N_keypoints` rows with `(frame, kp_idx, x_px, y_px, conf)`
  (`roboflow.py:560-574`).

**Test:** the above is itself the test (`test_tracking_roboflow.py`, new
`test_run_pipeline_core_with_canned_models`), gated to run without the heavy extra because the
fakes provide the modules.

### 3.4 Goalkeeper kit mis-classification [LOW]

**Where:** GKs go through the same `sports` kit classifier as outfielders
(`roboflow.py:497`, `:583`); positional `assign_goalkeepers` exists only in hygiene
(`core.py:263`).

**Fix: GK grounding rides on the §3.1 hygiene chain (recommended), plus a docstring note.** When
`own_kit` is supplied, `run_hygiene` already calls `assign_goalkeepers` (`run.py:234`) so GK team
is set positionally (nearest team's players), correctly, in the grounded deliverable. For the
ungrounded path, document in `RoboflowBackend` / `analyze_video` that GK `team` comes from the
outfield kit classifier and may be wrong, and that hygiene (`own_kit`) is the canonical GK path.
Re-implementing positional GK assignment inside the tracker would duplicate `assign_goalkeepers`
and still need pitch coords + grounded outfield labels — exactly what the chain already provides.

**Test:** in the §3.1 grounded `analyze_video` test, include a GK track and assert its deliverable
`team` matches the positional `assign_goalkeepers` result (the nearest-team assignment), not the
raw kit cluster.

### 3.5 Bare ByteTrack defaults [LOW]

**Where:** `roboflow.py:459` — `tracker = sv.ByteTrack()` with no tuning.

**Fix: document that downstream stitching is the stability mechanism, and make the params
injectable (recommended) — do not blind-tune.** Add a `tracker_kwargs: dict | None = None`
constructor arg on `RoboflowBackend`, forwarded as `sv.ByteTrack(**(self.tracker_kwargs or {}))`,
defaulting to `{}` (today's behavior). Comment at `:459` that the defaults are intentionally
untuned and that track stability is provided by `hygiene.stitch_tracks` (the audit's
867→400-track result), which §3.1 now chains into the default path.

**Why not tune:** there is no per-session tracking ground truth to tune against, so a sweep risks
clip-overfit; making the params injectable defers the choice to data without a code change.

**Test:** monkeypatch `sv.ByteTrack` (via the fake-supervision module from §3.3) to capture its
kwargs; assert `tracker_kwargs` is forwarded and that the default is `{}`.

### 3.6 Two full video decodes per run [LOW]

**Where:** `roboflow.py:436-450` (Pass 1: every-60th-frame decode just to *fit* the
`TeamClassifier`) then `:465+` (Pass 2: full detection/tracking decode).

**Fix: single streaming pass.** Drop Pass 1. In the main pass, the per-track top-area crops are
already collected (`_keep_top_crop`, `track_crops`, `:500`); after the pass, **fit** the
`TeamClassifier` on a sample of those collected crops, **then** `predict` per-track (the
classification at `:579-584` already runs post-pass). This removes the whole-video Pass-1 decode.

**Trade-off:** Pass 1 sampled temporally (every 60th frame); the single-pass fit uses the
per-track clearest (largest-area) crops already bounded by `_TEAM_CROPS_PER_TRACK`. That is an
equal-or-better fit sample (clearest kits) for a small bounded memory increase, and the
classifier only needs an unsupervised 2-cluster fit.

**Test:** in the §3.3 canned-model test, wrap the fake `get_video_frames_generator` to count
invocations; assert it is called **once** per `_run_pipeline` run.

### 3.7 Ball track-id collision within a frame [LOW]

**Where:** `roboflow.py:537` — `synthetic_id = _BALL_TRACK_ID_BASE - frame_idx`. With ≥2 ball
detections in one frame (possible at `ball_conf=0.05`), the rows share a `track_id`. Currently
benign (consumers key the ball by frame, and `interpolate_ball_gaps` / `assemble_phases` collapse
to the highest-conf ball per frame), but it violates the track_id invariant.

**Fix: emit only the highest-conf ball per frame.** At ball emission (`roboflow.py:530-552`),
select `argmax(conf)` among `b_dets` and emit that single row. The synthetic id is then unique by
construction, with no consumer impact (downstream already keeps the top-conf ball per frame).

**Why not a within-frame index:** adding a sub-index to the synthetic id complicates the negative
-id scheme for no consumer benefit, since nothing wants multiple balls per frame.

**Test:** in the §3.3 canned-model test, feed two ball detections in one frame with different conf
and assert exactly one ball row is emitted, that it is the higher-conf one, and that its
`track_id == _BALL_TRACK_ID_BASE - frame_idx`.

### 3.8 Stale metadata + dead code [LOW]

**Where:** `roboflow.py:282` — `version: Final = "main@2026-05-28"` (not a real ref);
`roboflow.py:23-25` — vestigial `if TYPE_CHECKING: ... pass`; `pipeline.py:91` —
`apply_modal_team_per_track` is a no-op *for this backend* (team is already per-track constant from
`:582-584`).

**Fix:**
- Pin `version` to the actual upstream `roboflow/sports` git ref the adapter wraps (real
  commit SHA or release tag), recorded from the lockfile/installed package — **the exact ref is a
  human input** (see Decisions).
- Delete the empty `if TYPE_CHECKING: pass` block (`:23-25`).
- Add a one-line comment at `pipeline.py:91` that `apply_modal_team_per_track` is idempotent for
  the roboflow backend (per-track-constant team) but is a real smoother for backends that predict
  team per frame — so it stays.

**Test:** a `version` regex assertion (matches a SHA or `vX.Y`/tag, rejects `main@<date>`). The
dead-block deletion and comment are covered by the existing `ruff`/`mypy --strict` clean gate; no
behavioral test.

### 3.9 Labeler infrastructure [LOW]

These four live in the labeler but are infrastructure, not calibration geometry.

**(a) `compute_chain` multiprocessing hang on macOS spawn — `chain.py:13,151,185-198`.**
The `multiprocessing.Pool` path (`with Pool(...) as pool:` `:195`) hangs when the labeler is run
backgrounded on macOS (spawn start-method + pickling), which is why the operational workaround is
`--workers 1`. **Fix: replace `multiprocessing.Pool` with `concurrent.futures.ThreadPoolExecutor`**
(keep the existing chunking and per-chunk `cv2.VideoCapture`). The per-chunk work is OpenCV/ORB +
numpy, which release the GIL, so threads give real parallelism without spawn/pickle fragility, and
do not hang when backgrounded. Keep `workers` default at `cores-1`.
**Test:** `compute_chain(tiny_video, workers=2)` returns the same interframe map as
`workers=1` (determinism/equivalence) and completes — the test running to completion is the
no-hang proof. (No `multiprocessing.Pool` is referenced after the change.)

**(b) `frame_jpeg` re-seeks a shared `VideoCapture` every request — `server.py:197-204`.**
Each `/api/frame/<idx>` does `cap.set(POS_FRAMES, idx)` + `read()`, an expensive seek per request.
**Fix:** wrap the JPEG bytes in a small LRU cache (`functools.lru_cache(maxsize≈64)` keyed on
`idx`) plus a sequential fast-path (when `idx == last_pos + 1`, `read()` without `set()`); add a
comment that this assumes ALL-INTRA clips (the documented Trace operating mode) so per-frame seeks
are cheap and decode order is stable.
**Test:** with a fake capture counting `set`/`read`, assert requesting the same `idx` twice decodes
once (cache hit) and that a sequential `idx` does not call `set`.

**(c) `_state_payload` computes status twice per poll — `server.py:54-66`.**
`status_buckets()` builds the full `status_list()` (`state.py:336`) and `coverage()` (`:326`)
walks `_status_of` over every frame again — two full status passes per `/api/state`.
**Fix:** add a `LabelerState` method that computes `status_list` once and derives both `coverage`
and `status_buckets` from that single list; `_state_payload` calls it once.
**Test:** monkeypatch `LabelerState._status_of` to count calls; assert one full pass (`n_frames`
calls), not two, per `_state_payload`.

**(d) Bulk `add_clicks` / `add_line_clicks` extend the lists without the lock —
`state.py:240-248`, `:259-264`.** Safe today only by the boot-ordering invariant (called
single-threaded before the worker serves), as the comments note; a future post-boot caller would
race the worker reading `self.clicks` / `self.line_clicks` / `self._seq`.
**Fix:** wrap the `extend`s of `clicks`/`line_clicks`/`_seq` in `with self._lock:` (cheap;
`_recompute_all` already runs after). This makes the methods unconditionally safe and removes the
implicit-ordering footgun.
**Test:** call `add_clicks` after the worker has started and assert `len(self.clicks)` and
`len(self._seq)` stay in lockstep and consistent (the lock makes the extend atomic vs the worker).

### 3.10 Bake-off methodology note [MED — documentation only]

No code change. Record in this spec that the tracking-backend bake-off decision hinged on a
1-point margin driven by a **"homography fidelity"** scoring axis, which (i) rested on the
per-frame camera-model premise that SP1 has since reframed, and (ii) is a property of pitch
detection, which is **separable** from the tracker (§3.2: one-pass coupling is a perf choice, not
a fidelity requirement). Conclusion for the record: homography fidelity should not have been an
intra-tracking-backend scoring axis; the backend choice is **reopenable** if evidence warrants,
but there is currently **no evidence the tracker is the bottleneck** — hygiene already addressed
track shredding (867→400 tracks, balance 1.11). No re-run is scheduled; this is a note so the
decision's provenance is honest, not a task.

## 4. Testing (TDD)

Tests are written before the fixes, in lockstep with §3. New / extended files:

- `tests/test_pipeline_analyze_video.py` — own/opp grounding wiring (§3.1: inversion guard +
  ungrounded warning + GK positional team via §3.4); `analyze_video(backend=MockBackend())`
  end-to-end (§3.2).
- `tests/test_hygiene_core.py` — `map_own_cluster` permutation-invariance (§3.1c). (Pure;
  extends the existing hygiene-core tests.)
- `tests/test_tracking_protocol.py` / `tests/test_tracking_mock.py` — extended protocol incl.
  `process_with_pitch`; `MockBackend.process_with_pitch` schema conformance (§3.2).
- `tests/test_tracking_roboflow.py` — `_run_pipeline` canned-model core test covering foot-point
  math, ball center, class map, per-track team application, synthetic ids, keypoint reshape
  (§3.3); single-decode assertion (§3.6); ByteTrack kwargs forwarding (§3.5); within-frame
  highest-conf ball (§3.7); `version` regex (§3.8).
- `tests/test_labeler_chain.py` — `workers=2` thread-pool equivalence + no-hang (§3.9a).
- `tests/test_labeler_server.py` — frame JPEG cache + sequential fast-path (§3.9b); single
  status pass per `_state_payload` (§3.9c).
- `tests/test_labeler_state.py` — locked bulk add consistency (§3.9d).

All new code paths must keep `mypy --strict` + `ruff` clean (the audit's standing gate).

## 5. Risks

- **Grounding depends on homography quality.** Pitch-space stitching/clustering in `run_hygiene`
  needs usable `x_pitch`; while SP1's calibration is in flight, grounding quality tracks
  whatever homographies exist. This is a pre-existing dependency of the (already-shipped) hygiene
  path, not new — but the §3.1 deliverable is only as grounded as the homographies fed in.
  Mitigation: the `hygiene_report.json` `balance_gate` already surfaces a bad grounding; surface
  its warning through `analyze_video`'s logs.
- **`analyze_video` → `hygiene.run` coupling.** Must use a lazy import to avoid the circular
  `pipeline ↔ hygiene.run` import; the cleaned-trajectories disk round-trip is the seam. If the
  extra `orig_track_id` column from `stitch_tracks` ever trips `validate_trajectories`, the schema
  validator must be confirmed to tolerate extra columns (it currently checks a required subset).
- **ThreadPoolExecutor throughput.** If ORB/cv2 hold the GIL more than expected on some builds,
  threaded chain precompute could be slower than processes — but it is *correct and non-hanging*,
  which is the priority; `workers=1` remains a fallback.
- **Canned-model test fragility.** The `_run_pipeline` fakes must track the `supervision`
  `Detections`/`KeyPoints` API shape; pin the asserted fields to what the code reads
  (`xyxy`, `class_id`, `confidence`, `tracker_id`, `xy`, `confidence`) so the fakes are minimal.

## 6. Out of scope (handled by sibling sub-projects)

Calibration global-homography rework + honest status/export (SP1); possession anisotropy, the
defend-third boundary, transition windows, and halftime/attack-direction normalization (SP3); the
remaining pipeline/eval lows and the Phase-4 metrics product (SP4/SP5). The refuted findings
(`_infer_fps` rounding; eval aspect-ratio 1.5-vs-1.54) are excluded entirely — they are not bugs.
