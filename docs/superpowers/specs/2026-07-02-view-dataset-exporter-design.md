# View-Dataset Exporter (annotation-scaling, Slice 1.5) — Design

**Date:** 2026-07-02
**Status:** Proposed — awaiting review
**Sub-project:** annotation-view-clustering, Slice 1.5 (classical, no GPU).
**Provenance:** synthesized by a 3-design judge panel (manifest-first won 28/30), then YAGNI-scoped.
Panel raw output archived in the workflow transcript.

## Goal
Turn a video + its `ViewDigest` (Slice 1) into a training dataset the Colab model consumes,
serving **both** downstream objectives from **one** export:
- **contrastive / metric-learning embedding** — same-view frames = positives, via a per-frame
  `view_id` **pseudo-label**;
- **masked autoencoder** — mask players, reconstruct the field, bottleneck = embedding.

Every frame (dense stride) is assigned a `view_id` by **nearest-representative ORB match**, with a
`confidence` and `margin`, so Colab can use the labels (contrastive) or ignore them (MAE).

## Validated foundation (measured, oceanside held-out)
Nearest-rep assignment is temporally coherent (5.1% switch rate, 60-frame median runs, ~no noise);
confidence never <0.15 (a QC/weight signal, **not** an abstain gate); **margin** is the real
ambiguity signal (~6% pan-transition frames); a majority-vote temporal filter removes the lone
singleton. No symmetry aliasing on this clip.

## Approach — manifest-first (zero pixels by default)
Default export writes **no image files** — a ~2 MB parquet manifest + a self-describing JSON
sidecar. Colab reads the manifest and decodes frames on demand from the **same ALL-INTRA mp4** the
labeler used (keyframe seeks are cheap under the ALL-INTRA re-encode convention). Re-export at a new
margin/smoothing/split is an **instant cached pure-pandas rewrite** (no re-decode). An optional
`materialize(layout='folders')` escape hatch writes a portable ImageFolder tree for non-ALL-INTRA
sources or GPU-I/O-bound Colab runs.

### Module: `labeler/view_dataset.py` (tests `tests/test_view_dataset.py`)
Sits beside `view_digest.py`, imports from it. Distinct from top-level `dataset_export.py` (the
deferred YOLO-pose exporter). Same house style: pure core + thin I/O + `.npz` cache + CLI.

**One DRY refactor in `view_digest.py`:** extract the per-pair scorer out of `similarity_matrix`
into `_pair_match_fraction(da, db, ca, cb, *, min_match_dist, min_keypoints) -> float`; call it from
**both** `similarity_matrix` (behavior unchanged — pinned by existing `test_similarity_*`) and the
new `cross_match_fractions`. A parity test asserts the two metrics can never drift.

### Public API
```python
# --- refactor in view_digest.py ---
def _pair_match_fraction(da, db, ca, cb, *, min_match_dist=48, min_keypoints=10) -> float

# --- new module labeler/view_dataset.py ---
@dataclass(frozen=True)
class ViewAssignment:
    manifest: pd.DataFrame
    representatives: dict[int, int]
    meta: dict[str, Any]
    # properties: n_frames, n_views, n_ambiguous, switch_rate
    def materialize(self, out_dir, *, video_path, image_downscale=0.5,
                    jpeg_quality=90, boxes=None) -> list[Path]  # layout='folders' only (v1)

# pure core (unit-tested on synthetic frames / matrices, no video):
def cross_match_fractions(query_desc, query_counts, ref_desc, ref_counts, *,
                          min_match_dist=48, min_keypoints=10) -> NDArray  # (Q,R), same metric
def assign_nearest_view(match, ref_view_ids) -> pd.DataFrame
    # cols: view_id_raw:int32, view_second:int32 (runner-up VIEW), confidence:float32 (best),
    #       margin:float32 (best - best-match-to-a-DIFFERENT-view); all-zero row -> view_id_raw=-1
def smooth_view_sequence(view_ids, *, window=5) -> NDArray[int]
    # odd-window sliding MAJORITY vote; window<=1 -> identity; tie -> keep original
def assign_splits(manifest, *, val_frac=0.1, policy='per_view_tail', holdout_views=None) -> DataFrame
    # per_view_tail: last val_frac of each view's temporally-sorted rows -> val (every view in both)
def build_manifest(query_frames, match, ref_view_ids, keypoint_counts, *, game, fps,
                   n_boxes=None, ambiguity_margin=0.05, smooth_window=5, val_frac=0.1,
                   split_policy='per_view_tail') -> pd.DataFrame  # PURE assembly from a match matrix

# video I/O:
def build_view_assignment(video_path, digest, *, game=None, assign_stride=5, n_features=1200,
                          downscale=0.5, min_match_dist=48, min_keypoints=10, ambiguity_margin=0.05,
                          smooth_window=5, val_frac=0.1, split_policy='per_view_tail',
                          player_boxes=None, chunk=512, cache_dir=None) -> ViewAssignment
    # SINGLE forward-grab streaming decode; chunked (frame_descriptors -> cross_match_fractions ->
    # append rows -> discard pixels); caches raw (Q,R) match in viewassign_<hash>.npz; masked runs
    # bypass the cache (mirrors compute_view_digest)
def write_export(assignment, out_dir, *, video_path) -> list[Path]  # parquet(index=False) + json

# Colab entrypoints:
def load_manifest(export_dir) -> tuple[pd.DataFrame, dict]
class ViewFrameReader:  # forward-grab fast path + POS_FRAMES seek fallback (mirrors _read_frames)
    def __init__(self, video_path, manifest, *, boxes=None, downscale=1.0)  # raises on video_hash mismatch
    def __len__(self) -> int
    def read(self, i) -> tuple[NDArray, NDArray | None, pd.Series]  # (frame_bgr, keep_mask|None, row)
    def close(self) -> None

def main(argv=None) -> None
    # CLI: --video [--digest-json] --out --game --assign-stride --smooth-window
    #      --ambiguity-margin --val-frac [--boxes] [--materialize --image-downscale --jpeg-quality]

# constants: DEFAULT_ASSIGN_STRIDE=5, DEFAULT_AMBIGUITY_MARGIN=0.05, DEFAULT_SMOOTH_WINDOW=5,
#            DEFAULT_VAL_FRAC=0.1, DEFAULT_CHUNK=512, SCHEMA_VERSION=1 (reuse view_digest DEFAULT_*)
```

### Manifest schema (`view_dataset.parquet`, index=False, sorted by frame)
`game:str, frame:int64, t_seconds:float64, view_id:int32 (smoothed pseudo-label; -1=unassigned),
view_id_raw:int32 (pre-smoothing, audit), view_second:int32 (runner-up view for 2-hot soft labels),
view_key:str ('{game}:{view_id}', cross-game-safe contrastive class), confidence:float32 (best match
fraction, QC), weight:float32 (=confidence, loss weight), margin:float32, ambiguous:bool
(margin<ambiguity_margin), n_keypoints:int32, n_boxes:int32 (0 unmasked), split:str`. ~20k rows → <2 MB.

### Sidecar (`view_dataset.json`)
`schema_version, dataset_fingerprint (sha1 of params); video:{path,abspath,video_hash,n_frames,fps,
width,height}; params:{...}; digest:{n_views,representatives}; stats:{per_view_counts,n_ambiguous,
n_train,n_val,confidence_median,margin_median,switch_rate}; boxes_source`. Reloadable from manifest +
mp4 alone.

### Cache (`viewassign_<hash>.npz`)
Keyed by video_hash + assign_stride + n_features + downscale + min_match_dist + min_keypoints +
reps_fingerprint. Stores `query_frames, match(Q×R), ref_view_ids, keypoint_counts`. Makes re-export
at a new margin/smoothing/split instant. Masked runs bypass it.

## Two ambiguity signals (from the evidence)
- **confidence** (best match fraction) → QC / loss-`weight` column, **not** an abstain gate.
- **margin** (best − best-other-view) → the real ambiguity signal → `ambiguous` bool + persisted
  `view_second` so Colab can build a 2-hot soft label at pan boundaries. Colab decides per objective.

## Masking hook (optional, additive)
`player_boxes` (trajectories parquet, `frame,bbox_x1/y1/x2/y2`) → per-chunk masks via
`view_digest._build_masks` (255 keep / 0 player) into `frame_descriptors`; records `n_boxes`.
Colab-side `ViewFrameReader` returns a `keep_mask` the MAE unions into its masked set. **Fully usable
unmasked now** (oceanside boxes blocked on the pinned GPU tracking run → `n_boxes=0, keep_mask=None`).
Unit-tested with synthetic boxes; unexercised on real data until tracking lands.

## Determinism
Fixed stride, ascending frame sort, stable dtypes, `index=False` parquet, `video_hash` provenance
(no wall-clock). Tests assert manifest *content* equality + decoded *shape* (not raw parquet/JPEG
bytes) to dodge codec/metadata nondeterminism.

## Testing (TDD)
Pure core on synthetic frames/matrices (no real video):
- `_pair_match_fraction` **parity**: `cross_match_fractions([f], [reps])` equals the cross-block row of
  `similarity_matrix([f, *reps])`.
- `assign_nearest_view`: crafted match matrix → correct view_id_raw / view_second / confidence /
  margin; all-zero row → -1.
- `smooth_view_sequence`: removes a singleton; window=1 identity; tie keeps original.
- `assign_splits`: per_view_tail puts every view in both train and val; val_frac respected;
  holdout_views sends whole views to val.
- `build_manifest`: pure assembly → schema, dtypes, ambiguous flag, view_key, sorted, deterministic.
- masking: synthetic boxes zero the masked region → `n_boxes` recorded, keypoints drop.
- I/O integration (synthetic mp4, skip if writer unavailable): `build_view_assignment` →
  `write_export` → parquet+json exist, schema matches; **cache hit** on 2nd call (no re-decode);
  `ViewFrameReader` round-trips a frame + raises on video_hash mismatch; `materialize` writes a
  relative-path ImageFolder tree.

## Acceptance
On `oceanside_clip.mp4`: `python -m soccer_vision.labeler.view_dataset --video … --game oceanside`
writes `view_dataset.parquet` (~530 rows at stride 5) + json; switch_rate ≈ measured ~5%; a 2nd run
hits the cache instantly. Full suite + mypy + ruff clean. A short Colab-loader snippet (contrastive +
MAE) documented in the module docstring.

## Out of scope (deferred, YAGNI)
WebDataset **tar-shard** materialize layout (build when a real DDP/scale need appears);
`boxes_to_normalized` sidecar rasterization; **cross-game shared view vocabulary** (would require
registering reps across games — flagged, not built); PNG materialize option for MAE fidelity.
