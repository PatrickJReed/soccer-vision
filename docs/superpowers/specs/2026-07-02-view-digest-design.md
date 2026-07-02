# View Digest (annotation-scaling, Slice 1) — Design

**Date:** 2026-07-02
**Status:** Approved — build now
**Sub-project:** annotation-view-clustering, Slice 1 (classical, no GPU).

## Goal
Cluster a video's frames into distinct camera **views** and pick one **representative frame per
view**, so a labeler covers a whole clip by annotating ~tens of representatives instead of
hundreds of temporal anchors. Also the **data engine + baseline** for the later Colab deep model.
Motivated + validated by `docs/superpowers/2026-07-02-revisit-structure-findings.md` (~13 views
on oceanside).

## Approach (classical, CPU)
1. **Sample** frames at `stride` (default 25).
2. **ORB descriptors** per sampled frame (n_features=1200) on a `downscale`d (0.5) gray frame;
   optional player masking when boxes are supplied.
3. **Pairwise similarity** = fraction of cross-checked Hamming matches (dist < `min_match_dist`),
   normalized by min keypoint count. Frames of the same view share static field/stand structure
   → high similarity regardless of temporal distance.
4. **Cluster** into views: scipy agglomerative (average linkage) on `1 − similarity`, cut at
   `dist_threshold` (default 0.5 — finer is the safe bias for calibration transfer: under-
   clustering assigns a representative's homography to frames it does not fit, over-clustering
   only costs a few extra labels; 0.5 reproduces the ~13-view oceanside finding).
5. **Representative** per view = the **medoid** (member with the highest summed similarity to its
   cluster).

## API (`labeler/view_digest.py`, pure core + CLI)
```python
@dataclass(frozen=True)
class ViewDigest:
    sample_frames: list[int]          # sampled frame indices (rows of `similarity`)
    view_of: dict[int, int]           # sampled frame -> view id
    representatives: dict[int, int]   # view id -> representative frame index
    similarity: NDArray[np.float64]   # NxN sampled-frame similarity (diagnostic)

# testable pure helpers:
def similarity_matrix(descriptors: list, keypoint_counts: list[int], *, min_match_dist=48) -> NDArray
def cluster_views(similarity: NDArray, *, dist_threshold=0.7) -> NDArray  # labels (1..K)
def medoid_representatives(labels, similarity, sample_frames) -> dict[int, int]

# video entry:
def compute_view_digest(video_path, *, stride=25, n_features=1200, downscale=0.5,
                        dist_threshold=0.7, min_match_dist=48, player_boxes=None) -> ViewDigest
def render_digest(digest, video_path, out_dir) -> list[Path]  # montage + similarity heatmap
def main(argv=None) -> None  # CLI: python -m soccer_vision.labeler.view_digest --video <mp4>
```
Core clustering (`similarity_matrix`, `cluster_views`, `medoid_representatives`) is pure and unit-
tested; `compute_view_digest`/`render_digest`/CLI are the only I/O (video decode + plot).

## Outputs
- `view_digest.json`: `{sample_frames, view_of, representatives, n_views, stride}` — the
  **representative frame list is "label these frames."**
- `views_montage.png`: the representative frames tiled (so the user sees the distinct views).
- `similarity.png`: the self-similarity heatmap (revisit bands).
- CLI prints: #sampled frames, #views, representative indices, and each view's temporal span/coverage.

## Testing (synthetic, pure core)
- `similarity_matrix`: 3 distinct seeded texture patterns, each duplicated → block-diagonal
  similarity (high within pattern, low across).
- `cluster_views`: a block-structured similarity → correct K labels; threshold controls K.
- `medoid_representatives`: picks the most-central member of each cluster.
- End-to-end on synthetic frames (in-memory arrays via a fake reader) → correct #views + reps.

## Acceptance
On `oceanside_clip.mp4`: reproduces ~tens of views (order of magnitude of the finding), writes the
montage + json + heatmap, CLI prints the representative frames to label. Full suite + mypy + ruff clean.

## Out of scope (later slices)
Dense per-frame view assignment (v1 clusters the sampled frames; representatives are the deliverable),
intra-view direct registration + drift-free propagation (Slice 2), learned embedding / pose
regression (Slice 3, Colab).
