# View-Registration Calibration (annotation-scaling, Slice 2) — Design

**Date:** 2026-07-06
**Status:** Proposed — awaiting review
**Sub-project:** annotation-view-clustering, Slice 2 (classical, no GPU).

## Goal
Give every frame of a clip a **drift-free** pitch homography by registering it *directly* to its
view's labeled representative — a **single** homography hop — instead of composing the long temporal
chain of pairwise inter-frame homographies (which accumulates drift: measured 8 px on short spans →
40–266 px on the own→opp pan). This is the calibration payoff of the view-clustering direction:
label ~13 representatives once, then propagate their homographies to the whole game with no drift.

Depends on Slice 1 (`view_digest.py` → the ~13 views + medoid representatives). Reuses the ORB
registration machinery in `pitch/propagation.py` and the homography I/O in `labeler/chain.py`.

## The math (why it's drift-free)
For a frame `f` assigned to the view whose representative is `R`:
```
G   = register(f_px → R_px)          # one ORB homography (this frame ↔ its rep)
H_f = H_R_pitch @ G                    # f_px → pitch
```
`H_R_pitch` (R_px → pitch) is the representative's **labeled** homography. The labeler exports
full-pixel→pitch homographies (`state.py:393` applies `denormalize_homography`), and `register()`
operates in full-pixel space, so `H_R_pitch @ G` is directly f_px → pitch — no normalization juggling.
Error is bounded by one registration's accuracy; it does **not** grow with temporal distance from a
labeled frame (unlike the chain).

## Approach — assignment + registration unified in one ORB pass
Per frame `f`:
1. ORB `f` once (full-res; masked over players when boxes are supplied).
2. Match `f`'s descriptors against each of the ~13 rep descriptors (precomputed); rank reps by
   good-match count.
3. Try `findHomography` (RANSAC) on the **top-2** candidate reps' matched inliers; keep the one with
   the most RANSAC inliers (≥ `min_inliers`). This handles view-boundary frames that match a neighbor
   rep better. → `G`, chosen rep `R`, inlier count.
4. `H_f = H_R_pitch @ G`; `confidence` from inlier count. If both candidates fail, `source="none"`
   (an honest gap, counted in coverage — no temporal fill in v1).
Rep frames themselves pass through with their labeled homography (`source="rep"`, confidence 1.0).
A view whose rep has no labeled homography is unusable (its frames fall back to other reps or gap).

## Module: `pitch/view_registration.py` (pure core + video I/O + CLI)
```python
from soccer_vision.pitch.propagation import HomographyEntry   # reuse (H, source, confidence)

@dataclass(frozen=True)
class RegisteredCalib:
    homographies: dict[int, HomographyEntry]   # frame -> (H f_px->pitch, source, confidence)
    rep_of: dict[int, int]                      # frame -> view id of the rep it registered through
    stats: dict[str, Any]                       # coverage, median_inliers, per_view_counts, n_gap

# --- pure core (unit-tested on synthetic descriptors/homographies, no video) ---
def compose_pitch_homography(H_rep_pitch, G_frame_to_rep) -> NDArray[np.float64]   # H_rep @ G, /H[2,2]
def register_to_best_rep(frame_kp, frame_desc, rep_kps, rep_descs, *, min_inliers=12,
                         top_k=2) -> tuple[int, NDArray, int] | None
    # match frame vs each rep, rank by good-match count, findHomography on top_k, return
    # (best_rep_index, G full-px, n_inliers) or None if all fail

# --- video I/O ---
def rep_homographies_from_parquet(homographies_parquet: Path,
                                  representatives: dict[int, int]) -> dict[int, NDArray]
    # extract the labeled full-px->pitch H at each rep's frame index (skip reps not present)
def register_clip(video_path, digest: ViewDigest, rep_homographies: dict[int, NDArray], *,
                  player_boxes=None, frames=None, n_features=3000, min_inliers=12,
                  cache_dir=None) -> RegisteredCalib
    # ORB each rep once (cached); for each frame -> register_to_best_rep -> compose. `frames`
    # defaults to all video frames; rep descriptors + per-frame results are the work.
def write_homographies(calib: RegisteredCalib, out_path: Path) -> None   # reuse homographies_to_parquet

# --- validation (the drift-free proof) ---
def cross_registration_error(video_path, digest, rep_homographies, *,
                             n_features=3000, min_inliers=12) -> pd.DataFrame
    # for each labeled rep R: register R's frame to EVERY other labeled rep R', compose
    # H = H_R'_pitch @ G(R->R'), reproject the 4 image corners to pitch, compare to R's own
    # manual H_R. Returns rows (rep_R, rep_Rprime, temporal_dist=|frame_R-frame_R'|,
    # corner_err_pitch). Drift-free => corner_err ~flat in temporal_dist.

def main(argv=None) -> None
    # CLI: --video --digest-json (from render_digest) --rep-homographies <labeled parquet>
    #      --out --boxes --min-inliers [--validate]  ; python -m soccer_vision.pitch.view_registration
```

## Output — drop-in `homographies.parquet`
Same schema the pipeline/metrics already consume (`frame, h00..h22, source, confidence`), written via
the existing `homographies_to_parquet`. `source ∈ {"rep","registered","none"}`. So Slice 2 output
slots straight into `pipeline.assemble_from_homographies` → the Phase-4 metrics run on a full game
with a drift-free calibration.

## Reused machinery (do not reinvent)
- `propagation.register` / `_orb_downscaled` / `_homography_from_descriptors` — ORB + findHomography.
- `propagation._frame_mask` — 255 background / 0 over dilated player boxes.
- `propagation.HomographyEntry` — the (H, source, confidence) record.
- `chain.homographies_to_parquet` / the `h00..h22` schema (from `state.py`/`pipeline.py`).

## Validation / acceptance (chosen: flat error vs temporal distance)
On oceanside (12/13 reps already labeled in `~/sv-labeler/out/homographies.parquet`):
- **`cross_registration_error`** → mean corner reprojection error (pitch units) is ~**flat** across
  temporal distance |R−R′| (e.g. within a small band), i.e. registering through a far rep is no worse
  than a near one — the definition of drift-free. Contrast the chain, whose error grows 8→266 over the
  pan. Report the numbers as facts (Patrick interprets any rendered plot).
- **Coverage**: report % frames registered vs gap, median inliers, per-view counts.
- Full suite + mypy + ruff clean.

## Testing (TDD, synthetic — no real video for unit tests)
- `compose_pitch_homography`: known H_rep + G → expected mapping; normalizes H[2,2].
- `register_to_best_rep`: synthetic frames — one warped copy of rep A + a distinct rep B → picks A,
  returns a G that maps the warp back (reproject a few points within tolerance); all-mismatch → None;
  `min_inliers` gate respected; top-2 fallback picks the higher-inlier rep.
- `rep_homographies_from_parquet`: builds a tiny homographies parquet, extracts H at rep frames,
  skips a rep whose frame is absent.
- `register_clip` (synthetic mp4, skip if no writer): 3 planted views + rep homographies → every
  frame gets a `registered`/`rep` homography for decodable frames; a blank/blurred frame → gap;
  output parquet has the right schema; rep frames keep `source="rep"`.
- `cross_registration_error`: on the synthetic clip, error is small and does not grow with distance.

## Out of scope (deferred)
Temporal-neighbor gap fill; re-labeling reps (existing labeler covers it; 12/13 already labeled);
learned/embedding-based assignment (Slice 3); multi-game; masking beyond the existing box path.
