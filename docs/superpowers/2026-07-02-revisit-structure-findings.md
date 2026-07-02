# Revisit-Structure Finding — annotation scales with VIEWS, not frames (2026-07-02)

## Question
Manual anchor labeling doesn't scale to a full game: an efficient full 60-min game is
~540 anchors / ~6,500 clicks / 2–3 hrs (measured from the oceanside session: 849 clicks for
90 s → 78% green). Is there structure to exploit? A fixed Trace camera pans back and forth over
the same field, so distinct fields-of-view should be far fewer than frames.

## Method (no GPU, classical)
On `oceanside_clip.mp4` (2700 frames / 90 s), sampled every 25 frames (108 frames):
- **ORB self-similarity matrix:** per sampled frame, ORB features (1200) on a 0.5-downscaled
  gray frame; pairwise similarity = fraction of cross-checked Hamming matches (dist < 48).
  Agglomerative clustering (scipy, average linkage) on `1 − similarity`.
- **Pan trajectory:** cumulative chain transform → x-offset at image centre vs time.
Script: `/tmp/revisit.py` (to be productionized).

## Result — hypothesis validated
- **Distinct views ≈ 13** (30 at a strict threshold, 4 at a loose one) across all 2700 frames —
  **tens, not thousands**, stable across thresholds.
- **66% of frames revisit a view elsewhere** in the clip; mean **4.1 temporally-distant revisits
  per frame**.
- **Compression ≈ 90–208×** (frames per distinct view).
- **Caveat:** the chain-derived pan trajectory is **drift-dominated** (cumulative registration
  error over 2700 frames) — unreliable as a pan measure, but a direct picture of the drift that
  motivates loop closure. The similarity matrix is drift-free (compares image content) — trust it.
- ORB was **unmasked** (players included); masking would sharpen clusters further.

## Implication (the productization unlock)
Annotation burden is bounded by the number of **distinct views (~tens), which is ~constant with
game length** (same fixed camera + field → views recur). So **per-view** calibration could take a
full game from ~540 anchors to **~tens of anchors, total, regardless of duration** — impractical →
minutes per game.

## Direction: view-clustering + loop-closure calibration
1. **Cluster** frames by view (masked ORB / global descriptor now; a learned embedding later).
2. **Calibrate one representative per view** (labeler, eventually a pose-regression model).
3. **Assign every frame its representative's homography by direct view→representative registration**
   — a single hop, **drift-free**, replacing the long temporal chain composition.

This reframes scaling as a **view-clustering + loop-closure** problem, not "click more anchors."
Distinct from the auto pitch-detection (pose-regression) path, which is the eventual zero-click
end state; view-clustering is the near-term, no-GPU, buildable-now reducer and a stepping stone.
