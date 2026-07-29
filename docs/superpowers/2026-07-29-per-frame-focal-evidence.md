# Per-frame focal: before/after evidence

Baseline numbers captured before the per-frame focal engine change described in
`docs/superpowers/specs/2026-07-28-per-frame-focal-design.md`. Both baselines below were run
against the unmodified shared-K physical engine on 2026-07-29.

## Baseline (shared-K engine)

**Frozen-sidecar metrics** (anchors / green count / median in-sample error / shared K), run via a
scratchpad script that builds `Click`/`LineClick` lists from
`~/sv-labeler/home_g4_oceanside/rep_video.clicks.frozen-2026-07-24.json`, calls
`physical_calib.solve_session(points, lines, (1920, 1080), {})`, and scores every click against its
frame's `anchor_h` with `_point_feet`/`_line_perp_feet`:

```
cd packages/soccer-vision && uv run python <scratchpad>/engine_metrics.py
```

```
anchors: 112  green: 86  median in-sample err: 2.58 ft  K: 1445.6px
```

Output was identical across two consecutive runs (deterministic). The script's per-frame focal
block (`calib.focal_of` / `calib.focal_source`) printed nothing, as expected for the shared-K
engine.

**Shipped-session held-out gate** (chain npz `da63d2bb640cc974.npz` matched the clicks session on
the first try):

```
cd packages/soccer-vision && uv run python -m soccer_vision.pitch.validate_session \
  --chain ~/sv-labeler/.sv_labeler_cache/da63d2bb640cc974.npz \
  --clicks ~/sv-labeler/.sv_labeler_cache/oceanside_clip.clicks.json \
  --engine physical
```

```
[physical] held-out acceptance gate (feet):
  foreground   median=  4.18  p90=  9.72  n=157
  propagation  median=  5.85  p90= 21.13  n=366
  NUMERIC (fg med<=5 & p90<=12, prop med<=5): FAIL
```

Note vs the 2026-07-24 audit expectations: anchors (112) and green (86) match exactly; shared K is
1445.6px vs the audit's 1444.1px, and median in-sample error is 2.58 ft vs the expected
~3.3-4.0 ft. Both runs here are the recorded ground truth for the after-comparison.

## After (per-frame focal)

(filled by Task 5)
