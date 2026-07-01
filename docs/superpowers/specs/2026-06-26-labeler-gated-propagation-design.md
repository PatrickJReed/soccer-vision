# Proximity-seeded, consistency-gated click propagation (design)

**Date:** 2026-06-26
**Status:** Design approved, pending implementation plan
**Depends on:** `pitch/calib_anchor.poses_by_click_propagation` (current per-frame engine),
`pitch/manual_anchor` (`propagate_clicks`, `cumulative_transforms`, `build_segments`),
`calib/field_model.field_points_3d`, the calibrated `LabelerState`
(`_compute_poses`/`_compute_dirty`), the async refit worker (just shipped).

## Problem

The labeler fits each frame's camera pose by propagating clicks across the registration
chain within a fixed `±window` (default 360) and solving one SQPNP pose from **all**
in-window landmarks (nearest source click per landmark). On a real session this projects
badly in densely-clicked regions, and the diagnosis is conclusive:

- The **focal is correct** (1471, matches prior games), the chain is one clean segment, and
  every clicked frame fits its **own** clicks to **8–17 px** — so the clicks are good, not
  mislabeled.
- But the **propagated** fit explodes on the early cluster (frames 0–193: **315–3370 px**)
  while later frames stay 11–110 px. Those early own-goal-end frames reach across a fast
  own→opp camera pan (~frames 152–173, where chain registration is shakiest) and pull in the
  opp-end clicks from frame 193. Propagated across that drift, those points scatter and drag
  the pose to garbage.

Root cause: **the engine aggregates every in-window click with equal trust**, so a far click
mapped across a drifted chain pollutes a frame that would have been perfect from its own
local clicks. Perversely, the user's *dense* clicking of both ends made it *worse*. A single
global `window` cannot serve both densely-clicked regions (want a tight, local fit) and
sparse gaps (want reach for coverage). Frame-distance is also a poor reliability proxy: the
own/opp clusters are only ~50–60 frames apart, but the pan between them is unreliable, so even
a ±90 seed window re-mixes them (frames 134/144/193 → 1296/482/1702 px).

## Goal

Make propagation **reliability-aware**: a frame's pose is defined by its nearest, most
reliable clicks; far clicks contribute only when the chain to them is demonstrably accurate;
and a frame with no reliable nearby anchor stays **red** (an honest "re-anchor here" signal)
rather than projecting a low-confidence overlay from drifted far clicks. Never let a far,
chain-drifted click corrupt a frame that has good local clicks — more clicks must only ever
help.

User decisions captured during brainstorming:
- **Gap behavior:** stay red / prompt re-anchor (do NOT fake coverage from drifted far clicks).
- **Mechanism:** proximity-seeded, consistency-gated fit (approach "A"), not coverage-weighting
  or a chain-reliability model.
- **`gap_dist` default 180** (≈6 s): validated 94% green with every clicked frame at 8–20 px;
  the reds are genuine >6 s-from-any-click gaps.

## Design

### The algorithm — `poses_by_gated_propagation` (per target frame `f`)

1. **Propagate with source tracking.** For each landmark, find its *nearest* source click
   (same chain segment, `|c − f| ≤ max_reach`, default `max_reach=360`), map it into frame
   `f` via the chain exactly as `propagate_clicks` does, and keep the source frame-distance
   `d` per landmark. (Today's `propagate_clicks` discards `d`; the gated engine needs it.)
2. **Gap cap (stay-red).** Let `d_near = min_landmarks d`. If `d_near > gap_dist`
   (default 180) → emit nothing for `f` (RED). No reliable nearby anchor.
3. **Seed.** Take the `min(seed_size, n_available)` landmarks with the smallest `d`
   (`seed_size` default 6; if the frame has fewer total landmarks, the seed is all of them) →
   SQPNP → seed pose. For a clicked frame these are its own clicks (`d=0`) → a clean local
   pose; the cross-pan-drifted far clicks never enter the seed. RED if `< min_points`
   (default 4) landmarks exist in total, or seed SQPNP fails.
4. **Consistency gate.** Reproject every propagated landmark (seed + farther candidates)
   under the seed pose; **accept** a non-seed landmark only if its reprojection error
   `≤ gate_px` (default 60). Consistent ⇒ the chain to it is reliable (use it for depth);
   scattered ⇒ dropped. Seed landmarks are always kept.
5. **Final fit.** SQPNP on `seed ∪ accepted`; if the frame has `line_obs`, run `refine_pose`
   on the kept point set + lines (the Phase-2 path, unchanged). RED if `< min_points` kept.
   `residual_px` / `fold_count` / `n_points` are computed on the kept set, so the existing
   green/yellow/red status is meaningful (a degenerate or shallow seed yields a high final
   residual → naturally not green, never a false "green").

Returns `{frame: FramePose}` over the non-red frames — same shape as
`poses_by_click_propagation`, so it is a drop-in replacement.

### Why this resolves the failure (validated on the real session)

| | clicked-frame residual | full-clip green |
|---|---|---|
| current (`window=360`) | 315–3370 px ✗ | 1755/2700 |
| gated (`gap_dist=180`) | **8–20 px ✓** (incl. 134/144/193) | **2532/2700 (94%)** |
| gated (`gap_dist=240`) | 8–20 px | 2676/2700 (99%) |
| gated (`gap_dist=120`) | 8–20 px | 1890/2700 (70%) |

The seed is always the nearest cluster (its own clicks for a clicked frame), so the pan-drift
never enters the pose; the gate drops the scattered far points (frame 0: 18 candidates → 8
kept → 8 px); and `gap_dist` makes true gaps honest reds. Because acceptance requires
reprojection consistency, "fit" ⇒ "green" — coverage is trustworthy coverage.

### Where it lives & integration

- **New function** `poses_by_gated_propagation(clicks, transforms, segment_of, k, size, *,
  max_reach, seed_size, gate_px, gap_dist, min_points, line_obs=None, frames=None)
  -> dict[int, FramePose]` in `pitch/calib_anchor.py`. It needs propagation that exposes the
  per-landmark source distance; factor a small helper out of `manual_anchor.propagate_clicks`
  (or add a `with_distance` variant) so both stay DRY.
- **`poses_by_click_propagation` is kept** unchanged for `pitch/calib_compare.compare_engines`
  (Phase 3a A/B tooling) — this is an additive engine, not a rewrite.
- **`LabelerState`** gains constructor params `seed_size=6`, `gate_px=60.0`, `gap_dist=180`
  (and reuses its existing `window` as `max_reach`); `_compute_poses` calls
  `poses_by_gated_propagation(...)` instead of `poses_by_click_propagation(...)`. Everything
  downstream — the async worker, `_compute_dirty`/`_apply_fits`, `line_obs`, autosave/export,
  `/api/state` coverage, the frontend — is unchanged.

### Edge handling

- `< min_points` propagated, seed SQPNP failure, or `< min_points` surviving the gate → the
  frame is omitted (RED) — `_apply_fits` already pops omitted frames.
- `d_near > gap_dist` → omitted (RED).
- Degenerate / shallow seed → high final residual → non-green via the existing threshold
  (no special-casing).
- `line_obs` present but the gated point set is too small for `refine_pose` → `CalibError`
  is caught (today's behavior), keeping the SQPNP pose.

## Testing

- **`poses_by_gated_propagation` (synthetic, the gate is the gate):**
  - *Corrupted-far-click rejected:* a frame with good near clicks plus one far landmark whose
    propagated pixel is deliberately offset (simulating chain drift) → the far landmark is
    gated out and the pose matches the near-only fit (vs `poses_by_click_propagation`, which is
    dragged off). This is the core regression test.
  - *Stay-red gap:* a frame whose only clicks are beyond `gap_dist` → not in the result.
  - *Consistent far accepted:* on a clean (identity-ish) chain, far clicks reproject within
    `gate_px` → accepted, and the kept count exceeds the seed size (cross-pan depth is used
    when reliable).
  - *Seed = nearest:* a clicked frame seeds from its own clicks (kept count ≥ its own click
    count), not from a distant cluster.
- **`LabelerState`:** `_compute_poses` now routes through the gated engine; a synthetic
  `_pan_session` with an injected mid-span drift shows a clicked frame's fit stays accurate
  where the old engine degraded. Existing state/async tests still pass (the swap is transparent
  to coverage/worker semantics).
- **Real-data validation (numbers + Patrick's eyes):** re-run the saved session through the
  gated engine; report the per-clicked-frame residuals (expect 8–20 px) and the green-coverage
  map; Patrick assesses whether the overlay now projects correctly. Claude renders, Patrick
  interprets.

## Non-goals

- A chain-reliability model from repeated clicks (approach "B") or coverage-maximizing
  down-weighting (the rejected gap option) — deferred; the consistency gate + `gap_dist`
  suffice.
- Rewriting or removing `poses_by_click_propagation` (kept for `compare_engines`).
- Changing the focal bootstrap, outlier preprocessing, line-click math, async worker, autosave,
  export, or frontend.
- New UI to distinguish "red: no clicks" from "red: unreliable" (the frame is simply red either
  way; a future polish).
- Per-session auto-tuning of `gap_dist`/`gate_px` (constructor params with validated defaults;
  a CLI flag can come later if needed).
