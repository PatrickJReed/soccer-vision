---
name: calibration-global-homography
description: Replace the per-frame independent SQPNP calibration with one global image→pitch homography plus per-frame 2D offsets (H_f = H_global ∘ T(−offset_f)), behind the labeler's existing _compute_poses seam. Fixes the "lines in the sky" under-constraint, chain drift, and the false-green status, validated against held-out cross-end clicks.
status: approved
date: 2026-06-26
---

# Calibration — Global-Homography Rework + Honest Status

Sub-project 1 of the post-audit program (see `docs/superpowers/2026-06-26-codebase-audit.md`).
This is the **critical path**: it is a prerequisite for trustworthy pitch coordinates and
therefore for the Phase-4 metrics product.

## 1. Problem

The shipped labeler solves a **per-frame INDEPENDENT 6-DOF camera pose** (SQPNP, focal shared
via `calibrate_camera`, but `rvec`+`tvec` free per frame) from each frame's propagated clicks.
For a fixed Trace virtual-PTZ camera — where the measured inter-frame motion is a pure 2D
translation (scale≈1.000, rotation/shear≈0, perspective≈1e-6) — this is over-parameterized and
under-constrained:

- Each Trace view sees ~one end of the field. A pose fit to clicks clustered on one end pins that
  end and extrapolates wildly to the unclicked end → the overlay lines project "into the sky"
  (e.g. frame 193, opp-end clicks only → `own_corner_R` y=−464). This is **structural to the
  per-frame-independent model, not a bug in it.**
- Cross-end constraint is only available through a frame-to-frame registration chain that
  accumulates error (8 px short-span, 40–266 px long-span), made worse by fitting full 8-DOF
  homographies per pair where ~2 DOF (translation) exist.
- The green/export status gates on **in-sample reprojection residual over only the gate-kept
  near landmarks**, so a single-end "sky" frame scores 8–20 px → green and is exported as
  `source="manual", conf≈0.87` — false confidence that would poison every downstream metric.

Findings closed by this sub-project: `calib-per-frame-6dof-illposed` (critical),
`calib-chain-drift-structural`, `per-frame-homography-ill-posed-in-spec`,
`labeler-perframe-engine-illposed-coupling`, `calib-over-engineering-vs-goal`,
`calib-green-false-confidence`, `labeler-false-green-export-poisons-metrics`,
`labeler-export-wait-idle-silent-truncation`, `geometry-over-modeled`,
`calib-camera-model-inconsistency`, `calib-orphaned-engines-and-ml-path`,
`calib-global-fix-unvalidated-assumptions`, `labeler-frontend-residual-color-stale-threshold`.

## 2. Goals & non-goals

**Goals**
- One globally-consistent image→pitch homography per session, jointly constrained by clicks at
  both ends of the field, with no per-frame under-constraint and no chained-error accumulation.
- An honest status/export gate that reflects **whole-field** correctness, not in-sample fit.
- A held-out cross-end validation that is the acceptance bar — we do not trust calibration (or
  start Phase 4) until it passes.
- Per-frame export format unchanged, so `pipeline.assemble_from_homographies` and all downstream
  code are untouched.

**Non-goals (YAGNI)**
- No physical fixed-center bundle (shared focal + per-frame rotation), no mosaic/panorama image,
  no bundle adjustment. The chosen model is the **pure global homography + per-frame 2D offset**
  (decision: 2026-06-26). If the cross-end validation (§7) fails, that is the signal to revisit
  the model — we do not pre-build the bundle fallback.
- No new ML. The pitch-keypoint model stays deferred.
- No change to the HTTP/UI transport or the export file schema.

## 3. Camera model

Treat the video as 2D crops of one underlying canvas (the fixed sensor). Then:

```
pitch[0,1]²  ←──H_global──  image_ref     (one homography for the whole session)
image_ref    ←──T(offset_f)── image_f     (each frame is image_ref shifted by a 2D offset)
⇒  H_f = H_global · T(−offset_f)          (per-frame image_f → pitch homography)
```

`offset_f` is the 2D translation of frame `f` relative to the reference frame (frame 0).
`H_global` is a plain 8-DOF projective homography mapping reference-frame pixels to canonical
pitch coordinates — **no camera intrinsics, no focal, no pose.**

## 4. Module layout & seam

New pure module **`pitch/global_calib.py`** (numpy in → homographies out; no I/O, no server, no
video — unit-testable in isolation, the established pattern from `eval/pitch_metrics.py` and
`calib/validate.py`).

Public surface:

```python
@dataclass(frozen=True)
class GlobalCalib:
    h_global: NDArray[np.float64]            # image_ref → pitch[0,1] (3×3)
    offsets: dict[int, NDArray[np.float64]]  # frame → (dx, dy) px, reference-relative
    reference_frame: int
    n_clicks: int                            # clicks used in the global fit
    rms_px: float                            # global fit residual over ALL clicks (diagnostic only)

    def frame_homography(self, frame: int) -> NDArray[np.float64]:
        """Full-pixel image_f → pitch[0,1] homography = h_global · T(−offset_f)."""

def solve_global(
    clicks: Sequence[Click],
    offsets: Mapping[int, NDArray[np.floating]],
    size: tuple[int, int],
    *,
    line_obs: Mapping[int, Sequence[tuple[str, float, float]]] | None = None,
    reference_frame: int = 0,
) -> GlobalCalib: ...
```

`LabelerState._compute_poses` (state.py:152) is rewired to:
1. compute `offsets` from the cached chain (§5),
2. call `solve_global(active_clicks, offsets, size, line_obs=...)`,
3. produce per-frame `CalibFrame`s from `gc.frame_homography(f)` + the honest status (§6).

The seam contracts are preserved: `frame_h` still returns a 3×3 image→pitch H; `export()` still
writes `homographies.parquet` rows `(frame, h00..h22, source="manual", confidence)`. The frontend
and `pipeline.assemble_from_homographies` are not modified.

## 5. Per-frame 2D offsets

`offset_f` is derived by reducing the **existing cached inter-frame chain** (the `.npz` produced
by `labeler/chain.py` + `manual_anchor.cumulative_transforms`) to its 2D-translation component —
no new video pass. Concretely: take the cumulative reference→frame transform `M[f]`, and define
`offset_f` as the image-space translation it induces (the displacement of the reference frame
centre under `M[f]`, in full pixels).

Rationale: the measured transforms are already near-pure translations; keeping only the
translation component removes the spurious rotation/scale/perspective DOF that amplify chain
drift (a 2-DOF translation chain is a bounded random walk, not a multiplicative projective
blow-up). This is the concrete form of fixing `geometry-over-modeled` without adding an estimator
or a video pass.

Disconnected chain segments: if the session's clicks do not all lie in one connected component
of the chain, each component is solved independently against its own reference. A component whose
clicks cover only one end yields a single-end-constrained `H_global` — which the honest gate (§6)
will correctly flag non-green rather than paint over. (No special-casing needed; the gate is the
backstop.)

## 6. Honest status / export gate

Replace the in-sample residual gate. Status combines a **per-frame** check (1) and a
**per-component** check (2):

1. **Whole-field plausibility (per frame):** `fold_count(np.linalg.inv(H_f), size)` is in the
   physical range (default 4 ≤ fold_count ≤ 14 — a real narrow view shows ~6–12 of the 21
   landmarks; the range is a config constant). Note `fold_count` (`calib/validate.py:20`) maps
   canonical pitch[0,1] landmarks → pixels, so it takes the **pitch→image** homography, which is
   `inv(H_f)` (our `H_f` is image→pitch). It already exists and is already computed per frame; it
   is simply consulted now.
2. **Cross-end held-out accuracy (per component):** the chain component this frame belongs to has
   a cross-end held-out median (§7) within the acceptance tolerance (the §7 bar).

Status mapping: **red** if (1) fails; otherwise **green** if (2) passes for the frame's
component; otherwise **yellow** (whole-field-plausible but cross-end-under-constrained — honest,
not green). The in-sample `rms_px` is retained as a displayed diagnostic only — it no longer
drives status.

`export()`:
- gates on the same green criterion (a non-green frame is **not** exported — no `conf≈0.87` sky
  frames in `homographies.parquet`),
- **blocks until the refit worker is idle** (`pending()==0`, no timeout) before reading fits, so
  it can never silently write a partial/stale set; if it must time out it reports the
  exported-vs-expected frame count rather than a bare success.

Frontend: the per-frame residual readout colour uses the server-provided threshold, not the
stale hard-coded `0.05` (always-orange bug).

## 7. Validation — the acceptance bar ("nail it")

A pure routine `cross_end_holdout(clicks, offsets, size) -> HoldoutReport` in `global_calib.py`:

- Identify the two field ends by canonical landmark x/y (own-end vs opp-end landmark index sets).
- For each end present in the clicks: hold out **all** clicks of that end, solve `H_global` from
  the remaining clicks, and for each held-out click at frame `f` (landmark `kp`, pixel `p`)
  compute `pitch_pred = H_f · p`, take `disp = pitch_pred − PITCH_LANDMARKS[kp]`, and convert to
  feet via `eval/pitch_metrics.displacement_to_feet` (the per-axis width/length scaling — the
  same convention the eval framework uses; NOT `validate.reproj_error_feet`, which uses a
  different world-metres-via-`inv(H)` path).
- Report median and p90 per end and overall.

**Acceptance bar:** overall held-out median **< 1.5 m** (≈ the finest downstream threshold, the
contested-possession margin 0.022 pu) on the test session, with p90 reported. We do not delete
the old engines (§8) or start Phase 4 until this passes. The report is run from a notebook/CLI on
the existing `training_clip` click session.

## 8. Cleanup (only after §7 passes)

Delete, in `pitch/calib_anchor.py` and friends:
- `poses_by_pose_propagation` + `_rotation_from_chain` (Engine B — dead, encodes the disproven
  rotation model),
- `poses_by_gated_propagation`, `poses_by_click_propagation`, and their windowing/`gap_dist`/
  `seed_size`/`gate_px` parameters,
- the per-frame SQPNP path and `calibrate_clicked_frames`'s pose outputs (keep nothing that
  re-solves a per-frame pose),
- collapse `pitch/calib_compare.py` (the engine A/B/gated comparison harness) and its test.

`flag_outlier_clicks` / `_robust_sqpnp` are retained (per-clicked-frame outlier detection is
still useful and is independent of the per-frame pose engines). `calib/calibrate.py`'s
`refine_pose` / `line_residual` are retained only if the global least-squares refine reuses them;
otherwise removed.

## 9. Testing (TDD)

Pure-function tests in `tests/test_pitch_global_calib.py`, written before implementation:

1. **Round-trip recovery:** build a known `H_global` + known per-frame offsets; generate clicks
   where each frame sees only one end; assert `solve_global` recovers `H_global` within tol **and
   that `frame_homography(f)` projects the UNCLICKED end correctly** (the single-end "sky"
   regression that the old suite never exercised).
2. **Cross-end holdout:** on synthetic data with a known-good global H, assert held-out median is
   ~0; on data where one end's clicks are corrupted, assert the report flags it.
3. **Honest gate:** a frame whose `H_f` sends most landmarks off-frame (a "sky" H) is flagged
   non-green; a physically-plausible H is green. Drive `_status_of` and assert.
4. **Offset reduction:** synthetic translated chain → `offset_f` recovers the translation;
   spurious rotation/scale in the input is discarded.
5. **Export honesty:** `export()` skips non-green frames and blocks until idle (extend
   `test_labeler_state` / `test_labeler_server`).

All existing tests that asserted the per-frame engines or the residual gate are updated or removed
in lockstep with §8.

## 10. Risks

- **Offset drift across the long own→opp pan.** Mitigated by the translation-only reduction
  (bounded random walk) and surfaced by §7. If §7 fails, the contingency is direct
  phase-correlation registration of the (few) clicked frames to the reference — still model A, no
  bundle. This contingency is documented, not pre-built.
- **Distortion / non-translation motion.** Out of scope by the model-A decision; §7 is the
  detector. If recovered held-out error is irreducibly large, that escalates to a new design
  decision (revisit model), not silent failure.
- **Disconnected chain components** with single-end clicks: handled honestly by the gate (§6),
  reported as yellow.

## 11. Out of scope (handled by sibling sub-projects)

own/opp grounding (SP2), phase/possession correctness + halftime (SP3), pipeline/eval/labeler
cleanup + lows (SP4), and the Phase-4 metrics product (SP5, blocked by this). They are
independent of this change and proceed in parallel where possible.
