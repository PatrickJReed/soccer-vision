# Global-Crop Calibration — Design (2026-07-14)

**Problem.** The live physical engine (`pitch/physical_calib.py`) shares one focal across
anchors but gives every anchor a free 3-DOF translation (audit F-C1,
`docs/superpowers/2026-07-14-codebase-audit.md`). A single-end frame's clicks cannot pin
that translation, so the unclicked field end extrapolates badly ("lines in the sky" —
reduced by the physical model, not eliminated). Export then stamps green frames
`confidence=1.0` although nothing measures their far end (F-C2), and propagated frames are
graded by the nearest green anchor rather than the anchors actually used (F-C3).

**Chosen fix (Patrick, 2026-07-14): model the acquisition process exactly.** Trace footage
is a fixed camera whose "pan" is a 2D crop of one fixed view — measured inter-frame
transforms are pure translations (scale ≈ 1.000, rotation/shear ≈ 0, perspective ≈ 1e-6),
and per-anchor solved camera centers agree across 23 frames. So there is ONE image↔pitch
homography per registration segment, and each frame is a 2-DOF offset into it.

**Relation to prior attempts (why this is not June's failed Model A):** the 44-ft failure
fed the global solve chain-derived offsets over long drifting spans, then patched with
per-frame affine corrections that reintroduced free non-physical DOF. This design's
non-negotiable rule: **offsets at clicked frames come from clicks; unclicked frames use
short-hop chain deltas relative to bracketing anchors only.** Long chain compositions never
enter. The untracked spec `2026-06-26-global-reference-calibration-design.md` proposed a
mosaic + global solve; this supersedes it — same core idea, no mosaic (coordinates only),
offsets-from-clicks rule added.

---

## 1. Model

Coordinate conventions (unchanged from the labeler): clicks and internal homographies live
in normalized [0,1] image coordinates; export denormalizes (`H_px = H_norm @
diag(1/W, 1/H, 1)`). The canonical pitch is [0,1]² (x = width, y = length, own goal y=0).

Per registration segment `s`:

- **Global canvas** = the coordinate system of the segment's reference frame (the existing
  chain convention: `transforms[f] = M[f]` maps frame-f coords → reference coords). No
  mosaic image is built.
- **`H_g^s`** : one homography, global-canvas (normalized) coords → canonical pitch.
- **`d_f ∈ R²`** : per-frame offset; a point `p` in frame `f` sits at `p + d_f` on the
  canvas. `T(d) = [[1,0,dx],[0,1,dy],[0,0,1]]`.
- **Per-frame homography:** `H_f = H_g^s @ T(d_f)` — the same composition shape as today's
  `_shift_h`, with pure translations and one global anchor.

DOF accounting: 8 shared + 2 per frame. Every click in the segment constrains the same
`H_g`; a clicked frame is fully determined by ONE click. Under-constraint at a clicked
frame is structurally impossible; the far end of every frame comes from `H_g` (i.e. from
both ends' clicks jointly), never from per-frame extrapolation.

**Implied physical camera (report-only):** decompose `H_g` (with square-pixel,
centered-principal-point assumptions) into (f, R, C) via the orthonormality constraints on
`K⁻¹H`. Expected f ≈ 1461–1471 px (three prior games) — a free cross-session sanity check.
Warn if f ∉ [0.5·W_px, 5·W_px] (tightens the near-vacuous 0.1–50 gate, audit L1).

## 2. Solve pipeline (`solve_crop_session`)

Inputs: `points: Sequence[Click]`, `lines: Sequence[LineClick]`, `size`, `transforms`
(chain cumulative M[f], normalized), `segment_of`. Output: `CropCalib` (see §5). All
residuals are computed in **pitch space** (uniform units — the convention that fixed
`refine_pose`).

**Step 0 — crop-assumption check (per segment, also exposed as a CLI diagnostic).**
Decompose chain pair transforms: report max |rotation|, |shear|, |scale−1|, perspective
magnitude. Thresholds (tunable constants): |scale−1| ≤ 0.005, rotation ≤ 0.2°,
perspective ≤ 1e-5 on the median pair. Violations → loud warning; a scale violation has a
contained escape hatch (extend offsets to translation+scale, 3 DOF/frame) noted as a
follow-up, not built now.

**Step 1 — offset init, short-hop only.** For every frame, the initial `d_f` is the
translation component of `M[f]` (crop model ⇒ M ≈ translation). These raw values are used
ONLY as (a) the optimizer init at clicked frames and (b) *relative deltas to bracketing
anchors* for unclicked frames: `d_f = blend( d_a + (t_f − t_a), d_b + (t_f − t_b) )` with
the existing bracket weight `(f−a)/(b−a)`, subject to the existing `gap_guard=200` (beyond
it: no homography, red — same as today). The chain delta to the nearest anchor is therefore
≤ 200 frames and typically ≤ 100 — the measured 8-px-class drift regime.

**Step 2 — robust global fit.** Map every point click at frame `f` to canvas coords
`p + d_f`; fit `H_g` canvas→pitch with normalized DLT + RANSAC, threshold **0.012 pitch
units (≈ 0.8 m)** — a real inlier gate, unlike the current default-3.0-in-pitch-units bug
(F-C5). Requires ≥ 4 distinct landmarks whose pitch positions are non-degenerate (convex
hull area ≥ 0.02 pitch-units², tunable); otherwise the segment stays uncalibrated (red) —
the bootstrap-wait semantic the labeler already has.

**Step 3 — per-anchor offset re-solve.** For each clicked frame: minimize Σ point residuals
(pitch-space distance to the clicked landmark) + Σ line residuals (perpendicular pitch-space
distance to the named line) over `d_f` (2-DOF Gauss-Newton, IRLS/Huber for robustness to a
single bad click; init from Step 1). Line-only frames: ≥ 2 non-parallel lines fully
constrain `d_f`; exactly one line constrains one direction — solve that direction, take the
other from the chain, and cap the frame at **yellow**.

**Step 4 — alternate.** Refit `H_g` (least_squares over its 8 params, point + line
pitch-space residuals, inliers only) using re-solved offsets → re-solve offsets → 2–3
rounds or until residual change < 1e-6. Warm start for the RefitWorker = previous
(`H_g`, offsets). Note the F-C4 consequence: **every line click on every frame now
constrains the solve** — no more silently inert line clicks.

**Step 5 — physicality checks on the result.**
- Whole-field w-sign: with `P = sign_normalized(inv(H_f_px))` (field-center w > 0, the
  `_fold` convention), require **w > 0 for all 21 canonical landmarks** — a camera above
  ground sees the whole field plane in front of it; any w ≤ 0 is a sky/fold pose. This is
  the explicit far-end gate (F-C2); `fold_count ∈ [4,15]` stays as the in-frame plausibility
  band.
- Horizon of `H_g` must not intersect the projected field hull.
- Decomposition must yield a camera above ground (C_z on the correct side).

**Session-level honesty rule.** If the y-span of `H_g`'s inlier landmarks < 0.5 (session
saw effectively one field end), cap ALL frames in the segment at yellow: the far half is
genuinely unverified and no per-frame number can claim otherwise.

## 3. Status and confidence (trust seam — F-C2/F-C3)

- **green** — a clicked frame whose offset solve used ≥ 1 point click (or ≥ 2 lines) with
  in-sample pitch residual median ≤ tolerance (constants: points ≤ 6 ft, lines ≤ 4 ft, as
  today), **and** the Step-5 physicality checks pass for its `H_f`; or an unclicked frame
  within `GREEN_RADIUS` (100) of a green anchor **among the anchors actually used for its
  bracket** (F-C3 fixed by construction — the bracket anchors ARE the graded ones).
- **yellow** — beyond `GREEN_RADIUS` but within the gap guard; one-line frames; frames in a
  one-end-capped segment.
- **red** — no `H_g` for the segment, beyond gap guard, or any physicality check fails.

**Export (`LabelerState.export`)**: still green-only, but confidence is honest:
anchors 0.9; propagated green ramps linearly 0.8 → 0.6 across `GREEN_RADIUS`; constant 1.0
is retired. The session `GateReport` (§4) is written alongside as `calib_gate.json` so
downstream consumers can see what the numbers rest on. The stale "green = whole-field
trustworthy" comment is corrected.

## 4. Acceptance gate (binding done-criterion, agreed 2026-07-14)

Run on BOTH real sessions — training_clip (105 points/75 lines, chain
`ef2546eaddd5e6fc.npz`; the hard single-end case) and oceanside_clip (the 78.7%-green case
that must not regress):

1. **Existing numeric gate holds** (`evaluate_gate` recomputed under the new engine):
   foreground held-out median ≤ 5 ft, p90 ≤ 12 ft; leave-one-anchor-out propagation
   median ≤ 5 ft.
2. **New far-end metrics:** 100% of green frames pass the all-21-landmarks w>0 check; the
   leave-one-anchor-out table is split by which end the held frame's clicks see (own / opp /
   both — classified by clicked-landmark pitch-y majority vs 0.5). The single-end rows are
   the direct F-C1 measurement; report them (target: within the same ≤ 5 ft median; if they
   miss, the number is reported honestly and Patrick decides).
3. **Visual:** rendered spot-checks including the previously-sky frames (training_clip 193,
   134) — Claude renders, Patrick assesses (standing preference).

The physical engine remains in-repo until all three pass; `validate_session` runs both
engines side-by-side during the transition.

## 5. Integration and module layout

- **New pure module `pitch/global_crop.py`**: `solve_crop_session(points, lines, size,
  transforms, *, segment_of, gap_guard, seed) → CropCalib`. `CropCalib` implements the same
  surface `LabelerState` consumes from `PhysicalCalib` — `frame_homography(frame)`,
  `status(frame)`, `is_anchor(frame)`, `nearest_anchor_gap(frame)` — plus `H_g`/offsets/
  implied-camera fields for diagnostics. Segments solve independently (today's isolation
  rule); sharing the implied camera across segments is a noted future option, not built.
- **`LabelerState`**: engine swap is a constructor change once the gate passes; RefitWorker
  warm-start via `seed`. Bootstrap improves: the crop engine needs ≥ 4 well-spread
  landmarks in a segment (vs ≥ 3 diverse frames for `calibrate_camera`) — overlay appears
  sooner.
- **`validate_session` CLI**: `--engine {physical,crop,both}`; prints gate reports
  side-by-side; `--crop-check` runs the Step-0 diagnostic standalone.
- **Riders (mechanical, same plan, separate tasks):**
  - `PitchMapper.transform`: NaN where `w ≤ eps` (F-C6) + regression test.
  - Explicit `ransac_thresh` at the two live `fit_homography` call sites
    (`manual_anchor.py:273`, `landmarks.py:136`) in pitch units (F-C5) + tests.

## 6. Testing

TDD against a synthetic crop camera (ground-truth `H_g` + offsets → generated clicks):

- Recovery: `H_g` and all offsets to tight tolerance from clean clicks; with outlier clicks
  (RANSAC/IRLS path); with noisy chain init.
- **F-C1 regression test (the point of the project):** a frame clicked ONLY at one end must
  place the other end's landmarks within tolerance, given other frames saw that end.
- Chain-drift immunity: corrupt the long-span chain translations; anchor offsets must
  correct them; unclicked-frame error bounded by short-hop drift only.
- Line handling: line-only frames (2 lines → full solve; 1 line → yellow + one-direction);
  line clicks on any frame influence `H_g`.
- Degenerates: < 4 landmarks, collinear landmarks (hull gate), one-end session (yellow cap),
  segment isolation, empty segments.
- Status/export: far-end w>0 gate flips a sky pose to red; confidence values match the
  mapping; `calib_gate.json` written.
- Real-session validation script (gate + renders) on both sessions — the §4 evidence.

## 7. Out of scope

Cross-segment camera sharing; the joint C-prior bundle (Model B — escalation path only);
ML keypoint path changes; `view_registration` changes; Phase-4 metrics fixes (audit
priority 2, separate work); deleting `physical_calib` (follow-up after the swap bakes).
