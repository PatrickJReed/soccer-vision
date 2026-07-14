# Global-Crop Calibration — Task 11 Real-Session Verdict: REFUTED (gate held, flip blocked)

**Date:** 2026-07-14. Plan: `docs/superpowers/plans/2026-07-14-global-crop-calibration.md`
(Tasks 1–10 built, double-reviewed, all green: 469 tests). This document records the
Task-11 acceptance-gate evidence and the model-level finding. **Task 12 (engine flip) was
NOT executed — its precondition failed.** The labeler default remains `physical`.

## Gate results (validate_session --engine both --crop-check)

**training_clip** (160 pts / 26 frames; 0 line clicks — see Data Loss below; chain
`ef2546eaddd5e6fc.npz`):

| engine | foreground med/p90 (ft) | propagation med/p90 (ft) | verdict |
|---|---|---|---|
| physical | unevaluable (n=0, no line clicks) | 4.40 / 25.33 (n=144) | FAIL (fg unevaluable) |
| crop | unevaluable | **65.57 / 149.15** (n=147) | **FAIL** |

Crop by-end: own 63.1 / opp 86.3 / both 62.9 ft medians. Implied camera: unrecoverable.
Crop-check: QUESTIONABLE (max rot 0.40°, max scale dev 6.9%, max persp 1.0e-4).

**oceanside_clip** (366 pts / 50 frames; 483 line clicks; chain `da63d2bb640cc974.npz`):

| engine | foreground med/p90 (ft) | propagation med/p90 (ft) | verdict |
|---|---|---|---|
| physical | 4.18 / 9.72 (n=157) — PASS | 5.85 / 21.13 (n=366) | **FAIL (prop med > 5.0, marginal)** |
| crop | **129.7 / 134.9** | **81.3 / 123.2** | **FAIL** |

Crop-check: QUESTIONABLE (max rot 0.17°, max scale dev 3.6%).

## The decisive experiment (fit-free, chain-free)

Under the crop model, for any two anchor frames a,b the difference
`click_a(ℓ) − click_b(ℓ)` must be ONE constant vector across all shared landmarks ℓ
(= d_b − d_a). Measured spread around that constant, real sessions, pixels:

| clip | pairs | max-deviation median | p90 | best-fit scale dev median | p90 |
|---|---|---|---|---|---|
| oceanside | 900 | **51.7 px** | 142.5 | **5.1%** | 17.6% |
| training | 110 | **27.4 px** | 89.7 | **2.6%** | 14.1% |

A true crop predicts ~click-noise (2–5 px) and scale ≡ 1.000. Deviations are ~flat in
frame gap (view-dependent, not drift-accumulated). **Conclusion: Trace's virtual PTZ is
not a 2D crop between distant views — the constellation rescales (~5%) and warps
(perspective), i.e., rotational rendering. The June "pure 2D translation" measurement was
correct for CONSECUTIVE pairs (median rot 0.008°, scale 6e-4 on this chain) but does not
integrate to a global crop.** Chain-noise outliers (152/2699 pairs violating) compound
the problem but are not its root; a 2-DOF-offset model cannot close a ~50 px / 5%-scale
view-dependent violation regardless of solver quality.

## What survives (shipped and live, engine-agnostic)

- **Honest trust seam (audit F-C2/F-C3)** — `CalibFrame.confidence` via
  `frame_confidence` (0.9 anchor / 0.8→0.6 ramp; constant-1.0 retired), export green-only
  with honest confidence, `calib_gate.json` sidecar, w-sign no-sky gate + fold band +
  used-anchor grading — wired for BOTH engines in the labeler.
- **Mechanical guards (F-C5/F-C6)** — PitchMapper behind-camera NaN; real RANSAC
  thresholds at both live `fit_homography` sites.
- **validate_session --engine both + --crop-check** — the dual-engine evidence tool that
  produced this verdict; the crop-assumption diagnostic is reusable for any future clip.
- **`pitch/global_crop.py` + 32 tests** — a correct implementation of the crop model
  (F-C1 regression proven on synthetic data), retained as the opt-in `--engine crop`
  with honest self-diagnostics; do NOT flip the default.

## New findings for the project

1. **Training-clip line clicks are LOST.** The current sidecar has 160 points/0 lines;
   June's session had 105 points + 75 lines. No `.bak` (insurance landed 2026-07-01,
   after the loss) and no `line_clicks.parquet` export exists. Foreground quality on
   training_clip is unevaluable until re-clicked (~23 near-TL clicks were the June set).
2. **Physical engine on oceanside misses the propagation bar marginally**
   (5.85 ft vs 5.0; foreground passes). Levers: more anchors near weak spans, or accept
   the marginal miss explicitly.
3. **Model direction:** the fit-free experiment is direct evidence FOR a rotational
   camera model. The audit's F-C1 recommendation (shared camera center + per-frame
   rotation, C FIXED at the measured value — which scored 2.5–3.0 ft in the 2026-07-01
   experiment; only free-C failed) is now the best-supported next step. The trust seam,
   gate machinery, and by-end LOO built here carry over unchanged.
