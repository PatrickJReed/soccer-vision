# Per-Frame Focal for the Physical Engine — Design (2026-07-28)

**Goal.** Replace the physical engine's single shared focal with per-anchor focal
fitting, and fix the ordering bug that lets outlier clicks poison the focal estimate.
Uniform scope: all anchor frames in both workflows (rep-pack and classic dense-clip),
evidence-gated against the shipped sessions (Patrick's scope decision, 2026-07-28).

**Motivation (audit evidence, 2026-07-24, home_g4_oceanside rep pack, 112 clicked
frames).** (1) Three catastrophic mislabeled clicks shifted the shared focal
1205 → 1444 px (~20%) because `solve_session` computes K from ALL clicks BEFORE
`flag_outlier_clicks` runs — every frame's pose was solved under a poisoned lens, so
each new label re-fit a skewed K and flipped borderline greens red (Patrick's observed
non-convergence). (2) Deeper: per-frame optimal focal spread is p90/p10 = 1.32 across
111 frames; 59/111 frames improve > 1.5 ft with their own focal (median frame error
3.65 → 2.04 ft). Trace's virtual PTZ changes effective zoom between views (the crop
refutation measured 2.6–5.1% scale spread on a 2-min clip; a full game is far wider).
One shared focal cannot fit a full game's views — no amount of careful clicking
converges. Diagnostic scripts: scratchpad `audit_clicks.py` / `focal_sweep.py`.

## 1. New solve order (`solve_session` in `pitch/physical_calib.py`)

1. **K₀** from all clicks via existing `calibrate_camera(obs, size, min_points=6)` —
   used ONLY to seed outlier flagging. Its `CalibError` fallback (empty calib, no
   anchors) is unchanged.
2. **Two-pass flagging:** `flag_outlier_clicks` (40 px, unchanged) with K₀ → refit
   **K₁** = `calibrate_camera` on the pass-1 clean clicks → re-flag the ORIGINAL
   click set with K₁. The pass-2 result alone is final (a click wrongly flagged by a
   skewed K₀ gets restored; a false keep gets caught); the final clean set and K₁
   flow onward. Poisoned-K path is dead: no click that fails the final flagging
   contributes to any focal used for poses.
3. **Per-anchor focal** (new pure module `pitch/focal.py`): for each clicked frame
   with ≥ 6 unique clean landmark ids, 1-D search over focal minimizing that frame's
   median in-sample residual (point feet + line perpendicular feet, exactly the
   residuals `_grade` uses): coarse sweep of 9 log-spaced candidates in
   [0.6, 1.6]·f₁ (f₁ = K₁[0,0]) then golden-section refine to ~1 px. Accept `f*`
   only if (a) it is an interior minimum (not within one coarse step of either sweep
   edge) and (b) it improves the frame's median residual vs f₁ by ≥ 0.15 ft
   (`MIN_FOCAL_GAIN_FT`); otherwise the frame is *unconstrained*.
4. **Fallback ladder:** unconstrained and < 6-id frames use `f_med` = median of
   accepted `f*`s; if < 3 accepted `f*`s in the session, every frame uses f₁ (exact
   current behavior — sparse sessions degrade to today's engine, never worse).
5. **Poses:** existing `_anchor_pose` (SQPNP + `refine_pose`) per frame at that
   frame's chosen focal (K built as [[f,0,W/2],[0,f,H/2],[0,0,1]] — square pixels,
   centered principal point, as today). Grading (`_grade`), fold gates, w-sign
   guards, bracket propagation, segment logic: all unchanged. Propagation composes
   anchor H's and is focal-agnostic.

## 2. `pitch/focal.py` API (pure, no I/O)

`focal.py` must NOT import `physical_calib` (which imports `focal` — no cycle). It is
a pure 1-D search unit: the caller supplies the frame's error function as a closure.

```python
@dataclass(frozen=True)
class FocalFit:
    f: float            # chosen focal (px)
    constrained: bool   # True iff accepted per §1.3 (interior min + gain)
    err_ft: float       # frame median residual at f

def fit_frame_focal(
    err_at: Callable[[float], float | None],  # median frame residual (ft) at focal f,
    f_init: float,                            #   None where no pose solves
) -> FocalFit | None    # None if err_at is None across the whole sweep

def session_focal(
    fits: Mapping[int, FocalFit | None], f_shared: float,
) -> dict[int, tuple[float, str]]   # fallback ladder of §1.4; EVERY input frame gets
                                    # (focal, source) with source "fit"|"median"|"shared"
```

`physical_calib` builds `err_at` per frame from its existing residual helpers
(`_point_feet`, `_line_perp_feet`) and `_anchor_pose` — the same in-sample residuals
`_grade` uses, never a duplicate implementation. Tests exercise `fit_frame_focal`
with synthetic error curves (counting evaluations, injecting None regions) without
any solver at all.

## 3. `PhysicalCalib` surface

- New fields `focal_of: dict[int, float]` and `focal_source: dict[int, str]`
  (every anchor frame present; empty dicts for the pre-focal empty-calib path) and
  accessor `frame_K(frame: int) -> NDArray`
  returning that frame's K (falls back to the nominal `K` for unknown frames).
- Existing `K` field KEPT as the session-nominal K (K₁; or f_med-based when ≥ 3
  accepted fits — pick ONE: nominal K uses **f_med when available, else f₁** so the
  nominal reflects the session's consensus zoom). No external consumers exist today
  (grep verified); internal uses switch to `frame_K`.

## 4. Holdout honesty (`foreground_holdout`, `propagation_holdout`)

- `foreground_holdout` currently computes its own shared K from ALL clicks (same
  ordering bug) and refits poses without near-touchline evidence. New behavior: run
  the §1 pipeline (two-pass flag + per-frame focal), and for each held-out frame
  **re-select the focal from the held-out fit set** (the po/lo WITHOUT near-touchline
  evidence, same `fit_frame_focal` sweep). A focal chosen using near-touchline clicks
  must not leak into a near-touchline prediction claim.
  **ERRATUM (2026-07-29, found by the Task 3 leak test):** the originally spec'd
  fallback — "use the frame's session focal when the sweep is unconstrained" — is
  itself a leak on clean sessions: `constrained` requires ≥ MIN_FOCAL_GAIN_FT of
  gain, which near-zero held-out residuals cannot produce, so clean frames always
  fell back to the (near-TL-contaminated) session focal. Actual rule: **held-out
  model selection** — accept the swept focal unless the session focal's held-out
  fit-set error is strictly better (the constrained case is a strict subset). The
  session focal remains a candidate (that much is unavoidable — it seeds the sweep)
  but can only win on held-out evidence.
  **ERRATUM 2 (2026-07-29, found by the §6b evidence gate — SUPERSEDES the rule
  above):** held-out focal re-selection is an over-correction, refuted on real data.
  On the oceanside session the re-sweep runs on thin far-field-only fit sets and
  OVERFITS (fg holdout 6.97/17.19 ft vs baseline 4.18/9.72; one frame pinned at the
  1.6× sweep edge, ±30% focal swings). Every leak-free variant measured worse
  (constrained-gate 5.29/15.29, interior-only 6.03/15.29, near-TL-blind ladder
  5.09/15.29, single median focal 3.96/13.79) because near-TL evidence contributes
  REAL depth diversity that constrains the focal — removing it removes constraint,
  not just leak. Meanwhile the shared-K BASELINE's holdout focal was itself fit on
  all clicks including near-TL, so the no-leak demand taxed the new engine against a
  baseline that never paid it. Final rule: **the holdout refits the pose without
  near-touchline evidence at the frame's SESSION focal (no re-sweep)** — measured
  fg 3.78/8.58, better than baseline on both. The residual focal-level optimism is
  bounded (one scalar shared by ~15+ residuals of which near-TL is a minority;
  synthetic worst-case with ALL near-TL clicks displaced 30 px absorbs ~2/3 of the
  displacement into reported error, direction always preserved) and is identical in
  kind to the baseline gate's. The leak regression test asserts DIRECTIONALITY
  (corrupted near-TL evidence must worsen the reported claim, never improve it)
  rather than full non-absorption. Diagnostic: scratchpad `fg_policy_diag.py`,
  numbers in `docs/superpowers/2026-07-29-per-frame-focal-evidence.md`.
- `propagation_holdout` calls `solve_session` on the remaining clicks and inherits
  the new pipeline automatically; no separate change beyond passing through.

## 5. Reporting & artifacts

- `calib_gate.json` gains `"focal": {"per_frame": {frame: f}, "source": {frame:
  "fit"|"median"|"shared"}, "spread_p90_p10": float}`.
- `validate_session` physical report prints one focal summary line (n fit / n median
  / n shared, f range, spread).
- `homographies.parquet` schema unchanged (H is already per-frame).

## 6. Acceptance evidence (must-pass, in the plan)

- **Synthetic** (extend the proven multi-view synthetic camera recipe): (a) views
  rendered at different true focals → per-view recovery within 2%; (b) poison test:
  one 200-ft-wrong click on frame A moves frame B's focal < 1% and B's pose grade is
  unchanged; (c) ordering regression: the K used for poses on a poisoned session
  equals (±1 px) the K of the same session with the poison click pre-removed;
  (d) fallback ladder: < 6-id frames get f_med; < 3 accepted fits → all f₁.
- **Real, before/after table required in the build report:**
  (a) a FROZEN copy of the 2026-07-24 home_g4_oceanside sidecar (poison clicks
  included, so before/after is apples-to-apples even as Patrick fixes clicks in the
  labeler): green count MUST rise from the 86/112 baseline and the all-frame median
  in-sample error MUST drop (audit predicts ~3.6 → ~2.2 ft);
  (b) shipped oceanside session via `validate_session --engine physical`: fg and
  prop gate metrics hold (within +0.3 ft) or improve. Regression here = stop and
  scope down to rep-pack-only (single-frame segments) before shipping.
- Full suite green (488 baseline), bare `uv run mypy` no NEW errors vs the 58
  pre-existing, ruff clean.

## 7. Out of scope

Lens-distortion modeling; per-frame principal point; the crop engine;
view_registration; labeler UI; auto-repair of the 3 catastrophic clicks (Patrick
fixes those in the labeler; the engine change merely stops them from poisoning
other frames).
