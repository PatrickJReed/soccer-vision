---
name: phase-possession-correctness
description: Reconcile the shipped possession proxy and phase splitter to the master spec §6.1/§3 — isotropic (length-normalized) possession distances, the opp-third defend split, turnover detection through contested/loose, manual halftime attack-direction normalization, an honest possession-agreement gate that exposes abstention, and highest-confidence ball selection. Pure-function fixes, extends existing test coverage, no calibration coupling.
status: approved
date: 2026-06-26
---

# Phase & Possession — Correctness Reconciliation

Sub-project 3 of the post-audit program (see `docs/superpowers/2026-06-26-codebase-audit.md`).
The possession proxy (`phase/possession.py`) is the one finished slice of real product logic and
is the cross-cutting input to every §6 metric. The audit flagged seven discrete correctness gaps
between the shipped code and the master spec (`docs/superpowers/specs/2026-05-27-soccer-vision-design.md`
§6.1 phase model, §3 pitch-relative units). This sub-project closes them. It is independent of the
calibration rework (SP1) and the own/opp grounding (SP2); it touches no homography, no camera model,
and no team-assignment logic.

Findings closed: `possession-distances-anisotropic` (MED), `defend-split-at-midfield-not-third` (MED),
`transition-misses-through-contested` (MED), `splitter-test-locks-wrong-boundary` (LOW),
`halftime-attack-direction-gap` (MED), `agreement-gate-cant-validate-contested` (MED),
`possession-arbitrary-ball` (LOW).

## 1. Problem

`phase/possession.py` and `phase/splitter.py` faithfully implement the *shape* of the spec §6.1
five-state model, but seven details diverge from the spec's own definitions, and each is silently
wrong on the target footage:

1. **Anisotropic distances.** Possession distances are computed on raw per-axis pitch coords
   (`sqrt(dx² + dy²)`), but x is a fraction of pitch *width* and y a fraction of pitch *length*
   (§3: `aspect_ratio≈1.5`, "a unit of x is physically 1/1.5 of a unit of y"). The spec's
   "pitch-length" thresholds are therefore ~1.5× too tight in the width axis — the
   contested/clump/loose decision boundary is an ellipse, not the intended circle. `hygiene/core.py`
   already normalizes (`x_len = x_pitch / aspect_ratio`); possession does not.
2. **Defend split at midfield.** The defend sub-labels split at y=0.5, but spec §6.1 splits at the
   opp third (0.667). The own side is correctly symmetric at 0.333; the opp side is not.
3. **Turnovers miss contested.** `transition` only fires on a *direct* own↔opp flip, but youth
   turnovers pass through `contested`/`loose_ball`, so the phase under-fires on real footage.
4. **A test locks the wrong boundary** (the 0.5 defend split), so fixing (2) without fixing the
   test fails CI for the wrong reason.
5. **No halftime awareness.** Phase sub-labels hard-code own goal at y≈0, so build↔defend_low and
   attack↔defend_high invert for the entire second half of any full game — the season use case.
6. **Agreement gate can't see contested.** The possession-agreement validator drops every frame
   where either side is not own/opp, so the headline `pct_contested` (§6.8) is unvalidatable and the
   gate is gameable by abstaining (widen the contested margin → conditional agreement rises).
7. **Arbitrary ball.** Possession reads the *first* ball row per frame, while the pipeline uses the
   *highest-confidence* ball for phase — so possession and phase can disagree on which ball they used.

## 2. Goals & non-goals

**Goals**
- Possession/phase outputs match the master spec §6.1/§3 definitions exactly, on the target footage.
- Distances are isotropic length-normalized pitch units, sharing the one convention `hygiene` uses,
  defined cleanly so SP5's §6 radius metrics reuse it rather than re-deriving it.
- Full-game (cross-halftime) phase labels are correct given a manually supplied `halftime_frame`.
- The possession-agreement gate is honest: abstention is visible, not rewarded.
- Every fix lands with extended tests in the existing, well-covered suites; no behavior change is
  left unguarded.

**Non-goals (YAGNI)**
- No automatic halftime detection. `halftime_frame` is a manual input (the memory note's decision).
- No change to `possession_state` semantics or the homography/camera model. `own/opp` is nearest-team-
  to-ball and is **direction-independent**; only the phase *sub-labels* (and future y-axis spatial
  metrics) need direction normalization. SP1/SP2 own those layers.
- No new GT schema fields *required*. Extending the agreement GT CSV to a `contested` state is a
  recommended-against-for-now option (§3.6); the minimal honest fix is coverage reporting.
- No metrics-layer work (SP5). We only make the convention reusable, we do not build §6.

## 3. Design (per finding)

All edits are to pure functions with existing unit tests. File:line references are against the
current tree (verified by reading; the audit's line numbers were approximate).

### 3.1 Isotropic, length-normalized possession distances `[MED]`

**Where.** `phase/possession.py:52-54` computes
`dx = x - bx; dy = y - by; dists = sqrt(dx*dx + dy*dy)` on raw pitch coords. Those `dists` feed the
loose-ball test (`:62`), the clump counts (`:66-67`), and the own/opp/contested margin (`:68-74`).

**Spec.** §3: "x is fraction of WIDTH, y is fraction of LENGTH … a unit of x is physically 1/1.5 of
a unit of y." §3.3 lists the possession thresholds (margin 0.022, clump 0.044, near-ball 0.073) as
**pitch-length** units. `PossessionThresholds`' docstring (`possession.py:17`) already says
"Thresholds in pitch-units (fractions of pitch length)." `hygiene/core.py:57` and `:287-288`
implement the correct convention (`x_len = x_pitch / aspect_ratio`, `y_len = y_pitch`).

**Fix.** Thread a `PitchSpec` into `classify_possession`:

```python
def classify_possession(
    detections: pd.DataFrame,
    thresholds: PossessionThresholds | None = None,
    pitch_spec: PitchSpec | None = None,   # NEW; default PitchSpec.standard_9v9()
) -> pd.Series: ...
```

Inside the loop, divide the x offset by `aspect_ratio` before the hypot:

```python
ar = (pitch_spec or PitchSpec.standard_9v9()).aspect_ratio
dx = (players["x_pitch"].to_numpy() - bx) / ar
dy = players["y_pitch"].to_numpy() - by
dists = np.hypot(dx, dy)            # now in pitch-LENGTH units, isotropic
```

The thresholds are unchanged (already length-units); only the metric they apply to is corrected.
Use `np.hypot` to match `hygiene` (`core.py:92`) and avoid the manual square. `pipeline.py:94` calls
`classify_possession(enriched, possession_thresholds)` — leave that call as-is (the new arg defaults
correctly); optionally pass `pitch_spec` through `assemble_phases` only if a non-9v9 spec is ever
wired, otherwise YAGNI.

**Reusable convention (for SP5).** This is the third site needing length-normalized distance
(`hygiene/extract_fragments`, `hygiene/assign_goalkeepers`, now possession), and §6.5/§6.8 radius
metrics will be the fourth. Define one helper to stop the convention drifting:

```python
# pitch/spec.py (or a new pitch/units.py)
def length_norm_xy(x_pitch, y_pitch, spec: PitchSpec):
    """Return (x/aspect_ratio, y): coords in isotropic pitch-LENGTH units."""
```

possession.py uses it; `hygiene` is *not* refactored in this sub-project (it is correct already and
out of scope), but the helper is the single source of truth SP5 imports. (Decision flagged in the
summary — helper now vs. inline-and-defer.)

### 3.2 Defend split at the opp third, not midfield `[MED]`

**Where.** `phase/splitter.py:63`:
`phases.loc[fi] = "defend_high" if by > OPP_HALF_MIN_Y else "defend_low"`, where
`OPP_HALF_MIN_Y = 0.5` (`:12`). `OPP_THIRD_MIN_Y = 0.667` is declared (`:13`) but **dead**.

**Spec.** §6.1: "`defend_low` = opp + ball in own 2/3 … `defend_high` = opp + ball in opp 1/3."
"Own 2/3" is y < 0.667; "opp 1/3" is y > 0.667. The own side is already correct and symmetric:
`build` (own 1/3, y < 0.333) vs `attack` (opp 2/3) at `OWN_THIRD_MAX_Y = 0.333` (`:61`).

**Fix.** Use the (currently dead) opp-third constant, making the opp side mirror the own side:

```python
else:  # "opp"
    phases.loc[fi] = "defend_high" if by > OPP_THIRD_MIN_Y else "defend_low"
```

Remove the now-unused `OPP_HALF_MIN_Y = 0.5` (`:12`) to prevent re-introduction.

### 3.3 Turnover on the last *committed* team label `[MED]`

**Where.** `phase/splitter.py:42-48` sets `is_turnover` only on adjacent own→opp / opp→own:

```python
state_prev = possession_state.shift(1)
is_turnover = (
    ((possession_state == "own") & (state_prev == "opp"))
    | ((possession_state == "opp") & (state_prev == "own"))
)
```

**Spec.** §6.1: "`transition` = 5-second window after each possession change." A real youth turnover
(`own → contested → opp`) is a possession change, but the adjacent-flip test never fires because the
prior state is `contested`, not `opp`. The transition phase is consequently near-dead on target
footage (audit). Smoothing preserves `contested` (`possession.py:98-99`), so this is structural, not
a smoothing artifact.

**Fix.** Define the turnover on the change of the **last committed** own/opp label: forward-fill the
own/opp labels through intervening `contested`/`loose_ball`/`unknown`, then diff.

```python
committed = possession_state.where(possession_state.isin(["own", "opp"])).ffill()
is_turnover = committed.ne(committed.shift(1)) & committed.notna() & committed.shift(1).notna()
turnover_frames = frames[is_turnover.to_numpy()]
```

- `own → contested → opp`: committed is `own,own,opp` → fires at the first `opp` frame (the moment
  the new possession is confirmed). The intervening `contested` frames keep their own phase label —
  the transition-overlay loop (`:71`) already only overwrites `build/attack/defend_*`, never
  `contested/loose_ball/unknown`.
- `own → loose_ball → own` (a scramble that stays own): committed is `own,own,own` → **no** false
  turnover. This is a strict improvement over the adjacent-flip rule, which also missed it.
- The leading `notna()` guards prevent firing on the first committed label (no prior possession).

### 3.4 Fix the splitter test that locks the wrong boundary `[LOW]`

**Where.** `tests/test_phase_splitter.py`:
- `test_transition_window_after_possession_change` (`:31-43`) sets every `ball_y = 0.6` and asserts
  `phases.loc[80] == "defend_high"` (`:42`) — which is only true under the wrong 0.5 split. Under the
  §6.1 0.667 split, y=0.6 is `defend_low`.
- `test_defend_low_when_opp_with_ball_in_own_half` (`:24-29`) is named for "own half" but its intent
  is "own 2/3"; its data (y=0.3, 0.4) passes under both splits, so it does not actually pin the
  boundary.

**Fix.**
- In the transition test, change the out-of-window assertion to the §6.1-correct label
  (`phases.loc[80] == "defend_low"` at y=0.6), keeping its real purpose (the transition window).
- Rename `test_defend_low_when_opp_with_ball_in_own_half` →
  `test_defend_low_when_opp_with_ball_in_own_two_thirds`.
- Add explicit boundary cases (these are the regression guards for §3.2):
  `opp + y=0.6 → defend_low`, `opp + y=0.7 → defend_high`, and a just-below/just-above-0.667 pair.

### 3.5 Halftime / attack-direction normalization `[MED]`

**Where.** `phase/splitter.py:60-63` maps `ball_y` to phase sub-labels assuming own goal at y≈0 for
the whole clip. `pipeline.py:107` calls
`label_phase(poss_full, ball_y_full, fps, transition_seconds=...)` with no halftime input.

**Spec / scope.** The season use case spans halftime; `landmarks.py` fixes the canonical y-axis to
the two physical goals ("own"/"opp" = goal at y=0 / y=1). After the teams switch ends, the *same*
canonical y now points at the opposite team's goal, so build↔defend_low and attack↔defend_high
invert for the whole second half. **Crucially, this is a phase-label-only problem:**
`possession_state` (nearest team to ball) and the homography are direction-independent and must not
change (audit; memory `project_halftime_direction_gap`).

**Fix.** Add a manual `halftime_frame` and normalize attack direction by flipping the y used for the
thirds mapping (a single reflection handles build↔attack and defend_low↔defend_high symmetrically):

```python
def label_phase(
    possession_state, ball_y_pitch, fps,
    transition_seconds=5.0,
    halftime_frame: int | None = None,   # NEW; None = single-half clip (no flip)
):
    eff_ball_y = ball_y_pitch
    if halftime_frame is not None:
        flip = ball_y_pitch.index >= halftime_frame
        eff_ball_y = ball_y_pitch.mask(flip, 1.0 - ball_y_pitch)  # NaN stays NaN
    # ... use eff_ball_y in the per-frame thirds branch (:60-63) ...
```

After the flip, "own goal at y=0" holds in every half, so the *existing* thirds logic (now via
`OPP_THIRD_MIN_Y`) is correct for both halves. Transition detection is unchanged (possession-driven,
direction-independent).

**Threading.** Add `halftime_frame: int | None = None` to `assemble_phases` (`pipeline.py:56-68`)
and forward it to the `label_phase` call (`:107`). `analyze_video` /
`assemble_from_parquet` / `assemble_from_homographies` already forward `**assemble_opts` to
`assemble_phases`, so a caller passing `halftime_frame=...` flows through with no further edits.
Default `None` keeps the single-half bake-off clip and all current tests unchanged.

The flip is deliberately localized to a one-line reflection so SP5's y-axis spatial metrics
(verticality, field_tilt, block height — §6.7/§6.2) can reuse the same `halftime_frame` convention
(out of scope here; noted for reuse).

### 3.6 Honest possession-agreement gate `[MED]`

**Where.** `hygiene/core.py:351-382`, `possession_agreement`. Line 359:
`both = pred.isin(["own", "opp"]) & gt_states.isin(["own", "opp"])`. Only frames where **both** sides
commit own/opp enter `n_compared`, `agreement`, and the disagreement spans. The GT CSV schema
(`hygiene/agreement.py`) is `{own, opp, none}` — there is no `contested` state. So contested/loose
frames are excluded from numerator and denominator, the §6.8 headline `pct_contested` is
unvalidatable, and the conditional `agreement` is **gameable by abstaining**: widening the contested
margin moves hard frames out of `both`, mechanically raising the reported percentage.

**Spec.** §6.8 lists `pct_contested` as a headline youth metric on the per-game report; §8.1 expects
contested-ball configurations to be exercised. A gate that cannot see the contested state cannot
validate the metric it gates.

**Fix (minimal honest version).** Keep the conditional agreement as the headline number, but surface
the abstention that currently hides behind it. Extend `AgreementResult` (`core.py:324-330`) and
report the new fields in the CLI (`agreement.py:27-32`):

```python
@dataclass(frozen=True)
class AgreementResult:
    n_compared: int                  # GT∈{own,opp} AND pred∈{own,opp}  (unchanged)
    agreement: float                 # conditional agreement over n_compared (unchanged)
    disagreements: list[...]         # (unchanged)
    n_gt_team: int                   # NEW: frames where GT∈{own,opp} (the attributable denom)
    pred_commit_rate: float          # NEW: n_compared / n_gt_team  (abstention -> drops)
    pred_contested_frac: float       # NEW: pred∈{contested,loose_ball} / all GT-covered frames
```

- `pred_commit_rate` is the anti-gaming number: of the frames the human attributed to a team, what
  fraction did the model also commit? Widening the contested margin lowers this even as `agreement`
  rises, so the trade-off is visible in one place.
- `pred_contested_frac` surfaces the model's contested/loose share so a human can sanity-check
  `pct_contested` against the footage, since the GT cannot.
- Update both `return AgreementResult(...)` sites (`:362` empty case and `:382`) to the new fields.
- CLI prints, e.g.: `committed 412/520 GT-team frames (79.2%); contested/loose share 18.4%` under the
  existing `agreement: …% (target >= 80%)` line. The 80% target now reads as *conditional on an
  adequate commit rate*, which the human can eyeball.

**Optional (recommended against now).** Extend the GT schema to `{own, opp, contested, none}` and
score a 3-way agreement on `pred,gt ∈ {own,opp,contested}`. This is the only way to *directly*
validate `pct_contested`, but it materially raises the hand-labeling cost (the labeler must adjudicate
"contested" frame-by-frame, which is exactly the judgment the proxy exists to automate) and is not
needed to make the gate honest. Defer unless the coverage numbers prove insufficient. (Decision
flagged in the summary.)

### 3.7 Highest-confidence ball, not the first row `[LOW]`

**Where.** `phase/possession.py:41-42` reads `ball_rows["x_pitch"].iloc[0]` /
`["y_pitch"].iloc[0]` — an arbitrary row when a frame has multiple ball detections (possible at the
conf=0.05 floor). `pipeline.py:99-100` independently uses the **highest-confidence** ball for the
phase `ball_y`. So the two consumers can read different balls in the same frame.

**Spec.** §4.1: detections carry a `conf` column; the pipeline already treats highest-conf as the
canonical ball. Possession and phase should agree on the ball they classify against.

**Fix.** Select the single highest-confidence ball *row* (atomic x/y pair — no column mixing):

```python
ball_rows = group[group["class"] == "ball"]
if ball_rows.empty:
    states[fkey] = "unknown"; continue
ball = ball_rows.loc[ball_rows["conf"].idxmax()]
bx, by = ball["x_pitch"], ball["y_pitch"]
if pd.isna(bx) or pd.isna(by):
    states[fkey] = "unknown"; continue
```

`conf` is guaranteed present on enriched trajectories (`io/schema.validate_trajectories`, run at
`pipeline.py:92`). Picking one atomic row (rather than a per-column reduction) also avoids the
"Frankenstein x/y" hazard the audit noted for the pipeline's `groupby().last()` path
(`pipeline.py:100`) — that pipeline-side reduction is a separate LOW belonging to SP4 and is left
as-is here; this change makes possession's selection the principled one.

## 4. Testing (TDD)

Extend the existing suites; write the assertion before the edit. All targets are pure functions.

**`tests/test_phase_possession.py`** (§3.1, §3.7):
1. *Anisotropy / loose-ball boundary:* ball at (0.5, 0.5), lone own player at (0.6, 0.5) → raw
   x-dist 0.10 > 0.073 (would be `loose_ball`), but length-normalized 0.10/1.5 ≈ 0.067 < 0.073 →
   assert `own`. This fails on the current code and passes after §3.1.
2. *Anisotropy / clump axis:* a width-axis-spread balanced clump that counts as a clump only after
   normalization → assert `contested`.
3. *Spec override:* same geometry under `PitchSpec.fifa_11v11()` (aspect_ratio 1.54) yields a
   different boundary → asserts the param is honored.
4. *Highest-conf ball:* one frame, two ball rows — a high-conf ball near `own` and a low-conf
   first-row ball near `opp`; assert `own` (proves `iloc[0]` is no longer used).
5. *NaN-safe selection:* highest-conf ball has valid coords, a lower-conf row is NaN → classified
   normally; highest-conf ball is NaN → `unknown`.

**`tests/test_phase_splitter.py`** (§3.2, §3.3, §3.4, §3.5):
6. Boundary cases: `opp + y=0.6 → defend_low`, `opp + y=0.7 → defend_high`, plus a just-below/
   just-above-0.667 pair. Rename the misleading `…in_own_half` test to `…in_own_two_thirds`.
7. Fix the transition test's out-of-window assertion to `defend_low` at y=0.6.
8. *Turnover through contested:* `["own"]*10 + ["contested"]*5 + ["opp"]*20` fires `transition`
   starting at the first `opp` frame; the `contested` frames remain `contested`.
9. *No false turnover:* `["own"]*10 + ["loose_ball"]*5 + ["own"]*10` produces **no** transition.
10. *Halftime flip:* a series spanning `halftime_frame` with constant `own` + constant `ball_y=0.2`
    → `build` before halftime, `attack` after; symmetric opp case (`defend_low`→`defend_high`).
    Plus `halftime_frame=None` reproduces today's labels (regression guard).

**`tests/test_hygiene_core.py`** (§3.6):
11. *Abstention is visible:* GT all `own` over N frames; model commits `own` on some and `contested`
    on the rest with high conditional agreement → assert `pred_commit_rate` < 1.0 and that widening
    abstention lowers it while `agreement` stays high. Update the existing agreement tests
    (`:269-322`) for the new fields (values unchanged for `n_compared`/`agreement`).
12. *Empty case:* `n_gt_team == 0` → `pred_commit_rate` defined (0.0), no div-by-zero (guards the
    `:362` return site).

**`tests/test_pipeline_assemble.py`** (§3.5 threading):
13. `assemble_phases(..., halftime_frame=k)` flows to `label_phase` and flips post-k labels on a
    minimal synthetic clip; default (no kwarg) is unchanged.

## 5. Risks

- **`halftime_frame` is manual and easy to omit.** If a full game is processed without it, the second
  half is mislabeled exactly as today — no regression, but no fix either. Mitigation: default `None`
  is safe for single-half clips; document the parameter on the assemble entry points; surface it in
  the per-game report metadata so an omitted value is conspicuous. Auto-detection is explicitly out
  of scope.
- **Threshold re-tuning.** Correcting the width-axis distance (§3.1) loosens the *effective* radius
  in x by ~1.5×, which will shift `pct_contested` and the loose/own/opp split on real footage. This
  is the spec-correct behavior, but the e2e golden snapshot (§8.2) will move; regenerate it
  intentionally and note it in the commit.
- **Agreement coverage interpretation.** `pred_commit_rate` makes abstention visible but does not by
  itself *forbid* it; the 80% target must be read alongside it. If a human later wants a hard gate,
  that is the schema-extension option (§3.6), deferred.
- **Helper placement (§3.1).** Introducing `length_norm_xy` now risks a trivial premature
  abstraction if SP5 ends up wanting a different signature. Mitigation: keep it a 1-line pure
  function with the exact convention `hygiene` already uses; if rejected, inline the `/ar` in
  possession and let SP5 extract later (flagged for the human).

## 6. Out of scope

- own/opp grounding / `--own-kit` threading through `analyze_video` (SP2).
- The calibration/homography rework and camera model (SP1) — untouched here by design.
- `pipeline.py`'s `groupby().last()` ball reduction (`:99-100`), the `TrackingBackend` protocol gap,
  and other pipeline/labeler LOWs (SP4).
- Building any §6 metric module (SP5); we only make the distance convention and `halftime_frame`
  reusable for it.
- Automatic halftime detection; extending the GT CSV with a `contested` state (recommended-against;
  see §3.6).
- The refuted findings (`_infer_fps` rounding; eval aspect-ratio 1.5 vs 1.54) — not bugs.
</content>
</invoke>
