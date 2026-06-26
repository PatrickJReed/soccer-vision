# Soccer-Vision — Project Status & Open Issues (session handoff)

**Date:** 2026-06-26
**Purpose:** Snapshot for resuming fresh. Captures what shipped, the active unsolved
problem (with all diagnostic evidence so it need not be re-derived), the decision made on
direction, and every outstanding issue.

---

## TL;DR — where we are

The interactive pitch-anchor **labeler** is the per-game homography path (ML pitch model was
deferred — see memory). This session shipped two labeler improvements, **both merged to
`master` locally (NOT pushed — `master` is 18 commits ahead of `origin/master`):**

1. **Async refit (responsiveness)** — clicks return instantly; the windowed pose refit runs
   on a background worker. **Works well; Patrick confirmed "responsiveness is fast."**
2. **Gated propagation (accuracy attempt)** — `poses_by_gated_propagation`. **This did NOT
   fix the real problem and introduced a false-green status. It optimized reprojection
   residual, which is the wrong metric. See the open issue below — it likely needs to be
   reverted or have its status made honest.**

**ACTIVE UNSOLVED PROBLEM:** on the training clip, most frames draw the pitch-overlay lines
"in the sky." Root cause is now firmly diagnosed (below): each frame sees **one end** of the
field, so its pose is well-constrained for that end and wrong for the other end; the cure is
cross-end constraint, which requires **long-span chain propagation**, and the registration
chain **drifts 40–266 px over long spans** (vs 8 px short-span). Patrick chose to **"go at the
root" (the chain)**. The fix is undesigned — that's the next work.

---

## ⭐ CRITICAL REFRAME (end of session) — read this FIRST

**The camera is FIXED (Trace virtual-PTZ). There is ONE global field↔image homography, and
every video frame is just a 2D-cropped slice of the whole.** Confirmed by the data: inter-frame
transforms are pure 2D translations (tx up to ±88 px, ty≈0, scale≈1.000, rotation/shear≈0,
perspective≈1e-6). No real pan, no rotation, no zoom.

**We drifted from this.** The shipped calibration (Phase-1 onward) solves **per-frame
INDEPENDENT 6-DOF SQPNP poses**, not one global homography. That independence is the ROOT of
*both* problems below:
- single-end "lines in the sky" = each frame constrained only by its own clicks;
- chain drift = we chain frame-to-frame 2D translations and accumulate error.
All the per-frame propagation work (incl. gated propagation) was patching symptoms of this
divergence.

**THE RIGHT FIX (supersedes everything in "Decision/fix directions" below):** return to the
global model. Build the *whole* — one globally-consistent reference image of the fixed sensor
(a mosaic/panorama; register each frame to it directly or via a bundle-adjusted, NON-chained
global alignment so error is bounded, not accumulated) — then solve **ONE global homography
(whole → field) from ALL clicks at once.** A frame is a known 2D translation within the whole,
so own-end clicks (frame 0) and opp-end clicks (frame 193) jointly constrain the single
homography. No per-frame under-constraint, no chain-drift accumulation. Per-frame output
homography = (frame's 2D offset in the whole) ∘ (global homography), so downstream
(`assemble_from_homographies`) is unchanged. This is the design basis for the next session.

(Why this also explains old findings: Engine B's "propagate the pose as a camera rotation"
drifted 100-268 ft because the motion is NOT a rotation — it's a 2D crop; modeling it as a
rotation was wrong.)

---

## What shipped this session (all on `master`, unpushed)

### A. Async refit — responsiveness (commits `034b4bc`..`0b7f41d`)
- New `labeler/refit_worker.py` `RefitWorker`: daemon thread, dirty-set, revision-based
  cancellation (rapid clicks coalesce), `wait_idle`/`pending`.
- `LabelerState`: fits the **clicked frame synchronously** (instant overlay) then
  `mark_dirty`s the ±window for the background worker; `_compute_poses`/`_compute_dirty`/
  `_apply_fits` under one `RLock`; **clicks snapshot is force-copied** and all mutators locked
  (a HIGH-severity race was caught + fixed in review, guarded by a concurrent-stress test);
  line-refine scoped to `±line_band` (60).
- Server: `pending` in `/api/state`, worker stopped on shutdown. Frontend: polls while
  `pending>0`.
- Status: **GOOD. Keep.** Spec/plan: `docs/superpowers/{specs,plans}/2026-06-26-labeler-async-refit*`.

### B. Gated propagation — accuracy attempt (commits `8a07fe0`..`a7df03b`)
- `pitch/manual_anchor.propagate_clicks_with_distance` (propagation tracking each landmark's
  source frame-distance; `propagate_clicks` is now a thin wrapper).
- `pitch/calib_anchor.poses_by_gated_propagation`: per frame — RED if nearest support
  > `gap_dist` (180); SQPNP **seed** from the nearest `seed_size` (6) landmarks; **consistency
  gate** keeps farther landmarks only within `gate_px` (60) of the seed; final SQPNP + line
  refine. `LabelerState` swapped `_compute_poses` to it; old `poses_by_click_propagation`
  kept for `compare_engines`.
- **Status: DID NOT WORK.** It drove reprojection residual down (8–20 px) but the poses are
  still only constrained by each frame's single end → the OTHER end projects to the sky. Worse,
  the low residual paints those single-end frames **green (false confidence)**, hiding the
  problem the old engine surfaced as honest high-residual yellow/red. Spec/plan:
  `docs/superpowers/{specs,plans}/2026-06-26-labeler-gated-propagation*`.
- **Decision needed:** revert gated (back to honest old engine), OR make the green status
  fold/conditioning/coverage-aware so it stops lying — independent of the root-cause fix.

### Earlier this session (already pushed): Phase 3b-2 line-click UI (`aab91aa`..`70c0b43`)
LineClick + propagate + `refine_pose`-with-lines, server endpoint, LINES palette, persistence.
On `master` AND `origin` (this was pushed). Patrick placed 75 line clicks in the test session.

---

## THE OPEN ISSUE — "lines in the sky" (root cause confirmed)

### Symptom
On the training clip, most frames show the reprojected pitch overlay with lines drawn above
the field ("in the sky"); coverage shows ~100% green; adding more points to a frame doesn't
visibly fix it.

### What it is NOT (ruled out with evidence)
- **NOT mislabeled clicks.** Every clicked frame fits its OWN clicks at **8–17 px**; focal is
  correct (**1461–1471**, matches prior games' 1469).
- **NOT folding.** `fold_count` (calib/validate.py:20) counts landmarks projecting IN-FRAME;
  its docstring: *folding pulls the far field in → count near 21; a physical camera keeps the
  visible slice → ~6–12*. Measured values are **6–10 = healthy**. The SQPNP physical poses do
  **not** fold (Phase-1's no-fold property holds). *(Two earlier mis-diagnoses in this session:
  first chasing reprojection residual, then misreading `fold_count` as "folds" — both wrong.)*
- **NOT a rendering/units bug.** The clicked landmarks reproject correctly (they land where the
  user clicked); only UNCLICKED regions are wrong.

### What it IS (confirmed by direct geometry)
Direct projection of all 21 landmarks under the real gated SQPNP pose:
- The **clicked** landmarks land correctly (never in the sky).
- The **unclicked end** of the field lands in the sky / off-frame. Examples:
  - frame 193 (clicked only OPP-end landmarks): own-end goes to sky — `own_corner_R` y=**−464**,
    `own_post_R` y=**−125**.
  - frame 134 (clicked only OWN-end): opp-end is garbage — `opp_corner_R` y=**2464**.
- **Mechanism:** each Trace view sees ~one end (~half the field). The clicks at a frame pin
  that end; the other end is under-constrained, so the pose is accurate locally and wrong for
  the unclicked end → its overlay lines project into the sky.

### Why propagation can't currently rescue it — the chain drift (quantified)
Cross-end constraint = borrow the opposite end's clicks from a frame that saw it, via the
registration chain. Measured chain self-consistency (propagate a landmark's click across the
chain, compare to its actual re-click at the target frame — pure chain error):
- **short spans (≤60 frames): 8 px median** — chain is excellent locally.
- **long spans (>120 frames): 41 px median, up to 235–266 px** — drifts badly with distance.

Own-end → opp-end is a **long span (~190+ frames)**, so borrowed cross-end clicks arrive
40–266 px off — too noisy to constrain the pose, and the gated engine's 60 px gate rejects
them outright (which is why gated leaves every frame stuck on one end). The chain's worst drift
is around the fast own→opp **pan (~frames 152–173)**, where registering a busy game scene
(moving players, motion blur) is hardest.

### Supporting earlier finding (window sweep)
With the old aggregate engine: early frames are clean at `window ≤ 60` (residual ~15–23 px) but
explode at `window ≥ 120` (612–1587 px), because a wider window pulls in the drifted cross-end
clicks. Dense two-end clicking + a wide window actively fight each other.

---

## Decision made + fix directions (next work)

**Patrick chose B — "go at the root" (fix the chain).** Two candidate directions; not yet
designed:

1. **Reduce chain drift** so long-span borrowing works. Drift comes from registering a busy
   scene. Levers: **player-masking** during chain registration (`--player-boxes` plumbing
   exists in `compute_chain`; masking is OFF by default), software H.264 decode (cleaner
   frames), better feature matching / RANSAC. Cheap to try; may or may not get long-span error
   under the gate. *Reduces but never eliminates drift.*
2. **Global field-anchored solve (recommended).** Stop trusting the chain over long spans. The
   no-parallax camera is a fixed center + per-frame rotation, and every click references the
   same rigid field. Jointly optimize one focal + all per-frame rotations to fit ALL clicks at
   once (bundle adjustment), with the chain only as a local prior. The field ties own-end and
   opp-end clicks together directly instead of through 190 frames of drift. More work; robust
   to chain drift; standard SfM practice.

**Claude's lean:** (2) is the principled fix; (1) is a cheap experiment worth running first to
see if masking alone suffices. Suggested next step: a quick chain-masking experiment to measure
long-span drift with players masked, THEN design the global solve if needed. Run the proper
brainstorm→spec→plan→subagent-build flow for the chosen direction.

**Also pending (status honesty, independent of the root fix):** the gated engine's green is
lying on single-end frames. Either revert gated to the old engine (honest high-residual), or
make status fold/conditioning/coverage-aware (e.g., flag frames whose unclicked region is
under-constrained — possibly by checking how many of the 21 landmarks the pose places sanely,
or the conditioning of the click geometry).

---

## All outstanding issues (full list)

### Blocking the labeler's usefulness on this clip
- **[P0] "Lines in the sky" / single-end under-constraint + chain drift** — the open issue
  above. Root cause confirmed; fix undesigned (Patrick chose B).
- **[P1] Gated-engine false-green status** — revert or make status honest.

### Project-level (from memory, still true)
- **`master` is 18 commits ahead of `origin/master` — UNPUSHED.** Push when ready.
- **Phase 4 metrics** (the actual goal) is blocked on a trustworthy per-game homography, which
  this clip doesn't yet have. Also needs Patrick's possession ground-truth CSV (≥80% agreement)
  + confirm which kit is his.
- **Halftime direction gap** — phase splitter assumes a fixed attacking direction → mirror-wrong
  across halftime on full games; Phase-4 fix = attack-direction normalization with a MANUAL
  `halftime_frame`. (Homography-independent.)

### Labeler polish / follow-ups (non-blocking)
- `frame_jpeg` re-seeks the video on every request (`cap.set(POS_FRAMES, idx)`); mooted by
  all-intra clips but worth fixing in code (sequential read + JPEG cache) + default clip
  extraction to short GOP.
- Parallel chain-precompute `Pool` **hangs when the labeler is launched as a backgrounded /
  stdin-detached process** on macOS (spawn). **Workaround: always launch with `--workers 1`.**
- 3b-2 follow-ups: the projected **midline isn't drawn** in the overlay (`EDGES` lacks the
  `(4,5)` edge — would aid the midline line-click check); `n_clicks` counts only points
  (cosmetic).
- "Adding points doesn't visibly help" is explained by the root cause (a new point changes the
  pose but doesn't add cross-end depth, so the unclicked end stays in the sky) — not a separate
  bug.

---

## Operational / runtime context

- **Test clip:** `~/sv-labeler/training_clip.mp4` = 12:00–13:30 of `training.mp4`, re-encoded
  **all-intra** (`ffmpeg -g 1 -keyint_min 1 -x264-params scenecut=0`) for fast frame scrubbing
  (0.06–0.13 s/frame vs 0.25–0.58 s for a long-GOP cut). 2700 frames, 1080p30. Chain cached at
  `~/sv-labeler/.sv_labeler_cache/ef2546eaddd5e6fc.npz` (copied from the long-GOP cut — same
  frames; valid).
- **Patrick's session:** `~/sv-labeler/.sv_labeler_cache/training_clip.clicks.json` — **105
  point clicks across 16 frames + 75 line clicks.** This is the data behind every number above.
  Clicked frames: 0,3,37,77,80,134,144,193 (early dense cluster) then 552,997,1356,1506,1716,
  2076,2425,2436 (sparse).
- **A labeler server is still running** on `http://127.0.0.1:8000` (gated engine, `--workers 1`,
  this session). It's a daemon; it will keep running after the session closes. Kill with
  `pkill -f soccer_vision.labeler` if desired.
- **Relaunch command:**
  ```bash
  cd /Users/patrickreed/Sandbox/soccer-vision/packages/soccer-vision
  uv run python -m soccer_vision.labeler --video /Users/patrickreed/sv-labeler/training_clip.mp4 \
    --export-dir /Users/patrickreed/sv-labeler/training_clip_out --port 8000 --workers 1
  ```
- **Diagnostic scripts** used this session live in `/tmp/diag_*.py` (ephemeral — chain drift,
  fold projection, free-fit, window sweep). They load the chain + sidecar and re-run the engines;
  re-creatable from the numbers above if needed.

## Key files
- `packages/soccer-vision/src/soccer_vision/pitch/calib_anchor.py` — engines
  (`poses_by_click_propagation` [old], `poses_by_gated_propagation` [new], `calibrate_clicked_frames`,
  `frame_homography`, `_fold_for_pose`, `_reproj_rms_px`).
- `packages/soccer-vision/src/soccer_vision/pitch/manual_anchor.py` — `propagate_clicks[_with_distance]`,
  `build_segments`, `cumulative_transforms`.
- `packages/soccer-vision/src/soccer_vision/labeler/{state.py,server.py,refit_worker.py,static/}`.
- `packages/soccer-vision/src/soccer_vision/labeler/chain.py` — `compute_chain` (registration,
  the drift source), `_video_hash`.
- `packages/soccer-vision/src/soccer_vision/calib/validate.py` — `fold_count` (counts in-frame
  landmarks; 6–12 healthy).

## Suggested first moves for the fresh session
1. Re-read this file + the memory (`MEMORY.md` → `project_interactive_anchor_labeler.md`).
2. Decide the status-honesty stopgap (revert gated vs fold-aware status) so green stops lying.
3. Run the cheap chain-masking experiment: re-`compute_chain` with player boxes / software
   decode and re-measure long-span drift (target: < ~30 px so cross-end clicks pass a gate).
4. If masking is insufficient, brainstorm + build the **global field-anchored bundle solve**
   (one focal + per-frame rotations fit to all clicks, chain as a local prior).
