# Colab Game-Prep Notebook — Design (2026-07-16)

**Goal.** Move the per-game line-dataset workflow onto Colab so Patrick's Air never holds
a full game video (limited local storage — Patrick's requirement, 2026-07-16). The ONLY
local step is the interactive rep-frame clicking, shrunk to a few-MB "rep pack".

**Deliverable.** `packages/soccer-vision/scripts/colab_game_prep.py` + exact `.ipynb`
twin (repo cell pattern, like the trainer). One parameterized notebook, reused per game.

## Flow (three stages, one notebook)

**Stage A — Colab (heavy, no local storage):**
1. Setup: clone the PUBLIC repo (`GIT_REF` config, default a pinned commit), install the
   `soccer-vision` package + yt-dlp. Mount Drive.
2. Config: `DRIVE_ROOT`, `PLAYLIST_FILE` (a Drive text file with playlist URLs, one per
   line — the unlisted URL is NEVER hardcoded in the committed notebook; public repo),
   `PLAYLIST_LINE` (1-based), `GAME_NUMBER`, `GAME_ID`, `FIELD_ID`.
3. Pull the game to Colab's EPHEMERAL disk (never Drive, never local) — reuse the
   mechanism in `examples/pull_trace_clip.py` (read it; adapt for Colab, e.g. plain
   yt-dlp if its helper needs tooling Colab lacks).
4. View digest (`labeler.view_digest` API) on the Colab-disk video; display
   `views_montage.png` INLINE for Patrick's eyeball.
5. Build the **rep pack** → `DRIVE_ROOT/<game_id>/rep_pack/`: `rep_video.mp4` (the ~13–25
   representative frames only, original resolution, all-intra so the labeler scrubs
   instantly), `rep_map.json` ({tiny_video_index → original_frame, view_id}), and the
   digest json. A few MB — the only download.

**Local — clicking only (existing labeler, zero new UI):**
Download rep_pack; `python -m soccer_vision.labeler --video rep_video.mp4 --export-dir
rep_export --workers 1`; click every frame to GREEN (5+ spread points + near-TL/midline
line clicks); Export; upload `rep_export/` to `DRIVE_ROOT/<game_id>/rep_export/`.
Each rep frame is its own one-frame registration segment — the physical engine handles
that natively (shared focal from ≥3 diverse rep frames; per-frame anchors; no propagation).

**Stage B — Colab (heavy again):**
6. Read `rep_export/homographies.parquet`; REMAP its frame indices tiny→original via
   `rep_map.json`; `rep_homographies_from_parquet` + `register_clip` against the
   Colab-disk video → `DRIVE_ROOT/<game_id>/session/homographies.parquet`; print
   coverage/inlier stats (warn < ~90%: some reps not green/registered — redo step Local
   for those views).
7. View manifest: run the `view_dataset` assignment on Colab; copy its parquet to
   `session/view_manifest.parquet`.
8. Append/refresh the game's entry in `DRIVE_ROOT/games.toml` (paths relative to it);
   run the line-dataset generator with `--out DRIVE_ROOT/line_dataset_v1`; display the
   contact sheet INLINE + print stats (`0 empty masks` expected).

## Constraints & notes

- YouTube bot-checks Colab's datacenter IPs (hit live 2026-07-16): pulls need
  logged-in cookies. `DRIVE_ROOT/youtube_cookies.txt` (Netscape export, throwaway
  account recommended) is wired into yt-dlp's default config by the setup cell when
  present; the pull-output filter redacts links (`<link>`) instead of dropping whole
  lines so yt-dlp ERROR text stays visible without ever echoing the unlisted URLs.
- Colab disk holds the full video (ephemeral, fine at ~80 GB); Drive receives only:
  rep pack (MBs), session parquets (KBs–MBs), and the dataset pairs (~0.3–0.7 GB/game —
  note Drive-space in the notebook).
- A Colab disconnect between stages only costs the ephemeral video (re-pull); everything
  durable is on Drive. Stage B re-pulls the video if absent (idempotent guard cell).
- Repo-side verification (no Colab in CI): `py_compile` + ruff clean; the rep-pack build
  and index-remap functions must be exercised LOCALLY against the existing oceanside
  clip + cached digest as build evidence (they're pure enough); `.ipynb` = exact cell twin.
- Runbook `line-dataset-labeling.md` is updated to the Colab-first flow (local-only path
  kept as an appendix).

## Out of scope

Running the labeler UI inside Colab (port-proxy — rejected: slow scrubbing, click loss on
disconnect); automation of the clicking; trainer changes (it already reads the Drive
dataset dir).
