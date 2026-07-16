# %% [markdown]
# # Game-prep (Colab) — pull, digest, rep pack; then registration + line dataset
#
# One parameterized notebook, reused per game, so the local machine never holds a
# full game video (spec: docs/superpowers/specs/2026-07-16-colab-game-prep-design.md):
#
#   Stage A (Colab): pull the game from the unlisted playlist onto Colab's
#     EPHEMERAL disk -> view digest (montage shown inline) -> write the few-MB
#     "rep pack" (rep_video.mp4 + rep_map.json + view_digest.json) to Drive.
#   LOCAL (the only local step): download rep_pack/, click every rep frame GREEN
#     in the existing labeler, upload rep_export/ back to Drive (see PAUSE cell).
#   Stage B (Colab): remap the labeled homographies tiny->original frame indices,
#     register every frame to its view rep, build the view manifest, register the
#     game in games.toml, run the line-dataset generator into Drive.
#
# Drive receives ONLY the rep pack (MBs), session parquets (KBs-MBs) and the
# dataset pairs (~0.3-0.7 GB/game — mind Drive space across games). The full game
# video stays on the ephemeral disk (re-pulled by the Stage B guard cell after a
# disconnect); it is NEVER written to Drive and never downloaded locally.
#
# Repo-side verification (no Colab in CI): py_compile + ruff clean, and the two
# pure pieces (build_rep_pack, remap_rep_homographies) exercised locally against
# the oceanside clip + cached digest.

# %%
# ---- CONFIG -----------------------------------------------------------------
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

IN_COLAB = "google.colab" in sys.modules
# The action cells below no-op outside Colab (repo-side compile/import stays
# side-effect-free). Set True to run the cells on another Linux box.
RUN_CELLS = IN_COLAB

# The public repo is cloned + pip-installed INSIDE Colab. GIT_REF pins the exact
# code this notebook was validated against — bump it (a newer sha, or
# "origin/master") when you need newer soccer_vision code.
REPO_URL = "https://github.com/PatrickJReed/soccer-vision.git"
GIT_REF = "738f17abf93ec219ea9ae80879d7df4ee341eef0"
REPO_DIR = Path("/content/soccer-vision")

# ---- Per-game parameters (edit these per run) ---------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive/soccer-vision")
# The playlist URL is READ FROM DRIVE, never hardcoded here: this repo is PUBLIC
# and the playlist is unlisted family video — the URL must never appear in the
# committed notebook. Put the URL(s) in this text file on your Drive, one per
# line; PLAYLIST_LINE picks the line (1-based).
PLAYLIST_FILE = DRIVE_ROOT / "trace_playlist_url.txt"
# YouTube bot-checks datacenter IPs (Colab): downloads need logged-in cookies.
# Export youtube.com cookies (Netscape format; a throwaway account is fine —
# the videos are unlisted, any signed-in account passes) and put them here.
# The setup cell wires them into yt-dlp's default config when the file exists.
COOKIES_FILE = DRIVE_ROOT / "youtube_cookies.txt"
PLAYLIST_LINE = 1
GAME_NUMBER = 1                     # 1-based game index in the playlist (--list order)
GAME_ID = "riverside_2026_07a"      # dataset game id (bare TOML key: [A-Za-z0-9_-])
FIELD_ID = "riverside"              # FIELD id — powers field-held-out evaluation

# ---- Knobs ---------------------------------------------------------------------
DIGEST_STRIDE = 25       # view-digest sampling stride. The ORB similarity matrix is
#                          O(samples^2): a full 90-min game at stride 25 is ~6500
#                          samples (hours) — the digest cell AUTO-BUMPS the stride
#                          to keep ~900 samples and says so loudly. To force denser
#                          sampling anyway, set FORCE_DIGEST_STRIDE = True.
FORCE_DIGEST_STRIDE = False   # True: use DIGEST_STRIDE as-is, no auto-adjust
DIGEST_DIST_THRESHOLD = 0.5
REGISTER_STRIDE = 5      # frame stride for view registration. Keep it a divisor of
#                          round(fps) (30 for Trace) so the generator's 1-second
#                          stride anchor (frame % 30 == 0) lands on registered rows.
ASSIGN_STRIDE = 5        # dense view-assignment stride for the view manifest
COVERAGE_WARN = 0.90     # registration coverage below this -> redo the weak views
FORCE_STAGE_A = False    # True: recompute digest + rep pack even if already on Drive
DATASET_NAME = "line_dataset_v1"

# ---- Derived paths (Drive = durable; /content = EPHEMERAL) ----------------------
GAME_DIR = DRIVE_ROOT / GAME_ID
REP_PACK_DIR = GAME_DIR / "rep_pack"      # Stage A output — the only download
REP_EXPORT_DIR = GAME_DIR / "rep_export"  # you upload this after local clicking
SESSION_DIR = GAME_DIR / "session"        # homographies.parquet + view manifest
DATASET_DIR = DRIVE_ROOT / DATASET_NAME
GAMES_TOML = DRIVE_ROOT / "games.toml"
VIDEO_PATH = Path(f"/content/{GAME_ID}.mp4")   # full game: ephemeral disk ONLY
WORK_DIR = Path(f"/content/{GAME_ID}_work")    # digest render + ORB caches (ephemeral)


# %%
# ---- Rep pack + remap (pure helpers — exercised locally in repo verification) ---
def _read_frames_at(video_path, indices):
    """Decode the requested ASCENDING frame indices -> {index: BGR frame}.

    Sequential forward grab (same pattern as view_digest._read_frames), but with
    no soccer_vision import: this cell must run before the setup cell installs
    the package, and the repo-side local exercise imports this file directly.
    """
    cap = cv2.VideoCapture(str(video_path))
    out = {}
    pos = 0
    try:
        for idx in indices:
            if idx < pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                pos = idx
            while pos < idx:
                if not cap.grab():
                    break
                pos += 1
            ok, frame = cap.read()
            pos += 1
            if ok:
                out[idx] = frame
    finally:
        cap.release()
    return out


def _pack_hash(rep_map):
    """Content stamp for a rep pack: sha1 over the sorted (tiny, frame, view)
    triples. Written into rep_map.json as "_pack_hash"; load_rep_map verifies it."""
    triples = sorted((int(t), int(m["frame"]), int(m["view"]))
                     for t, m in rep_map.items())
    return hashlib.sha1(json.dumps(triples).encode()).hexdigest()[:16]


def build_rep_pack(video_path, representatives, out_dir, *, fps=2.0):
    """Write the few-MB rep pack: rep_video.mp4 + rep_map.json; return the map.

    rep_video.mp4 holds ONE original-resolution frame per view, ordered by
    ORIGINAL frame index (chronological — scrubs naturally in the labeler),
    written via cv2.VideoWriter mp4v then re-encoded truly ALL-INTRA (ffmpeg
    -g 1) so the labeler's frame-exact seeking assumption holds literally.
    rep_map.json maps the tiny video's frame index to
    {"frame": original_frame, "view": view_id} (+ a "_pack_hash" stamp); Stage
    B's remap inverts it. Raises on any undecodable rep frame (a missing rep
    silently drops its whole view) and re-decodes the FINAL written video to
    verify the frame count.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_items = sorted((int(f), int(v)) for v, f in representatives.items())
    frames = _read_frames_at(video_path, [f for f, _ in rep_items])
    missing = [f for f, _ in rep_items if f not in frames]
    if missing:
        raise ValueError(f"could not decode representative frame(s) {missing} — "
                         f"a missing rep drops its whole view; check the video")
    h, w = frames[rep_items[0][0]].shape[:2]
    video_out = out_dir / "rep_video.mp4"
    tmp_out = out_dir / "rep_video_mp4v.mp4"  # cv2 write target; replaced below
    writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    if not writer.isOpened():
        raise ValueError(f"cv2.VideoWriter failed to open {tmp_out} (mp4v)")
    rep_map = {}
    for tiny, (orig, view) in enumerate(rep_items):
        img = frames[orig]
        if img.shape[:2] != (h, w):
            writer.release()
            raise ValueError(f"frame {orig} size {img.shape[:2]!r} != {(h, w)!r}")
        writer.write(img)
        rep_map[tiny] = {"frame": orig, "view": view}
    writer.release()
    # True all-intra re-encode: mp4v's keyframe cadence is codec-default, so force
    # -g 1 via ffmpeg (preinstalled on Colab). If ffmpeg is absent (bare local
    # run), keep the mp4v file with a warning — a ~13-25-frame video still scrubs.
    if shutil.which("ffmpeg"):
        subprocess.check_call(
            ["ffmpeg", "-y", "-v", "error", "-i", str(tmp_out), "-c:v", "libx264",
             "-g", "1", "-keyint_min", "1", "-x264-params", "scenecut=0",
             "-crf", "18", "-pix_fmt", "yuv420p", str(video_out)])
        tmp_out.unlink()
    else:
        print("WARNING: ffmpeg not found — keeping the mp4v encode (not strictly "
              "all-intra; fine for a tiny rep video)")
        tmp_out.replace(video_out)
    n_out = 0  # verify the FINAL file by decoding (after the remux): a silent
    cap = cv2.VideoCapture(str(video_out))  # codec failure would waste a click round
    while cap.read()[0]:
        n_out += 1
    cap.release()
    if n_out != len(rep_items):
        raise ValueError(f"rep_video decodes {n_out} frames, expected {len(rep_items)}")
    (out_dir / "rep_map.json").write_text(
        json.dumps({**rep_map, "_pack_hash": _pack_hash(rep_map)}, indent=2))
    return rep_map


def load_rep_map(path):
    """rep_map.json -> {tiny_index (int): {"frame": int, "view": int}}.

    Verifies the "_pack_hash" stamp when present — a corrupted or hand-edited
    map must never silently mis-anchor views.
    """
    raw = json.loads(Path(path).read_text())
    out = {int(k): {"frame": int(v["frame"]), "view": int(v["view"])}
           for k, v in raw.items() if not k.startswith("_")}
    stamp = raw.get("_pack_hash")
    if stamp is not None and stamp != _pack_hash(out):
        raise ValueError(f"{path}: _pack_hash mismatch — rep_map.json does not "
                         f"match its stamp (corrupted or hand-edited)")
    return out


def remap_rep_homographies(tiny_parquet, rep_map, out_path=None):
    """Remap a labeler homographies.parquet from tiny rep_video frame indices back
    to ORIGINAL game frame indices via rep_map; returns the remapped DataFrame.

    Every input row's frame must be a known tiny index, and no two rows may land
    on the same original frame — both raise (never silently mis-anchor a view).
    All other columns (h00..h22, source, confidence) pass through untouched.
    """
    df = pd.read_parquet(tiny_parquet)
    mapping = {int(t): int(m["frame"]) for t, m in rep_map.items()}
    tiny = df["frame"].astype("int64")
    unknown = sorted(set(tiny) - set(mapping))
    if unknown:
        raise ValueError(f"frame indices {unknown} not in rep_map — was this "
                         f"parquet exported from a different rep_video?")
    out = df.copy()
    out["frame"] = tiny.map(mapping)
    if out["frame"].duplicated().any():
        dupes = sorted(out.loc[out["frame"].duplicated(), "frame"].tolist())
        raise ValueError(f"duplicate original frames after remap: {dupes}")
    out = out.sort_values("frame").reset_index(drop=True)
    if out_path is not None:
        out.to_parquet(out_path, index=False)
    return out


# %%
# ---- Pull mechanism + stage predicates (shared by Stage A and the Stage B guard)
def read_playlist_url():
    """Line PLAYLIST_LINE (1-based) of PLAYLIST_FILE on Drive.

    The unlisted playlist URL must never appear in this committed notebook
    (public repo, unlisted family videos) — it lives only in the Drive file.
    """
    if not PLAYLIST_FILE.exists():
        raise FileNotFoundError(
            f"{PLAYLIST_FILE} not found — create it on Drive with the playlist "
            f"URL(s), one per line (it is intentionally NOT in this notebook)")
    lines = [ln.strip() for ln in PLAYLIST_FILE.read_text().splitlines() if ln.strip()]
    if not 1 <= PLAYLIST_LINE <= len(lines):
        raise ValueError(f"PLAYLIST_LINE={PLAYLIST_LINE} out of range "
                         f"({len(lines)} line(s) in {PLAYLIST_FILE.name})")
    return lines[PLAYLIST_LINE - 1]


def probe_video(path):
    """(n_frames, fps, width, height) via cv2."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


_LINK_RE = re.compile(r"https?://\S+")


def _run_url_safe(cmd, url):
    """Run a command whose argv contains the unlisted URL without ever echoing it.

    pull_trace_clip has no quiet flag and yt-dlp prints lines like
    'Extracting URL: https://...', so output is captured and re-printed with
    every link redacted to '<link>' (dropping whole http-bearing lines hid the
    real yt-dlp ERROR text — the 2026-07-16 bot-check failure was invisible);
    any line still carrying the playlist id after redaction is dropped outright.
    Failures re-raise a clean RuntimeError with the URL redacted from argv
    (never a raw CalledProcessError carrying the URL). This keeps the unlisted
    family-playlist URL out of saved cell output.
    """
    list_id = url.split("list=")[-1].split("&")[0] if "list=" in url else url
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for stream in (proc.stdout, proc.stderr):
        for ln in (stream or "").splitlines():
            ln = ln.replace(url, "<playlist-url>")
            ln = _LINK_RE.sub("<link>", ln)
            if list_id in ln:
                continue
            print(ln)
    if proc.returncode != 0:
        redacted = [("<playlist-url>" if a == url else a) for a in cmd]
        raise RuntimeError(f"pull step failed (exit {proc.returncode}): {redacted}")


def ensure_game_video():
    """Pull game GAME_NUMBER onto the EPHEMERAL disk unless already there.

    Reuses examples/pull_trace_clip.py from the cloned repo (yt-dlp + ffmpeg +
    deno for YouTube's n-challenge full-res formats — all provided by the setup
    cell; Colab preinstalls ffmpeg). The video lands on /content ONLY: never on
    Drive, never local. Prints the playlist titles (not URLs) for an eyeball
    check that GAME_NUMBER is the right game.

    Re-pull determinism: YouTube serves a stable 1080p encode, so frame indices
    are stable across pulls — a Stage B re-pull anchors to the same frames the
    rep pack was built from.
    """
    # >50 MB is a heuristic completeness check only: yt-dlp downloads to .part
    # files and renames on completion, so a partial at the final path is unlikely.
    if VIDEO_PATH.exists() and VIDEO_PATH.stat().st_size > 50 * 2**20:
        print(f"video already on ephemeral disk: {VIDEO_PATH} "
              f"({VIDEO_PATH.stat().st_size / 2**30:.2f} GB) — skipping pull")
        return
    url = read_playlist_url()
    script = REPO_DIR / "examples" / "pull_trace_clip.py"
    print(f"playlist entries (game {GAME_NUMBER} will be pulled):")
    _run_url_safe([sys.executable, str(script), url, "--list"], url)
    _run_url_safe([sys.executable, str(script), url,
                   "--game", str(GAME_NUMBER), "--out", str(VIDEO_PATH)], url)
    n, fps, w, h = probe_video(VIDEO_PATH)
    print(f"pulled {VIDEO_PATH.name}: {n} frames @ {fps:.4g} fps, {w}x{h}, "
          f"{VIDEO_PATH.stat().st_size / 2**30:.2f} GB (ephemeral disk)")


def stage_a_done():
    """The rep pack is already on Drive (Stage A output is durable)."""
    return ((REP_PACK_DIR / "rep_video.mp4").exists()
            and (REP_PACK_DIR / "rep_map.json").exists())


def stage_b_ready():
    """The locally-labeled rep export has been uploaded back to Drive."""
    ok = (REP_EXPORT_DIR / "homographies.parquet").exists()
    if not ok:
        print(f"SKIP: {REP_EXPORT_DIR / 'homographies.parquet'} not on Drive yet — "
              f"do the LOCAL labeling step (PAUSE cell), upload rep_export/, then "
              f"re-run the Stage B cells")
    return ok


# %%
# ---- Colab setup: clone repo @ GIT_REF, install package + pull tooling, mount Drive
if RUN_CELLS:
    if not (REPO_DIR / ".git").exists():
        subprocess.check_call(["git", "clone", REPO_URL, str(REPO_DIR)])
    subprocess.check_call(["git", "-C", str(REPO_DIR), "fetch", "--quiet", "origin"])
    subprocess.check_call(["git", "-C", str(REPO_DIR), "checkout", "--quiet", GIT_REF])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           str(REPO_DIR / "packages" / "soccer-vision"), "yt-dlp"])
    # yt-dlp needs a JS runtime (deno) for YouTube's "n challenge" — without it
    # most videos cap at 360p; pull_trace_clip auto-finds ~/.deno/bin.
    if shutil.which("deno") is None and not (Path.home() / ".deno/bin/deno").exists():
        subprocess.check_call("curl -fsSL https://deno.land/install.sh | sh", shell=True)
    if IN_COLAB:
        from google.colab import drive
        drive.mount("/content/drive")
    for d in (DRIVE_ROOT, GAME_DIR, WORK_DIR):
        d.mkdir(parents=True, exist_ok=True)
    # Wire Drive cookies into yt-dlp's default config (read by every yt-dlp
    # invocation, incl. pull_trace_clip's) — Colab's datacenter IPs hit
    # YouTube's "Sign in to confirm you're not a bot" without them.
    if COOKIES_FILE.exists():
        _ytdlp_cfg = Path.home() / ".config" / "yt-dlp"
        _ytdlp_cfg.mkdir(parents=True, exist_ok=True)
        (_ytdlp_cfg / "config").write_text(f"--cookies {COOKIES_FILE}\n")
        print(f"yt-dlp cookies wired from {COOKIES_FILE}")
    else:
        print(f"NOTE: no {COOKIES_FILE.name} on Drive — if the pull fails with "
              f"'Sign in to confirm you're not a bot', export youtube.com "
              f"cookies (Netscape format; throwaway account fine) to "
              f"{COOKIES_FILE} and re-run this cell (see the runbook)")
    print(f"repo @ {GIT_REF[:12]} | drive root: {DRIVE_ROOT} | game: {GAME_ID}")

# %% [markdown]
# ## Stage A — pull -> digest -> rep pack (all Colab; output = a few MB on Drive)
# Run-all safe: these cells skip themselves once the rep pack exists on Drive
# (set `FORCE_STAGE_A = True` to redo). Eyeball the montage: each tile is one
# distinct camera view, labeled with its representative's ORIGINAL frame number.

# %%
if RUN_CELLS:
    if FORCE_STAGE_A or not stage_a_done():
        ensure_game_video()
    else:
        print(f"rep pack already on Drive ({REP_PACK_DIR}) — Stage A cells skip; "
              f"set FORCE_STAGE_A = True to redo")

# %%
# ---- Stage A: view digest + inline montage --------------------------------------
if RUN_CELLS and (FORCE_STAGE_A or not stage_a_done()):
    from soccer_vision.labeler.view_digest import compute_view_digest, render_digest

    n_frames, fps, _w, _h = probe_video(VIDEO_PATH)
    digest_stride = DIGEST_STRIDE
    n_samples = len(range(0, max(n_frames, 1), digest_stride))
    if n_samples > 1200 and not FORCE_DIGEST_STRIDE:
        # The ORB similarity matrix is O(samples^2): ~6500 samples (a full game at
        # stride 25) means HOURS on Colab CPU. Auto-adjust keeps Run-all usable.
        digest_stride = max(DIGEST_STRIDE, -(-n_frames // 900))
        n_samples = len(range(0, max(n_frames, 1), digest_stride))
        print(f"AUTO-ADJUST: DIGEST_STRIDE {DIGEST_STRIDE} -> {digest_stride} "
              f"({n_samples} samples; O(samples^2) similarity would take hours at "
              f"the configured stride). To force denser sampling set DIGEST_STRIDE "
              f"explicitly AND FORCE_DIGEST_STRIDE = True.")
    elif n_samples > 1200:
        print(f"WARNING: FORCE_DIGEST_STRIDE=True keeps {n_samples} samples at "
              f"stride {digest_stride} — the O(n^2) similarity matrix will take "
              f"hours on Colab CPU")
    digest = compute_view_digest(
        VIDEO_PATH, stride=digest_stride, dist_threshold=DIGEST_DIST_THRESHOLD,
        cache_dir=WORK_DIR / "cache")
    digest_dir = WORK_DIR / "digest"
    render_digest(digest, VIDEO_PATH, digest_dir)
    span = {v: sum(1 for vv in digest.view_of.values() if vv == v)
            for v in digest.representatives}
    print(f"{n_samples} sampled frames -> {digest.n_views} views "
          f"(expect ~13-25 on a full game)")
    for view, frame in sorted(digest.representatives.items()):
        print(f"  view {view:2d}: rep frame {frame:6d}  ({span[view]} sampled frames)")
    from IPython.display import Image, display
    display(Image(filename=str(digest_dir / "views_montage.png")))

# %%
# ---- Stage A: build the rep pack -> Drive ----------------------------------------
if RUN_CELLS and (FORCE_STAGE_A or not stage_a_done()):
    if REP_EXPORT_DIR.exists():
        # Any export that predates the pack being (re)built was clicked against a
        # DIFFERENT rep_video: its tiny indices would silently mis-anchor views.
        stale = REP_EXPORT_DIR.with_name(
            f"rep_export_stale_{datetime.now():%Y%m%d_%H%M%S}")
        REP_EXPORT_DIR.rename(stale)
        print(f"STALE EXPORT MOVED: {REP_EXPORT_DIR.name} predates this rep-pack "
              f"build -> {stale.name}. Re-click the NEW rep_video.mp4 and upload a "
              f"fresh rep_export/ (Stage B cannot detect overlap-but-different "
              f"maps on its own — this move is what prevents mis-anchoring).")
    rep_map = build_rep_pack(VIDEO_PATH, digest.representatives, REP_PACK_DIR)
    for name in ("view_digest.json", "views_montage.png"):
        shutil.copy2(WORK_DIR / "digest" / name, REP_PACK_DIR / name)
    size_mb = sum(p.stat().st_size for p in REP_PACK_DIR.iterdir()) / 2**20
    print(f"rep pack -> {REP_PACK_DIR} ({size_mb:.1f} MB, {len(rep_map)} rep frames "
          f"ordered by original frame index)")
    print(f"next: PAUSE cell below — download rep_pack/, click locally, upload "
          f"rep_export/ to {REP_EXPORT_DIR}")

# %% [markdown]
# ## PAUSE — the LOCAL clicking step (the only local work)
#
# 1. Download this game's `rep_pack/` folder from Drive (a few MB:
#    `soccer-vision/<GAME_ID>/rep_pack/` -> rep_video.mp4, rep_map.json,
#    view_digest.json, views_montage.png).
# 2. Label EVERY frame of the tiny video with the existing labeler
#    (from `packages/soccer-vision/`):
#
#        uv run python -m soccer_vision.labeler --video rep_pack/rep_video.mp4 \
#          --export-dir rep_export --workers 1
#
#    Each tiny frame is one view's representative, ordered by ORIGINAL frame
#    index (rep_map.json says which view/original frame each one is; the
#    montage's frame numbers are ORIGINAL indices). Every frame must come out
#    GREEN: 5+ spread point landmarks (corners, box corners, posts) plus
#    near-touchline / midline LINE clicks where visible. A frame that is not
#    green drops its WHOLE VIEW from registration. Each rep is its own one-frame
#    segment — the physical engine handles that natively (the shared focal wants
#    >= 3 diverse green frames).
# 3. Hit Export, then upload the resulting `rep_export/` folder to Drive at
#    `soccer-vision/<GAME_ID>/rep_export/` (must contain homographies.parquet).
# 4. Run the Stage B cells below. After a Colab restart, run the CONFIG + setup
#    cells first; the guard cell re-pulls the video onto the ephemeral disk.

# %%
# ---- Stage B guard: rep_export uploaded? re-pull the ephemeral video if absent --
if RUN_CELLS and stage_b_ready():
    ensure_game_video()

# %%
# ---- Stage B: tiny->original remap + register every frame to its view rep -------
if RUN_CELLS and stage_b_ready():
    from soccer_vision.pitch.view_registration import (
        _digest_from_json,
        register_clip,
        rep_homographies_from_parquet,
        write_homographies,
    )

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    rep_map = load_rep_map(REP_PACK_DIR / "rep_map.json")
    remapped_path = SESSION_DIR / "rep_homographies.parquet"
    # Stale-export guards: remap raises unless the export's tiny indices are a
    # SUBSET of the current rep_map, and Stage A renames any pre-existing
    # rep_export/ to rep_export_stale_* whenever it rebuilds the pack. An export
    # whose indices overlap a DIFFERENT map (rebuilt pack, similar size) is NOT
    # auto-detectable here — the Stage A stale-move is what prevents that case.
    remap_rep_homographies(
        REP_EXPORT_DIR / "homographies.parquet", rep_map, out_path=remapped_path)
    digest = _digest_from_json(REP_PACK_DIR / "view_digest.json")
    rep_h = rep_homographies_from_parquet(remapped_path, digest.representatives)
    print(f"labeled rep homographies: {len(rep_h)}/{digest.n_views} views")
    missing_views = sorted(set(digest.representatives) - set(rep_h))
    if missing_views:
        print(f"WARNING: views {missing_views} have no green labeled rep — their "
              f"frames can only register to OTHER views (or gap). Re-click those "
              f"frames locally and re-upload rep_export/.")
    n_frames, fps, _w, _h = probe_video(VIDEO_PATH)
    assert round(fps) % REGISTER_STRIDE == 0, (
        f"REGISTER_STRIDE={REGISTER_STRIDE} must divide round(fps)={round(fps)} "
        f"or the generator's 1s anchor (frame % {round(fps)} == 0) misses every "
        f"registered frame")
    frames = sorted(set(range(0, max(n_frames, 1), REGISTER_STRIDE))
                    | {digest.representatives[v] for v in rep_h
                       if digest.representatives[v] < n_frames})
    calib = register_clip(VIDEO_PATH, digest, rep_h, frames=frames)
    write_homographies(calib, SESSION_DIR / "homographies.parquet")
    s = calib.stats
    # 0.6 = the generator's default --min-confidence trust gate (line_dataset)
    usable = sum(1 for e in calib.homographies.values() if e.confidence >= 0.6)
    print(f"registered {s['n_registered']} + {s['n_rep']} rep of {s['n_frames']} "
          f"frames (stride {REGISTER_STRIDE}); coverage {s['coverage']:.1%} "
          f"(generator-usable @ conf>=0.6: {usable / max(s['n_frames'], 1):.1%}), "
          f"median inliers {s['median_inliers']:.0f}")
    if s["coverage"] < COVERAGE_WARN:
        print(f"WARNING: coverage {s['coverage']:.1%} < {COVERAGE_WARN:.0%} — some "
              f"views did not label/register; redo the local step for those views")

# %%
# ---- Stage B: dense per-frame view manifest -> session/view_manifest.parquet ----
if RUN_CELLS and stage_b_ready():
    from soccer_vision.labeler.view_dataset import build_view_assignment, write_export
    from soccer_vision.pitch.view_registration import _digest_from_json

    digest = _digest_from_json(REP_PACK_DIR / "view_digest.json")
    va = build_view_assignment(
        VIDEO_PATH, digest, game=GAME_ID, assign_stride=ASSIGN_STRIDE,
        cache_dir=WORK_DIR / "cache")
    write_export(va, WORK_DIR / "view_dataset_out", video_path=VIDEO_PATH)
    shutil.copy2(WORK_DIR / "view_dataset_out" / "view_dataset.parquet",
                 SESSION_DIR / "view_manifest.parquet")
    shutil.copy2(WORK_DIR / "view_dataset_out" / "view_dataset.json",
                 SESSION_DIR / "view_manifest.json")  # provenance sidecar
    print(f"view manifest: {va.n_frames} rows, {va.n_views} views, switch_rate "
          f"{va.switch_rate:.4f}, {va.n_ambiguous} ambiguous "
          f"-> {SESSION_DIR / 'view_manifest.parquet'}")


# %%
# ---- Stage B: register the game in DRIVE_ROOT/games.toml (idempotent upsert) ----
def _toml_value(v):
    """TOML literal for a scalar (JSON string escaping is valid TOML basic-string).

    Only str/int/float are supported — anything else (incl. bool, whose repr is
    not valid TOML) raises rather than silently writing a broken registry.
    """
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"unsupported games.toml value type "
                        f"{type(v).__name__}: {v!r}")
    return repr(v)


def upsert_games_toml(toml_path, game_id, entry):
    """Replace/add [game_id] in games.toml, preserving every other game's table.

    The registry schema is flat scalar-valued tables (line_dataset.load_games),
    so a hand-rolled writer is safe; unknown keys on other entries are preserved.
    """
    import tomllib
    ok = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    games = tomllib.loads(toml_path.read_text()) if toml_path.exists() else {}
    games[game_id] = entry
    lines = []
    for gid in sorted(games):
        if not set(gid) <= ok:
            raise ValueError(f"game id {gid!r} is not a bare TOML key")
        lines.append(f"[{gid}]")
        for k, v in games[gid].items():
            lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")
    toml_path.write_text("\n".join(lines))


if RUN_CELLS and stage_b_ready():
    # video = the ABSOLUTE ephemeral path (the video is never on Drive by design):
    # the generator runs in this same session, and load_games resolves absolute
    # paths as-is. After a reconnect the Stage B guard re-pulls to this same path.
    upsert_games_toml(GAMES_TOML, GAME_ID, {
        "field": FIELD_ID,
        "video": str(VIDEO_PATH),
        "session": f"{GAME_ID}/session",  # relative to games.toml (Drive)
    })
    print(f"games.toml updated: [{GAME_ID}] -> {GAMES_TOML}")

# %%
# ---- Stage B: run the line-dataset generator (pairs land on Drive) --------------
if RUN_CELLS and stage_b_ready():
    from soccer_vision import line_dataset

    line_dataset.main(["--games", str(GAMES_TOML), "--out", str(DATASET_DIR),
                       "--game", GAME_ID])

# %%
# ---- Stage B: eyeball the labels — stats + contact sheet inline -----------------
if RUN_CELLS and stage_b_ready():
    from IPython.display import Image, display

    stats = json.loads((DATASET_DIR / "dataset_stats.json").read_text())
    g = stats["games"].get(GAME_ID, {})
    print(f"{GAME_ID}: {g.get('n_written', 0)} pairs written, "
          f"{g.get('n_empty_masks', 0)} empty masks (expected 0), "
          f"{g.get('n_undecodable', 0)} undecodable")
    print("class pixel fractions:",
          {k: round(v, 6) for k, v in g.get("class_pixel_frac", {}).items()})
    if g.get("n_empty_masks", 0):
        print("WARNING: empty masks — check homography signs/coverage before training")
    sheet = DATASET_DIR / f"contact_{GAME_ID}.jpg"
    if sheet.exists():
        display(Image(filename=str(sheet)))
    else:
        print(f"no contact sheet at {sheet}")
