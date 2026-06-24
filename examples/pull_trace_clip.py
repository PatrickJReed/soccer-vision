#!/usr/bin/env python3
"""Pull Trace games (or short clips) out of a YouTube playlist.

The Trace playlist is the full training + testing corpus, so this is a reusable
corpus tool, not a one-off:

  list the games (pick which field is which):
      python pull_trace_clip.py PLAYLIST_URL --list

  grab a 2-minute UNSEEN test clip (for acceptance_pitch):
      python pull_trace_clip.py PLAYLIST_URL --game 3 --start 12:00 --duration 120 \
          --out heldout_gameC.mp4

  grab a WHOLE game (for labeling -> more training data):
      python pull_trace_clip.py PLAYLIST_URL --game 3 --out game3.mp4

Requires yt-dlp (pip install yt-dlp) and ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys


def _require_tools() -> None:
    for tool in ("yt-dlp", "ffmpeg"):
        if shutil.which(tool) is None:
            sys.exit(
                f"missing '{tool}' on PATH — install it "
                f"(pip install yt-dlp; brew/apt install ffmpeg)"
            )


def list_games(playlist: str) -> list[tuple[int, str, str]]:
    """Return (index, video_id, title) for every entry, without downloading."""
    out = subprocess.run(
        ["yt-dlp", "--flat-playlist", "-J", playlist],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    return [
        (i, e.get("id", "?"), e.get("title", "<no title>"))
        for i, e in enumerate(data.get("entries", []), start=1)
    ]


def _video_url(playlist: str, game: int) -> str:
    rows = list_games(playlist)
    if not 1 <= game <= len(rows):
        sys.exit(f"--game {game} out of range; playlist has {len(rows)} games")
    return f"https://www.youtube.com/watch?v={rows[game - 1][1]}"


def _add_offset(start: str, duration: int) -> str:
    """start (HH:MM:SS or MM:SS) + duration seconds -> HH:MM:SS."""
    secs = 0
    for part in start.split(":"):
        secs = secs * 60 + int(part)
    secs += duration
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def pull(url: str, out: str, start: str | None, duration: int | None) -> None:
    # native resolution up to 1080p (Trace is 1080p/30) so the clip matches
    # what the model trained on; remux to mp4.
    cmd = [
        "yt-dlp",
        "-f", "bv*[height<=1080]+ba/b[height<=1080]",
        "--merge-output-format", "mp4",
        "-o", out, url,
    ]
    if start is not None and duration is not None:
        end = _add_offset(start, duration)
        cmd[1:1] = [
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",  # clean cut at the boundaries
        ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("playlist", help="YouTube playlist URL")
    ap.add_argument("--list", action="store_true", help="list games and exit")
    ap.add_argument("--game", type=int, help="1-based game index in the playlist")
    ap.add_argument("--start", help="clip start HH:MM:SS or MM:SS (omit for whole game)")
    ap.add_argument("--duration", type=int, help="clip length in seconds (omit for whole game)")
    ap.add_argument("--out", help="output .mp4 path")
    args = ap.parse_args(argv)

    _require_tools()

    if args.list:
        for i, _vid, title in list_games(args.playlist):
            print(f"{i:2d}  {title}")
        return

    if args.game is None or args.out is None:
        ap.error("need --game and --out (or --list)")
    if (args.start is None) != (args.duration is None):
        ap.error("--start and --duration go together (omit both for the whole game)")

    url = _video_url(args.playlist, args.game)
    pull(url, args.out, args.start, args.duration)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
