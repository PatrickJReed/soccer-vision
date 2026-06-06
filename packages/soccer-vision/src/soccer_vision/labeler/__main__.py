"""CLI entry: python -m soccer_vision.labeler --video game.mp4 [--port 8000]."""

from __future__ import annotations

import argparse
from pathlib import Path

from soccer_vision.labeler.server import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive pitch anchor labeler")
    ap.add_argument("--video", required=True, type=Path, help="path to an H.264 mp4")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--export-dir", type=Path, default=None,
                    help="where Export writes parquets (default: cwd)")
    args = ap.parse_args()
    run(args.video, port=args.port, export_dir=args.export_dir)


if __name__ == "__main__":
    main()
