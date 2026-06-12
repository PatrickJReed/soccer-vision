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
    ap.add_argument("--window", type=int, default=360,
                    help="max frame distance a click propagates (drift caught "
                         "by the residual gate; default 360 = ±12s at 30fps, "
                         "the measured accuracy knee)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="previously exported keypoints.parquet to load as clicks")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel chain precompute workers (default: cores-1)")
    args = ap.parse_args()
    run(args.video, port=args.port, export_dir=args.export_dir,
        window=args.window, resume=args.resume, workers=args.workers)


if __name__ == "__main__":
    main()
