"""CLI: python -m soccer_vision.hygiene --traj ... --homographies ... --video ...

See docs/superpowers/specs/2026-06-11-track-hygiene-design.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from soccer_vision.hygiene.run import run_hygiene


def main() -> None:
    ap = argparse.ArgumentParser(description="Track hygiene & team re-clustering")
    ap.add_argument("--traj", required=True, type=Path, help="trajectories_px.parquet")
    ap.add_argument("--homographies", required=True, type=Path)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--own-kit", required=True,
                    help="your shirt color (e.g. white, dark blue)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-gap-s", type=float, default=2.0)
    ap.add_argument("--max-speed", type=float, default=8.0)
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()
    run_hygiene(
        traj_path=args.traj, homographies_path=args.homographies,
        video_path=args.video, out_dir=args.out_dir, own_kit=args.own_kit,
        max_gap_s=args.max_gap_s, max_speed_ms=args.max_speed, margin=args.margin,
    )


if __name__ == "__main__":
    main()
