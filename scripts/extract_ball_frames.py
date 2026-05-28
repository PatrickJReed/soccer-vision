"""Extract sampled frames from multiple game videos for ball-detector fine-tuning.

Strategy: per-video uniform stride sampling. The fine-tune dataset needs
~500 frames total spanning ball states (rolling, in-flight, occluded, near touchline).

Usage:
    uv run python scripts/extract_ball_frames.py \\
        --inputs ~/Downloads/Game1.mov ~/Downloads/Game2.mov \\
        --out data/labeled/ball_v1/raw \\
        --per-video 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract(video_path: Path, out_dir: Path, n_frames: int, prefix: str) -> int:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise RuntimeError(f"Video {video_path} reports 0 frames")
    stride = max(1, total // n_frames)
    saved = 0
    for frame_idx in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out = out_dir / f"{prefix}_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        if saved >= n_frames:
            break
    cap.release()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="paths to game .mov/.mp4 files")
    ap.add_argument("--out", type=Path, required=True, help="output dir for sampled frames")
    ap.add_argument("--per-video", type=int, default=100, help="frames per video")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for video in args.inputs:
        video_path = Path(video).expanduser()
        prefix = video_path.stem.replace(" ", "_")
        saved = extract(video_path, args.out, args.per_video, prefix)
        print(f"{video_path.name}: saved {saved} frames")
        total += saved
    print(f"\nTotal: {total} frames in {args.out}")


if __name__ == "__main__":
    main()
