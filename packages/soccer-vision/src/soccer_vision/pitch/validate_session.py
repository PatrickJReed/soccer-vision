"""CLI: acceptance gate for a labeler session's PHYSICAL calibration.

Loads a cached inter-frame chain + a clicks sidecar (points AND lines) and runs the
held-out gate: (1) near-touchline FOREGROUND (remove it, predict it) and (2) leave-one-
anchor-out bracket PROPAGATION, both in feet. With --video it also renders a few clipped
overlay spot-checks (including a sparse/no-line frame) for a human to confirm before the
session is trusted -- numeric metrics under-measure the "lines in the sky" failure.

Usage:
  uv run python -m soccer_vision.pitch.validate_session \\
      --chain  ~/sv-labeler/.sv_labeler_cache/<videohash>.npz \\
      --clicks ~/sv-labeler/.sv_labeler_cache/<clip>.clicks.json \\
      [--video ~/sv-labeler/<clip>.mp4 --spot-out ~/sv-labeler/gate_spotcheck]

Acceptance (numeric): foreground median <= 5 ft & p90 <= 12 ft AND propagation median <= 5 ft.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from soccer_vision.labeler.chain import load_chain
from soccer_vision.labeler.state import clicks_from_sidecar, line_clicks_from_sidecar
from soccer_vision.pitch.manual_anchor import build_segments, cumulative_transforms
from soccer_vision.pitch.physical_calib import (
    GateReport,
    evaluate_gate,
    solve_session,
)

# canonical field skeleton (x = width frac, y = length frac) as polylines to overlay
_SKELETON: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (
    ((0.0, 0.0), (0.0, 1.0)),   # near touchline
    ((1.0, 0.0), (1.0, 1.0)),   # far touchline
    ((0.0, 0.0), (1.0, 0.0)),   # own goal line
    ((0.0, 1.0), (1.0, 1.0)),   # opp goal line
    ((0.0, 0.5), (1.0, 0.5)),   # midline
)


def run_gate(chain_path: Path, clicks_path: Path) -> GateReport | None:
    """Load the chain + clicks and evaluate the numeric acceptance gate. None if the chain
    cannot be loaded."""
    loaded = load_chain(chain_path)
    if loaded is None:
        return None
    interframe, n_frames, size = loaded
    points = clicks_from_sidecar(clicks_path)
    lines = line_clicks_from_sidecar(clicks_path)
    segment_of = build_segments(interframe, n_frames)
    transforms = cumulative_transforms(interframe, segment_of)
    return evaluate_gate(points, lines, size, transforms, segment_of=segment_of)


def _spot_frames(clicked: list[int], n_frames: int, k: int = 8) -> list[int]:
    """A spread of frames to render: the clicked anchors + evenly-spaced samples across the
    clip (so a viewer sees both anchored and propagated frames)."""
    spread = [round(i * (n_frames - 1) / (k - 1)) for i in range(k)] if n_frames > 1 else [0]
    return sorted(set(clicked) | set(spread))


def render_spotcheck(
    chain_path: Path, clicks_path: Path, video_path: Path, out_dir: Path
) -> list[Path]:
    """Render clipped field-overlay PNGs for a spread of frames. Returns the written paths.
    Off-screen / behind-camera field parts are clipped out (clipped_polyline)."""
    import cv2

    from soccer_vision.viz.pitch_overlay import clipped_polyline

    loaded = load_chain(chain_path)
    if loaded is None:
        return []
    interframe, n_frames, size = loaded
    w, h = size
    points = clicks_from_sidecar(clicks_path)
    lines = line_clicks_from_sidecar(clicks_path)
    segment_of = build_segments(interframe, n_frames)
    transforms = cumulative_transforms(interframe, segment_of)
    calib = solve_session(points, lines, size, transforms, segment_of=segment_of)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    written: list[Path] = []
    for f in _spot_frames(sorted({c.frame for c in points}), n_frames):
        h_norm = calib.frame_homography(f)
        if h_norm is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        h_pitch_to_px = np.diag([float(w), float(h), 1.0]) @ np.linalg.inv(h_norm)
        for a, b in _SKELETON:
            samples = np.linspace(a, b, 50)
            # Clip per-segment so a line that exits the frame / dips behind the camera
            # BREAKS instead of drawing a spurious chord across the dropped gap.
            for s0, s1 in itertools.pairwise(samples):
                seg = clipped_polyline(h_pitch_to_px, np.array([s0, s1]), size=size, margin=60)
                if len(seg) == 2:
                    cv2.line(frame, seg[0], seg[1], (0, 255, 255), 2, cv2.LINE_AA)
        tag = f"frame {f}  {calib.status(f)}  {'anchor' if calib.is_anchor(f) else 'propagated'}"
        cv2.putText(frame, tag, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(frame, tag, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        path = out_dir / f"spot_{f:05d}.png"
        cv2.imwrite(str(path), frame)
        written.append(path)
    cap.release()
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", type=Path, required=True, help="cached inter-frame chain .npz")
    ap.add_argument("--clicks", type=Path, required=True, help="clicks sidecar .json")
    ap.add_argument("--video", type=Path, default=None, help="video for spot-check renders")
    ap.add_argument("--spot-out", type=Path, default=None, help="dir for spot-check PNGs")
    args = ap.parse_args()

    report = run_gate(args.chain, args.clicks)
    if report is None:
        raise SystemExit(f"chain cache not found: {args.chain}")

    print("held-out acceptance gate (feet):")
    print(f"  foreground   median={report.fg_median_ft:6.2f}  p90={report.fg_p90_ft:6.2f}"
          f"  n={report.fg_n}")
    print(f"  propagation  median={report.prop_median_ft:6.2f}  p90={report.prop_p90_ft:6.2f}"
          f"  n={report.prop_n}")
    print(f"  NUMERIC (fg med<=5 & p90<=12, prop med<=5): "
          f"{'PASS' if report.passed_numeric else 'FAIL'}")

    if args.video is not None:
        out = args.spot_out or Path("gate_spotcheck")
        paths = render_spotcheck(args.chain, args.clicks, args.video, out)
        print(f"\nwrote {len(paths)} spot-check overlays -> {out}")
        print("REQUIRED: review the spot-checks (incl. a sparse/no-line frame) and confirm "
              "the foreground/overlay before trusting green.")


if __name__ == "__main__":
    main()
