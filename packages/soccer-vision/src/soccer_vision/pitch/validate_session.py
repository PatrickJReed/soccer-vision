"""CLI: measure the global-homography held-out cross-end accuracy for a labeler session.

This is the ACCEPTANCE GATE for the calibration global-homography rework (Sub-project 1).
It loads a cached inter-frame chain + a clicks sidecar, runs leave-one-clicked-frame-out
cross-end validation, and reports the error in feet. We do not trust the new calibration
(or start the metrics product / delete the old engines) until this passes.

Usage:
  uv run python -m soccer_vision.pitch.validate_session \\
      --chain  ~/sv-labeler/.sv_labeler_cache/<videohash>.npz \\
      --clicks ~/sv-labeler/.sv_labeler_cache/training_clip.clicks.json

The .npz cache stores the normalized inter-frame chain plus n_frames and (w, h), so no
size/frame-count flags are needed. Acceptance bar: overall held-out median < 1.5 m
(~4.92 ft) — the finest downstream metric threshold (the contested-possession margin).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from soccer_vision.labeler.chain import load_chain
from soccer_vision.labeler.state import clicks_from_sidecar
from soccer_vision.pitch.global_calib import cross_end_holdout
from soccer_vision.pitch.manual_anchor import build_segments, cumulative_transforms

ACCEPT_FT = 4.92  # 1.5 m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", type=Path, required=True, help="cached inter-frame chain .npz")
    ap.add_argument("--clicks", type=Path, required=True, help="clicks sidecar .json")
    args = ap.parse_args()

    loaded = load_chain(args.chain)
    if loaded is None:
        raise SystemExit(f"chain cache not found: {args.chain}")
    interframe, n_frames, size = loaded

    clicks = clicks_from_sidecar(args.clicks)
    segment_of = build_segments(interframe, n_frames)
    transforms = cumulative_transforms(interframe, segment_of)

    report = cross_end_holdout(clicks, transforms, segment_of, size)
    if report is None:
        print("Not enough clicked frames (need >= 2) for a held-out report.")
        return

    print("leave-one-clicked-frame-out cross-end error (feet):")
    print(f"  overall   median={report.median_ft:6.2f}  p90={report.p90_ft:6.2f}  n={report.n}")
    print(f"  own-end   median={report.own_median_ft}")
    print(f"  opp-end   median={report.opp_median_ft}")
    verdict = "PASS" if report.median_ft < ACCEPT_FT else "FAIL"
    print(f"  ACCEPTANCE (< {ACCEPT_FT:.2f} ft = 1.5 m): {verdict}")


if __name__ == "__main__":
    main()
