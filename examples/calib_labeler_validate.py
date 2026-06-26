"""Validate the calibrated LabelerState backend on a full-game session.

Loads a chain .npz + an exported keypoints.parquet, runs the calibrated LabelerState,
and reports the focal, coverage, fold-free behaviour, and the flagged outlier clicks.
The user (or Claude, locally) runs this on the real data — it is the 3b-1 go/no-go.

Usage: python examples/calib_labeler_validate.py CHAIN.npz KEYPOINTS.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(chain_path: str, keypoints_path: str) -> None:
    from soccer_vision.labeler.chain import load_chain
    from soccer_vision.labeler.state import LabelerState, clicks_from_keypoints_parquet

    loaded = load_chain(Path(chain_path))
    assert loaded is not None, f"chain not found: {chain_path}"
    interframe, n_frames, size = loaded
    clicks = clicks_from_keypoints_parquet(Path(keypoints_path), size)

    st = LabelerState(interframe=interframe, n_frames=n_frames, size=size, window=360)
    st.add_clicks(clicks)

    n_covered = sum(1 for f in range(n_frames) if st._status_of(f) != "red")
    flagged = st._outliers
    focal = None if st._K is None else round(float(st._K[0, 0]), 1)
    print(f"calibrated: {st._calibrated}   focal: {focal} px")
    print(f"coverage: {n_covered}/{n_frames} frames ({100 * n_covered / n_frames:.1f}%), "
          f"green: {st.coverage() * 100:.1f}%")
    print(f"frames with flagged outlier clicks: {len(flagged)}")
    for f in sorted(flagged):
        print(f"  frame {f}: dropped kp {sorted(flagged[f])}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
