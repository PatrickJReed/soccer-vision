"""CLI: compare hand-labeled possession CSV against a phases.parquet.

CSV format (change points): header `t_seconds,possession`, one row whenever
possession changes; possession in {own, opp, none}.

Usage: python -m soccer_vision.hygiene.agreement --phases phases.parquet --gt gt.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soccer_vision.hygiene.core import possession_agreement


def main() -> None:
    ap = argparse.ArgumentParser(description="Possession ground-truth agreement")
    ap.add_argument("--phases", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    args = ap.parse_args()
    phases = pd.read_parquet(args.phases)
    gt = pd.read_csv(args.gt)
    res = possession_agreement(gt, phases)
    print(f"compared frames (both attribute a team): {res.n_compared}")
    print(f"agreement: {res.agreement * 100:.1f}%   (target >= 80%)")
    print(
        f"committed {res.n_compared}/{res.n_gt_team} GT-team frames "
        f"({res.pred_commit_rate * 100:.1f}%); "
        f"contested/loose share {res.pred_contested_frac * 100:.1f}%"
    )
    if res.disagreements:
        print("disagreement spans (t_start..t_end  gt -> pred):")
        for t0, t1, g, p in res.disagreements:
            print(f"  {t0:7.1f}..{t1:7.1f}s   {g} -> {p}")


if __name__ == "__main__":
    main()
