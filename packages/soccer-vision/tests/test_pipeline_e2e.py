"""End-to-end regression scaffold for the §6 metrics product (Phase 4 / SP5).

SKIPPED until the metrics layer exists. The spec's §8.2
(2026-05-27-soccer-vision-design.md:539-546) calls for this test to run the full
pipeline on the bake-off clip, compute the §6 summary statistics (mean_area_own,
pct_contested, ...), and assert each is within 5% of a golden JSON.

It cannot exist yet: soccer_vision.metrics is a one-line stub (no §6 metrics). This
file is the ready SEAM for SP5 — fill the body when Phase 4 metrics land.

COVERAGE NOTE: the repo's reported coverage is SUBSYSTEM-scoped, not product-readiness.
There is no end-to-end metric gate until SP5 fills this scaffold.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 4 metrics not built — see SP5 (the §6 metrics product)")
def test_pipeline_e2e_summary_stats_within_tolerance() -> None:
    """Intended shape (SP5 fills this in):

      1. result = analyze_video(BAKEOFF_CLIP, tmp_out)   # or assemble_from_homographies
         on the cached 5-parquet checkpoint to avoid GPU.
      2. summary = compute_summary_stats(result)         # §6 metrics: mean_area_own,
         pct_contested, swarm_index, pitch_coverage, ...
      3. golden = json.load(GOLDEN_PATH)
      4. for k, v in summary.items():
             assert abs(v - golden[k]) <= 0.05 * abs(golden[k])  # 5% drift tolerance

    Upstream model weights are pinned (model drift out of scope); intentional version
    bumps regenerate the golden file.
    """
    raise AssertionError("scaffold — must not run while skipped")
