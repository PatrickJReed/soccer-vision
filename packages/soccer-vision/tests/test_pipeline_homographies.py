"""Tests for homographies checkpoint I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
from soccer_vision.pipeline import (
    PipelineResult,
    assemble_from_homographies,
    homographies_from_parquet,
    homographies_to_parquet,
)
from soccer_vision.pitch.propagation import HomographyEntry

FPS = 1.0


def test_homographies_parquet_roundtrip(tmp_path) -> None:
    entries = {
        3: HomographyEntry(np.eye(3), "anchor", 1.0),
        4: HomographyEntry(np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]]),
                           "propagated", 0.7),
    }
    path = tmp_path / "homographies.parquet"
    homographies_to_parquet(entries, path)
    back = homographies_from_parquet(path)

    assert set(back) == {3, 4}
    assert back[3].source == "anchor" and back[3].confidence == 1.0
    assert back[4].source == "propagated" and abs(back[4].confidence - 0.7) < 1e-9
    assert np.allclose(back[4].H, entries[4].H)
    assert np.allclose(back[3].H, entries[3].H)


def test_homographies_to_parquet_columns(tmp_path) -> None:
    homographies_to_parquet({3: HomographyEntry(np.eye(3), "anchor", 1.0)},
                            tmp_path / "h.parquet")
    df = pd.read_parquet(tmp_path / "h.parquet")
    assert list(df.columns) == [
        "frame", *[f"h{i}{j}" for i in range(3) for j in range(3)], "source", "confidence",
    ]


def _traj() -> pd.DataFrame:
    rows = []
    for f in range(3):
        rows.append({"frame": f, "t_seconds": f / FPS, "track_id": 1,
                     "x_px": 0.5, "y_px": 0.25, "bbox_x1": 0.49, "bbox_y1": 0.24,
                     "bbox_x2": 0.51, "bbox_y2": 0.26, "class": "player", "team": "own", "conf": 0.9})
        rows.append({"frame": f, "t_seconds": f / FPS, "track_id": -1 - f,
                     "x_px": 0.5, "y_px": 0.27, "bbox_x1": 0.49, "bbox_y1": 0.26,
                     "bbox_x2": 0.51, "bbox_y2": 0.28, "class": "ball", "team": "unknown", "conf": 0.9})
    return pd.DataFrame(rows).astype({"frame": "int64", "track_id": "int64"})


def test_assemble_from_homographies_roundtrip(tmp_path) -> None:
    traj = _traj()
    traj_path = tmp_path / "trajectories_px.parquet"
    h_path = tmp_path / "homographies.parquet"
    traj.to_parquet(traj_path, index=False)
    homographies_to_parquet({f: HomographyEntry(np.eye(3), "anchor", 1.0) for f in range(3)}, h_path)
    out_dir = tmp_path / "out"

    result = assemble_from_homographies(traj_path, h_path, out_dir)
    assert isinstance(result, PipelineResult)
    assert (out_dir / "trajectories.parquet").exists()
    assert (out_dir / "phases.parquet").exists()
    assert result.anchor_coverage == 1.0


def test_manual_source_counts_toward_coverage(tmp_path) -> None:
    # I1 lock: homographies with source="manual" must contribute to coverage.
    traj = _traj()
    traj_path = tmp_path / "trajectories_px.parquet"
    h_path = tmp_path / "homographies.parquet"
    traj.to_parquet(traj_path, index=False)
    homographies_to_parquet(
        {f: HomographyEntry(np.eye(3), "manual", 1.0) for f in range(3)}, h_path
    )
    out_dir = tmp_path / "out"

    result = assemble_from_homographies(traj_path, h_path, out_dir)
    assert result.homography_coverage > 0
