"""Tests for homographies checkpoint I/O."""

from __future__ import annotations

import numpy as np
from soccer_vision.pipeline import homographies_from_parquet, homographies_to_parquet
from soccer_vision.pitch.propagation import HomographyEntry


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


def test_homographies_to_parquet_columns(tmp_path) -> None:
    import pandas as pd
    homographies_to_parquet({3: HomographyEntry(np.eye(3), "anchor", 1.0)},
                            tmp_path / "h.parquet")
    df = pd.read_parquet(tmp_path / "h.parquet")
    assert list(df.columns) == [
        "frame", *[f"h{i}{j}" for i in range(3) for j in range(3)], "source", "confidence",
    ]
