"""Round-trip tests for the frozen-benchmark manifest."""

from __future__ import annotations

from pathlib import Path

from soccer_vision.eval.benchmark import BenchmarkField, BenchmarkManifest, load_manifest


def test_manifest_round_trip(tmp_path: Path) -> None:
    m = BenchmarkManifest(fields=[
        BenchmarkField(field="chula_vista", game_id="game8",
                       homographies="chula/homographies.parquet",
                       frame_indices=[100, 200, 300]),
        BenchmarkField(field="carlsbad", game_id="game3",
                       homographies="carlsbad/homographies.parquet",
                       frame_indices=[50, 60]),
    ])
    p = tmp_path / "benchmark.json"
    m.save(p)
    loaded = load_manifest(p)
    assert loaded == m
    assert [f.field for f in loaded.fields] == ["chula_vista", "carlsbad"]
    assert loaded.fields[0].frame_indices == [100, 200, 300]
