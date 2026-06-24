"""Round-trip tests for the frozen-benchmark manifest."""

from __future__ import annotations

from pathlib import Path

from soccer_vision.eval.benchmark import BenchmarkField, BenchmarkManifest, load_manifest


def test_manifest_round_trip(tmp_path: Path) -> None:
    m = BenchmarkManifest(fields=[
        BenchmarkField(field="chula_vista", split="unseen_field", game_id="game8",
                       homographies="chula/homographies.parquet",
                       keypoints="chula/keypoints.parquet",
                       frame_indices=[100, 200, 300]),
        BenchmarkField(field="carlsbad", split="unseen_time", game_id="game3",
                       homographies="carlsbad/homographies.parquet",
                       keypoints="carlsbad/keypoints.parquet",
                       frame_indices=[50, 60]),
    ])
    p = tmp_path / "benchmark.json"
    m.save(p)
    loaded = load_manifest(p)
    assert loaded == m
    assert [f.field for f in loaded.fields] == ["chula_vista", "carlsbad"]
    assert loaded.fields[0].frame_indices == [100, 200, 300]
    assert loaded.fields[0].split == "unseen_field"
    assert loaded.fields[1].split == "unseen_time"


def test_sample_frames_even_spread() -> None:
    from soccer_vision.eval.benchmark import sample_frames
    out = sample_frames(list(range(0, 101, 5)), 5)   # 21 available
    assert out[0] == 0 and out[-1] == 100
    assert len(out) == 5
    assert out == sorted(set(out))


def test_sample_frames_fewer_than_n() -> None:
    from soccer_vision.eval.benchmark import sample_frames
    assert sample_frames([3, 9, 12], 10) == [3, 9, 12]
