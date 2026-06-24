"""Frozen-benchmark manifest: which held-out fields/frames define the eval set.

Versioned so every retrain scores the identical frames. Paths are relative to the
manifest file's directory (portable across machines/Colab).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkField:
    field: str               # human label, e.g. "chula_vista"
    game_id: str             # source game identifier
    homographies: str        # path to the labeler homographies.parquet (relative)
    keypoints: str           # path to the labeler keypoints.parquet (relative)
    frame_indices: list[int]  # the sampled frames scored for this field


@dataclass(frozen=True)
class BenchmarkManifest:
    fields: list[BenchmarkField]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(
            {"fields": [asdict(f) for f in self.fields]}, indent=2))


def load_manifest(path: Path) -> BenchmarkManifest:
    data = json.loads(Path(path).read_text())
    return BenchmarkManifest(fields=[
        BenchmarkField(
            field=f["field"], game_id=f["game_id"],
            homographies=f["homographies"], keypoints=f["keypoints"],
            frame_indices=list(f["frame_indices"]),
        )
        for f in data["fields"]
    ])
