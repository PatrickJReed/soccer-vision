"""Wiring test for analyze_video using a stub backend (no GPU / no real video)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from soccer_vision.pipeline import PipelineResult, analyze_video
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

FPS = 1.0


def _identity_keypoints(n_frames: int) -> pd.DataFrame:
    idxs = [0, 5, 13, 16, 24, 29]
    pts = PITCH_LANDMARKS[idxs]
    rows = []
    for f in range(n_frames):
        for k, (x, y) in zip(idxs, pts, strict=True):
            rows.append({"frame": f, "kp_idx": k, "x_px": float(x), "y_px": float(y), "conf": 0.9})
    return pd.DataFrame(rows)


def _trajectories() -> pd.DataFrame:
    rows = []
    for f in range(3):
        rows.append({
            "frame": f, "t_seconds": f / FPS, "track_id": 1,
            "x_px": 0.5, "y_px": 0.25,
            "bbox_x1": 0.49, "bbox_y1": 0.24, "bbox_x2": 0.51, "bbox_y2": 0.26,
            "class": "player", "team": "own", "conf": 0.9,
        })
        rows.append({
            "frame": f, "t_seconds": f / FPS, "track_id": -1 - f,
            "x_px": 0.5, "y_px": 0.27,
            "bbox_x1": 0.49, "bbox_y1": 0.26, "bbox_x2": 0.51, "bbox_y2": 0.28,
            "class": "ball", "team": "unknown", "conf": 0.9,
        })
    return pd.DataFrame(rows).astype({"frame": "int64", "track_id": "int64"})


class _StubBackend:
    name = "stub"
    version = "0"

    def __init__(self, traj: pd.DataFrame, kp: pd.DataFrame) -> None:
        self._traj = traj
        self._kp = kp

    def process_with_pitch(self, video_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._traj, self._kp


def test_analyze_video_writes_four_parquets(tmp_path: Path) -> None:
    backend = _StubBackend(_trajectories(), _identity_keypoints(3))
    out_dir = tmp_path / "game1"

    result = analyze_video(Path("unused.mp4"), out_dir, backend=backend)

    assert isinstance(result, PipelineResult)
    for name in ("trajectories_px.parquet", "keypoints.parquet",
                 "trajectories.parquet", "phases.parquet"):
        assert (out_dir / name).exists(), f"missing {name}"
    # Checkpoint is the verbatim tracker output.
    checkpoint = pd.read_parquet(out_dir / "trajectories_px.parquet")
    assert "x_pitch" not in checkpoint.columns
    assert checkpoint.shape == _trajectories().shape
    assert list(checkpoint.columns) == list(_trajectories().columns)
    # Deliverable is enriched.
    enriched = pd.read_parquet(out_dir / "trajectories.parquet")
    assert "x_pitch" in enriched.columns
    assert list(result.phases["frame"]) == [0, 1, 2]
