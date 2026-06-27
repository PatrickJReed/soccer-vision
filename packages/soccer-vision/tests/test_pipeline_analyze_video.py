"""Wiring test for analyze_video using a stub backend (no GPU / no real video)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from soccer_vision.pipeline import PipelineResult, analyze_video
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.tracking.mock import MockBackend

FPS = 1.0


def _identity_keypoints(n_frames: int) -> pd.DataFrame:
    idxs = [0, 3, 6, 11, 16, 19]
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

    def process(self, video_path: Path) -> pd.DataFrame:
        return self._traj

    def process_with_pitch(self, video_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._traj, self._kp


def test_analyze_video_writes_five_parquets(tmp_path: Path) -> None:
    backend = _StubBackend(_trajectories(), _identity_keypoints(3))
    out_dir = tmp_path / "game1"

    result = analyze_video(Path("unused.mp4"), out_dir, backend=backend)

    assert isinstance(result, PipelineResult)
    for name in ("trajectories_px.parquet", "keypoints.parquet",
                 "homographies.parquet", "trajectories.parquet", "phases.parquet"):
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


def _write_tiny_video(path: Path, n: int = 8) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))  # type: ignore[attr-defined]
    for _ in range(n):
        vw.write(np.zeros((240, 320, 3), dtype=np.uint8))
    vw.release()


def test_analyze_video_runs_end_to_end_with_mock_backend(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_tiny_video(video)
    out_dir = tmp_path / "mockgame"
    result = analyze_video(video, out_dir, backend=MockBackend())
    assert isinstance(result, PipelineResult)
    for name in ("trajectories_px.parquet", "keypoints.parquet",
                 "homographies.parquet", "trajectories.parquet", "phases.parquet"):
        assert (out_dir / name).exists(), f"missing {name}"


def _trajectories_with_gk() -> pd.DataFrame:
    """Player (track 1, own), a goalkeeper (track 2), and a per-frame ball."""
    rows: list[dict[str, object]] = []
    for f in range(3):
        rows.append({"frame": f, "t_seconds": f / FPS, "track_id": 1,
                     "x_px": 0.5, "y_px": 0.25,
                     "bbox_x1": 0.49, "bbox_y1": 0.24, "bbox_x2": 0.51, "bbox_y2": 0.26,
                     "class": "player", "team": "own", "conf": 0.9})
        rows.append({"frame": f, "t_seconds": f / FPS, "track_id": 2,
                     "x_px": 0.5, "y_px": 0.30,
                     "bbox_x1": 0.49, "bbox_y1": 0.29, "bbox_x2": 0.51, "bbox_y2": 0.31,
                     "class": "goalkeeper", "team": "own", "conf": 0.9})
        rows.append({"frame": f, "t_seconds": f / FPS, "track_id": -1 - f,
                     "x_px": 0.5, "y_px": 0.27,
                     "bbox_x1": 0.49, "bbox_y1": 0.26, "bbox_x2": 0.51, "bbox_y2": 0.28,
                     "class": "ball", "team": "unknown", "conf": 0.9})
    return pd.DataFrame(rows).astype({"frame": "int64", "track_id": "int64"})


def test_analyze_video_grounds_and_inverts_to_kit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """own_kit set -> the hygiene grounding decides team; the deliverable matches the
    GROUNDED (here inverted) labels, not the raw cluster labels. The anti-inversion guarantee."""
    raw = _trajectories_with_gk()  # player+GK raw team 'own'
    backend = _StubBackend(raw, _identity_keypoints(3))
    out_dir = tmp_path / "grounded"

    def fake_run_hygiene(*, traj_path: Path, homographies_path: Path, video_path: Path,
                         out_dir: Path, own_kit: str, **kw: object) -> dict[str, object]:
        df = pd.read_parquet(traj_path).copy()
        df.loc[df["class"] == "player", "team"] = "opp"       # invert outfield
        df.loc[df["class"] == "goalkeeper", "team"] = "opp"   # positional GK -> opp
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(Path(out_dir) / "trajectories_px_clean.parquet", index=False)
        return {"balance": {"ratio": 1.0, "passed": True}, "warning": None}

    monkeypatch.setattr("soccer_vision.hygiene.run.run_hygiene", fake_run_hygiene)
    analyze_video(Path("unused.mp4"), out_dir, backend=backend, own_kit="white")

    deliver = pd.read_parquet(out_dir / "trajectories.parquet")
    assert (deliver.loc[deliver["class"] == "player", "team"] == "opp").all()
    assert (deliver.loc[deliver["class"] == "goalkeeper", "team"] == "opp").all()


def test_analyze_video_ungrounded_warns_and_passes_through(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw = _trajectories_with_gk()
    backend = _StubBackend(raw, _identity_keypoints(3))
    out_dir = tmp_path / "ungrounded"
    with caplog.at_level(logging.WARNING):
        analyze_video(Path("unused.mp4"), out_dir, backend=backend)  # own_kit defaults to None
    deliver = pd.read_parquet(out_dir / "trajectories.parquet")
    assert (deliver.loc[deliver["class"] == "player", "team"] == "own").all()
    assert not (out_dir / "trajectories_px_clean.parquet").exists()
    assert any("UNGROUNDED" in r.message for r in caplog.records)
