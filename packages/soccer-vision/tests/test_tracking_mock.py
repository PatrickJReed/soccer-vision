"""Tests for the MockBackend — exists to prove the Protocol allows swapping."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from soccer_vision.io.schema import validate_trajectories
from soccer_vision.tracking.base import TrackingBackend
from soccer_vision.tracking.mock import MockBackend


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    out = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))  # type: ignore[attr-defined]
    for _ in range(10):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    return out


def test_mock_satisfies_protocol() -> None:
    backend: TrackingBackend = MockBackend()
    assert backend.name == "mock"
    assert backend.version == "0.1.0"


def test_mock_returns_schema_conformant_df(tiny_video: Path) -> None:
    backend = MockBackend()
    df = backend.process(tiny_video)
    validate_trajectories(df)


def test_mock_emits_two_teams_with_eight_players_each(tiny_video: Path) -> None:
    backend = MockBackend()
    df = backend.process(tiny_video)
    frame0 = df[df["frame"] == 0]
    assert (frame0["team"] == "own").sum() == 8
    assert (frame0["team"] == "opp").sum() == 8


def test_mock_process_with_pitch_returns_schema_conformant_pair(tiny_video: Path) -> None:
    backend = MockBackend()
    df, kp_df = backend.process_with_pitch(tiny_video)
    validate_trajectories(df)
    assert list(kp_df.columns) == ["frame", "kp_idx", "x_px", "y_px", "conf"]
    assert len(kp_df) == 0  # mock emits no keypoints, but the frame is schema-conformant


def test_mock_satisfies_extended_protocol() -> None:
    backend = MockBackend()
    assert isinstance(backend, TrackingBackend)  # now requires process_with_pitch too
    assert hasattr(backend, "process_with_pitch")
