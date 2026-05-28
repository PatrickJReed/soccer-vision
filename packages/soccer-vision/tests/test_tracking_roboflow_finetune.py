"""Tests for RoboflowBackend's ball_weights_path override."""

from __future__ import annotations

from pathlib import Path

import pytest
from soccer_vision.tracking.roboflow import RoboflowBackend


def test_default_ball_weights_path_is_none() -> None:
    backend = RoboflowBackend()
    assert backend.ball_weights_path is None


def test_custom_ball_weights_path_stored(tmp_path: Path) -> None:
    fake_weights = tmp_path / "ball_v1.pt"
    fake_weights.write_bytes(b"")  # dummy file
    backend = RoboflowBackend(ball_weights_path=fake_weights)
    assert backend.ball_weights_path == fake_weights


def test_nonexistent_ball_weights_path_raises_immediately(tmp_path: Path) -> None:
    bogus = tmp_path / "does-not-exist.pt"
    with pytest.raises(FileNotFoundError):
        RoboflowBackend(ball_weights_path=bogus)
