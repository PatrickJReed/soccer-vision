"""Tests for RoboflowBackend's ball_weights_path override."""

from __future__ import annotations

from pathlib import Path

import pytest
from soccer_vision.tracking.roboflow import BALL_V1_URL, WEIGHTS, RoboflowBackend


def test_default_ball_weights_path_is_none() -> None:
    backend = RoboflowBackend()
    assert backend.ball_weights_path is None


def test_ball_role_defaults_to_finetuned_release_asset() -> None:
    """The default ball model is the Phase 2 fine-tune, not the roboflow baseline."""
    kind, locator, filename = WEIGHTS["ball"]
    assert kind == "url"
    assert locator == BALL_V1_URL
    assert filename == "ball_yolov8_v1.pt"
    assert BALL_V1_URL.endswith("/ball-v1/ball_yolov8_v1.pt")


def test_custom_ball_weights_path_stored(tmp_path: Path) -> None:
    fake_weights = tmp_path / "ball_v1.pt"
    fake_weights.write_bytes(b"")  # dummy file
    backend = RoboflowBackend(ball_weights_path=fake_weights)
    assert backend.ball_weights_path == fake_weights


def test_nonexistent_ball_weights_path_raises_immediately(tmp_path: Path) -> None:
    bogus = tmp_path / "does-not-exist.pt"
    with pytest.raises(FileNotFoundError):
        RoboflowBackend(ball_weights_path=bogus)
