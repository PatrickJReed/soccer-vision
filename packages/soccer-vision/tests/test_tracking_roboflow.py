"""Tests for the RoboflowBackend adapter.

Protocol conformance is tested unconditionally. The heavy detection path
requires roboflow/sports + ultralytics installed (the 'roboflow' optional
extra); those tests are skipped if the extras aren't available.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from soccer_vision.io.schema import validate_trajectories
from soccer_vision.tracking.base import TrackingBackend
from soccer_vision.tracking.roboflow import RoboflowBackend

try:
    import sports  # type: ignore[import-not-found]  # noqa: F401
    import ultralytics  # type: ignore[import-not-found]  # noqa: F401
    _HEAVY_AVAILABLE = True
except ImportError:
    _HEAVY_AVAILABLE = False


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    """30-frame, 320x240, single-colour video — exercises the codec path only."""
    out = tmp_path / "tiny.mp4"
    fps = 30
    w, h = 320, 240
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    for _ in range(30):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()
    return out


def test_adapter_satisfies_protocol() -> None:
    backend = RoboflowBackend()
    assert isinstance(backend, TrackingBackend)
    assert backend.name == "roboflow-sports"
    assert backend.version


def test_adapter_module_import_does_not_require_heavy_deps() -> None:
    """Importing the adapter module must not fail when ultralytics/sports aren't installed."""
    import soccer_vision.tracking.roboflow as _  # noqa: F401


@pytest.mark.skipif(not _HEAVY_AVAILABLE, reason="roboflow extras not installed")
def test_adapter_returns_schema_conformant_df(tiny_video: Path) -> None:
    """Heavy path: full process() on a blank video; empty df is acceptable."""
    backend = RoboflowBackend()
    df = backend.process(tiny_video)
    validate_trajectories(df)


def test_default_detect_pitch_false() -> None:
    backend = RoboflowBackend()
    assert backend.detect_pitch is False


@pytest.mark.skipif(not _HEAVY_AVAILABLE, reason="roboflow extras not installed")
def test_adapter_with_pitch_returns_keypoints(tiny_video: Path) -> None:
    """When detect_pitch=True, process_with_pitch() returns (df, keypoints_df)."""
    from soccer_vision.io.schema import validate_trajectories
    backend = RoboflowBackend(detect_pitch=True)
    df, kp_df = backend.process_with_pitch(tiny_video)
    validate_trajectories(df)
    assert {"frame", "kp_idx", "x_px", "y_px", "conf"}.issubset(kp_df.columns)


def test_pitch_weights_use_release_url() -> None:
    from soccer_vision.tracking.roboflow import PITCH_V1_URL, WEIGHTS

    kind, locator, filename = WEIGHTS["pitch"]
    assert kind == "url"
    assert locator == PITCH_V1_URL
    assert filename == "pitch_yolov8_v1.pt"
    assert PITCH_V1_URL.endswith("pitch_yolov8_v1.pt")
    assert "releases/download/pitch-v1/" in PITCH_V1_URL


def test_pitch_weights_path_override_missing_raises(tmp_path) -> None:
    from soccer_vision.tracking.roboflow import RoboflowBackend

    missing = tmp_path / "nope.pt"
    try:
        RoboflowBackend(pitch_weights_path=missing)
    except FileNotFoundError as e:
        assert "pitch_weights_path" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_pitch_weights_path_override_accepted(tmp_path) -> None:
    from soccer_vision.tracking.roboflow import RoboflowBackend

    w = tmp_path / "custom_pitch.pt"
    w.write_bytes(b"stub")
    backend = RoboflowBackend(pitch_weights_path=w, detect_pitch=True)
    assert backend.pitch_weights_path == w
