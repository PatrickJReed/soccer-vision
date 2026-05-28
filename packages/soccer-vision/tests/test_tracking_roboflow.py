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
