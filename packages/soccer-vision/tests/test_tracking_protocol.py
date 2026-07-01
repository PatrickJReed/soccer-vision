"""Tests for the TrackingBackend protocol contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from soccer_vision.io.schema import validate_trajectories
from soccer_vision.tracking.base import TrackingBackend


class _MockBackend:
    """A minimal backend that satisfies the protocol structurally."""

    name = "mock"
    version = "0.0.0"

    def process(self, video_path: Path) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame": [0],
                "t_seconds": [0.0],
                "track_id": [1],
                "x_px": [100.0],
                "y_px": [100.0],
                "bbox_x1": [90.0],
                "bbox_y1": [90.0],
                "bbox_x2": [110.0],
                "bbox_y2": [110.0],
                "class": ["player"],
                "team": ["own"],
                "conf": [0.9],
            }
        )

    def process_with_pitch(self, video_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.process(video_path), pd.DataFrame(
            columns=["frame", "kp_idx", "x_px", "y_px", "conf"]
        )


def test_mock_backend_is_a_tracking_backend() -> None:
    backend: TrackingBackend = _MockBackend()
    assert backend.name == "mock"
    assert backend.version == "0.0.0"


def test_mock_backend_output_passes_schema() -> None:
    backend = _MockBackend()
    df = backend.process(Path("/dev/null"))
    validate_trajectories(df)  # should not raise


def test_protocol_is_runtime_checkable() -> None:
    """The protocol should be runtime-checkable for adapter conformance."""
    backend = _MockBackend()
    assert isinstance(backend, TrackingBackend)


def test_protocol_requires_process_with_pitch() -> None:
    """A backend lacking process_with_pitch is NOT a TrackingBackend (runtime_checkable)."""

    class _ProcessOnly:
        name = "x"
        version = "0"

        def process(self, video_path: Path) -> pd.DataFrame:  # pragma: no cover - structural
            return pd.DataFrame()

    assert not isinstance(_ProcessOnly(), TrackingBackend)
