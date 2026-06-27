"""TrackingBackend Protocol: the contract every detection backend must satisfy."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class TrackingBackend(Protocol):
    """A detection + tracking pipeline that consumes a video and returns trajectories.

    Implementations wrap an upstream repo (roboflow/sports, abdullahtarek, etc.)
    or a local model. The returned DataFrame must validate against
    soccer_vision.io.schema.validate_trajectories.
    """

    @property
    def name(self) -> str:
        """Stable identifier for the backend, e.g. 'roboflow-sports'.

        Declared read-only so implementers may expose it as a Final/class
        attribute (RoboflowBackend), an instance attribute (MockBackend), or a
        property; all satisfy this member.
        """
        ...

    @property
    def version(self) -> str:
        """Version string, e.g. the upstream commit SHA or release tag."""
        ...

    def process(self, video_path: Path) -> pd.DataFrame:
        """Run detection + tracking on the video at `video_path`.

        Returns a DataFrame with columns defined in
        soccer_vision.io.schema.REQUIRED_COLUMNS.
        """
        ...

    def process_with_pitch(self, video_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run detection + tracking AND pitch-keypoint detection in ONE video pass.

        Returns (trajectories, keypoints). trajectories validates against
        soccer_vision.io.schema.validate_trajectories; keypoints has columns
        frame, kp_idx, x_px, y_px, conf and may be empty but must be schema-conformant.
        """
        ...
