"""LabelerState: hold the registration chain + clicks, recompute per-frame
homographies on demand, and export the keypoints/homographies parquets.

Separated from the HTTP server so it is testable without a socket or a video.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from soccer_vision.pipeline import homographies_to_parquet
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import (
    Click,
    FrameFit,
    build_segments,
    clicks_to_keypoints_df,
    coverage_fraction,
    cumulative_transforms,
    fit_frame_homographies,
    frame_status,
    to_homography_entries,
)


class LabelerState:
    """Mutable session: clicks in, per-frame homographies + coverage out."""

    def __init__(
        self,
        interframe: Mapping[int, NDArray[np.floating]],
        n_frames: int,
        *,
        window: int = 60,
        residual_threshold: float = 0.05,
    ) -> None:
        self.n_frames = n_frames
        self.window = window
        self.residual_threshold = residual_threshold
        self._segment_of = build_segments(interframe, n_frames)
        self._transforms = cumulative_transforms(interframe, self._segment_of)
        self.clicks: list[Click] = []
        self._fits: dict[int, FrameFit] = {}

    def _recompute(self) -> None:
        self._fits = fit_frame_homographies(
            self.clicks, self._transforms, self._segment_of,
            PITCH_LANDMARKS, window=self.window,
        )

    def add_click(self, frame: int, kp_idx: int, x: float, y: float) -> None:
        self.clicks.append(Click(frame=frame, kp_idx=kp_idx, x=x, y=y))
        self._recompute()

    def remove_last(self) -> None:
        if self.clicks:
            self.clicks.pop()
            self._recompute()

    def coverage(self) -> float:
        return coverage_fraction(
            self._fits, self.n_frames, residual_threshold=self.residual_threshold
        )

    def status_list(self) -> list[str]:
        status = frame_status(
            self._fits, self.n_frames, residual_threshold=self.residual_threshold
        )
        return [status[f] for f in range(self.n_frames)]

    def frame_homography(self, frame: int) -> FrameFit | None:
        return self._fits.get(frame)

    def export(self, out_dir: Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        clicks_to_keypoints_df(self.clicks).to_parquet(
            out / "keypoints.parquet", index=False
        )
        entries = to_homography_entries(
            self._fits, residual_threshold=self.residual_threshold
        )
        homographies_to_parquet(entries, out / "homographies.parquet")
