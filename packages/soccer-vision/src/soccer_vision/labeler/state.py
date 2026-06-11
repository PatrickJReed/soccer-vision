"""LabelerState: hold the registration chain + clicks, recompute per-frame
homographies on demand, and export the keypoints/homographies parquets.

Separated from the HTTP server so it is testable without a socket or a video.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.labeler.chain import denormalize_homography
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
from soccer_vision.pitch.propagation import HomographyEntry


class LabelerState:
    """Mutable session: clicks in, per-frame homographies + coverage out."""

    def __init__(
        self,
        interframe: Mapping[int, NDArray[np.floating]],
        n_frames: int,
        *,
        size: tuple[int, int],
        window: int = 360,
        residual_threshold: float = 0.05,
    ) -> None:
        self.n_frames = n_frames
        self.size = size
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

    def add_clicks(self, clicks: Sequence[Click]) -> None:
        """Bulk-add clicks with a single recompute (used by --resume)."""
        self.clicks.extend(clicks)
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
        w, h = self.size
        px_clicks = [Click(c.frame, c.kp_idx, c.x * w, c.y * h) for c in self.clicks]
        clicks_to_keypoints_df(px_clicks).to_parquet(
            out / "keypoints.parquet", index=False
        )
        entries = to_homography_entries(
            self._fits, residual_threshold=self.residual_threshold
        )
        px_entries = {
            f: HomographyEntry(
                denormalize_homography(e.H, self.size), e.source, e.confidence
            )
            for f, e in entries.items()
        }
        homographies_to_parquet(px_entries, out / "homographies.parquet")


def clicks_from_keypoints_parquet(path: Path, size: tuple[int, int]) -> list[Click]:
    """Load an exported keypoints.parquet (full-pixel) back into normalized Clicks.

    Inverse of LabelerState.export's keypoints write: x_px/y_px divide by the
    video (width, height) to recover the normalized [0,1] click coordinates.
    """
    df = pd.read_parquet(path)
    w, h = size
    return [
        Click(frame=int(f), kp_idx=int(k), x=float(x) / w, y=float(y) / h)
        for f, k, x, y in zip(
            df["frame"].to_numpy(),
            df["kp_idx"].to_numpy(),
            df["x_px"].to_numpy(),
            df["y_px"].to_numpy(),
            strict=True,
        )
    ]
