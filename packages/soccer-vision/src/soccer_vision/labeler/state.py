"""LabelerState: hold the registration chain + clicks, recompute per-frame
homographies on demand, and export the keypoints/homographies parquets.

Separated from the HTTP server so it is testable without a socket or a video.
"""

from __future__ import annotations

import json
import os
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
        autosave_path: Path | None = None,
    ) -> None:
        self.n_frames = n_frames
        self.size = size
        self.window = window
        self.residual_threshold = residual_threshold
        self.autosave_path = autosave_path
        self._segment_of = build_segments(interframe, n_frames)
        self._transforms = cumulative_transforms(interframe, self._segment_of)
        self.clicks: list[Click] = []
        self._fits: dict[int, FrameFit] = {}

    def _refit(self, frames: list[int]) -> None:
        """Refit exactly `frames`, replacing/removing their cached fits."""
        sub = fit_frame_homographies(
            self.clicks, self._transforms, self._segment_of,
            PITCH_LANDMARKS, window=self.window, frames=frames,
        )
        for f in frames:
            if f in sub:
                self._fits[f] = sub[f]
            else:
                self._fits.pop(f, None)

    def _affected(self, frame: int) -> list[int]:
        """Frames whose fit can change when a click at `frame` mutates."""
        seg = self._segment_of.get(frame)
        lo = max(0, frame - self.window)
        hi = min(self.n_frames - 1, frame + self.window)
        return [f for f in range(lo, hi + 1) if self._segment_of.get(f) == seg]

    def _recompute_chunked(self, chunk: int = 5000) -> None:
        self._fits = {}
        all_frames = sorted(self._transforms)
        for i in range(0, len(all_frames), chunk):
            part = all_frames[i:i + chunk]
            self._fits.update(fit_frame_homographies(
                self.clicks, self._transforms, self._segment_of,
                PITCH_LANDMARKS, window=self.window, frames=part,
            ))

    def _autosave(self) -> None:
        """Atomically persist normalized clicks to the sidecar (if configured)."""
        if self.autosave_path is None:
            return
        self.autosave_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"frame": c.frame, "kp_idx": c.kp_idx, "x": c.x, "y": c.y}
            for c in self.clicks
        ]
        tmp = self.autosave_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.autosave_path)

    def add_click(self, frame: int, kp_idx: int, x: float, y: float) -> None:
        self.clicks.append(Click(frame=frame, kp_idx=kp_idx, x=x, y=y))
        self._refit(self._affected(frame))
        self._autosave()

    def add_clicks(self, clicks: Sequence[Click], *, chunk: int = 5000) -> None:
        """Bulk-add (resume/sidecar load) with one chunked full recompute."""
        self.clicks.extend(clicks)
        self._recompute_chunked(chunk=chunk)
        self._autosave()

    def remove_last(self) -> None:
        if self.clicks:
            removed = self.clicks.pop()
            self._refit(self._affected(removed.frame))
            self._autosave()

    def nudge_click(self, frame: int, kp_idx: int, x: float, y: float) -> bool:
        """Move the MOST RECENT click matching (frame, kp_idx). False if none."""
        for i in range(len(self.clicks) - 1, -1, -1):
            c = self.clicks[i]
            if c.frame == frame and c.kp_idx == kp_idx:
                self.clicks[i] = Click(frame=frame, kp_idx=kp_idx, x=x, y=y)
                self._refit(self._affected(frame))
                self._autosave()
                return True
        return False

    def coverage(self) -> float:
        return coverage_fraction(
            self._fits, self.n_frames, residual_threshold=self.residual_threshold
        )

    def status_list(self) -> list[str]:
        status = frame_status(
            self._fits, self.n_frames, residual_threshold=self.residual_threshold
        )
        return [status[f] for f in range(self.n_frames)]

    def status_buckets(self, *, n_buckets: int = 1200) -> tuple[list[str], int]:
        """Downsampled timeline: worst status per bucket (red > yellow > green)."""
        full = self.status_list()
        if len(full) <= n_buckets:
            return full, 1
        bucket_size = -(-len(full) // n_buckets)  # ceil division
        out: list[str] = []
        for i in range(0, len(full), bucket_size):
            chunk = full[i:i + bucket_size]
            if "red" in chunk:
                out.append("red")
            elif "yellow" in chunk:
                out.append("yellow")
            else:
                out.append("green")
        return out, bucket_size

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


def clicks_from_sidecar(path: Path) -> list[Click]:
    """Load the autosave sidecar (normalized coords) back into Clicks."""
    data = json.loads(Path(path).read_text())
    return [
        Click(frame=int(d["frame"]), kp_idx=int(d["kp_idx"]),
              x=float(d["x"]), y=float(d["y"]))
        for d in data
    ]


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
