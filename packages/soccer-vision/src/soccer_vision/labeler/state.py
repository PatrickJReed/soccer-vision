"""LabelerState: hold the registration chain + clicks, recompute per-frame
homographies on demand, and export the keypoints/homographies parquets.

Separated from the HTTP server so it is testable without a socket or a video.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.calib.calibrate import CalibError
from soccer_vision.labeler.chain import denormalize_homography
from soccer_vision.pipeline import homographies_to_parquet
from soccer_vision.pitch.calib_anchor import (
    FramePose,
    calibrate_clicked_frames,
    flag_outlier_clicks,
    frame_homography,
    poses_by_click_propagation,
)
from soccer_vision.pitch.manual_anchor import (
    Click,
    build_segments,
    clicks_to_keypoints_df,
    cumulative_transforms,
)
from soccer_vision.pitch.propagation import HomographyEntry


@dataclass(frozen=True, eq=False)
class CalibFrame:
    """A calibrated per-frame result in the labeler's normalized space."""

    H: NDArray[np.float64]  # NORMALIZED image -> pitch[0,1] (frontend overlay)
    residual: float         # inlier reprojection RMS (px)
    n_points: int
    fold_count: int


class LabelerState:
    """Mutable session: clicks in, per-frame homographies + coverage out."""

    def __init__(
        self,
        interframe: Mapping[int, NDArray[np.floating]],
        n_frames: int,
        *,
        size: tuple[int, int],
        window: int = 360,
        # green threshold on the per-frame reprojection RMS (px). Measured on the
        # window-PROPAGATED clicks, so chain drift inflates it well above the true
        # ~7 ft pose accuracy; ~60 px ≈ "close to an anchor / trustworthy" vs
        # "re-anchor here" (62% green on the full game). Tunable per session.
        residual_px_threshold: float = 60.0,
        outlier_px: float = 40.0,
        autosave_path: Path | None = None,
    ) -> None:
        self.n_frames = n_frames
        self.size = size
        self.window = window
        self.residual_px_threshold = residual_px_threshold
        self.outlier_px = outlier_px
        self.autosave_path = autosave_path
        self._segment_of = build_segments(interframe, n_frames)
        self._transforms = cumulative_transforms(interframe, self._segment_of)
        self.clicks: list[Click] = []
        self._fits: dict[int, CalibFrame] = {}
        self._K: NDArray[np.float64] | None = None
        self._calibrated = False
        self._outliers: dict[int, list[int]] = {}
        w, h = size
        self._to_norm = np.diag([float(w), float(h), 1.0])  # H_norm = H_px @ this

    def _active_clicks(self) -> list[Click]:
        """Clicks with the flagged outliers removed (used for fitting)."""
        if not self._outliers:
            return self.clicks
        return [c for c in self.clicks if c.kp_idx not in self._outliers.get(c.frame, [])]

    def _try_bootstrap(self) -> bool:
        """Estimate + freeze the shared focal once >=3 anchor frames each have >=6
        landmarks (calibrate_camera's min), and flag outlier clicks. Returns True if
        calibrated. The focal is physically constant (fixed Trace lens), so once frozen
        it stays valid; `recalibrate()` re-estimates K + outliers on demand."""
        if self._calibrated:
            return True
        try:
            k, _poses = calibrate_clicked_frames(self.clicks, self.size, min_points=6)
        except CalibError:
            return False
        self._K = k
        _clean, self._outliers = flag_outlier_clicks(
            self.clicks, k, self.size, thr=self.outlier_px)
        self._calibrated = True
        return True

    def _calib_frame(self, pose: FramePose) -> CalibFrame:
        assert self._K is not None
        h_px = frame_homography(self._K, pose.rvec, pose.tvec)  # full-pixel image->pitch
        h_norm = np.asarray(h_px @ self._to_norm, dtype=np.float64)  # normalized
        return CalibFrame(H=h_norm, residual=pose.residual_px,
                          n_points=pose.n_points, fold_count=pose.fold_count)

    def _refit(self, frames: list[int]) -> None:
        if not self._calibrated or self._K is None:
            for f in frames:
                self._fits.pop(f, None)
            return
        sub = poses_by_click_propagation(
            self._active_clicks(), self._transforms, self._segment_of, self._K,
            self.size, window=self.window, frames=frames)
        for f in frames:
            if f in sub:
                self._fits[f] = self._calib_frame(sub[f])
            else:
                self._fits.pop(f, None)

    def _recompute_all(self, chunk: int = 5000) -> None:
        self._fits = {}
        if not self._calibrated or self._K is None:
            return
        allf = sorted(self._transforms)
        active = self._active_clicks()
        for i in range(0, len(allf), chunk):
            part = allf[i:i + chunk]
            sub = poses_by_click_propagation(
                active, self._transforms, self._segment_of, self._K, self.size,
                window=self.window, frames=part)
            for f, pose in sub.items():
                self._fits[f] = self._calib_frame(pose)

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

    def _affected(self, frame: int) -> list[int]:
        seg = self._segment_of.get(frame)
        lo = max(0, frame - self.window)
        hi = min(self.n_frames - 1, frame + self.window)
        return [f for f in range(lo, hi + 1) if self._segment_of.get(f) == seg]

    def add_click(self, frame: int, kp_idx: int, x: float, y: float) -> None:
        self.clicks.append(Click(frame=frame, kp_idx=kp_idx, x=x, y=y))
        if not self._calibrated and self._try_bootstrap():
            self._recompute_all()
        elif self._calibrated:
            self._refit(self._affected(frame))
        self._autosave()

    def add_clicks(self, clicks: Sequence[Click], *, chunk: int = 5000) -> None:
        self.clicks.extend(clicks)
        self._try_bootstrap()
        self._recompute_all(chunk=chunk)
        self._autosave()

    def remove_last(self) -> None:
        # The frozen focal stays valid after an undo (constant lens); it is NOT
        # re-estimated here even if the undo drops below the bootstrap count.
        # `recalibrate()` refreshes K + outlier flags when the user wants.
        if self.clicks:
            removed = self.clicks.pop()
            if self._calibrated:
                self._refit(self._affected(removed.frame))
            self._autosave()

    def nudge_click(self, frame: int, kp_idx: int, x: float, y: float) -> bool:
        for i in range(len(self.clicks) - 1, -1, -1):
            c = self.clicks[i]
            if c.frame == frame and c.kp_idx == kp_idx:
                self.clicks[i] = Click(frame=frame, kp_idx=kp_idx, x=x, y=y)
                if self._calibrated:
                    self._refit(self._affected(frame))
                self._autosave()
                return True
        return False

    def recalibrate(self) -> bool:
        self._calibrated = False
        self._K = None
        self._outliers = {}
        if not self._try_bootstrap():
            return False
        self._recompute_all()
        return True

    def _status_of(self, f: int) -> str:
        cf = self._fits.get(f)
        if cf is None:
            return "red"
        return "green" if cf.residual <= self.residual_px_threshold else "yellow"

    def coverage(self) -> float:
        if self.n_frames == 0:
            return 0.0
        green = sum(1 for f in range(self.n_frames) if self._status_of(f) == "green")
        return green / self.n_frames

    def status_list(self) -> list[str]:
        return [self._status_of(f) for f in range(self.n_frames)]

    def status_buckets(self, *, n_buckets: int = 1200) -> tuple[list[str], int]:
        full = self.status_list()
        if len(full) <= n_buckets:
            return full, 1
        bucket = -(-len(full) // n_buckets)
        out: list[str] = []
        for i in range(0, len(full), bucket):
            chunk = full[i:i + bucket]
            out.append("red" if "red" in chunk else "yellow" if "yellow" in chunk else "green")
        return out, bucket

    def frame_homography(self, frame: int) -> CalibFrame | None:
        return self._fits.get(frame)

    def export(self, out_dir: Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        w, h = self.size
        px_clicks = [Click(c.frame, c.kp_idx, c.x * w, c.y * h) for c in self.clicks]
        clicks_to_keypoints_df(px_clicks).to_parquet(out / "keypoints.parquet", index=False)
        entries: dict[int, HomographyEntry] = {}
        for f in range(self.n_frames):
            if self._status_of(f) != "green":
                continue
            cf = self._fits[f]
            conf = float(np.clip(1.0 - cf.residual / self.residual_px_threshold, 0.0, 1.0))
            entries[f] = HomographyEntry(
                denormalize_homography(cf.H, self.size), "manual", conf)
        homographies_to_parquet(entries, out / "homographies.parquet")


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
