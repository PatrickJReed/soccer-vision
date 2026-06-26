"""LabelerState: hold the registration chain + clicks, recompute per-frame
homographies on demand, and export the keypoints/homographies parquets.

Separated from the HTTP server so it is testable without a socket or a video.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.labeler.chain import denormalize_homography
from soccer_vision.labeler.refit_worker import RefitWorker
from soccer_vision.pipeline import homographies_to_parquet
from soccer_vision.pitch.global_calib import (
    fold_of_norm,
    frame_status,
    solve_global,
    two_ended_segments,
)
from soccer_vision.pitch.manual_anchor import (
    Click,
    LineClick,
    build_segments,
    clicks_to_keypoints_df,
    cumulative_transforms,
    propagate_line_clicks,
)
from soccer_vision.pitch.propagation import HomographyEntry


@dataclass(frozen=True, eq=False)
class CalibFrame:
    """A calibrated per-frame result in the labeler's normalized space."""

    H: NDArray[np.float64]  # NORMALIZED image -> pitch[0,1] (frontend overlay)
    residual: float         # in-sample global-fit RMS (norm px) — diagnostic only
    n_points: int           # clicks in this frame's segment
    fold_count: int
    two_ended: bool         # the segment saw both field ends (drives green vs yellow)


class LabelerState:
    """Mutable session: clicks in, per-frame homographies + coverage out."""

    def __init__(
        self,
        interframe: Mapping[int, NDArray[np.floating]],
        n_frames: int,
        *,
        size: tuple[int, int],
        window: int = 360,
        line_band: int = 60,
        seed_size: int = 6,
        gate_px: float = 60.0,
        gap_dist: int = 180,
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
        self.line_band = line_band
        self.seed_size = seed_size
        self.gate_px = gate_px
        self.gap_dist = gap_dist
        self._lock = threading.RLock()
        self._refit_chunk = 256
        self.residual_px_threshold = residual_px_threshold
        self.outlier_px = outlier_px
        self.autosave_path = autosave_path
        self._segment_of = build_segments(interframe, n_frames)
        self._transforms = cumulative_transforms(interframe, self._segment_of)
        self.clicks: list[Click] = []
        self.line_clicks: list[LineClick] = []
        self._seq: list[str] = []  # insertion order across clicks ("pt") + line_clicks ("ln")
        self._fits: dict[int, CalibFrame] = {}
        self._K: NDArray[np.float64] | None = None
        self._calibrated = False
        self._outliers: dict[int, list[int]] = {}
        w, h = size
        self._to_norm = np.diag([float(w), float(h), 1.0])  # H_norm = H_px @ this
        self._worker: RefitWorker[CalibFrame | None] = RefitWorker(
            self._compute_dirty, self._apply_fits)
        self._worker.start()

    def _active_clicks(self) -> list[Click]:
        """Clicks with the flagged outliers removed (used for fitting)."""
        if not self._outliers:
            return self.clicks
        return [c for c in self.clicks if c.kp_idx not in self._outliers.get(c.frame, [])]

    def _try_bootstrap(self) -> bool:
        """Calibrated once any segment has >= 4 clicks (a homography is fittable).
        No focal/K: the global model is a plain image->pitch homography."""
        if self._calibrated:
            return True
        with self._lock:
            clicks = list(self._active_clicks())
        gc = solve_global(clicks, self._transforms, self._segment_of, self.size)
        if not gc.h_by_segment:
            return False
        with self._lock:
            self._calibrated = True
        return True

    def _line_obs(self, frames: Sequence[int] | None) -> dict[int, list[tuple[str, float, float]]]:
        w, h = self.size
        prop = propagate_line_clicks(
            self.line_clicks, self._transforms, self._segment_of,
            window=self.line_band, frames=frames)
        return {f: [(lid, x * w, y * h) for (lid, x, y) in lst] for f, lst in prop.items()}

    def _compute_poses(self, frames: Sequence[int]) -> dict[int, CalibFrame]:
        """Solve the global homography (per segment) from ALL clicks, then emit a
        CalibFrame for each requested frame. Snapshots inputs under the lock and solves
        off the lock. The global solve is cheap (one DLT per segment), so recomputing it
        per chunk is acceptable."""
        with self._lock:
            if not self._calibrated:
                return {}
            clicks = list(self._active_clicks())  # stable COPY for the lock-free solve
        gc = solve_global(clicks, self._transforms, self._segment_of, self.size)
        two_ended = two_ended_segments(clicks, self._segment_of)
        out: dict[int, CalibFrame] = {}
        for f in frames:
            h_norm = gc.frame_homography(f)
            if h_norm is None:
                continue
            seg = self._segment_of.get(f)
            out[f] = CalibFrame(
                H=h_norm,
                residual=gc.rms_by_segment.get(seg, float("nan"))
                if seg is not None else float("nan"),
                n_points=gc.n_by_segment.get(seg, 0) if seg is not None else 0,
                fold_count=fold_of_norm(h_norm, self.size),
                two_ended=(seg in two_ended),
            )
        return out

    def _compute_dirty(
        self, frames: Sequence[int], is_cancelled: Callable[[], bool]
    ) -> dict[int, CalibFrame | None] | None:
        """Compute `frames` in chunks (checking cancellation between chunks). Returns a
        map over EVERY requested frame -> CalibFrame-or-None (None = no longer solvable,
        so the applier pops any stale fit). None return = the whole pass was cancelled."""
        out: dict[int, CalibFrame | None] = {}
        ordered = list(frames)
        for i in range(0, len(ordered), self._refit_chunk):
            if is_cancelled():
                return None
            chunk = ordered[i:i + self._refit_chunk]
            solved = self._compute_poses(chunk)
            for f in chunk:
                out[f] = solved.get(f)
        return out

    def _apply_fits(self, results: dict[int, CalibFrame | None]) -> None:
        """Merge computed fits into _fits under the lock: set solved frames, pop the rest."""
        with self._lock:
            for f, cf in results.items():
                if cf is None:
                    self._fits.pop(f, None)
                else:
                    self._fits[f] = cf

    def _refit_one(self, frame: int) -> None:
        """Synchronously fit just `frame` (instant overlay for the clicked frame)."""
        result = self._compute_dirty([frame], lambda: False)
        assert result is not None
        self._apply_fits(result)

    def wait_idle(self, timeout: float | None = None) -> None:
        self._worker.wait_idle(timeout)

    def pending(self) -> int:
        return self._worker.pending()

    def stop_worker(self) -> None:
        self._worker.stop()

    def _recompute_all(self) -> None:  # chunking is governed by self._refit_chunk
        with self._lock:
            self._fits = {}
        if not self._calibrated:
            return
        result = self._compute_dirty(sorted(self._transforms), lambda: False)
        assert result is not None
        self._apply_fits(result)

    def _autosave(self) -> None:
        """Atomically persist normalized clicks + line clicks to the sidecar."""
        if self.autosave_path is None:
            return
        self.autosave_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "clicks": [{"frame": c.frame, "kp_idx": c.kp_idx, "x": c.x, "y": c.y}
                       for c in self.clicks],
            "line_clicks": [{"frame": lc.frame, "line_id": lc.line_id, "x": lc.x, "y": lc.y}
                            for lc in self.line_clicks],
        }
        tmp = self.autosave_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.autosave_path)

    def _affected(self, frame: int) -> list[int]:
        """A click changes the global homography for its ENTIRE segment, so every frame
        in that segment must be recomputed (not just a window)."""
        seg = self._segment_of.get(frame)
        return [f for f in range(self.n_frames) if self._segment_of.get(f) == seg]

    def add_click(self, frame: int, kp_idx: int, x: float, y: float) -> None:
        with self._lock:
            self.clicks.append(Click(frame=frame, kp_idx=kp_idx, x=x, y=y))
            self._seq.append("pt")
        if not self._calibrated and self._try_bootstrap():
            self._refit_one(frame)                       # clicked frame instant
            self._worker.mark_dirty(range(self.n_frames))  # everything else in background
        elif self._calibrated:
            self._refit_one(frame)
            self._worker.mark_dirty(self._affected(frame))
        self._autosave()

    def add_clicks(self, clicks: Sequence[Click]) -> None:
        # Bulk boot/resume path: called single-threaded before the server accepts
        # requests, and finishes with a synchronous _recompute_all (no live worker
        # pass concurrently iterating the lists), so the unlocked extend is safe.
        self.clicks.extend(clicks)
        self._seq.extend("pt" for _ in clicks)
        self._try_bootstrap()
        self._recompute_all()
        self._autosave()

    def add_line_click(self, frame: int, line_id: str, x: float, y: float) -> None:
        with self._lock:
            self.line_clicks.append(LineClick(frame=frame, line_id=line_id, x=x, y=y))
            self._seq.append("ln")
        if self._calibrated:
            self._refit_one(frame)
            self._worker.mark_dirty(self._affected(frame))
        self._autosave()

    def add_line_clicks(self, line_clicks: Sequence[LineClick]) -> None:
        # Bulk boot/resume path (see add_clicks): single-threaded, synchronous.
        self.line_clicks.extend(line_clicks)
        self._seq.extend("ln" for _ in line_clicks)
        self._recompute_all()
        self._autosave()

    def remove_last(self) -> None:
        # The frozen focal stays valid after an undo (constant lens); it is NOT
        # re-estimated here even if the undo drops below the bootstrap count.
        # `recalibrate()` refreshes K + outlier flags when the user wants.
        with self._lock:  # pop _seq + the matching list atomically (worker may be reading)
            if not self._seq:
                return
            kind = self._seq.pop()
            # `_seq` is kept in lockstep with the two lists, so the matching list is
            # always non-empty here; assert the invariant rather than silently popping
            # the wrong kind.
            if kind == "ln":
                assert self.line_clicks, "_seq/line_clicks out of sync"
                removed_frame = self.line_clicks.pop().frame
            else:
                assert self.clicks, "_seq/clicks out of sync"
                removed_frame = self.clicks.pop().frame
        if self._calibrated:
            self._refit_one(removed_frame)
            self._worker.mark_dirty(self._affected(removed_frame))
        self._autosave()

    def nudge_click(self, frame: int, kp_idx: int, x: float, y: float) -> bool:
        with self._lock:  # scan + replace atomically (worker may be reading self.clicks)
            found = False
            for i in range(len(self.clicks) - 1, -1, -1):
                c = self.clicks[i]
                if c.frame == frame and c.kp_idx == kp_idx:
                    self.clicks[i] = Click(frame=frame, kp_idx=kp_idx, x=x, y=y)
                    found = True
                    break
        if not found:
            return False
        if self._calibrated:
            self._refit_one(frame)
            self._worker.mark_dirty(self._affected(frame))
        self._autosave()
        return True

    def recalibrate(self) -> bool:
        with self._lock:  # reset calibration state atomically vs the worker's reads
            self._calibrated = False
            self._K = None
            self._outliers = {}
        if not self._try_bootstrap():
            with self._lock:
                self._fits = {}
            return False
        with self._lock:
            self._fits = {}
        self._worker.mark_dirty(range(self.n_frames))
        return True

    def _status_of(self, f: int) -> str:
        with self._lock:
            cf = self._fits.get(f)
        if cf is None:
            return "red"
        return frame_status(cf.H, self.size, segment_two_ended=cf.two_ended)

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
        with self._lock:
            return self._fits.get(frame)

    def export(self, out_dir: Path) -> None:
        self.wait_idle(timeout=30)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        w, h = self.size
        px_clicks = [Click(c.frame, c.kp_idx, c.x * w, c.y * h) for c in self.clicks]
        clicks_to_keypoints_df(px_clicks).to_parquet(out / "keypoints.parquet", index=False)
        entries: dict[int, HomographyEntry] = {}
        for f in range(self.n_frames):
            with self._lock:  # single locked read: no green-then-missing TOCTOU window
                cf = self._fits.get(f)
            if cf is None or cf.residual > self.residual_px_threshold:  # not green
                continue
            conf = float(np.clip(1.0 - cf.residual / self.residual_px_threshold, 0.0, 1.0))
            entries[f] = HomographyEntry(
                denormalize_homography(cf.H, self.size), "manual", conf)
        homographies_to_parquet(entries, out / "homographies.parquet")
        if self.line_clicks:
            pd.DataFrame(
                [{"frame": lc.frame, "line_id": lc.line_id, "x_px": lc.x * w, "y_px": lc.y * h}
                 for lc in self.line_clicks],
                columns=["frame", "line_id", "x_px", "y_px"],
            ).to_parquet(out / "line_clicks.parquet", index=False)


def clicks_from_sidecar(path: Path) -> list[Click]:
    """Load the autosave sidecar's POINT clicks (handles the old bare-list format)."""
    data = json.loads(Path(path).read_text())
    rows = data["clicks"] if isinstance(data, dict) else data
    return [Click(frame=int(d["frame"]), kp_idx=int(d["kp_idx"]),
                  x=float(d["x"]), y=float(d["y"])) for d in rows]


def line_clicks_from_sidecar(path: Path) -> list[LineClick]:
    """Load the autosave sidecar's LINE clicks ([] for the old bare-list format)."""
    data = json.loads(Path(path).read_text())
    rows = data.get("line_clicks", []) if isinstance(data, dict) else []
    return [LineClick(frame=int(d["frame"]), line_id=str(d["line_id"]),
                      x=float(d["x"]), y=float(d["y"])) for d in rows]


def line_clicks_from_parquet(path: Path, size: tuple[int, int]) -> list[LineClick]:
    """Load an exported line_clicks.parquet (full-pixel) back into normalized LineClicks."""
    df = pd.read_parquet(path)
    w, h = size
    return [LineClick(frame=int(f), line_id=str(lid), x=float(x) / w, y=float(y) / h)
            for f, lid, x, y in zip(df["frame"].to_numpy(), df["line_id"].to_numpy(),
                                    df["x_px"].to_numpy(), df["y_px"].to_numpy(), strict=True)]


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
