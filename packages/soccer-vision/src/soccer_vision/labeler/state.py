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
from soccer_vision.pitch.manual_anchor import (
    Click,
    LineClick,
    build_segments,
    clicks_to_keypoints_df,
    cumulative_transforms,
    propagate_line_clicks,
)
from soccer_vision.pitch.physical_calib import (
    DEFAULT_GAP_GUARD,
    PhysicalCalib,
    solve_session,
)
from soccer_vision.pitch.propagation import HomographyEntry


@dataclass(frozen=True, eq=False)
class CalibFrame:
    """A calibrated per-frame result in the labeler's normalized space."""

    H: NDArray[np.float64]  # NORMALIZED image -> pitch[0,1] (frontend overlay)
    status: str             # "green" | "yellow" | "red" (physical per-frame status)
    is_anchor: bool         # True if this frame was directly clicked and solved as a pose
    residual: float | None  # diagnostic only (unused by the physical gate); None = n/a
    n_points: int           # point clicks on this anchor frame (0 for propagated frames)


class LabelerState:
    """Mutable session: clicks in, per-frame homographies + coverage out."""

    def __init__(
        self,
        interframe: Mapping[int, NDArray[np.floating]],
        n_frames: int,
        *,
        size: tuple[int, int],
        line_band: int = 60,
        # Diagnostic-only display threshold served to the frontend for colouring the
        # per-frame residual readout. The physical engine does not use an in-sample residual
        # gate (export is gated on whole-field GREEN status), so this affects UI colour only.
        residual_px_threshold: float = 60.0,
        outlier_px: float = 40.0,
        autosave_path: Path | None = None,
    ) -> None:
        self.n_frames = n_frames
        self.size = size
        self.line_band = line_band
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
        self._last_calib: PhysicalCalib | None = None  # warm-start seed for the next solve
        self._gap_guard = DEFAULT_GAP_GUARD
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
        """Calibrated once solve_session produces >= 1 physical anchor. The physical model
        needs >= 3 diverse clicked frames (>= 6 points each) to estimate the shared focal;
        below that there is no anchor yet and this stays False (the bootstrap waits)."""
        if self._calibrated:
            return True
        calib = self._solve()
        if not calib.anchor_h:
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

    def _solve(self) -> PhysicalCalib:
        """Run the physical per-frame solve ONCE over a stable snapshot of the active clicks
        (points + lines). Snapshots under the lock, then runs solve_session OFF the lock
        (shared-focal calibrate + per-anchor SQPNP+refine_pose; bracket-propagation is lazy
        in frame_homography). Callers solve once per recompute and reuse the result — the
        solve is expensive. solve_session handles the empty / too-few-views case by returning
        a PhysicalCalib with no anchors, so frame_homography is None for every frame.

        The previous PhysicalCalib (self._last_calib) is passed as a warm-start seed so each
        anchor's refine_pose starts from its prior pose; it only affects iteration count, not
        the optimum, and a stale seed is safe (each anchor re-refines to the current clicks).
        """
        with self._lock:
            clicks = list(self._active_clicks())  # stable COPY for the lock-free solve
            lines = list(self.line_clicks)
            seed = self._last_calib               # warm-start from the prior solution
        calib = solve_session(
            clicks, lines, self.size, self._transforms,
            segment_of=self._segment_of, gap_guard=self._gap_guard, seed=seed)
        with self._lock:
            self._last_calib = calib
        return calib

    def _build_frame(
        self, calib: PhysicalCalib, counts: Mapping[int, int], f: int
    ) -> CalibFrame | None:
        """Build one frame's CalibFrame from an already-solved PhysicalCalib (cheap: an
        anchor lookup or a bracket propagation). None if the frame has no homography
        (uncalibrated / beyond the gap guard). `counts` is frame -> point-click count."""
        h = calib.frame_homography(f)
        if h is None:
            return None
        anchor = calib.is_anchor(f)
        return CalibFrame(
            H=h,
            status=calib.status(f),
            is_anchor=anchor,
            residual=None,
            n_points=counts.get(f, 0) if anchor else 0,
        )

    def _compute_dirty(
        self, frames: Sequence[int], is_cancelled: Callable[[], bool]
    ) -> dict[int, CalibFrame | None] | None:
        """Solve the physical session ONCE, then build each requested frame's CalibFrame
        (cheap). Chunks ONLY to check cancellation between chunks — the solve itself runs
        once, never re-run per chunk. Returns a map over EVERY requested frame ->
        CalibFrame-or-None (None = no longer solvable, so the applier pops any stale fit).
        None return = the whole pass was cancelled."""
        calib = self._solve()
        with self._lock:
            counts: dict[int, int] = {}
            for c in self.clicks:
                counts[c.frame] = counts.get(c.frame, 0) + 1
        out: dict[int, CalibFrame | None] = {}
        ordered = list(frames)
        for i in range(0, len(ordered), self._refit_chunk):
            if is_cancelled():
                return None
            for f in ordered[i:i + self._refit_chunk]:
                out[f] = self._build_frame(calib, counts, f)
        return out

    def _apply_fits(self, results: dict[int, CalibFrame | None]) -> None:
        """Merge computed fits into _fits under the lock: set solved frames, pop the rest."""
        with self._lock:
            for f, cf in results.items():
                if cf is None:
                    self._fits.pop(f, None)
                else:
                    self._fits[f] = cf

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
        """Frames whose homography a LINE edit at `frame` can change: its own registration
        segment (the line refines that segment's anchor poses / propagation). A POINT edit is
        NOT scoped here — it re-estimates the shared focal K over ALL frames, so point paths
        mark every frame dirty instead."""
        seg = self._segment_of.get(frame)
        return [f for f in range(self.n_frames) if self._segment_of.get(f) == seg]

    def add_click(self, frame: int, kp_idx: int, x: float, y: float) -> None:
        # Non-blocking: append the click and mark frames dirty, then return immediately. The
        # background RefitWorker re-solves off the request thread; the clicked frame keeps its
        # cached (pre-click) overlay until the worker drains (~100-300 ms). A point click feeds
        # the SHARED focal K (estimated over all frames), so every frame is marked dirty.
        with self._lock:
            self.clicks.append(Click(frame=frame, kp_idx=kp_idx, x=x, y=y))
            self._seq.append("pt")
        if self._calibrated or self._try_bootstrap():
            self._worker.mark_dirty(range(self.n_frames))
        self._autosave()

    def add_clicks(self, clicks: Sequence[Click]) -> None:
        # Bulk boot/resume path. Lock the extend so it is atomic vs the worker reading
        # self.clicks / self._seq, making this unconditionally safe (not just under the
        # boot-ordering invariant). _recompute_all still runs after.
        with self._lock:
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
            self._worker.mark_dirty(self._affected(frame))  # non-blocking; worker re-solves
        self._autosave()

    def add_line_clicks(self, line_clicks: Sequence[LineClick]) -> None:
        # Bulk boot/resume path (see add_clicks): lock the extend vs the worker.
        with self._lock:
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
            # a removed POINT changes the shared K (all frames); a removed LINE is segment-scoped
            affected = self._affected(removed_frame) if kind == "ln" else range(self.n_frames)
            self._worker.mark_dirty(affected)  # non-blocking
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
            self._worker.mark_dirty(range(self.n_frames))  # point moved -> shared K changes
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
        return cf.status if cf is not None else "red"

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

    def status_summary(
        self, *, n_buckets: int = 1200
    ) -> tuple[float, list[str], int]:
        """Compute the per-frame status list ONCE and derive both coverage and buckets
        from it. _state_payload calls this instead of coverage() + status_buckets(),
        which each walk every frame (two full _status_of passes per /api/state poll)."""
        full = self.status_list()
        coverage = (
            sum(1 for s in full if s == "green") / self.n_frames
            if self.n_frames else 0.0
        )
        if len(full) <= n_buckets:
            return coverage, full, 1
        bucket = -(-len(full) // n_buckets)
        out: list[str] = []
        for i in range(0, len(full), bucket):
            chunk = full[i:i + bucket]
            out.append("red" if "red" in chunk else "yellow" if "yellow" in chunk else "green")
        return coverage, out, bucket

    def frame_homography(self, frame: int) -> CalibFrame | None:
        with self._lock:
            return self._fits.get(frame)

    def export(self, out_dir: Path) -> None:
        # Block until the background worker has fully drained — never write a partial
        # set (no 30s timeout: a partial export of in-flight fits is worse than waiting).
        while self.pending() > 0:
            self.wait_idle()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        w, h = self.size
        px_clicks = [Click(c.frame, c.kp_idx, c.x * w, c.y * h) for c in self.clicks]
        clicks_to_keypoints_df(px_clicks).to_parquet(out / "keypoints.parquet", index=False)
        entries: dict[int, HomographyEntry] = {}
        for f in range(self.n_frames):
            with self._lock:  # single locked read: no green-then-missing TOCTOU window
                cf = self._fits.get(f)
            # Honest gate: only export whole-field GREEN frames (anchor that passed its own
            # foreground self-check + plausible fold), never yellow/red "sky" frames.
            if cf is None or cf.status != "green":
                continue
            # A green frame is whole-field trustworthy -> confidence 1.0. The physical gate
            # (foreground self-check + fold), not an in-sample residual, decides trust.
            entries[f] = HomographyEntry(
                denormalize_homography(cf.H, self.size), "manual", 1.0)
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
