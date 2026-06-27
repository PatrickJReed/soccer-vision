"""Pipeline orchestrator: chains pitch + phase modules into enriched outputs.

assemble_phases is pure (no models, no GPU, no ultralytics/sports import) so the
integration logic is testable without a GPU. analyze_video / assemble_from_parquet / assemble_from_homographies add model
invocation and parquet I/O around it.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from soccer_vision.io.schema import validate_trajectories
from soccer_vision.phase.possession import (
    PossessionThresholds,
    classify_possession,
    smooth_possession,
)
from soccer_vision.phase.splitter import label_phase
from soccer_vision.phase.team_mode import apply_modal_team_per_track
from soccer_vision.pitch.filter import filter_outside_pitch
from soccer_vision.pitch.homography import smooth_homographies
from soccer_vision.pitch.landmarks import build_frame_homographies
from soccer_vision.pitch.mapper import PitchMapper
from soccer_vision.pitch.propagation import (
    HomographyEntry,
    compute_interframe_homographies,
    propagate_homographies,
)
from soccer_vision.tracking.base import TrackingBackend

logger = logging.getLogger(__name__)

_H_COLS = [f"h{i}{j}" for i in range(3) for j in range(3)]


@dataclass(frozen=True)
class PipelineResult:
    """Enriched outputs of the pipeline plus coverage diagnostics."""

    trajectories: pd.DataFrame   # per-detection, +x_pitch/+y_pitch, team modal-cleaned
    phases: pd.DataFrame         # per-frame over [0, total_frames); +homography_source/conf
    homography_coverage: float   # fraction of frames with a homography (anchor + propagated)
    ball_coverage: float         # fraction of frames with a non-NaN ball pitch coord
    anchor_coverage: float       # fraction from detected landmarks
    propagated_coverage: float   # fraction filled by propagation


def assemble_phases(
    trajectories_px: pd.DataFrame,
    keypoints: pd.DataFrame,
    fps: float,
    total_frames: int,
    *,
    homographies: dict[int, HomographyEntry] | None = None,
    kp_conf_threshold: float = 0.5,
    homography_alpha: float = 0.5,
    filter_margin: float = 0.05,
    possession_thresholds: PossessionThresholds | None = None,
    transition_seconds: float = 5.0,
    halftime_frame: int | None = None,
) -> PipelineResult:
    """Run the full pitch + phase chain on tracker output. Pure; no I/O.

    When `homographies` (a precomputed {frame: HomographyEntry} from the propagation
    stage) is given, it is used directly; otherwise homographies are computed from
    keypoints (landmark anchors + carry-forward smoothing) exactly as before.
    """
    if homographies is not None:
        h_entries = homographies
        h_map = {f: e.H for f, e in h_entries.items()}
    else:
        raw_h = build_frame_homographies(keypoints, conf_threshold=kp_conf_threshold)
        smoothed = smooth_homographies(raw_h, alpha=homography_alpha)
        # Only landmark frames are provenance-bearing anchors; carry-forward frames are
        # still mapped for pitch coords but excluded from provenance/coverage (preserves
        # the raw-anchor coverage metric).
        h_entries = {f: HomographyEntry(smoothed[f], "anchor", 1.0) for f in raw_h}
        h_map = dict(smoothed)
    if not h_entries:
        logger.warning("No homographies fitted; pitch coords NaN, phases all 'unknown'.")

    enriched = PitchMapper().transform(trajectories_px, h_map)
    enriched = filter_outside_pitch(enriched, margin=filter_margin)
    enriched = apply_modal_team_per_track(enriched)
    validate_trajectories(enriched)

    poss = classify_possession(enriched, possession_thresholds).sort_index()
    window = max(1, round(fps))
    poss_smoothed = smooth_possession(poss, window_frames=window)

    # Highest-confidence ball per frame (multiple low-conf detections are possible at conf=0.05).
    ball = enriched[enriched["class"] == "ball"]
    ball_by_frame = ball.sort_values("conf").groupby("frame")[["x_pitch", "y_pitch"]].last()

    full_index = pd.RangeIndex(0, total_frames, name="frame")
    poss_full = poss_smoothed.reindex(full_index, fill_value="unknown")
    ball_x_full = ball_by_frame["x_pitch"].reindex(full_index)
    ball_y_full = ball_by_frame["y_pitch"].reindex(full_index)

    phase_series = label_phase(
        poss_full, ball_y_full, fps,
        transition_seconds=transition_seconds, halftime_frame=halftime_frame,
    )

    valid = {f: e for f, e in h_entries.items() if 0 <= f < total_frames}
    src = pd.Series("none", index=full_index)
    conf = pd.Series(0.0, index=full_index)
    if valid:
        src.update(pd.Series({f: e.source for f, e in valid.items()}))
        conf.update(pd.Series({f: e.confidence for f, e in valid.items()}))

    phases = pd.DataFrame({
        "frame": full_index,
        "t_seconds": full_index.to_numpy() / fps,
        "possession_state": poss_full.to_numpy(),
        "phase": phase_series.to_numpy(),
        "ball_x_pitch": ball_x_full.to_numpy(),
        "ball_y_pitch": ball_y_full.to_numpy(),
        "homography_source": src.to_numpy(),
        "homography_conf": conf.to_numpy(),
    }).astype({
        "frame": "int64", "t_seconds": "float64",
        "possession_state": "object", "phase": "object",
        "ball_x_pitch": "float64", "ball_y_pitch": "float64",
        "homography_source": "object", "homography_conf": "float64",
    })

    counts = Counter(e.source for e in valid.values())
    # Manual-anchor frames are direct high-confidence observations; count as anchor coverage.
    n_anchor = counts["anchor"] + counts["manual"]
    n_prop = counts["propagated"]
    anchor_cov = n_anchor / total_frames if total_frames else 0.0
    prop_cov = n_prop / total_frames if total_frames else 0.0
    ball_cov = float(ball_y_full.notna().sum()) / total_frames if total_frames else 0.0

    return PipelineResult(
        trajectories=enriched,
        phases=phases,
        homography_coverage=anchor_cov + prop_cov,
        ball_coverage=ball_cov,
        anchor_coverage=anchor_cov,
        propagated_coverage=prop_cov,
    )


def _infer_fps(trajectories_px: pd.DataFrame) -> float:
    """Recover fps from a row's frame / t_seconds (t = frame / fps).

    Uses the first row with frame > 0 and t_seconds > 0. Falls back to 30.0
    (common broadcast default) when no such row exists; callers who know the
    source fps should pass it explicitly.
    """
    nonzero = trajectories_px[(trajectories_px["t_seconds"] > 0) & (trajectories_px["frame"] > 0)]
    if nonzero.empty:
        return 30.0
    row = nonzero.iloc[0]
    return float(row["frame"]) / float(row["t_seconds"])


def _resolve_fps_and_frames(
    trajectories_px: pd.DataFrame, fps_override: float | None = None
) -> tuple[float, int]:
    """Resolve (fps, total_frames) from tracker output.

    fps comes from fps_override when given, else inferred. total_frames is the
    last detected frame + 1; trailing detection-free frames are omitted.
    """
    fps = fps_override if fps_override is not None else _infer_fps(trajectories_px)
    total_frames = int(trajectories_px["frame"].max()) + 1 if not trajectories_px.empty else 0
    return fps, total_frames


def _write_deliverables(result: PipelineResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.trajectories.to_parquet(out_dir / "trajectories.parquet", index=False)
    result.phases.to_parquet(out_dir / "phases.parquet", index=False)


def assemble_from_parquet(
    trajectories_px_path: Path,
    keypoints_path: Path,
    out_dir: Path,
    *,
    fps: float | None = None,
    **assemble_opts: object,
) -> PipelineResult:
    """Re-run the pure assembly stage from a Stage-1 checkpoint and write deliverables.

    This is the cheap-recompute path: tweak thresholds without re-running GPU tracking.
    """
    trajectories_px = pd.read_parquet(trajectories_px_path)
    keypoints = pd.read_parquet(keypoints_path)
    resolved_fps, total_frames = _resolve_fps_and_frames(trajectories_px, fps)
    result = assemble_phases(
        trajectories_px, keypoints, fps=resolved_fps, total_frames=total_frames, **assemble_opts  # type: ignore[arg-type]
    )
    _write_deliverables(result, Path(out_dir))
    return result


def analyze_video(
    video_path: Path,
    out_dir: Path,
    *,
    backend: TrackingBackend | None = None,
    own_kit: str | None = None,
    **assemble_opts: object,
) -> PipelineResult:
    """Run the full pipeline on a video and write checkpoints + deliverables.

    Stage 1 (GPU): pitch-aware tracking + checkpoints. Stage 2 (CPU): propagate
    homographies from anchors → homographies.parquet. Stage 3 (pure): assemble +
    write deliverables.

    own_kit grounds own/opp. When given (e.g. "white", "dark blue"), the verified
    hygiene grounding is chained (stitch fragments -> kit-cluster -> map_own_cluster
    -> positional goalkeeper assignment -> balance gate); the cleaned, grounded
    trajectories are assembled, so possession/phase are computed on grounded labels.
    When None, behaviour is unchanged BUT a loud warning is emitted: the tracker's
    own/opp comes from an arbitrary KMeans cluster index and may be globally inverted.
    """
    if backend is None:
        from soccer_vision.tracking.roboflow import (
            RoboflowBackend,  # lazy: avoids roboflow extra at import
        )

        backend = RoboflowBackend(detect_pitch=True)

    trajectories_px, keypoints = backend.process_with_pitch(video_path)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trajectories_px.to_parquet(out / "trajectories_px.parquet", index=False)
    keypoints.to_parquet(out / "keypoints.parquet", index=False)

    homographies = build_homographies(keypoints, video_path, trajectories_px)
    homographies_to_parquet(homographies, out / "homographies.parquet")

    resolved_fps, total_frames = _resolve_fps_and_frames(trajectories_px)

    if own_kit is not None:
        # Lazy import: hygiene.run imports homographies_from_parquet from THIS module,
        # so a top-level import would be circular (mirrors the lazy RoboflowBackend above).
        from soccer_vision.hygiene.run import run_hygiene

        report = run_hygiene(
            traj_path=out / "trajectories_px.parquet",
            homographies_path=out / "homographies.parquet",
            video_path=video_path,
            out_dir=out,
            own_kit=own_kit,
        )
        balance = report.get("balance") or {}
        if not balance.get("passed", False):
            logger.warning(
                "hygiene balance gate FAILED (own:opp ratio=%s) — own/opp grounding may be "
                "unreliable; inspect %s and the team_cluster_*.png contact sheets",
                balance.get("ratio"), out / "hygiene_report.json",
            )
        if report.get("warning"):
            logger.warning("hygiene grounding warning: %s", report["warning"])
        trajectories_for_assembly = pd.read_parquet(out / "trajectories_px_clean.parquet")
    else:
        logger.warning(
            "own/opp UNGROUNDED: team labels come from an arbitrary KMeans cluster index and "
            "may be GLOBALLY INVERTED. Pass own_kit=\"<your shirt colour>\" (or run "
            "`python -m soccer_vision.hygiene`) for trustworthy own-team analytics."
        )
        trajectories_for_assembly = trajectories_px

    result = assemble_phases(
        trajectories_for_assembly, keypoints, fps=resolved_fps, total_frames=total_frames,
        homographies=homographies, **assemble_opts,  # type: ignore[arg-type]
    )
    _write_deliverables(result, out)
    return result


def homographies_to_parquet(entries: dict[int, HomographyEntry], path: Path) -> None:
    """Serialize {frame: HomographyEntry} to a flat parquet (frame, h00..h22, source, conf)."""
    rows = []
    for frame, e in sorted(entries.items()):
        flat = np.asarray(e.H, dtype=np.float64).reshape(9)
        rows.append({"frame": frame, **dict(zip(_H_COLS, flat, strict=True)),
                     "source": e.source, "confidence": e.confidence})
    df = pd.DataFrame(rows, columns=["frame", *_H_COLS, "source", "confidence"])
    df = df.astype({"frame": "int64", "source": "object", "confidence": "float64",
                    **{c: "float64" for c in _H_COLS}})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def homographies_from_parquet(path: Path) -> dict[int, HomographyEntry]:
    """Read a homographies.parquet checkpoint back into {frame: HomographyEntry}."""
    df = pd.read_parquet(path)
    out: dict[int, HomographyEntry] = {}
    for _, r in df.iterrows():
        H = np.array([r[c] for c in _H_COLS], dtype=np.float64).reshape(3, 3)
        out[int(r["frame"])] = HomographyEntry(H, str(r["source"]), float(r["confidence"]))
    return out


def build_homographies(
    keypoints: pd.DataFrame,
    video_path: Path,
    trajectories_px: pd.DataFrame,
    *,
    kp_conf_threshold: float = 0.5,
    max_gap: int = 45,
    disagreement_tau: float = 0.10,
    downscale: float = 1.0,
) -> dict[int, HomographyEntry]:
    """Anchors from keypoints + propagation into the gaps.

    Computes the needed consecutive inter-frame homographies in one sequential video
    pass (each frame decoded once via grab/retrieve, ORB on a downscaled copy), then
    composes them with the pure propagate_homographies. CPU; reads frames sequentially.
    """
    anchors = build_frame_homographies(keypoints, conf_threshold=kp_conf_threshold)
    keys = sorted(anchors)
    needed_pairs: set[int] = set()
    for a, b in itertools.pairwise(keys):
        if 1 <= b - a - 1 <= max_gap:
            needed_pairs.update(range(a, b))          # G[a..b-1] span the gap

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    pos = 0

    def read_frame(idx: int) -> np.ndarray | None:
        nonlocal pos
        if idx < pos:                                 # backward jump (rare) -> seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            pos = idx
        while pos < idx:                              # skip forward cheaply (no decode)
            if not cap.grab():
                return None
            pos += 1
        ok, frame = cap.read()
        pos += 1
        return frame if ok else None

    try:
        interframe = compute_interframe_homographies(
            read_frame, needed_pairs, trajectories_px, downscale=downscale,
        )
    finally:
        cap.release()

    return propagate_homographies(
        anchors, interframe, max_gap=max_gap, disagreement_tau=disagreement_tau,
        frame_size=(width, height),
    )


def assemble_from_homographies(
    trajectories_px_path: Path,
    homographies_path: Path,
    out_dir: Path,
    *,
    fps: float | None = None,
    **assemble_opts: object,
) -> PipelineResult:
    """Stage-3 re-run from precomputed homographies (instant; no video, no GPU)."""
    trajectories_px = pd.read_parquet(trajectories_px_path)
    homographies = homographies_from_parquet(homographies_path)
    resolved_fps, total_frames = _resolve_fps_and_frames(trajectories_px, fps)
    # keypoints are unused when homographies are supplied; pass an empty frame.
    empty_kp = pd.DataFrame(columns=["frame", "kp_idx", "x_px", "y_px", "conf"])
    result = assemble_phases(
        trajectories_px, empty_kp, fps=resolved_fps, total_frames=total_frames,
        homographies=homographies, **assemble_opts,  # type: ignore[arg-type]
    )
    _write_deliverables(result, Path(out_dir))
    return result
