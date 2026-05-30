"""Pipeline orchestrator: chains pitch + phase modules into enriched outputs.

assemble_phases is pure (no models, no GPU, no ultralytics/sports import) so the
integration logic is testable without a GPU. analyze_video / assemble_from_parquet
add model invocation and parquet I/O around it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Enriched outputs of the pipeline plus coverage diagnostics."""

    trajectories: pd.DataFrame   # per-detection, +x_pitch/+y_pitch, team modal-cleaned
    phases: pd.DataFrame         # per-frame over [0, total_frames)
    homography_coverage: float   # fraction of frames where the pitch model fit a homography (pre-smoothing)
    ball_coverage: float         # fraction of frames with a non-NaN ball pitch coord


def assemble_phases(
    trajectories_px: pd.DataFrame,
    keypoints: pd.DataFrame,
    fps: float,
    total_frames: int,
    *,
    kp_conf_threshold: float = 0.5,
    homography_alpha: float = 0.5,
    filter_margin: float = 0.05,
    possession_thresholds: PossessionThresholds | None = None,
    transition_seconds: float = 5.0,
) -> PipelineResult:
    """Run the full pitch + phase chain on tracker output. Pure; no I/O."""
    raw_h = build_frame_homographies(keypoints, conf_threshold=kp_conf_threshold)
    homographies = smooth_homographies(raw_h, alpha=homography_alpha)
    if not homographies:
        logger.warning("No homographies fitted; pitch coords NaN, phases all 'unknown'.")

    enriched = PitchMapper().transform(trajectories_px, homographies)
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
        poss_full, ball_y_full, fps, transition_seconds=transition_seconds
    )

    phases = pd.DataFrame({
        "frame": full_index,
        "t_seconds": full_index.to_numpy() / fps,
        "possession_state": poss_full.to_numpy(),
        "phase": phase_series.to_numpy(),
        "ball_x_pitch": ball_x_full.to_numpy(),
        "ball_y_pitch": ball_y_full.to_numpy(),
    }).astype({
        "frame": "int64",
        "t_seconds": "float64",
        "possession_state": "object",
        "phase": "object",
        "ball_x_pitch": "float64",
        "ball_y_pitch": "float64",
    })

    hom_cov = len(raw_h) / total_frames if total_frames else 0.0
    ball_cov = float(ball_y_full.notna().sum()) / total_frames if total_frames else 0.0

    return PipelineResult(
        trajectories=enriched,
        phases=phases,
        homography_coverage=hom_cov,
        ball_coverage=ball_cov,
    )
