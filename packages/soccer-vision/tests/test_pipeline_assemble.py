"""End-to-end tests for assemble_phases (the integration that was missing)."""

from __future__ import annotations

import pandas as pd
from soccer_vision.pipeline import PipelineResult, assemble_phases
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

FPS = 1.0  # window_frames = round(1.0) = 1 -> smoothing is a no-op, so per-frame states show through


def _identity_keypoints(n_frames: int) -> pd.DataFrame:
    """6 landmarks per frame with image points == pitch coords -> identity H."""
    idxs = [0, 5, 13, 16, 24, 29]
    pts = PITCH_LANDMARKS[idxs]
    rows = []
    for f in range(n_frames):
        for k, (x, y) in zip(idxs, pts, strict=True):
            rows.append({"frame": f, "kp_idx": k, "x_px": float(x), "y_px": float(y), "conf": 0.9})
    return pd.DataFrame(rows)


def _det(frame, track_id, x, y, cls, team, conf=0.9):
    # Identity homography -> x_px/y_px ARE the pitch coords.
    return {
        "frame": frame, "t_seconds": frame / FPS, "track_id": track_id,
        "x_px": x, "y_px": y,
        "bbox_x1": x - 0.01, "bbox_y1": y - 0.01, "bbox_x2": x + 0.01, "bbox_y2": y + 0.01,
        "class": cls, "team": team, "conf": conf,
    }


def _scene() -> pd.DataFrame:
    rows = []
    # own player track 1: own third, team flickers own/opp/own -> modal "own"
    for f, team in zip(range(3), ["own", "opp", "own"], strict=True):
        rows.append(_det(f, 1, 0.50, 0.25, "player", team))
    # opp player track 101: opp end, always "opp"
    for f in range(3):
        rows.append(_det(f, 101, 0.50, 0.85, "player", "opp"))
    # ball: f0 near own (build), f1 mid loose, f2 near opp (defend_high)
    rows.append(_det(0, -1, 0.50, 0.27, "ball", "unknown"))
    rows.append(_det(1, -2, 0.50, 0.55, "ball", "unknown"))
    rows.append(_det(2, -3, 0.50, 0.80, "ball", "unknown"))
    # adjacent-game (off-pitch) detection in f0: x_pitch 1.40 > 1+margin -> dropped
    rows.append(_det(0, 999, 1.40, 0.50, "player", "unknown"))
    df = pd.DataFrame(rows)
    return df.astype({"frame": "int64", "track_id": "int64"})


def test_assemble_phases_end_to_end() -> None:
    traj = _scene()
    kp = _identity_keypoints(3)
    result = assemble_phases(traj, kp, fps=FPS, total_frames=3)

    assert isinstance(result, PipelineResult)
    # Enriched trajectories carry pitch coords.
    assert "x_pitch" in result.trajectories.columns
    assert "y_pitch" in result.trajectories.columns
    # Off-pitch adjacent-game detection was dropped.
    assert (result.trajectories["track_id"] == 999).sum() == 0
    # Team flicker on track 1 resolved to modal "own".
    own_rows = result.trajectories[result.trajectories["track_id"] == 1]
    assert set(own_rows["team"]) == {"own"}
    # Per-frame phases over the full [0, 3) range.
    phases = result.phases.set_index("frame")
    assert list(result.phases["frame"]) == [0, 1, 2]
    assert phases.loc[0, "possession_state"] == "own"
    assert phases.loc[0, "phase"] == "build"
    assert phases.loc[1, "possession_state"] == "loose_ball"
    assert phases.loc[1, "phase"] == "loose_ball"
    assert phases.loc[2, "possession_state"] == "opp"
    assert phases.loc[2, "phase"] == "defend_high"
    # Coverage stats.
    assert result.homography_coverage == 1.0
    assert result.ball_coverage == 1.0


def test_assemble_phases_no_homography_degrades_to_unknown() -> None:
    traj = _scene()
    empty_kp = pd.DataFrame(columns=["frame", "kp_idx", "x_px", "y_px", "conf"])
    result = assemble_phases(traj, empty_kp, fps=FPS, total_frames=3)
    assert result.homography_coverage == 0.0
    assert set(result.phases["possession_state"]) == {"unknown"}
    assert set(result.phases["phase"]) == {"unknown"}
    assert set(result.phases["homography_source"]) == {"none"}
    assert result.anchor_coverage == 0.0 and result.propagated_coverage == 0.0


def test_assemble_phases_fills_full_frame_range() -> None:
    traj = _scene()
    kp = _identity_keypoints(3)
    result = assemble_phases(traj, kp, fps=FPS, total_frames=5)  # 2 trailing empty frames
    assert list(result.phases["frame"]) == [0, 1, 2, 3, 4]
    assert result.phases.set_index("frame").loc[4, "possession_state"] == "unknown"


def test_assemble_from_parquet_roundtrip(tmp_path) -> None:
    from soccer_vision.pipeline import assemble_from_parquet

    traj = _scene()
    kp = _identity_keypoints(3)
    traj_path = tmp_path / "trajectories_px.parquet"
    kp_path = tmp_path / "keypoints.parquet"
    traj.to_parquet(traj_path, index=False)
    kp.to_parquet(kp_path, index=False)
    out_dir = tmp_path / "out"

    result = assemble_from_parquet(traj_path, kp_path, out_dir)

    # fps inferred from t_seconds (= 1.0 here), total_frames from max(frame)+1 (= 3).
    assert list(result.phases["frame"]) == [0, 1, 2]
    # Deliverables written.
    assert (out_dir / "trajectories.parquet").exists()
    assert (out_dir / "phases.parquet").exists()
    reloaded = pd.read_parquet(out_dir / "trajectories.parquet")
    assert "x_pitch" in reloaded.columns
    reloaded_phases = pd.read_parquet(out_dir / "phases.parquet")
    assert list(reloaded_phases.columns) == [
        "frame", "t_seconds", "possession_state", "phase", "ball_x_pitch", "ball_y_pitch",
        "homography_source", "homography_conf",
    ]


def test_infer_fps_falls_back_to_30_when_no_positive_frame() -> None:
    from soccer_vision.pipeline import _infer_fps

    df = pd.DataFrame({"frame": [0], "t_seconds": [0.0]})
    assert _infer_fps(df) == 30.0


def test_assemble_phases_applies_possession_smoothing() -> None:
    # fps=5 -> window=5; a single-frame opp flicker (frame 2) smooths back to own.
    rows = []
    for f in range(5):
        rows.append(_det(f, 1, 0.50, 0.25, "player", "own"))    # own player, own third
        rows.append(_det(f, 101, 0.50, 0.75, "player", "opp"))  # opp player
    # ball near own except frame 2 near opp -> raw possession own,own,opp,own,own
    for f, by in enumerate([0.27, 0.27, 0.73, 0.27, 0.27]):
        rows.append(_det(f, -1 - f, 0.50, by, "ball", "unknown"))
    traj = pd.DataFrame(rows).astype({"frame": "int64", "track_id": "int64"})
    kp = _identity_keypoints(5)

    result = assemble_phases(traj, kp, fps=5.0, total_frames=5)
    states = result.phases.set_index("frame")["possession_state"]
    # Frame 2 is 'opp' before smoothing; the window=5 mode smooths it back to 'own'.
    assert states.loc[2] == "own"
    assert states.loc[0] == "own"


def test_assemble_phases_accepts_precomputed_homographies() -> None:
    import numpy as np
    from soccer_vision.pitch.propagation import HomographyEntry

    traj = _scene()
    kp = _identity_keypoints(3)
    # identity-H entries: frame 0 anchor, frames 1-2 propagated (lower conf)
    homs = {
        0: HomographyEntry(np.eye(3), "anchor", 1.0),
        1: HomographyEntry(np.eye(3), "propagated", 0.6),
        2: HomographyEntry(np.eye(3), "propagated", 0.6),
    }
    result = assemble_phases(traj, kp, fps=FPS, total_frames=3, homographies=homs)

    ph = result.phases.set_index("frame")
    assert ph.loc[0, "homography_source"] == "anchor"
    assert ph.loc[1, "homography_source"] == "propagated"
    assert abs(ph.loc[1, "homography_conf"] - 0.6) < 1e-9
    assert result.anchor_coverage == 1 / 3
    assert abs(result.propagated_coverage - 2 / 3) < 1e-9
    assert abs(result.homography_coverage - 1.0) < 1e-9


def test_assemble_phases_legacy_path_marks_anchors() -> None:
    traj = _scene()
    kp = _identity_keypoints(3)
    result = assemble_phases(traj, kp, fps=FPS, total_frames=3)  # no homographies arg
    ph = result.phases.set_index("frame")
    assert set(ph["homography_source"]) <= {"anchor", "none"}
    assert result.propagated_coverage == 0.0
    assert result.anchor_coverage == 1.0   # all 3 frames are landmark anchors in the fixture
