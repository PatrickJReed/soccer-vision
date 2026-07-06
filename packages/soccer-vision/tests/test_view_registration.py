"""Tests for drift-free view-registration calibration (Slice 2)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from soccer_vision.labeler.view_digest import ViewDigest
from soccer_vision.pitch.view_registration import (
    RegisteredCalib,
    compose_pitch_homography,
    register_clip,
    register_to_best_rep,
    rep_homographies_from_parquet,
    view_consistency,
    write_homographies,
)

H, W = 240, 320


def _pattern(seed: int) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 30, np.uint8)
    for _ in range(60):
        x1, x2 = sorted(rng.integers(0, W, size=2).tolist())
        y1, y2 = sorted(rng.integers(0, H, size=2).tolist())
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        else:
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
    return img


def _orb(img):
    o = cv2.ORB_create(3000)  # type: ignore[attr-defined]
    return o.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)


def test_compose_pitch_homography() -> None:
    H_rep = np.array([[2.0, 0, 1], [0, 3.0, 2], [0, 0, 1]])   # rep_px -> pitch
    G = np.array([[1.0, 0, 5], [0, 1.0, 7], [0, 0, 1]])        # frame_px -> rep_px
    Hf = compose_pitch_homography(H_rep, G)
    p = Hf @ np.array([0, 0, 1.0])
    p = p[:2] / p[2]
    assert np.allclose(p, [11.0, 23.0])
    assert abs(Hf[2, 2] - 1.0) < 1e-9


def test_register_to_best_rep_picks_matching_view() -> None:
    rep_a = _pattern(1)
    rep_b = _pattern(99)
    M = np.array([[1, 0, 8.0], [0, 1, -5.0], [0, 0, 1]])
    frame = cv2.warpPerspective(rep_a, M, (W, H))
    fkp, fdesc = _orb(frame)
    akp, adesc = _orb(rep_a)
    bkp, bdesc = _orb(rep_b)
    out = register_to_best_rep(fkp, fdesc, [akp, bkp], [adesc, bdesc], min_inliers=12)
    assert out is not None
    idx, G, n_in = out
    assert idx == 0 and n_in >= 12
    p = G @ np.array([8.0, 0.0, 1.0])
    p = p[:2] / p[2]
    assert np.linalg.norm(p - np.array([0.0, 5.0])) < 3.0


def test_register_to_best_rep_no_match_returns_none() -> None:
    frame = _pattern(3)
    fkp, fdesc = _orb(frame)
    blank = np.full((H, W, 3), 30, np.uint8)
    bkp, bdesc = _orb(blank)
    assert register_to_best_rep(fkp, fdesc, [bkp], [bdesc], min_inliers=12) is None


def _identity_digest(reps: dict[int, int]) -> ViewDigest:
    return ViewDigest(sample_frames=list(reps.values()),
                      view_of={f: v for v, f in reps.items()},
                      representatives=reps, similarity=np.zeros((1, 1)))


def _write_video(path: Path, frames: list) -> bool:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    vw = cv2.VideoWriter(str(path), fourcc, 30.0, (W, H))
    if not vw.isOpened():
        return False
    for fr in frames:
        vw.write(fr)
    vw.release()
    return path.exists() and path.stat().st_size > 0


def test_rep_homographies_from_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame([
        {"frame": 5, "h00": 1.0, "h01": 0, "h02": 0, "h10": 0, "h11": 1.0, "h12": 0,
         "h20": 0, "h21": 0, "h22": 1.0, "source": "manual", "confidence": 1.0},
        {"frame": 9, "h00": 2.0, "h01": 0, "h02": 1, "h10": 0, "h11": 2.0, "h12": 3,
         "h20": 0, "h21": 0, "h22": 1.0, "source": "manual", "confidence": 1.0}])
    p = tmp_path / "h.parquet"
    df.to_parquet(p, index=False)
    reps = rep_homographies_from_parquet(p, {0: 5, 1: 9, 2: 999})
    assert set(reps) == {0, 1}
    assert np.allclose(reps[1], [[2, 0, 1], [0, 2, 3], [0, 0, 1]])


def test_register_clip_end_to_end(tmp_path: Path) -> None:
    A, B = _pattern(1), _pattern(2)
    frames = [A] * 10 + [B] * 10
    video = tmp_path / "clip.mp4"
    if not _write_video(video, frames):
        pytest.skip("no mp4 writer")
    reps = {0: 0, 1: 10}
    digest = _identity_digest(reps)
    rep_h = {0: np.eye(3), 1: np.array([[1.0, 0, 100], [0, 1.0, 0], [0, 0, 1]])}
    calib = register_clip(video, digest, rep_h, cache_dir=tmp_path)
    assert isinstance(calib, RegisteredCalib)
    assert calib.homographies[0].source == "rep"
    assert np.allclose(calib.homographies[0].H, np.eye(3))
    e = calib.homographies[15]
    assert e.source in ("registered", "rep")
    assert calib.rep_of[15] == 1
    assert calib.stats["coverage"] > 0.8


def test_write_homographies_schema(tmp_path: Path) -> None:
    from soccer_vision.pitch.propagation import HomographyEntry
    calib = RegisteredCalib(
        homographies={0: HomographyEntry(np.eye(3), "rep", 1.0),
                      1: HomographyEntry(np.eye(3), "registered", 0.7)},
        rep_of={0: 0, 1: 0}, stats={})
    out = tmp_path / "homographies.parquet"
    write_homographies(calib, out)
    df = pd.read_parquet(out)
    assert set(["frame", "h00", "h22", "source", "confidence"]).issubset(df.columns)
    assert set(df["source"]) == {"rep", "registered"}


def test_pipeline_coverage_counts_rep_and_registered() -> None:
    """view_registration sources 'rep'/'registered' must count toward homography_coverage;
    a fully-calibrated run previously reported 0.0 -> false 'calibration FAILED' signal."""
    from soccer_vision.pipeline import assemble_phases
    from soccer_vision.pitch.propagation import HomographyEntry

    def det(frame: int, tid: int, x: float, y: float, cls: str, team: str) -> dict:
        return {"frame": frame, "t_seconds": float(frame), "track_id": tid,
                "x_px": x, "y_px": y, "bbox_x1": x - 0.01, "bbox_y1": y - 0.01,
                "bbox_x2": x + 0.01, "bbox_y2": y + 0.01,
                "class": cls, "team": team, "conf": 0.9}
    rows = []
    for f in range(3):
        rows.append(det(f, 1, 0.5, 0.25, "player", "own"))
        rows.append(det(f, 2, 0.5, 0.85, "player", "opp"))
        rows.append(det(f, -1, 0.5, 0.5, "ball", "unknown"))
    traj = pd.DataFrame(rows).astype({"frame": "int64", "track_id": "int64"})
    kp = pd.DataFrame(columns=["frame", "kp_idx", "x_px", "y_px", "conf"])
    homs = {0: HomographyEntry(np.eye(3), "rep", 1.0),           # anchor-equivalent
            1: HomographyEntry(np.eye(3), "registered", 0.8),    # derived-equivalent
            2: HomographyEntry(np.eye(3), "registered", 0.8)}
    result = assemble_phases(traj, kp, fps=1.0, total_frames=3, homographies=homs)
    assert abs(result.anchor_coverage - 1 / 3) < 1e-9        # "rep" -> anchor coverage
    assert abs(result.propagated_coverage - 2 / 3) < 1e-9    # "registered" -> prop coverage
    assert abs(result.homography_coverage - 1.0) < 1e-9      # not 0.0 (the false-FAIL bug)


def test_view_consistency_flags_drift() -> None:
    from soccer_vision.pitch.propagation import HomographyEntry
    # view 0 has 3 frames far apart in time; registration maps them ~identically (tight),
    # a synthetic "chain" maps them with growing offset (drift) -> larger spread.
    def H(dx):  # translation in pitch
        return np.array([[1.0, 0, dx], [0, 1.0, 0], [0, 0, 1]])
    reg = {10: HomographyEntry(H(0.0), "rep", 1.0),
           500: HomographyEntry(H(0.01), "registered", 0.9),
           2000: HomographyEntry(H(-0.01), "registered", 0.9)}
    chain = {10: H(0.0), 500: H(0.30), 2000: H(0.90)}   # drifts with time
    calib = RegisteredCalib(homographies=reg, rep_of={10: 0, 500: 0, 2000: 0}, stats={})
    df = view_consistency(calib, chain=chain)
    row = df[df["view"] == 0].iloc[0]
    assert row["n_frames"] == 3 and row["temporal_span"] == 1990
    assert row["reg_spread"] < 0.05          # registration is tight
    assert row["chain_spread"] > row["reg_spread"] * 3   # chain drifts much more


def test_view_consistency_skips_singleton_views() -> None:
    from soccer_vision.pitch.propagation import HomographyEntry
    calib = RegisteredCalib(
        homographies={1: HomographyEntry(np.eye(3), "rep", 1.0)},
        rep_of={1: 0}, stats={})
    assert view_consistency(calib).empty


def test_nondegenerate_rejects_extreme_scale() -> None:
    from soccer_vision.pitch.view_registration import _nondegenerate
    assert _nondegenerate(np.eye(3)) is True                  # det 1 -> ok
    assert _nondegenerate(np.diag([100.0, 100.0, 1.0])) is False   # det 1e4 > 50 -> degenerate
    assert _nondegenerate(np.diag([1e-3, 1e-3, 1.0])) is False      # det 1e-6 < 0.02 -> degenerate


def test_cli_writes_homographies(tmp_path: Path) -> None:
    import json

    from soccer_vision.pitch.view_registration import main
    A, B = _pattern(1), _pattern(2)
    frames = [A] * 6 + [B] * 6
    video = tmp_path / "clip.mp4"
    if not _write_video(video, frames):
        pytest.skip("no mp4 writer")
    digest_json = tmp_path / "view_digest.json"
    digest_json.write_text(json.dumps({
        "n_views": 2, "stride": 6, "sample_frames": [0, 6],
        "representatives": {"0": 0, "1": 6}, "view_of": {"0": 0, "6": 1}}))
    hp = tmp_path / "rep_h.parquet"
    pd.DataFrame([
        {"frame": 0, "h00": 1.0, "h01": 0, "h02": 0, "h10": 0, "h11": 1.0, "h12": 0,
         "h20": 0, "h21": 0, "h22": 1.0, "source": "manual", "confidence": 1.0},
        {"frame": 6, "h00": 1.0, "h01": 0, "h02": 9, "h10": 0, "h11": 1.0, "h12": 0,
         "h20": 0, "h21": 0, "h22": 1.0, "source": "manual", "confidence": 1.0},
    ]).to_parquet(hp, index=False)
    out = tmp_path / "out"
    main(["--video", str(video), "--digest-json", str(digest_json),
          "--rep-homographies", str(hp), "--out", str(out)])
    assert (out / "homographies.parquet").exists()
