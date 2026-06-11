"""End-to-end driver test on a tiny synthetic video with two kit colors."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from soccer_vision.hygiene.run import run_hygiene
from soccer_vision.pipeline import homographies_to_parquet
from soccer_vision.pitch.propagation import HomographyEntry

_W, _H, _N = 320, 240, 30
# pixels -> pitch: x_pitch = x_px / W, y_pitch = y_px / H (all bboxes in-bounds).
_H_PX = np.diag([1.0 / _W, 1.0 / _H, 1.0])

# two "players": white shirt/blue shorts at left, dark-blue shirt/yellow shorts right
_KITS: dict[int, tuple[tuple[int, int, int], tuple[int, int, int], int]] = {
    1: ((255, 255, 255), (150, 60, 20), 60),
    2: ((110, 40, 10), (40, 210, 230), 220),
}


def _write_video(path: Path) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (_W, _H))  # type: ignore[attr-defined]
    for _ in range(_N):
        frame = np.full((_H, _W, 3), (40, 120, 40), dtype=np.uint8)
        for _tid, (shirt, shorts, x) in _KITS.items():
            cv2.rectangle(frame, (x, 100), (x + 30, 130), shirt, -1)
            cv2.rectangle(frame, (x, 130), (x + 30, 150), shorts, -1)
        vw.write(frame)
    vw.release()


def _traj() -> pd.DataFrame:
    rows = []
    for f in range(_N):
        for tid, (_, _, x) in _KITS.items():
            rows.append({
                "frame": f, "t_seconds": f / 30.0, "track_id": tid,
                "x_px": x + 15.0, "y_px": 125.0,
                "bbox_x1": float(x), "bbox_y1": 100.0,
                "bbox_x2": float(x + 30), "bbox_y2": 150.0,
                "class": "player", "team": "stale", "conf": 0.9,
            })
    return pd.DataFrame(rows).astype({"frame": "int64", "track_id": "int64"})


def test_run_hygiene_end_to_end(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    _write_video(video)
    traj_path = tmp_path / "traj.parquet"
    _traj().to_parquet(traj_path, index=False)
    hom_path = tmp_path / "hom.parquet"
    homographies_to_parquet(
        {f: HomographyEntry(_H_PX, "manual", 1.0) for f in range(_N)}, hom_path
    )
    out = tmp_path / "out"

    report = run_hygiene(
        traj_path=traj_path, homographies_path=hom_path, video_path=video,
        out_dir=out, own_kit="white",
    )

    clean = pd.read_parquet(out / "trajectories_px_clean.parquet")
    assert "orig_track_id" in clean.columns
    teams = clean.groupby("track_id")["team"].first()
    assert set(teams.values) == {"own", "opp"}
    assert teams[1] == "own"   # white-shirt track
    assert (out / "hygiene_report.json").exists()
    saved = json.loads((out / "hygiene_report.json").read_text())
    assert saved["balance"]["passed"] is True
    sheets = list(out.glob("team_cluster_*.png"))
    assert len(sheets) == 2
    assert any("_OWN" in s.name for s in sheets)
    assert report["balance"]["passed"] is True


def test_run_hygiene_wrong_video_raises(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    _write_video(video)  # 30 frames
    traj = _traj()
    traj.loc[0, "frame"] = 500  # references frame beyond the video
    traj_path = tmp_path / "traj.parquet"
    traj.to_parquet(traj_path, index=False)
    hom_path = tmp_path / "hom.parquet"
    homographies_to_parquet({0: HomographyEntry(_H_PX, "manual", 1.0)}, hom_path)
    try:
        run_hygiene(traj_path=traj_path, homographies_path=hom_path,
                    video_path=video, out_dir=tmp_path / "out", own_kit="white")
    except ValueError as e:
        assert "wrong video" in str(e)
    else:
        raise AssertionError("expected ValueError")
