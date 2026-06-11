"""Impure driver for the hygiene stage: artifact I/O, crop features, orchestration.

Reads trajectories_px + homographies parquets and the local video; writes
trajectories_px_clean.parquet, contact sheets, and hygiene_report.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.hygiene.core import (
    apply_team_labels,
    assign_goalkeepers,
    balance_gate,
    cluster_teams,
    map_own_cluster,
    stitch_tracks,
)
from soccer_vision.pipeline import homographies_from_parquet
from soccer_vision.pitch.mapper import PitchMapper

_CROPS_PER_TRACK = 10
_MIN_CROP_PX = 8
_SHEET_COLS = 10
_SHEET_CELL = 64


def _resolve_fps(traj: pd.DataFrame) -> float:
    by_frame = traj.drop_duplicates("frame").sort_values("frame")
    dt = float(by_frame["t_seconds"].diff().median())
    dframe = float(by_frame["frame"].diff().median())
    if not dt or pd.isna(dt) or dt <= 0:
        return 30.0
    return dframe / dt


def _sample_frames(track_rows: pd.DataFrame) -> list[int]:
    frames = sorted(int(f) for f in track_rows["frame"].unique())
    if len(frames) <= _CROPS_PER_TRACK:
        return frames
    idx = np.linspace(0, len(frames) - 1, _CROPS_PER_TRACK).astype(int)
    return [frames[i] for i in idx]


def _region_lab(
    crop: NDArray[np.uint8], y0: float, y1: float
) -> NDArray[np.float64] | None:
    h, w = crop.shape[:2]
    if h < _MIN_CROP_PX or w < _MIN_CROP_PX:
        return None
    region = crop[int(h * y0):int(h * y1), int(w * 0.2):int(w * 0.8)]
    if region.size == 0:
        return None
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.asarray(np.median(lab, axis=0), dtype=np.float64)


def extract_track_features(
    video_path: Path, traj: pd.DataFrame
) -> tuple[
    dict[int, NDArray[np.float64]],
    dict[int, float],
    dict[int, list[NDArray[np.uint8]]],
]:
    """Per stitched player track: 6-dim shirt+shorts Lab feature, weight, crops.

    Only rows flagged on_pitch contribute (adjacent-field and no-homography rows
    are excluded from features). Reads the video once, sequentially.
    """
    players = traj[(traj["class"] == "player") & traj["on_pitch"]]
    wanted: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for tid, g in players.groupby("track_id", sort=True):
        for f in _sample_frames(g):
            row = g[g["frame"] == f].iloc[0]
            box = (float(row["bbox_x1"]), float(row["bbox_y1"]),
                   float(row["bbox_x2"]), float(row["bbox_y2"]))
            wanted.setdefault(f, []).append((cast(int, tid), box))

    shirt_feats: dict[int, list[NDArray[np.float64]]] = {}
    shorts_feats: dict[int, list[NDArray[np.float64]]] = {}
    crops: dict[int, list[NDArray[np.uint8]]] = {}
    cap = cv2.VideoCapture(str(video_path))
    pos = 0
    try:
        for f in sorted(wanted):
            while pos < f:
                if not cap.grab():
                    break
                pos += 1
            ok, frame = cap.read()
            pos += 1
            if not ok:
                continue
            for tid, (x1, y1, x2, y2) in wanted[f]:
                raw = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                crop: NDArray[np.uint8] = np.asarray(raw, dtype=np.uint8)
                shirt = _region_lab(crop, 0.15, 0.50)
                shorts = _region_lab(crop, 0.50, 0.80)
                if shirt is None or shorts is None:
                    continue
                shirt_feats.setdefault(tid, []).append(shirt)
                shorts_feats.setdefault(tid, []).append(shorts)
                crops.setdefault(tid, []).append(crop)
    finally:
        cap.release()

    features: dict[int, NDArray[np.float64]] = {}
    weights: dict[int, float] = {}
    track_len = players.groupby("track_id")["frame"].nunique()
    for tid, shirts in shirt_feats.items():
        features[tid] = np.concatenate([
            np.median(np.stack(shirts), axis=0),
            np.median(np.stack(shorts_feats[tid]), axis=0),
        ])
        weights[tid] = float(track_len.get(tid, 1))
    return features, weights, crops


def write_contact_sheet(
    crops: list[NDArray[np.uint8]], path: Path, *, max_cells: int = 40
) -> None:
    """Grid of resized crops for at-a-glance cluster verification."""
    cells = [cv2.resize(c, (_SHEET_CELL, _SHEET_CELL)) for c in crops[:max_cells]
             if c.size > 0]
    if not cells:
        cells = [np.zeros((_SHEET_CELL, _SHEET_CELL, 3), dtype=np.uint8)]
    rows = []
    for i in range(0, len(cells), _SHEET_COLS):
        row = cells[i:i + _SHEET_COLS]
        row += [np.zeros_like(cells[0])] * (_SHEET_COLS - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows))


def run_hygiene(
    *,
    traj_path: Path,
    homographies_path: Path,
    video_path: Path,
    out_dir: Path,
    own_kit: str,
    max_gap_s: float = 2.0,
    max_speed_ms: float = 8.0,
    margin: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Full hygiene pass; returns (and writes) the report dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj = pd.read_parquet(traj_path)
    entries = homographies_from_parquet(homographies_path)
    h_map = {f: e.H for f, e in entries.items()}

    cap = cv2.VideoCapture(str(video_path))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_video < int(traj["frame"].max()):
        raise ValueError(
            f"video has {n_video} frames but trajectories reference frame "
            f"{int(traj['frame'].max())} — wrong video?"
        )

    fps = _resolve_fps(traj)
    mapped = PitchMapper().transform(traj, h_map)

    # Step 1: on-pitch filter for player/GK rows; keep-but-flag no-homography rows.
    is_person = mapped["class"].isin(["player", "goalkeeper"])
    has_pitch = mapped["x_pitch"].notna()
    person = mapped[is_person]
    lo, hi = -margin, 1.0 + margin
    on_mask = person["x_pitch"].between(lo, hi) & person["y_pitch"].between(lo, hi)
    mapped["on_pitch"] = False
    mapped.loc[person.index[on_mask], "on_pitch"] = True
    drop = is_person & has_pitch & ~mapped["on_pitch"]
    n_dropped = int(drop.sum())
    kept = mapped[~drop].copy()

    # Step 2: stitch fragments in pitch space.
    is_person_kept = kept["class"].isin(["player", "goalkeeper"])
    n_tracks_before = int(kept.loc[is_person_kept, "track_id"].nunique())
    stitched = stitch_tracks(kept, fps=fps, max_gap_s=max_gap_s,
                             max_speed_ms=max_speed_ms)
    n_tracks_after = int(
        stitched.loc[stitched["class"].isin(["player", "goalkeeper"]),
                     "track_id"].nunique()
    )

    # Step 3: cluster teams from kit colors.
    features, weights, crops = extract_track_features(video_path, stitched)
    warning: str | None = None
    centroid_list: list[list[float]] = []
    if len(features) < 2:
        team_by_track: dict[int, str] = {}
        warning = "fewer than 2 tracks with features; all players unknown"
    else:
        _features_f: dict[int, NDArray[np.floating[Any]]] = cast(
            dict[int, NDArray[np.floating[Any]]], features
        )
        cluster_of, centroids = cluster_teams(_features_f, weights, seed=seed)
        centroid_list = [[float(v) for v in c] for c in centroids]
        if float(np.linalg.norm(centroids[0] - centroids[1])) < 10.0:
            team_by_track = {tid: "unknown" for tid in features}
            warning = "cluster centroids nearly identical; all players set unknown"
        else:
            own_cluster, warning = map_own_cluster(centroids, own_kit)
            label = {own_cluster: "own", 1 - own_cluster: "opp"}
            team_by_track = {tid: ("unknown" if c is None else label[c])
                             for tid, c in cluster_of.items()}
            for c_idx in (0, 1):
                tids = [t for t, c in cluster_of.items() if c == c_idx]
                sheet = [crop for t in tids for crop in crops.get(t, [])[:3]]
                suffix = "_OWN" if c_idx == own_cluster else ""
                write_contact_sheet(sheet, out / f"team_cluster_{c_idx}{suffix}.png")

    # Step 4: goalkeepers positionally.
    team_by_track.update(assign_goalkeepers(stitched, team_by_track))

    clean = apply_team_labels(stitched, team_by_track)
    ratio, passed = balance_gate(clean)
    clean = clean.drop(columns=["x_pitch", "y_pitch", "on_pitch"])
    clean.to_parquet(out / "trajectories_px_clean.parquet", index=False)

    person_clean = clean[clean["class"].isin(["player", "goalkeeper"])]
    spans = person_clean.groupby("track_id")["frame"].agg(["min", "max"])
    report: dict[str, Any] = {
        "n_dropped_off_pitch": n_dropped,
        "tracks_before": n_tracks_before,
        "tracks_after": n_tracks_after,
        "median_track_span_frames": (
            float((spans["max"] - spans["min"] + 1).median()) if len(spans) else 0.0
        ),
        "unknown_fraction": (
            float((person_clean["team"] == "unknown").mean())
            if len(person_clean) else 0.0
        ),
        "cluster_centroids_lab": centroid_list,
        "balance": {"ratio": ratio, "passed": passed},
        "warning": warning,
        "params": {"own_kit": own_kit, "max_gap_s": max_gap_s,
                   "max_speed_ms": max_speed_ms, "margin": margin, "fps": fps},
    }
    (out / "hygiene_report.json").write_text(json.dumps(report, indent=2))
    print(f"tracks {n_tracks_before} -> {n_tracks_after}; "
          f"dropped off-pitch rows: {n_dropped}")
    print(f"balance own:opp = {ratio:.2f}  [{'PASS' if passed else 'FAIL'}]")
    if warning:
        print(f"WARNING: {warning}")
    return report
