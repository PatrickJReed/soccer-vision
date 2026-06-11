"""Pure core of the track-hygiene stage.

Operates on trajectories DataFrames that already carry x_pitch/y_pitch columns
(added by PitchMapper). Distances are isotropic length-normalized pitch units:
x_len = x_pitch / aspect_ratio, y_len = y_pitch (y is the goal-to-goal axis).

No I/O. See docs/superpowers/specs/2026-06-11-track-hygiene-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.pitch.spec import PitchSpec

# Tolerances are physical, so they need a nominal pitch length to convert m/s
# into pitch-length units. US Soccer 9v9 mid-range; a tolerance, not a metric.
PITCH_LENGTH_M_DEFAULT = 68.5


@dataclass(frozen=True)
class Fragment:
    """A contiguous tracker fragment's pitch-space endpoints (length-normalized)."""

    track_id: int
    start_frame: int
    end_frame: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]


def extract_fragments(
    traj: pd.DataFrame, *, aspect_ratio: float | None = None
) -> list[Fragment]:
    """One Fragment per track_id from its first/last rows with valid pitch coords.

    Tracks with no valid pitch coordinate are omitted (they cannot stitch).
    """
    ar = aspect_ratio if aspect_ratio is not None else PitchSpec.standard_9v9().aspect_ratio
    frags: list[Fragment] = []
    valid = traj.dropna(subset=["x_pitch", "y_pitch"])
    for tid, group in valid.groupby("track_id", sort=True):
        g = group.sort_values("frame")
        first = g.iloc[0]
        last = g.iloc[-1]
        frags.append(
            Fragment(
                track_id=int(cast(int, tid)),
                start_frame=int(first["frame"]),
                end_frame=int(last["frame"]),
                start_xy=(float(first["x_pitch"]) / ar, float(first["y_pitch"])),
                end_xy=(float(last["x_pitch"]) / ar, float(last["y_pitch"])),
            )
        )
    return frags


def stitch_fragments(
    fragments: list[Fragment],
    *,
    fps: float,
    max_gap_s: float = 2.0,
    max_speed_ms: float = 8.0,
    pitch_length_m: float = PITCH_LENGTH_M_DEFAULT,
    slack: float = 0.02,
) -> dict[int, int]:
    """Greedy chain assembly: {orig track_id -> stitched id (chain's first id)}.

    Fragment B continues a chain if it starts 1..max_gap frames after the chain
    ends AND its start is within max_speed*gap + slack (length-normalized pitch
    units) of the chain end. Nearest-in-space eligible chain wins. Overlapping
    fragments never stitch (two people visible at once are not one person).
    """
    max_gap_frames = round(max_gap_s * fps)
    speed_pu = max_speed_ms / pitch_length_m  # pitch-length units per second
    chains: list[tuple[int, int, tuple[float, float]]] = []  # (sid, end_frame, end_xy)
    mapping: dict[int, int] = {}
    for frag in sorted(fragments, key=lambda f: (f.start_frame, f.track_id)):
        best_i = -1
        best_dist = float("inf")
        for i, (_, end_frame, end_xy) in enumerate(chains):
            gap = frag.start_frame - end_frame
            if gap < 1 or gap > max_gap_frames:
                continue
            dist = float(
                np.hypot(
                    float(frag.start_xy[0]) - float(end_xy[0]),
                    float(frag.start_xy[1]) - float(end_xy[1]),
                )
            )
            if dist > speed_pu * (gap / fps) + slack:
                continue
            if dist < best_dist:
                best_dist = dist
                best_i = i
        if best_i >= 0:
            sid, _, _ = chains[best_i]
            chains[best_i] = (sid, frag.end_frame, frag.end_xy)
            mapping[frag.track_id] = sid
        else:
            chains.append((frag.track_id, frag.end_frame, frag.end_xy))
            mapping[frag.track_id] = frag.track_id
    return mapping


def stitch_tracks(
    traj: pd.DataFrame,
    *,
    fps: float,
    classes: tuple[str, ...] = ("player", "goalkeeper"),
    max_gap_s: float = 2.0,
    max_speed_ms: float = 8.0,
    pitch_length_m: float = PITCH_LENGTH_M_DEFAULT,
    slack: float = 0.02,
) -> pd.DataFrame:
    """Rewrite track_id with stitched ids (per class); keep orig_track_id.

    Rows of other classes (ball, referee) pass through with track_id unchanged.
    """
    out = traj.copy()
    out["orig_track_id"] = out["track_id"]
    for cls in classes:
        sub = out[out["class"] == cls]
        if sub.empty:
            continue
        mapping = stitch_fragments(
            extract_fragments(sub),
            fps=fps,
            max_gap_s=max_gap_s,
            max_speed_ms=max_speed_ms,
            pitch_length_m=pitch_length_m,
            slack=slack,
        )
        mask = out["class"] == cls
        out.loc[mask, "track_id"] = (
            out.loc[mask, "orig_track_id"]
            .map(mapping)
            .fillna(out.loc[mask, "orig_track_id"])
            .astype("int64")
        )
    return out


def weighted_kmeans2(
    x: NDArray[np.floating],
    weights: NDArray[np.floating],
    *,
    n_iter: int = 50,
    seed: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Deterministic weighted 2-means (Lloyd's, farthest-point init).

    Returns (labels (n,), centroids (2, d)). Problem sizes are tiny (hundreds of
    tracks), so no sklearn dependency.
    """
    pts = np.asarray(x, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(pts)))
    d0 = np.linalg.norm(pts - pts[first], axis=1)
    second = int(d0.argmax())
    centroids = np.stack([pts[first], pts[second]])
    labels = np.full(len(pts), -1, dtype=np.int64)
    for _it in range(n_iter):
        d = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = d.argmin(axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in (0, 1):
            mask = labels == c
            if mask.any():
                centroids[c] = np.average(pts[mask], axis=0, weights=w[mask])
    return labels, centroids


def cluster_teams(
    features: dict[int, NDArray[np.floating]],
    weights: dict[int, float],
    *,
    boundary_ratio: float = 0.8,
    seed: int = 0,
) -> tuple[dict[int, int | None], NDArray[np.float64]]:
    """Cluster per-track kit features into 2 teams.

    Returns ({track_id: 0|1|None}, centroids). None = near the decision boundary
    (d_near/d_far > boundary_ratio) -> better unknown than wrong.
    """
    ids = sorted(features)
    x = np.stack([np.asarray(features[i], dtype=np.float64) for i in ids])
    w = np.array([weights[i] for i in ids], dtype=np.float64)
    labels, centroids = weighted_kmeans2(x, w, seed=seed)
    out: dict[int, int | None] = {}
    for row, tid in enumerate(ids):
        d = np.linalg.norm(centroids - x[row], axis=1)
        near = int(d.argmin())
        far = 1 - near
        if d[far] > 0 and float(d[near]) / float(d[far]) > boundary_ratio:
            out[tid] = None
        else:
            out[tid] = int(labels[row])
    return out, centroids


# BGR anchors for --own-kit color words; converted to Lab at runtime.
KIT_ANCHORS_BGR: dict[str, tuple[int, int, int]] = {
    "white": (245, 245, 245),
    "black": (20, 20, 20),
    "blue": (200, 80, 30),
    "dark blue": (110, 40, 10),
    "navy": (80, 30, 5),
    "red": (40, 40, 200),
    "yellow": (40, 210, 230),
    "green": (60, 160, 40),
    "orange": (30, 130, 240),
    "purple": (150, 50, 110),
}


def _lab_anchor(color_word: str) -> NDArray[np.float64]:
    key = color_word.strip().lower()
    if key not in KIT_ANCHORS_BGR:
        raise ValueError(
            f"unknown kit color {color_word!r}; known: {sorted(KIT_ANCHORS_BGR)}"
        )
    bgr = np.array([[KIT_ANCHORS_BGR[key]]], dtype=np.uint8)
    lab_raw = cast(NDArray[np.uint8], cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    return np.asarray(lab_raw[0, 0], dtype=np.float64)


def map_own_cluster(
    centroids: NDArray[np.floating],
    own_kit: str,
    *,
    warn_margin: float = 1.2,
) -> tuple[int, str | None]:
    """Pick the cluster whose SHIRT Lab centroid (first 3 dims) matches own_kit.

    Returns (own_cluster_index, warning|None). Warns when the two clusters are
    nearly equidistant from the hint (ratio < warn_margin) — contact sheets
    arbitrate then.
    """
    anchor = _lab_anchor(own_kit)
    shirt = np.asarray(centroids, dtype=np.float64)[:, :3]
    d = np.linalg.norm(shirt - anchor, axis=1)
    own = int(d.argmin())
    other = 1 - own
    warning: str | None = None
    if d[own] > 0 and float(d[other]) / float(d[own]) < warn_margin:
        warning = (
            f"--own-kit {own_kit!r} matches both clusters similarly "
            f"(d={d[own]:.1f} vs {d[other]:.1f}); verify the contact sheets"
        )
    return own, warning


def assign_goalkeepers(
    traj: pd.DataFrame, player_team_by_track: dict[int, str]
) -> dict[int, str]:
    """Assign each goalkeeper track the team whose players are nearest on average.

    Uses frames where the GK has pitch coords; compares mean distance (length-
    normalized) to own- vs opp-assigned players in the same frames. GK tracks
    with no usable frames or no co-visible teammates -> 'unknown'.
    """
    ar = PitchSpec.standard_9v9().aspect_ratio
    players = traj[traj["class"] == "player"].dropna(subset=["x_pitch", "y_pitch"]).copy()
    players["team_label"] = players["track_id"].map(
        lambda t: player_team_by_track.get(int(t), "unknown")
    )
    players = players[players["team_label"].isin(["own", "opp"])]
    out: dict[int, str] = {}
    gks = traj[traj["class"] == "goalkeeper"].dropna(subset=["x_pitch", "y_pitch"])
    for tid, g in gks.groupby("track_id", sort=True):
        merged = g.merge(players, on="frame", suffixes=("_gk", "_pl"))
        if merged.empty:
            out[int(cast(int, tid))] = "unknown"
            continue
        dx = (merged["x_pitch_gk"] - merged["x_pitch_pl"]) / ar
        dy = merged["y_pitch_gk"] - merged["y_pitch_pl"]
        merged = merged.copy()
        merged["dist"] = np.hypot(dx, dy)
        mean_by_team = merged.groupby("team_label")["dist"].mean()
        out[int(cast(int, tid))] = str(mean_by_team.idxmin())
    return out


def apply_team_labels(
    traj: pd.DataFrame, team_by_track: dict[int, str]
) -> pd.DataFrame:
    """Overwrite player/GK rows' team from {stitched track_id -> own/opp/unknown}.

    Unmapped player/GK tracks become 'unknown'. Ball/referee rows untouched.
    """
    out = traj.copy()
    mask = out["class"].isin(["player", "goalkeeper"])
    out.loc[mask, "team"] = out.loc[mask, "track_id"].map(
        lambda t: team_by_track.get(int(t), "unknown")
    )
    return out


def balance_gate(
    traj: pd.DataFrame, *, lo: float = 0.6, hi: float = 1.6
) -> tuple[float, bool]:
    """Own:opp player-detection ratio over all frames and whether it is in [lo, hi]."""
    players = traj[traj["class"].isin(["player", "goalkeeper"])]
    n_own = float((players["team"] == "own").sum())
    n_opp = float((players["team"] == "opp").sum())
    if n_opp == 0:
        return float("inf"), False
    ratio = n_own / n_opp
    return ratio, lo <= ratio <= hi


@dataclass(frozen=True)
class AgreementResult:
    """Frame-level possession agreement vs hand-labeled ground truth."""

    n_compared: int
    agreement: float
    disagreements: list[tuple[float, float, str, str]]  # (t_start, t_end, gt, pred)


def expand_ground_truth(gt: pd.DataFrame, t_seconds: pd.Series) -> pd.Series:
    """Expand change-point ground truth (t_seconds, possession) to a per-time series.

    Each row's state holds from its t_seconds until the next row. Times before
    the first row are 'none'.
    """
    gt_sorted = gt.sort_values("t_seconds")
    idx = (
        np.searchsorted(gt_sorted["t_seconds"].to_numpy(), t_seconds.to_numpy(), side="right") - 1
    )
    states = np.where(
        idx >= 0,
        gt_sorted["possession"].to_numpy()[np.clip(idx, 0, None)],
        "none",
    )
    return pd.Series(states, index=t_seconds.index)


def possession_agreement(gt: pd.DataFrame, phases: pd.DataFrame) -> AgreementResult:
    """Compare predicted possession_state with ground truth on team-attributed frames.

    Only frames where BOTH sides say own/opp are compared (loose/contested/
    unknown/none excluded). Disagreements are merged into contiguous spans.
    """
    pred = phases["possession_state"]
    gt_states = expand_ground_truth(gt, phases["t_seconds"])
    both = pred.isin(["own", "opp"]) & gt_states.isin(["own", "opp"])
    n = int(both.sum())
    if n == 0:
        return AgreementResult(0, 0.0, [])
    agree = pred[both] == gt_states[both]
    spans: list[tuple[float, float, str, str]] = []
    cur: tuple[float, str, str] | None = None
    last_t = 0.0
    for i in phases.index[both]:
        t = float(phases.loc[i, "t_seconds"])
        if bool(agree.loc[i]):
            if cur is not None:
                spans.append((cur[0], last_t, cur[1], cur[2]))
                cur = None
        else:
            g, p = str(gt_states.loc[i]), str(pred.loc[i])
            if cur is None or (cur[1], cur[2]) != (g, p):
                if cur is not None:
                    spans.append((cur[0], last_t, cur[1], cur[2]))
                cur = (t, g, p)
        last_t = t
    if cur is not None:
        spans.append((cur[0], last_t, cur[1], cur[2]))
    return AgreementResult(n, float(agree.mean()), spans)
