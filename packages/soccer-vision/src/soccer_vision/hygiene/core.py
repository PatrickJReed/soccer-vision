"""Pure core of the track-hygiene stage.

Operates on trajectories DataFrames that already carry x_pitch/y_pitch columns
(added by PitchMapper). Distances are isotropic length-normalized pitch units:
x_len = x_pitch / aspect_ratio, y_len = y_pitch (y is the goal-to-goal axis).

No I/O. See docs/superpowers/specs/2026-06-11-track-hygiene-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
                track_id=int(str(tid)),
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
