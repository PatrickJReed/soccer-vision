"""Tests for the hygiene pure core: stitching, clustering, teams, gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from soccer_vision.hygiene.core import Fragment, extract_fragments, stitch_tracks


def _rows(
    track_id: int,
    frames: list[int],
    xy: list[tuple[float, float]],
    cls: str = "player",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": frames,
            "t_seconds": [f / 30.0 for f in frames],
            "track_id": track_id,
            "x_pitch": [p[0] for p in xy],
            "y_pitch": [p[1] for p in xy],
            "class": cls,
        }
    )


def test_extract_fragments_uses_first_last_valid_pitch_rows() -> None:
    df = _rows(7, [10, 11, 12], [(0.5, 0.2), (np.nan, np.nan), (0.5, 0.3)])
    frags = extract_fragments(df)
    assert len(frags) == 1
    f = frags[0]
    assert isinstance(f, Fragment)
    assert (f.track_id, f.start_frame, f.end_frame) == (7, 10, 12)
    # x is length-normalized: 0.5 / 1.5 aspect
    assert np.isclose(f.start_xy[0], 0.5 / 1.5)
    assert np.isclose(f.end_xy[1], 0.3)


def test_extract_fragments_skips_tracks_without_pitch_coords() -> None:
    df = _rows(7, [10, 11], [(np.nan, np.nan), (np.nan, np.nan)])
    assert extract_fragments(df) == []


def test_stitch_joins_close_fragments() -> None:
    # same player: fragment A frames 0-10 ending at (0.5,0.5); B starts frame 20
    # (0.33s later) a tiny distance away -> stitched.
    a = _rows(1, [0, 10], [(0.5, 0.5), (0.5, 0.5)])
    b = _rows(2, [20, 30], [(0.5, 0.52), (0.5, 0.6)])
    df = pd.concat([a, b], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    assert out.loc[out.orig_track_id == 2, "track_id"].unique().tolist() == [1]
    assert out.loc[out.orig_track_id == 1, "track_id"].unique().tolist() == [1]


def test_stitch_refuses_teleport() -> None:
    # B starts 0.33s later but across the pitch -> speed bound violated -> no stitch.
    a = _rows(1, [0, 10], [(0.5, 0.1), (0.5, 0.1)])
    b = _rows(2, [20, 30], [(0.5, 0.9), (0.5, 0.9)])
    df = pd.concat([a, b], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    assert out.track_id.nunique() == 2


def test_stitch_refuses_long_gap() -> None:
    # B starts 4s later (> max_gap_s=2.0) right next door -> no stitch.
    a = _rows(1, [0, 10], [(0.5, 0.5), (0.5, 0.5)])
    b = _rows(2, [130, 140], [(0.5, 0.51), (0.5, 0.51)])
    df = pd.concat([a, b], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    assert out.track_id.nunique() == 2


def test_stitch_refuses_overlapping_fragments() -> None:
    # two people visible simultaneously can't be the same track.
    a = _rows(1, [0, 20], [(0.5, 0.5), (0.5, 0.5)])
    b = _rows(2, [10, 30], [(0.5, 0.52), (0.5, 0.52)])
    df = pd.concat([a, b], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    assert out.track_id.nunique() == 2


def test_stitch_picks_nearest_candidate() -> None:
    a = _rows(1, [0, 10], [(0.5, 0.5), (0.5, 0.5)])
    near = _rows(2, [20, 30], [(0.5, 0.52), (0.5, 0.52)])
    far = _rows(3, [20, 30], [(0.5, 0.58), (0.5, 0.58)])
    df = pd.concat([a, near, far], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    by_orig = out.groupby("orig_track_id")["track_id"].first()
    assert by_orig[2] == 1          # nearest joined the chain
    assert by_orig[3] == 3          # other stays its own chain


def test_stitch_classes_do_not_mix() -> None:
    a = _rows(1, [0, 10], [(0.5, 0.5), (0.5, 0.5)], cls="player")
    b = _rows(2, [20, 30], [(0.5, 0.52), (0.5, 0.52)], cls="goalkeeper")
    df = pd.concat([a, b], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    assert out.track_id.nunique() == 2
