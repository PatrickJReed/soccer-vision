"""Tests for the hygiene pure core: stitching, clustering, teams, gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from soccer_vision.hygiene.core import (
    Fragment,
    apply_team_labels,
    assign_goalkeepers,
    balance_gate,
    cluster_teams,
    expand_ground_truth,
    extract_fragments,
    map_own_cluster,
    possession_agreement,
    stitch_tracks,
    weighted_kmeans2,
)


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


def test_weighted_kmeans2_separates_two_blobs() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal((0, 0, 0, 0, 0, 0), 0.5, size=(20, 6))
    b = rng.normal((10, 10, 10, 10, 10, 10), 0.5, size=(20, 6))
    x = np.vstack([a, b])
    w = np.ones(40)
    labels, centroids = weighted_kmeans2(x, w, seed=0)
    assert set(labels[:20].tolist()) != set(labels[20:].tolist())
    assert len(set(labels.tolist())) == 2
    assert centroids.shape == (2, 6)


def test_weighted_kmeans2_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, size=(30, 6))
    w = np.ones(30)
    l1, _ = weighted_kmeans2(x, w, seed=7)
    l2, _ = weighted_kmeans2(x, w, seed=7)
    assert np.array_equal(l1, l2)


def test_cluster_teams_boundary_tracks_are_unknown() -> None:
    # two tight blobs + one feature midway between them. The midpoint gets a
    # tiny weight so it cannot drag its cluster's centroid (mirrors real data,
    # where dozens of tracks anchor each centroid) -> ratio ~1 -> unknown.
    feats = {
        1: np.zeros(6), 2: np.zeros(6) + 0.1,
        3: np.full(6, 10.0), 4: np.full(6, 10.1),
        5: np.full(6, 5.0),  # equidistant midpoint
    }
    weights = {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 0.01}
    teams, _centroids = cluster_teams(feats, weights, seed=0)
    assert teams[5] is None
    assert teams[1] is not None and teams[3] is not None
    assert teams[1] != teams[3]


def test_map_own_cluster_picks_nearer_shirt_color() -> None:
    # cluster 0 shirt ~ white (Lab L high, a/b neutral), cluster 1 ~ dark blue.
    centroids = np.array([
        [250.0, 128.0, 128.0, 100.0, 128.0, 128.0],   # white shirt
        [40.0, 130.0, 80.0, 180.0, 120.0, 190.0],     # dark-blue shirt
    ])
    own, warning = map_own_cluster(centroids, "white")
    assert own == 0
    assert warning is None


def test_map_own_cluster_unknown_color_word_raises() -> None:
    centroids = np.zeros((2, 6))
    try:
        map_own_cluster(centroids, "tartan")
    except ValueError as e:
        assert "tartan" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_stitch_true_tie_break_prefers_nearer_of_two_eligible() -> None:
    # two ELIGIBLE chains (both within speed bound); the nearer one wins.
    a = _rows(1, [0, 10], [(0.5, 0.50), (0.5, 0.50)])
    b = _rows(2, [0, 10], [(0.5, 0.56), (0.5, 0.56)])
    c = _rows(3, [20, 30], [(0.5, 0.51), (0.5, 0.51)])  # near a's end, eligible for both
    df = pd.concat([a, b, c], ignore_index=True)
    out = stitch_tracks(df, fps=30.0)
    by_orig = out.groupby("orig_track_id")["track_id"].first()
    assert by_orig[3] == 1  # joined the NEARER chain (a), not b


def _scene_with_gks() -> pd.DataFrame:
    rows = []
    for f in range(3):
        for i, (tid, team_cluster) in enumerate([(1, 0), (2, 0), (3, 1), (4, 1)]):
            y = 0.3 if team_cluster == 0 else 0.7
            rows.append({"frame": f, "t_seconds": f / 30.0, "track_id": tid,
                         "x_pitch": 0.4 + 0.05 * i, "y_pitch": y, "class": "player"})
        rows.append({"frame": f, "t_seconds": f / 30.0, "track_id": 90,
                     "x_pitch": 0.5, "y_pitch": 0.05, "class": "goalkeeper"})
        rows.append({"frame": f, "t_seconds": f / 30.0, "track_id": 91,
                     "x_pitch": 0.5, "y_pitch": 0.95, "class": "goalkeeper"})
    return pd.DataFrame(rows)


def test_assign_goalkeepers_by_nearest_team() -> None:
    df = _scene_with_gks()
    teams = {1: "own", 2: "own", 3: "opp", 4: "opp"}
    gk = assign_goalkeepers(df, teams)
    assert gk[90] == "own"   # near the y=0.3 group
    assert gk[91] == "opp"


def test_assign_goalkeepers_unknown_without_covisible_players() -> None:
    rows = [{"frame": 0, "t_seconds": 0.0, "track_id": 90,
             "x_pitch": 0.5, "y_pitch": 0.05, "class": "goalkeeper"}]
    gk = assign_goalkeepers(pd.DataFrame(rows), {})
    assert gk[90] == "unknown"


def test_apply_team_labels_inherits_and_passes_through() -> None:
    df = _scene_with_gks()
    df["team"] = "stale"
    teams = {1: "own", 2: "own", 3: "opp", 4: "opp", 90: "own", 91: "opp"}
    out = apply_team_labels(df, teams)
    assert (out.loc[out.track_id == 1, "team"] == "own").all()
    assert (out.loc[out.track_id == 91, "team"] == "opp").all()


def test_apply_team_labels_unknown_for_unmapped() -> None:
    df = _scene_with_gks()
    df["team"] = "stale"
    out = apply_team_labels(df, {1: "own"})
    assert (out.loc[out.track_id == 3, "team"] == "unknown").all()


def test_apply_team_labels_leaves_ball_and_ref_untouched() -> None:
    df = _scene_with_gks()
    ball = pd.DataFrame([{"frame": 0, "t_seconds": 0.0, "track_id": -1,
                          "x_pitch": 0.5, "y_pitch": 0.5, "class": "ball"}])
    df = pd.concat([df, ball], ignore_index=True)
    df["team"] = "keepme"
    out = apply_team_labels(df, {1: "own"})
    assert (out.loc[out["class"] == "ball", "team"] == "keepme").all()


def test_balance_gate() -> None:
    df = _scene_with_gks()
    df = apply_team_labels(df, {1: "own", 2: "own", 3: "opp", 4: "opp"})
    ratio, passed = balance_gate(df)
    assert np.isclose(ratio, 1.0)
    assert passed
    df_bad = apply_team_labels(df, {1: "own", 2: "opp", 3: "opp", 4: "opp"})
    _ratio_bad, passed_bad = balance_gate(df_bad)
    assert not passed_bad


def test_map_own_cluster_warns_when_ambiguous() -> None:
    # two centroids equally near the hint -> warn_margin fires.
    # White anchor Lab is [246, 128, 128]; place both centroids equidistant.
    centroids = np.array([
        [247.0, 128.0, 128.0, 0.0, 0.0, 0.0],
        [245.0, 128.0, 128.0, 0.0, 0.0, 0.0],
    ])
    _, warning = map_own_cluster(centroids, "white", warn_margin=1.2)
    assert warning is not None
    assert "verify" in warning


def test_expand_ground_truth_change_points() -> None:
    gt = pd.DataFrame({"t_seconds": [0.0, 4.0, 10.0],
                       "possession": ["own", "opp", "none"]})
    t = pd.Series([0.0, 2.0, 4.0, 9.9, 10.0, 12.0])
    states = expand_ground_truth(gt, t)
    assert states.tolist() == ["own", "own", "opp", "opp", "none", "none"]


def test_possession_agreement_counts_team_frames_only() -> None:
    phases = pd.DataFrame({
        "frame": range(6),
        "t_seconds": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "possession_state": ["own", "own", "opp", "loose_ball", "opp", "own"],
    })
    gt = pd.DataFrame({"t_seconds": [0.0, 2.0],
                       "possession": ["own", "opp"]})
    res = possession_agreement(gt, phases)
    # comparable frames: t=0,1 (own/own, own/own), 2,4 (opp/opp, opp/opp), 5 (opp gt vs own pred)
    # loose_ball frame excluded.
    assert res.n_compared == 5
    assert np.isclose(res.agreement, 4 / 5)
    assert len(res.disagreements) == 1
    _t0, _t1, gt_s, pred_s = res.disagreements[0]
    assert (gt_s, pred_s) == ("opp", "own")


def test_possession_agreement_empty_when_nothing_comparable() -> None:
    phases = pd.DataFrame({
        "frame": range(3),
        "t_seconds": [0.0, 1.0, 2.0],
        "possession_state": ["loose_ball", "loose_ball", "contested"],
    })
    gt = pd.DataFrame({"t_seconds": [0.0], "possession": ["own"]})
    res = possession_agreement(gt, phases)
    assert res.n_compared == 0
    assert res.disagreements == []


def test_possession_agreement_span_closes_on_agree() -> None:
    # disagree, disagree, agree -> one span covering the two disagreeing frames.
    phases = pd.DataFrame({
        "frame": range(3),
        "t_seconds": [0.0, 1.0, 2.0],
        "possession_state": ["own", "own", "opp"],
    })
    gt = pd.DataFrame({"t_seconds": [0.0], "possession": ["opp"]})
    res = possession_agreement(gt, phases)
    assert res.n_compared == 3
    assert res.disagreements == [(0.0, 1.0, "opp", "own")]


def test_possession_agreement_pair_change_opens_new_span() -> None:
    # gt own vs pred opp, then gt opp vs pred own -> two distinct spans.
    phases = pd.DataFrame({
        "frame": range(2),
        "t_seconds": [0.0, 1.0],
        "possession_state": ["opp", "own"],
    })
    gt = pd.DataFrame({"t_seconds": [0.0, 1.0], "possession": ["own", "opp"]})
    res = possession_agreement(gt, phases)
    assert res.n_compared == 2
    assert res.disagreements == [(0.0, 0.0, "own", "opp"), (1.0, 1.0, "opp", "own")]


def test_expand_ground_truth_before_first_row_is_none() -> None:
    gt = pd.DataFrame({"t_seconds": [5.0], "possession": ["own"]})
    t = pd.Series([0.0, 4.9, 5.0])
    assert expand_ground_truth(gt, t).tolist() == ["none", "none", "own"]


def test_balance_gate_no_opp_returns_inf_false() -> None:
    df = pd.DataFrame([{"class": "player", "team": "own"}])
    ratio, passed = balance_gate(df)
    assert ratio == float("inf")
    assert not passed


def test_map_own_cluster_invariant_to_kmeans_label_order() -> None:
    """KMeans labels its clusters in an arbitrary order; map_own_cluster must pick the SAME
    physical kit regardless. Same two kits, two centroid orderings -> identical chosen centroid."""
    white = [250.0, 128.0, 128.0, 100.0, 128.0, 128.0]
    blue = [40.0, 130.0, 80.0, 180.0, 120.0, 190.0]
    c01 = np.array([white, blue])
    c10 = np.array([blue, white])
    own01, _ = map_own_cluster(c01, "white")
    own10, _ = map_own_cluster(c10, "white")
    assert own01 != own10                       # index flips with the label order
    assert np.allclose(c01[own01], c10[own10])  # but the chosen physical centroid is identical
