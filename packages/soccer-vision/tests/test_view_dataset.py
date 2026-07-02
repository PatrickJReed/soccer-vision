"""Tests for the view-dataset exporter (annotation-scaling Slice 1.5)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from soccer_vision.labeler.view_dataset import (
    assign_nearest_view,
    assign_splits,
    build_manifest,
    cross_match_fractions,
    smooth_view_sequence,
)
from soccer_vision.labeler.view_digest import (
    _pair_match_fraction,
    frame_descriptors,
    similarity_matrix,
)

H, W = 240, 320


def _pattern(seed: int) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 30, np.uint8)
    for _ in range(45):
        x1, x2 = sorted(rng.integers(0, W, size=2).tolist())
        y1, y2 = sorted(rng.integers(0, H, size=2).tolist())
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        else:
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
    return img


def test_pair_match_fraction_matches_similarity_matrix() -> None:
    descs, counts = frame_descriptors([_pattern(0), _pattern(1), _pattern(0)])
    s = similarity_matrix(descs, counts)
    assert abs(_pair_match_fraction(descs[0], descs[1], counts[0], counts[1]) - s[0, 1]) < 1e-12
    assert abs(_pair_match_fraction(descs[0], descs[2], counts[0], counts[2]) - s[0, 2]) < 1e-12


def test_cross_match_fractions_parity_with_similarity_matrix() -> None:
    frames = [_pattern(5), _pattern(1), _pattern(2), _pattern(3)]
    q, refs = frames[:1], frames[1:]
    qd, qc = frame_descriptors(q)
    rd, rc = frame_descriptors(refs)
    cross = cross_match_fractions(qd, qc, rd, rc)          # (1, 3)
    full = similarity_matrix(*frame_descriptors(q + refs))  # (4, 4)
    assert cross.shape == (1, 3)
    assert np.allclose(cross[0], full[0, 1:], atol=1e-12)


def test_cross_match_fractions_low_keypoints_zero() -> None:
    qd, qc = frame_descriptors([_pattern(0)])
    rd, _rc = frame_descriptors([_pattern(1)])
    cross = cross_match_fractions(qd, qc, rd, [3], min_keypoints=10)  # ref "blind"
    assert cross[0, 0] == 0.0


def test_assign_nearest_view_fields() -> None:
    match = np.array([
        [0.8, 0.3, 0.1],   # -> view 0, second view 1, conf .8, margin .5
        [0.2, 0.2, 0.9],   # -> view 2, conf .9, margin .7
        [0.0, 0.0, 0.0],   # -> unassigned
    ])
    df = assign_nearest_view(match, [0, 1, 2])
    assert list(df["view_id_raw"]) == [0, 2, -1]
    assert df.loc[0, "view_second"] == 1
    assert abs(df.loc[0, "confidence"] - 0.8) < 1e-9
    assert abs(df.loc[0, "margin"] - 0.5) < 1e-9      # best - best-of-a-different-view
    assert df.loc[2, "confidence"] == 0.0 and df.loc[2, "view_second"] == -1


def test_assign_margin_is_best_minus_other_view_not_other_rep() -> None:
    # two reps share view 0; a third is view 1. margin compares across VIEWS.
    match = np.array([[0.8, 0.75, 0.1]])
    df = assign_nearest_view(match, [0, 0, 1])
    assert df.loc[0, "view_id_raw"] == 0
    assert abs(df.loc[0, "margin"] - (0.8 - 0.1)) < 1e-9


def test_smooth_view_sequence_removes_singleton() -> None:
    seq = [0, 0, 0, 1, 0, 0, 0]
    out = smooth_view_sequence(seq, window=5)
    assert list(out) == [0, 0, 0, 0, 0, 0, 0]


def test_smooth_window_one_is_identity() -> None:
    seq = [0, 1, 0, 2]
    assert list(smooth_view_sequence(seq, window=1)) == seq


def test_smooth_tie_keeps_original() -> None:
    seq = [0, 0, 1, 1]
    out = smooth_view_sequence(seq, window=3)
    assert list(out) == [0, 0, 1, 1]


def test_smooth_real_tie_keeps_center_when_among_winners() -> None:
    # window=3 at idx1 sees {0,1,2}: all tie at count 1; center label 1 is a winner -> kept.
    out = smooth_view_sequence([0, 1, 2], window=3)
    assert out[1] == 1


def test_smooth_tie_center_not_winner_falls_back_to_min() -> None:
    # window=5 at idx2 sees {4,4,7,5,5}: 4 and 5 tie at count 2, center 7 is NOT a winner
    # -> deterministic fallback to the smallest winner, 4.
    out = smooth_view_sequence([4, 4, 7, 5, 5], window=5)
    assert out[2] == 4


def test_assign_splits_per_view_tail_every_view_in_both() -> None:
    df = pd.DataFrame({"frame": list(range(20)), "view_id": [0]*10 + [1]*10})
    out = assign_splits(df, val_frac=0.2, policy="per_view_tail")
    for v in (0, 1):
        sub = out[out["view_id"] == v]
        assert (sub["split"] == "train").any() and (sub["split"] == "val").any()
        assert (sub["split"] == "val").sum() == 2
    assert set(out[(out["view_id"] == 0) & (out["split"] == "val")]["frame"]) == {8, 9}


def test_assign_splits_holdout_views() -> None:
    df = pd.DataFrame({"frame": list(range(20)), "view_id": [0]*10 + [1]*10})
    out = assign_splits(df, policy="holdout_views", holdout_views={1})
    assert (out[out["view_id"] == 1]["split"] == "val").all()
    assert (out[out["view_id"] == 0]["split"] == "train").all()


def test_build_manifest_schema_and_content() -> None:
    match = np.array([[0.8, 0.2], [0.15, 0.7], [0.75, 0.1]])   # frames -> views [0,1,0]
    df = build_manifest(query_frames=[0, 5, 10], match=match, ref_view_ids=[0, 1],
                        keypoint_counts=[300, 280, 310], game="oceanside", fps=30.0,
                        ambiguity_margin=0.05, smooth_window=1, val_frac=0.34)
    assert list(df.columns) == ["game", "frame", "t_seconds", "view_id", "view_id_raw",
        "view_second", "view_key", "confidence", "weight", "margin", "ambiguous",
        "n_keypoints", "n_boxes", "split"]
    assert df["view_id"].dtype == np.int32 and df["confidence"].dtype == np.float32
    assert list(df["view_id"]) == [0, 1, 0]
    assert list(df["view_key"]) == ["oceanside:0", "oceanside:1", "oceanside:0"]
    assert (df["weight"].to_numpy() == df["confidence"].to_numpy()).all()
    assert abs(df.loc[0, "t_seconds"] - 0.0) < 1e-9 and abs(df.loc[2, "t_seconds"] - 10/30) < 1e-6
    assert list(df["n_boxes"]) == [0, 0, 0]              # None -> 0
    assert df["frame"].is_monotonic_increasing


def test_build_manifest_ambiguous_flag() -> None:
    match = np.array([[0.50, 0.49]])                    # margin 0.01 < 0.05 -> ambiguous
    df = build_manifest([0], match, [0, 1], [300], game="g", fps=30.0, ambiguity_margin=0.05)
    assert bool(df.loc[0, "ambiguous"]) is True


def test_build_manifest_smoothing_preserves_raw() -> None:
    match = np.array([[0.9, 0.1]]*3 + [[0.1, 0.9]] + [[0.9, 0.1]]*3)  # lone view1 at idx3
    df = build_manifest(list(range(7)), match, [0, 1], [300]*7, game="g", fps=30.0,
                        smooth_window=5)
    assert df.loc[3, "view_id_raw"] == 1 and df.loc[3, "view_id"] == 0   # raw kept, smoothed fixed


def test_build_manifest_records_boxes() -> None:
    match = np.array([[0.8, 0.2]])
    df = build_manifest([0], match, [0, 1], [300], game="g", fps=30.0, n_boxes=[4])
    assert df.loc[0, "n_boxes"] == 4


def test_build_manifest_unsorted_input_matches_sorted() -> None:
    # smoothing must run in frame order: scrambling the Q-aligned rows must not change
    # the per-frame smoothed view_id / view_key / split.
    frames = list(range(7))
    match = np.array([[0.9, 0.1]] * 3 + [[0.1, 0.9]] + [[0.9, 0.1]] * 3)  # lone view1 at idx3
    kp = [300, 301, 302, 303, 304, 305, 306]
    sorted_df = build_manifest(frames, match, [0, 1], kp, game="g", fps=30.0, smooth_window=5)
    perm = [3, 0, 6, 1, 5, 2, 4]
    scrambled = build_manifest([frames[i] for i in perm], match[perm], [0, 1],
                               [kp[i] for i in perm], game="g", fps=30.0, smooth_window=5)
    for col in ("frame", "view_id", "view_id_raw", "view_key", "n_keypoints", "split"):
        assert list(sorted_df[col]) == list(scrambled[col]), col
