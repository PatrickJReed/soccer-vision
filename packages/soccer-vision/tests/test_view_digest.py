"""Tests for the classical view-digest (annotation-scaling Slice 1).

The pure core (frame_descriptors / similarity_matrix / cluster_views /
medoid_representatives / digest_frames) is exercised on synthetic feature-rich
frames — no video file needed. The video I/O + render path gets one integration
test guarded to skip if the platform's OpenCV cannot write an mp4.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from soccer_vision.labeler.view_digest import (
    ViewDigest,
    cluster_views,
    compute_view_digest,
    digest_frames,
    frame_descriptors,
    medoid_representatives,
    render_digest,
    similarity_matrix,
)

H, W = 240, 320


def _pattern(seed: int) -> NDArray[np.uint8]:
    """A deterministic, feature-rich BGR frame (bold shapes give ORB stable corners)."""
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


# three distinct views, each shown three times, temporally interleaved so a
# temporal-only method would fail but content clustering recovers the 3 views.
_ORDER = [0, 1, 2, 0, 1, 2, 0, 1, 2]
_FRAMES = [_pattern(v) for v in _ORDER]
_SAMPLE = [i * 25 for i in range(len(_ORDER))]  # pretend they were sampled at stride 25


# ----- similarity_matrix -------------------------------------------------------

def test_similarity_identical_high_distinct_low() -> None:
    descs, counts = frame_descriptors([_pattern(0), _pattern(0), _pattern(7)])
    s = similarity_matrix(descs, counts)
    assert s.shape == (3, 3)
    assert s[0, 1] > 0.9          # identical frames match almost perfectly
    assert s[0, 2] < 0.2          # unrelated frames barely match
    assert np.allclose(s, s.T)    # symmetric


def test_similarity_low_keypoints_is_zero() -> None:
    descs, counts = frame_descriptors([_pattern(0), _pattern(1)])
    s = similarity_matrix(descs, [counts[0], 3], min_keypoints=10)  # 2nd frame "blind"
    assert s[0, 1] == 0.0


def test_frame_descriptors_respects_mask() -> None:
    frame = _pattern(0)
    full = np.full((H, W), 255, np.uint8)
    blank = np.zeros((H, W), np.uint8)
    _, counts_full = frame_descriptors([frame], masks=[full])
    _, counts_masked = frame_descriptors([frame], masks=[blank])
    assert counts_full[0] > 0
    assert counts_masked[0] == 0


# ----- cluster_views -----------------------------------------------------------

def _block_similarity() -> NDArray[np.float64]:
    # 6 items: {0,1,2} one view, {3,4,5} another. High within, low across.
    s = np.full((6, 6), 0.05)
    for grp in ([0, 1, 2], [3, 4, 5]):
        for a in grp:
            for b in grp:
                s[a, b] = 0.8
    np.fill_diagonal(s, 1.0)
    return s


def test_cluster_views_recovers_blocks() -> None:
    labels = cluster_views(_block_similarity(), dist_threshold=0.7)
    assert len(set(labels.tolist())) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_cluster_views_ids_are_zero_based_by_first_appearance() -> None:
    labels = cluster_views(_block_similarity(), dist_threshold=0.7)
    assert labels[0] == 0            # first item -> view 0
    assert set(labels.tolist()) == {0, 1}


def test_cluster_views_threshold_controls_granularity() -> None:
    s = _block_similarity()
    loose = cluster_views(s, dist_threshold=0.99)   # everything merges
    strict = cluster_views(s, dist_threshold=0.1)   # everything splits
    assert len(set(loose.tolist())) < len(set(strict.tolist()))


def test_cluster_views_single_frame() -> None:
    labels = cluster_views(np.array([[1.0]]), dist_threshold=0.7)
    assert labels.tolist() == [0]


# ----- medoid_representatives --------------------------------------------------

def test_medoid_picks_most_central_member() -> None:
    # view 0 = {0,1,2}; item 1 is most similar to the others -> medoid.
    s = np.array([
        [1.0, 0.9, 0.3],
        [0.9, 1.0, 0.8],
        [0.3, 0.8, 1.0],
    ])
    labels = np.array([0, 0, 0])
    reps = medoid_representatives(labels, s, [100, 200, 300])
    assert reps == {0: 200}


def test_medoid_singleton_returns_itself() -> None:
    reps = medoid_representatives(np.array([0, 1]), np.eye(2), [10, 20])
    assert reps == {0: 10, 1: 20}


# ----- digest_frames (end-to-end pure core) ------------------------------------

def test_digest_frames_recovers_three_views() -> None:
    digest = digest_frames(_FRAMES, _SAMPLE, dist_threshold=0.7)
    assert isinstance(digest, ViewDigest)
    assert len(set(digest.view_of.values())) == 3
    # every sampled frame is assigned
    assert set(digest.view_of) == set(_SAMPLE)
    # frames of the same underlying pattern share a view
    by_pattern: dict[int, set[int]] = {}
    for frame, order in zip(_SAMPLE, _ORDER, strict=True):
        by_pattern.setdefault(order, set()).add(digest.view_of[frame])
    assert all(len(v) == 1 for v in by_pattern.values())


def test_digest_representatives_are_sampled_frames() -> None:
    digest = digest_frames(_FRAMES, _SAMPLE, dist_threshold=0.7)
    assert set(digest.representatives) == set(digest.view_of.values())
    assert all(rep in _SAMPLE for rep in digest.representatives.values())


def test_digest_similarity_shape() -> None:
    digest = digest_frames(_FRAMES, _SAMPLE, dist_threshold=0.7)
    assert digest.similarity.shape == (len(_SAMPLE), len(_SAMPLE))


# ----- video I/O + render (integration) ----------------------------------------

def _write_video(path: Path, frames: list[NDArray[np.uint8]]) -> bool:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    vw = cv2.VideoWriter(str(path), fourcc, 30.0, (W, H))
    if not vw.isOpened():
        return False
    for fr in frames:
        vw.write(fr)
    vw.release()
    return path.exists() and path.stat().st_size > 0


def test_compute_and_render_from_video(tmp_path: Path) -> None:
    # 30 frames: 3 views in contiguous blocks of 10.
    frames = [_pattern(v) for v in ([0] * 10 + [1] * 10 + [2] * 10)]
    video = tmp_path / "clip.mp4"
    if not _write_video(video, frames):
        pytest.skip("platform OpenCV cannot write mp4")

    digest = compute_view_digest(video, stride=3, dist_threshold=0.7,
                                 cache_dir=tmp_path)
    n_views = len(set(digest.view_of.values()))
    assert 2 <= n_views <= 5           # ~3 distinct views recovered (tolerant to codec noise)
    assert len(digest.sample_frames) == len(range(0, 30, 3))

    out = tmp_path / "digest_out"
    paths = render_digest(digest, video, out)
    assert any(p.name == "views_montage.png" for p in paths)
    assert any(p.name == "similarity.png" for p in paths)
    assert all(p.exists() for p in paths)

    data = json.loads((out / "view_digest.json").read_text())
    assert data["n_views"] == n_views
    assert len(data["representatives"]) == n_views
