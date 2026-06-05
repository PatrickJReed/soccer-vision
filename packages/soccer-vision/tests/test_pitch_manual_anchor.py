"""Tests for the manual-anchor point-propagation core."""

from __future__ import annotations

import numpy as np
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import (
    Click,
    FrameFit,
    build_segments,
    clicks_to_keypoints_df,
    coverage_fraction,
    cumulative_transforms,
    fit_frame_homographies,
    frame_status,
    map_point,
    to_homography_entries,
)

_SCALE = 1000.0
_FIT_IDXS = [0, 3, 6, 11, 16, 19]


def test_click_and_framefit_fields() -> None:
    c = Click(frame=3, kp_idx=0, x=10.0, y=20.0)
    assert (c.frame, c.kp_idx, c.x, c.y) == (3, 0, 10.0, 20.0)
    f = FrameFit(H=np.eye(3), residual=0.01, n_points=5)
    assert f.n_points == 5 and f.residual == 0.01
    assert np.array_equal(f.H, np.eye(3))


def test_segments_single_connected_run() -> None:
    interframe = {0: np.eye(3), 1: np.eye(3), 2: np.eye(3)}  # links 0-1-2-3
    seg = build_segments(interframe, n_frames=4)
    assert seg == {0: 0, 1: 0, 2: 0, 3: 0}


def test_segments_split_on_missing_link() -> None:
    interframe = {0: np.eye(3), 2: np.eye(3)}  # link 0-1, gap at 1-2, link 2-3
    seg = build_segments(interframe, n_frames=4)
    assert seg == {0: 0, 1: 0, 2: 1, 3: 1}


def test_segments_all_isolated() -> None:
    seg = build_segments({}, n_frames=3)
    assert seg == {0: 0, 1: 1, 2: 2}


def test_cumulative_identity_chain() -> None:
    interframe = {0: np.eye(3), 1: np.eye(3)}
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    for f in range(3):
        assert np.allclose(M[f], np.eye(3))


def test_cumulative_translation_chain() -> None:
    # each frame shifts +10px in x relative to the previous (i -> i+1).
    g = np.eye(3)
    g[0, 2] = 10.0
    interframe = {0: g, 1: g}
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    # M[f] maps frame f -> reference(0). frame 2 is +20px from frame 0, so
    # mapping a point back to ref subtracts 20.
    assert np.allclose(M[2] @ np.array([20.0, 0.0, 1.0]), [0.0, 0.0, 1.0])


def test_cumulative_resets_per_segment() -> None:
    g = np.eye(3)
    g[0, 2] = 10.0
    interframe = {0: g}  # link 0-1 only; frame 2 is a new segment
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    assert np.allclose(M[2], np.eye(3))  # segment start -> identity


def test_map_point_through_translation() -> None:
    g = np.eye(3)
    g[0, 2] = 10.0
    interframe = {0: g, 1: g}
    seg = build_segments(interframe, 3)
    M = cumulative_transforms(interframe, seg)
    # a point at x=5 in frame 0 appears at x=25 in frame 2 (camera moved +20).
    x, y = map_point(M[0], M[2], 5.0, 0.0)
    assert np.isclose(x, 25.0) and np.isclose(y, 0.0)


def _identity_chain(n: int) -> dict[int, np.ndarray]:
    return {i: np.eye(3) for i in range(n - 1)}


def _clicks_one_per_frame() -> list[Click]:
    clicks = []
    for f, idx in enumerate(_FIT_IDXS):
        px, py = PITCH_LANDMARKS[idx] * _SCALE
        clicks.append(Click(frame=f, kp_idx=idx, x=float(px), y=float(py)))
    return clicks


def test_fit_recovers_homography_from_spread_clicks() -> None:
    n = 6
    interframe = _identity_chain(n)
    seg = build_segments(interframe, n)
    transforms = cumulative_transforms(interframe, seg)
    fits = fit_frame_homographies(
        _clicks_one_per_frame(), transforms, seg, PITCH_LANDMARKS, window=10
    )
    assert set(fits) == set(range(n))
    f3 = fits[3]
    assert f3.n_points == 6
    assert f3.residual < 1e-6
    pt = np.array([PITCH_LANDMARKS[0, 0] * _SCALE, PITCH_LANDMARKS[0, 1] * _SCALE, 1.0])
    mapped = f3.H @ pt
    mapped = mapped[:2] / mapped[2]
    assert np.allclose(mapped, PITCH_LANDMARKS[0], atol=1e-6)


def test_window_excludes_distant_clicks() -> None:
    n = 6
    interframe = _identity_chain(n)
    seg = build_segments(interframe, n)
    transforms = cumulative_transforms(interframe, seg)
    fits = fit_frame_homographies(
        _clicks_one_per_frame(), transforms, seg, PITCH_LANDMARKS, window=1
    )
    assert 5 not in fits


def test_fewer_than_four_landmarks_uncovered() -> None:
    n = 3
    interframe = _identity_chain(n)
    seg = build_segments(interframe, n)
    transforms = cumulative_transforms(interframe, seg)
    clicks = [
        Click(0, _FIT_IDXS[0], *(PITCH_LANDMARKS[_FIT_IDXS[0]] * _SCALE)),
        Click(0, _FIT_IDXS[1], *(PITCH_LANDMARKS[_FIT_IDXS[1]] * _SCALE)),
        Click(0, _FIT_IDXS[2], *(PITCH_LANDMARKS[_FIT_IDXS[2]] * _SCALE)),
    ]
    fits = fit_frame_homographies(clicks, transforms, seg, PITCH_LANDMARKS, window=10)
    assert fits == {}


def test_clicks_do_not_cross_segments() -> None:
    interframe = {0: np.eye(3)}  # links 0-1; frame 2 isolated
    seg = build_segments(interframe, 3)
    transforms = cumulative_transforms(interframe, seg)
    clicks = [
        Click(0 if i < 2 else 1, idx, *(PITCH_LANDMARKS[idx] * _SCALE))
        for i, idx in enumerate(_FIT_IDXS[:4])
    ]
    fits = fit_frame_homographies(clicks, transforms, seg, PITCH_LANDMARKS, window=10)
    assert 2 not in fits
    assert 0 in fits and 1 in fits


def test_degenerate_collinear_clicks_yield_large_residual() -> None:
    # 4 landmarks whose image points are collinear cannot define a valid pitch
    # homography; the fit's residual is far above the coverage threshold (0.05),
    # so the residual gate downstream rejects it even though no error is raised.
    interframe = _identity_chain(2)
    seg = build_segments(interframe, 2)
    transforms = cumulative_transforms(interframe, seg)
    clicks = [
        Click(0, _FIT_IDXS[0], 0.0, 0.0),
        Click(0, _FIT_IDXS[1], 1.0, 1.0),
        Click(0, _FIT_IDXS[2], 2.0, 2.0),
        Click(0, _FIT_IDXS[3], 3.0, 3.0),
    ]
    fits = fit_frame_homographies(clicks, transforms, seg, PITCH_LANDMARKS, window=10)
    assert 0 in fits
    assert fits[0].residual > 0.05


def _fits() -> dict[int, FrameFit]:
    return {
        0: FrameFit(np.eye(3), residual=0.01, n_points=6),
        1: FrameFit(np.eye(3), residual=0.09, n_points=5),
    }


def test_frame_status_green_yellow_red() -> None:
    status = frame_status(_fits(), n_frames=3, residual_threshold=0.05)
    assert status == {0: "green", 1: "yellow", 2: "red"}


def test_coverage_fraction_counts_green_only() -> None:
    assert coverage_fraction(_fits(), n_frames=3, residual_threshold=0.05) == 1 / 3


def test_to_homography_entries_keeps_green_with_source_manual() -> None:
    entries = to_homography_entries(_fits(), residual_threshold=0.05)
    assert set(entries) == {0}
    assert entries[0].source == "manual"
    assert 0.0 <= entries[0].confidence <= 1.0


def test_clicks_to_keypoints_df_schema() -> None:
    clicks = [Click(2, 0, 10.0, 20.0), Click(5, 3, 30.0, 40.0)]
    df = clicks_to_keypoints_df(clicks)
    assert list(df.columns) == ["frame", "kp_idx", "x_px", "y_px", "conf"]
    assert (df["conf"] == 1.0).all()
    assert df.iloc[0].to_dict() == {
        "frame": 2, "kp_idx": 0, "x_px": 10.0, "y_px": 20.0, "conf": 1.0
    }
