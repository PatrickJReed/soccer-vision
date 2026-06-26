from __future__ import annotations

import numpy as np
from soccer_vision.pitch.global_calib import (
    GlobalCalib,
    cross_end_holdout,
    fold_of_norm,
    frame_status,
    solve_global,
    two_ended_segments,
)
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click

SIZE = (1920, 1080)


def _build_session() -> tuple[
    list[Click], dict[int, np.ndarray], dict[int, int], np.ndarray, list[int], list[int]
]:
    """A synthetic fixed-camera session: one known global homography (pitch->image_ref),
    each frame a pure 2D translation of the reference, and each frame clicks ONLY the
    landmarks that fall on 'its' end (so no single frame sees the whole field)."""
    # Ground-truth pitch[0,1] -> reference-image (normalized) homography.
    # A gentle perspective, zoomed so the reference frame is a realistic CROP that
    # only shows ~the own half (fold ~13 of 21, not the whole field) — a frame is a
    # 2D crop of the canvas, not a world map.
    H_pitch_to_ref = np.array(
        [[0.8, 0.0, 0.1],
         [0.0, 1.6, 0.1],
         [0.0, 0.15, 1.0]],
        dtype=np.float64,
    )
    H_ref_to_pitch = np.linalg.inv(H_pitch_to_ref)

    # Per-frame normalized 2D offsets (frame pixel = ref pixel + offset_f).
    offsets = {0: np.array([0.0, 0.0]), 1: np.array([0.30, 0.0]),
               2: np.array([0.55, 0.0]), 3: np.array([-0.20, 0.0])}
    # M[f] maps frame-f -> ref. A landmark at reference pixel `ref` appears at
    # `ref + offset_f` in frame f, so mapping frame -> ref SUBTRACTS offset_f.
    transforms: dict[int, np.ndarray] = {
        f: np.array([[1.0, 0.0, -off[0]], [0.0, 1.0, -off[1]], [0.0, 0.0, 1.0]])
        for f, off in offsets.items()
    }
    segment_of = {f: 0 for f in offsets}

    own_end = [i for i in range(len(PITCH_LANDMARKS)) if PITCH_LANDMARKS[i, 1] < 0.5 and i != 5]
    opp_end = [i for i in range(len(PITCH_LANDMARKS)) if PITCH_LANDMARKS[i, 1] > 0.5 and i != 5]

    def click_for(frame: int, kp: int) -> Click:
        # true reference pixel for this landmark
        ref = H_pitch_to_ref @ np.array([*PITCH_LANDMARKS[kp], 1.0])
        ref = ref[:2] / ref[2]
        frame_pt = ref + offsets[frame]  # frame pixel = ref + offset
        return Click(frame=frame, kp_idx=kp, x=float(frame_pt[0]), y=float(frame_pt[1]))

    clicks = [click_for(0, kp) for kp in own_end]          # frame 0 sees own end only
    clicks += [click_for(2, kp) for kp in opp_end]         # frame 2 sees opp end only
    return clicks, transforms, segment_of, H_ref_to_pitch, own_end, opp_end


def test_solve_global_recovers_homography_and_projects_unclicked_end() -> None:
    clicks, transforms, segment_of, H_ref_to_pitch, _own_end, opp_end = _build_session()
    gc = solve_global(clicks, transforms, segment_of, SIZE)
    assert isinstance(gc, GlobalCalib)
    assert set(gc.h_by_segment) == {0}

    # Frame 0 clicked ONLY the own end; its H must still project the OPP end correctly
    # (this is the "lines in the sky" regression the old per-frame engine failed).
    H0 = gc.frame_homography(0)
    assert H0 is not None
    for kp in opp_end:
        ref = np.linalg.inv(H_ref_to_pitch) @ np.array([*PITCH_LANDMARKS[kp], 1.0])
        ref = ref[:2] / ref[2]
        frame_pt = ref + transforms[0][:2, 2]  # offset_0 = 0, but keep general
        proj = H0 @ np.array([frame_pt[0], frame_pt[1], 1.0])
        proj = proj[:2] / proj[2]
        assert np.allclose(proj, PITCH_LANDMARKS[kp], atol=1e-6)


def test_two_ended_segments_detects_both_ends() -> None:
    clicks, _transforms, segment_of, *_ = _build_session()
    assert two_ended_segments(clicks, segment_of) == {0}
    # own end only -> not two-ended
    own_only = [c for c in clicks if c.kp_idx in
                {i for i in range(len(PITCH_LANDMARKS)) if PITCH_LANDMARKS[i, 1] < 0.5}]
    assert two_ended_segments(own_only, segment_of) == set()


def test_cross_end_holdout_is_small_for_consistent_session() -> None:
    clicks, transforms, segment_of, *_ = _build_session()
    report = cross_end_holdout(clicks, transforms, segment_of, SIZE)
    assert report is not None
    # synthetic data is exactly planar+translation, so leave-one-frame-out should
    # reconstruct each end to within a few feet.
    assert report.median_ft < 5.0


def test_solve_global_skips_segment_with_too_few_points() -> None:
    transforms: dict[int, np.ndarray] = {0: np.eye(3), 1: np.eye(3)}
    segment_of = {0: 0, 1: 0}
    clicks = [Click(0, 0, 0.1, 0.1), Click(0, 1, 0.2, 0.2)]  # only 2 points
    gc = solve_global(clicks, transforms, segment_of, SIZE)
    assert gc.h_by_segment == {}
    assert gc.frame_homography(0) is None


def test_fold_of_norm_counts_in_frame_landmarks() -> None:
    clicks, transforms, segment_of, *_ = _build_session()
    gc = solve_global(clicks, transforms, segment_of, SIZE)
    fold = fold_of_norm(gc.frame_homography(0), SIZE)
    assert 0 <= fold <= len(PITCH_LANDMARKS)


def test_frame_status_green_yellow_red() -> None:
    clicks, transforms, segment_of, *_ = _build_session()
    gc = solve_global(clicks, transforms, segment_of, SIZE)
    two_ended = two_ended_segments(clicks, segment_of)
    # frame 0: plausible H + two-ended segment -> green
    assert frame_status(gc.frame_homography(0), SIZE, segment_two_ended=True) == "green"
    # plausible H but single-ended -> yellow
    assert frame_status(gc.frame_homography(0), SIZE, segment_two_ended=False) == "yellow"
    # no H -> red
    assert frame_status(None, SIZE, segment_two_ended=True) == "red"
    # a wildly-folding H (maps everything off-frame) -> red
    sky = np.array([[1e-6, 0, 0], [0, 1e-6, 0], [0, 0, 1.0]])
    assert frame_status(sky, SIZE, segment_two_ended=True) == "red"
    assert two_ended == {0}
