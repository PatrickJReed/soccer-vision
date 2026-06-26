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
    from soccer_vision.pitch.homography import HomographyError, fit_homography

    clicks, transforms, segment_of, H_ref_to_pitch, _own_end, opp_end = _build_session()

    # Strengthen the test so it genuinely beats a per-frame fit (not just matches it):
    # keep only 3 of frame 0's own-end clicks, so frame 0 ALONE is under-determined
    # (a homography needs >=4 points). A per-frame engine cannot calibrate frame 0 at
    # all; the pooled global solve still does, via frame 2's opp-end clicks.
    f0 = [c for c in clicks if c.frame == 0][:3]
    f2 = [c for c in clicks if c.frame == 2]
    sparse = f0 + f2

    # A per-frame fit from frame 0's clicks alone genuinely fails (under-determined).
    img0 = np.array([[c.x, c.y] for c in f0], dtype=np.float64)
    pitch0 = PITCH_LANDMARKS[[c.kp_idx for c in f0]]
    try:
        fit_homography(img0, pitch0)
        per_frame_succeeded = True
    except HomographyError:
        per_frame_succeeded = False
    assert not per_frame_succeeded, "frame 0 alone must be under-determined for this test"

    gc = solve_global(sparse, transforms, segment_of, SIZE)
    assert isinstance(gc, GlobalCalib)
    assert set(gc.h_by_segment) == {0}

    # Frame 0 (3 own-end clicks, never saw the opp end) STILL projects the OPP end
    # correctly — only possible because the global solve pooled frame 2's opp clicks.
    # This is the "lines in the sky" regression the per-frame engine could not fix.
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
