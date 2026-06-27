from __future__ import annotations

import numpy as np
from soccer_vision.eval.pitch_metrics import displacement_to_feet
from soccer_vision.pitch.global_calib import (
    OPP_END_IDX,
    OWN_END_IDX,
    BundleCalib,
    GlobalCalib,
    _affine_from,
    _apply_h,
    _interp_affine_params,
    cross_end_holdout,
    fold_of_norm,
    frame_status,
    solve_bundle,
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


def test_solve_global_rejects_grossly_mislabeled_click() -> None:
    # cv2.findHomography's default ransacReprojThreshold (3.0) is in the DESTINATION
    # space, which here is pitch [0,1] — so 3.0 makes EVERY click an inlier and no
    # outlier is ever rejected. solve_global must pass a pitch-unit threshold so that
    # one grossly-mislabeled click does not corrupt the whole segment's global H.
    from soccer_vision.eval.pitch_metrics import displacement_to_feet

    clicks, transforms, segment_of, H_ref_to_pitch, own_end, opp_end = _build_session()
    H_pitch_to_ref = np.linalg.inv(H_ref_to_pitch)

    # Corrupt exactly ONE existing frame-0 click by ~0.25 in normalized image space.
    bad = list(clicks)
    corrupted = False
    for i, c in enumerate(bad):
        if c.frame == 0:
            bad[i] = Click(c.frame, c.kp_idx, c.x + 0.25, c.y + 0.25)
            corrupted = True
            break
    assert corrupted

    dirty = solve_global(bad, transforms, segment_of, SIZE)
    h_dirty = dirty.h_by_segment[0]  # reference-image-norm -> pitch[0,1]

    # Every true landmark's reference pixel must STILL map to its canonical pitch
    # coord within a few feet -> the one gross outlier was rejected by the RANSAC fit.
    # Without the pitch-unit threshold the bad click pulls the fit and this blows up.
    for kp in own_end + opp_end:
        ref = H_pitch_to_ref @ np.array([*PITCH_LANDMARKS[kp], 1.0])
        ref = ref[:2] / ref[2]
        proj = h_dirty @ np.array([ref[0], ref[1], 1.0])
        proj = proj[:2] / proj[2]
        ft = float(displacement_to_feet(proj - PITCH_LANDMARKS[kp]))
        assert ft < 3.0, f"kp {kp}: {ft:.2f} ft off -> outlier not rejected"


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


# --- Field-anchored bundle (solve_bundle) ---------------------------------------------

_H_PITCH_TO_REF = np.array([[0.8, 0.0, 0.1], [0.0, 1.6, 0.1], [0.0, 0.15, 1.0]])
_OWN_END = [i for i in range(len(PITCH_LANDMARKS)) if PITCH_LANDMARKS[i, 1] < 0.5 and i != 5]
_OPP_END = [i for i in range(len(PITCH_LANDMARKS)) if PITCH_LANDMARKS[i, 1] > 0.5 and i != 5]


def _click_for(frame: int, kp: int, offset: np.ndarray) -> Click:
    """A ground-truth click: the kp's true reference pixel shifted by the frame's pan
    offset. The click is independent of any chain drift (it's a real observation)."""
    ref = _H_PITCH_TO_REF @ np.array([*PITCH_LANDMARKS[kp], 1.0])
    ref = ref[:2] / ref[2]
    frame_pt = ref + offset
    return Click(frame=frame, kp_idx=kp, x=float(frame_pt[0]), y=float(frame_pt[1]))


def _build_drift_session() -> tuple[
    list[Click], dict[int, np.ndarray], dict[int, int]
]:
    """A both-ends pan over 4 frames where the registration chain M[f] has injected
    affine DRIFT that grows with frame index (accumulated chain error). Each end is
    clicked REDUNDANTLY in two differently-drifted frames (own in 0 & 1, opp in 2 & 3),
    so a single global homography with only per-frame translation (solve_global) cannot
    reconcile them, but a per-frame affine correction (solve_bundle) can.

    M[f] = D[f] @ T_true[f], where T_true[f] is the true (drift-free) frame->ref
    translation and D[f] = I + f * P is a small affine drift in reference space.
    """
    offsets = {0: np.array([0.0, 0.0]), 1: np.array([0.18, 0.06]),
               2: np.array([0.40, 0.14]), 3: np.array([0.58, 0.20])}
    drift_gen = np.array([[0.04, 0.03, 0.03], [0.025, -0.05, 0.04], [0.0, 0.0, 0.0]])
    t_true = {f: np.array([[1.0, 0.0, -off[0]], [0.0, 1.0, -off[1]], [0.0, 0.0, 1.0]])
              for f, off in offsets.items()}
    transforms = {f: (np.eye(3) + f * drift_gen) @ t_true[f] for f in offsets}
    segment_of = {f: 0 for f in offsets}
    clicks = (
        [_click_for(0, kp, offsets[0]) for kp in _OWN_END]
        + [_click_for(1, kp, offsets[1]) for kp in _OWN_END]
        + [_click_for(2, kp, offsets[2]) for kp in _OPP_END]
        + [_click_for(3, kp, offsets[3]) for kp in _OPP_END]
    )
    return clicks, transforms, segment_of


def _end_medians(
    get_h: object, clicks: list[Click]
) -> tuple[float, float, float]:
    """In-sample own / opp / overall median feet error using a frame->H callable."""
    own, opp = set(OWN_END_IDX), set(OPP_END_IDX)
    own_ft: list[float] = []
    opp_ft: list[float] = []
    for c in clicks:
        h = get_h(c.frame)  # type: ignore[operator]
        proj = _apply_h(h, np.array([[c.x, c.y]]))[0]
        ft = float(displacement_to_feet(proj - PITCH_LANDMARKS[c.kp_idx]))
        (own_ft if c.kp_idx in own else opp_ft).append(ft)
    return (float(np.median(own_ft)), float(np.median(opp_ft)),
            float(np.median(own_ft + opp_ft)))


def test_solve_bundle_corrects_chain_drift_on_both_ends() -> None:
    clicks, transforms, segment_of = _build_drift_session()

    bc = solve_bundle(clicks, transforms, segment_of, SIZE)
    assert isinstance(bc, BundleCalib)
    assert set(bc.h_by_segment) == {0}
    assert sorted(bc.a_params_by_segment[0]) == [0, 1, 2, 3]

    b_own, b_opp, b_all = _end_medians(bc.frame_homography, clicks)
    # The per-frame affine absorbs the injected drift: in-sample reprojection is ~0,
    # well under 1 ft on BOTH ends.
    assert b_own < 1.0, f"bundle own {b_own:.3f} ft"
    assert b_opp < 1.0, f"bundle opp {b_opp:.3f} ft"

    gc = solve_global(clicks, transforms, segment_of, SIZE)
    g_own, g_opp, g_all = _end_medians(gc.frame_homography, clicks)
    # solve_global has only a per-frame translation, so it cannot reconcile the two
    # differently-drifted views of each end -> materially worse on both ends.
    assert g_all > 2.5, f"solve_global overall {g_all:.3f} ft (drift not exercised)"
    assert g_own > b_own and g_opp > b_opp
    assert g_all > 10 * b_all


def test_frame_homography_interpolates_affine_between_clicked_frames() -> None:
    # Two clicked frames (0 own, 8 opp) with growing drift, plus an UNCLICKED frame 4
    # exactly halfway between them in frame index. frame_homography(4) must use the
    # midpoint of the two clicked frames' affine params.
    offsets = {0: np.array([0.0, 0.0]), 4: np.array([0.40, 0.14]),
               8: np.array([0.58, 0.20])}
    drift_gen = np.array([[0.03, 0.02, 0.02], [0.02, -0.04, 0.03], [0.0, 0.0, 0.0]])
    t_true = {f: np.array([[1.0, 0.0, -off[0]], [0.0, 1.0, -off[1]], [0.0, 0.0, 1.0]])
              for f, off in offsets.items()}
    transforms = {f: (np.eye(3) + f * drift_gen) @ t_true[f] for f in offsets}
    segment_of = {f: 0 for f in offsets}
    clicks = ([_click_for(0, kp, offsets[0]) for kp in _OWN_END]
              + [_click_for(8, kp, offsets[8]) for kp in _OPP_END])

    bc = solve_bundle(clicks, transforms, segment_of, SIZE)
    a0 = bc.a_params_by_segment[0][0]
    a8 = bc.a_params_by_segment[0][8]
    # the two clicked frames really do have different corrections (drift differs)
    assert not np.allclose(a0, a8, atol=1e-3)

    # frame 4 is unclicked, exactly halfway -> A is the midpoint of a0 and a8.
    interp = _interp_affine_params(bc.a_params_by_segment[0], 4)
    expected = 0.5 * (a0 + a8)
    assert np.allclose(interp, expected, atol=1e-9)
    # each component lies between the two clicked frames' values
    lo = np.minimum(a0, a8)
    hi = np.maximum(a0, a8)
    assert np.all(interp >= lo - 1e-9) and np.all(interp <= hi + 1e-9)

    # frame_homography wires that interpolated affine in: H = H_g @ A_interp @ M[4].
    expected_h = bc.h_by_segment[0] @ _affine_from(expected) @ transforms[4]
    assert np.allclose(bc.frame_homography(4), expected_h, atol=1e-9)


def test_solve_bundle_skips_segment_with_too_few_points() -> None:
    transforms: dict[int, np.ndarray] = {0: np.eye(3), 1: np.eye(3)}
    segment_of = {0: 0, 1: 0}
    clicks = [Click(0, 0, 0.1, 0.1), Click(0, 1, 0.2, 0.2)]  # only 2 points
    bc = solve_bundle(clicks, transforms, segment_of, SIZE)
    assert bc.h_by_segment == {}
    assert bc.a_params_by_segment == {}
    assert bc.frame_homography(0) is None
