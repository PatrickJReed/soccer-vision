"""Phase 3a tests: shared propagation + calibration registration engines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np
from soccer_vision.calib.calibrate import homography_from_pose, pitch_homography
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.labeler.chain import normalize_homography
from soccer_vision.pitch.calib_anchor import (
    FramePose,
    _reproj_rms_px,
    _robust_sqpnp,
    _rotation_from_chain,
    calibrate_clicked_frames,
    frame_homography,
    poses_by_click_propagation,
    poses_by_gated_propagation,
    poses_by_pose_propagation,
)
from soccer_vision.pitch.manual_anchor import (
    Click,
    build_segments,
    cumulative_transforms,
    propagate_clicks,
)


def test_propagate_clicks_carries_a_click_along_an_identity_chain() -> None:
    # Identity inter-frame transforms -> a click at frame 0 propagates UNCHANGED to
    # every frame within the window, for its own landmark.
    interframe = {i: np.eye(3) for i in range(5)}  # frames 0..5 linked
    seg = build_segments(interframe, 6)
    transforms = cumulative_transforms(interframe, seg)
    clicks = [Click(frame=0, kp_idx=3, x=0.4, y=0.6)]
    prop = propagate_clicks(clicks, transforms, seg, window=10)
    assert prop[2][3] == (0.4, 0.6)  # frame 2, landmark 3
    assert prop[5][3] == (0.4, 0.6)
    # window is INCLUSIVE (|Δframe| <= window): frame 1 (distance 1) is in, frame 5 is out
    prop_small = propagate_clicks(clicks, transforms, seg, window=1)
    assert prop_small[1][3] == (0.4, 0.6)  # boundary: distance == window, included
    assert 3 not in prop_small.get(5, {})


def test_propagate_clicks_respects_segments() -> None:
    # A gap (missing link at 2) splits segments; a click in segment 0 does not reach
    # segment 1.
    interframe = {0: np.eye(3), 1: np.eye(3), 3: np.eye(3)}  # link missing at 2
    seg = build_segments(interframe, 5)
    transforms = cumulative_transforms(interframe, seg)
    clicks = [Click(frame=0, kp_idx=1, x=0.5, y=0.5)]
    prop = propagate_clicks(clicks, transforms, seg, window=10)
    assert 1 in prop[1]   # same segment
    assert 4 not in prop  # frame 4 is a different segment -> never receives the click


_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)


def _look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    e, t, u = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    fwd = t - e
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, u)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rmat = np.vstack([right, down, fwd])
    rvec, _ = cv2.Rodrigues(rmat)
    return rvec, (-rmat @ e).reshape(3, 1)


def test_frame_homography_round_trips_pixel_to_pitch() -> None:
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    h_pitch = pitch_homography(homography_from_pose(_K, rvec, tvec))  # canon[0,1]^2 -> px
    h_img2pitch = frame_homography(_K, rvec, tvec)                    # px -> canon[0,1]^2
    canon = np.array([0.3, 0.6, 1.0])
    px = h_pitch @ canon
    px = px / px[2]
    back = h_img2pitch @ np.array([px[0], px[1], 1.0])
    back = back / back[2]
    assert np.hypot(back[0] - 0.3, back[1] - 0.6) < 1e-9


def test_calibrate_clicked_frames_recovers_focal() -> None:
    # synthetic clicks (NORMALIZED) on several elevated views -> shared focal ~1400
    eyes = [(8.0, 4, 70), (33, 14, 80), (18, 59, 75), (40, 44, 85), (23, -1, 90), (3, 34, 78)]
    fp = field_points_3d()
    clicks: list[Click] = []
    for fidx, e in enumerate(eyes):
        rvec, tvec = _look_at(e, (22.85, 34.25, 0.0))
        px = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            # 5 = halfway_near, hidden under the camera -> never labeled
            if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080:
                clicks.append(Click(fidx, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080))
    k, poses = calibrate_clicked_frames(clicks, (1920, 1080), min_points=6)
    assert abs(k[0, 0] - 1400.0) < 40.0
    assert len(poses) >= 3
    f0 = sorted(poses)[0]
    assert poses[f0][0].shape == (3, 1) and poses[f0][1].shape == (3, 1)


def test_frame_pose_dataclass_fields() -> None:
    fp_ = FramePose(rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)),
                    residual_px=1.5, n_points=7, fold_count=9)
    assert fp_.residual_px == 1.5 and fp_.n_points == 7 and fp_.fold_count == 9


def test_reproj_rms_px_zero_for_exact_and_nan_for_empty() -> None:
    import math
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    fp = field_points_3d()
    ids = [0, 1, 3, 9, 14]
    px = cv2.projectPoints(fp[ids], rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    point_obs = [(i, float(px[n, 0]), float(px[n, 1])) for n, i in enumerate(ids)]
    assert _reproj_rms_px(_K, rvec, tvec, point_obs) < 1e-6  # exact projection -> ~0
    assert math.isnan(_reproj_rms_px(_K, rvec, tvec, []))    # no points -> nan


def _pan_sequence(
    n: int = 13,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, np.ndarray]]:
    """A fixed-centre camera panning across n frames (the Trace motion model).

    Returns ({frame: (rvec, tvec)}, {i: normalized inter-frame homography}); the
    chain is the physically-correct pan homography G = K R_{i+1} R_i^-1 K^-1, so
    clicks/poses propagate consistently (unlike an identity chain over different
    views).
    """
    center = (-8.0, 34.0, 9.0)
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for f, dy in enumerate(np.linspace(-10.0, 10.0, n)):
        poses[f] = _look_at(center, (22.85, 34.0 + float(dy), 0.0))
    interframe: dict[int, np.ndarray] = {}
    for i in range(n - 1):
        ri, _ = cv2.Rodrigues(poses[i][0])
        rj, _ = cv2.Rodrigues(poses[i + 1][0])
        g_px = _K @ rj @ np.linalg.inv(ri) @ np.linalg.inv(_K)
        interframe[i] = normalize_homography(g_px, (1920, 1080))
    return poses, interframe


def _clicks_at(
    poses: Mapping[int, tuple[np.ndarray, np.ndarray]],
    frames: Sequence[int],
    size: tuple[int, int] = (1920, 1080),
) -> list[Click]:
    """Normalized clicks at the given frames from their true poses."""
    w, h = size
    fp = field_points_3d()
    clicks: list[Click] = []
    for f in frames:
        rvec, tvec = poses[f]
        px = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            if j != 5 and 0 < px[j, 0] < w and 0 < px[j, 1] < h:  # 5 = hidden landmark
                clicks.append(Click(f, j, float(px[j, 0]) / w, float(px[j, 1]) / h))
    return clicks


def test_engine_a_recovers_clicked_and_propagated_frames() -> None:
    # Realistic fixed-camera pan: a VALID chain links all frames. Clicks at a few
    # frames calibrate the focal; Engine A propagates to recover EVERY frame's pose
    # (clicked AND unclicked) against the field -- no fold, no drift.
    poses, interframe = _pan_sequence(13)
    clicks = _clicks_at(poses, [0, 4, 8, 12])
    seg = build_segments(interframe, 13)
    transforms = cumulative_transforms(interframe, seg)
    k, _kp = calibrate_clicked_frames(clicks, (1920, 1080), min_points=6)
    out = poses_by_click_propagation(clicks, transforms, seg, k, (1920, 1080),
                                     window=360, min_points=4)
    assert len(out) == 13  # every frame covered (clicked + propagated)
    fp = field_points_3d()
    for f in range(13):
        truth = cv2.projectPoints(fp, poses[f][0], poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        rec = cv2.projectPoints(fp, out[f].rvec, out[f].tvec, _K, np.zeros(5))[0].reshape(-1, 2)
        assert np.median(np.linalg.norm(truth - rec, axis=1)) < 1.0  # recovered ~= truth
        assert out[f].fold_count < 21  # not a fold


def test_engine_a_line_obs_path_runs() -> None:
    # an extra line observation at a frame still solves (refine_pose line path
    # exercised; real line propagation is Phase 3b).
    poses, interframe = _pan_sequence(9)
    clicks = _clicks_at(poses, [0, 4, 8])
    seg = build_segments(interframe, 9)
    transforms = cumulative_transforms(interframe, seg)
    k, _kp = calibrate_clicked_frames(clicks, (1920, 1080), min_points=6)
    line_obs = {0: [("midline", 960.0, 540.0)]}
    out = poses_by_click_propagation(clicks, transforms, seg, k, (1920, 1080),
                                     window=360, min_points=4, line_obs=line_obs)
    assert 0 in out


def test_engine_b_recovers_panned_poses_from_one_clicked_frame() -> None:
    poses, interframe = _pan_sequence(7)
    seg = build_segments(interframe, 7)
    transforms = cumulative_transforms(interframe, seg)
    # only frame 0 is "clicked" (calibrated); propagate its pose to all others
    clicked_poses = {0: poses[0]}
    out = poses_by_pose_propagation(transforms, seg, _K, clicked_poses, (1920, 1080))
    fp = field_points_3d()
    for f in range(7):
        assert f in out
        truth = cv2.projectPoints(fp, poses[f][0], poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        rec = cv2.projectPoints(fp, out[f].rvec, out[f].tvec, _K, np.zeros(5))[0].reshape(-1, 2)
        assert np.max(np.linalg.norm(truth - rec, axis=1)) < 0.5  # prototype: ~1e-11
        assert out[f].n_points == 0


def test_rotation_from_chain_clamps_reflection_to_proper_rotation() -> None:
    # a reflection (det -1) must be clamped to a PROPER rotation in SO(3) (det +1) --
    # this exercises the det-sign-flip branch of _rotation_from_chain.
    g_px = _K @ np.diag([-1.0, 1.0, 1.0]) @ np.linalg.inv(_K)
    r = _rotation_from_chain(g_px, _K)
    assert abs(np.linalg.det(r) - 1.0) < 1e-9          # proper rotation
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)  # orthonormal


def test_engine_b_uncovered_without_clicked_neighbor() -> None:
    _poses, interframe = _pan_sequence(5)
    seg = build_segments(interframe, 5)
    transforms = cumulative_transforms(interframe, seg)
    # no clicked frames at all -> nothing covered
    out = poses_by_pose_propagation(transforms, seg, _K, {}, (1920, 1080))
    assert out == {}


def test_engine_b_nonrotation_chain_does_not_crash() -> None:
    # a degenerate inter-frame transform -> SVD nearest-rotation, still returns a pose
    poses, interframe = _pan_sequence(3)
    interframe[1] = normalize_homography(
        np.array([[1.0, 0.3, 5.0], [0.0, 1.2, 2.0], [1e-4, 0.0, 1.0]]), (1920, 1080))
    seg = build_segments(interframe, 3)
    transforms = cumulative_transforms(interframe, seg)
    out = poses_by_pose_propagation(transforms, seg, _K, {0: poses[0]}, (1920, 1080))
    assert all(np.all(np.isfinite(p.rvec)) and np.all(np.isfinite(p.tvec)) for p in out.values())


def test_robust_sqpnp_drops_a_gross_outlier_click() -> None:
    # 8 in-frame landmarks from a known pose; corrupt one click by 400px -> the helper
    # must drop exactly that landmark and return a clean inlier pose.
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    fp = field_points_3d()
    ids = [1, 3, 4, 6, 7, 8, 14, 16]
    img = cv2.projectPoints(fp[ids], rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    img[2] = img[2] + np.array([400.0, -300.0])  # gross outlier on ids[2]
    res = _robust_sqpnp(_K, ids, img, thr=40.0, min_points=4)
    assert res is not None
    rv, tv, inliers, outliers = res
    assert outliers == [ids[2]]          # exactly the corrupted landmark dropped
    assert set(inliers) == set(ids) - {ids[2]}
    proj = cv2.projectPoints(fp[inliers], rv, tv, _K, np.zeros(5))[0].reshape(-1, 2)
    truth = cv2.projectPoints(fp[inliers], rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    assert np.max(np.linalg.norm(proj - truth, axis=1)) < 2.0  # clean pose


def test_robust_sqpnp_keeps_all_clean_clicks() -> None:
    # noiseless clicks -> nothing dropped
    rvec, tvec = _look_at((-8.0, 34.0, 8.0), (22.85, 34.0, 0.0))
    fp = field_points_3d()
    ids = [1, 3, 4, 6, 7, 8, 14, 16]
    img = cv2.projectPoints(fp[ids], rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    res = _robust_sqpnp(_K, ids, img, thr=40.0, min_points=4)
    assert res is not None
    _rv, _tv, inliers, outliers = res
    assert outliers == [] and set(inliers) == set(ids)


def test_engine_a_frames_restricts_targets() -> None:
    # frames= recomputes only the requested targets (windowed recompute path).
    poses, interframe = _pan_sequence(9)
    clicks = _clicks_at(poses, [0, 4, 8])
    seg = build_segments(interframe, 9)
    transforms = cumulative_transforms(interframe, seg)
    k, _kp = calibrate_clicked_frames(clicks, (1920, 1080), min_points=6)
    out = poses_by_click_propagation(clicks, transforms, seg, k, (1920, 1080),
                                     window=360, min_points=4, frames=[2, 4])
    assert set(out) == {2, 4}


def test_flag_outlier_clicks_removes_and_flags_a_mislabel() -> None:
    # one clicked landmark corrupted at frame 4 -> flag_outlier_clicks removes it from
    # clean_clicks and records it; other frames are untouched.
    from soccer_vision.pitch.calib_anchor import flag_outlier_clicks
    poses, _interframe = _pan_sequence(9)
    clicks = _clicks_at(poses, [0, 4, 8])
    assert any(c.frame == 4 and c.kp_idx == 6 for c in clicks)
    clicks = [c if not (c.frame == 4 and c.kp_idx == 6)
              else Click(c.frame, c.kp_idx, c.x + 0.25, c.y) for c in clicks]
    k, _kp = calibrate_clicked_frames(clicks, (1920, 1080), min_points=6)
    clean, flagged = flag_outlier_clicks(clicks, k, (1920, 1080), thr=40.0)
    assert flagged.get(4) == [6]                                  # mislabel flagged
    assert not any(c.frame == 4 and c.kp_idx == 6 for c in clean)  # removed from clean
    assert len(clean) == len(clicks) - 1                          # only that one dropped


def _static_chain(n: int) -> tuple[dict[int, np.ndarray], dict[int, int], dict[int, np.ndarray]]:
    interframe = {i: np.eye(3) for i in range(n)}
    seg = build_segments(interframe, n)
    transforms = cumulative_transforms(interframe, seg)
    return interframe, seg, transforms


def test_gated_propagation_rejects_corrupted_far_click() -> None:
    # Identity chain, static camera: a frame with good NEAR clicks plus a CORRUPTED
    # far click (simulating chain drift) -> the far click is gated out, pose stays true,
    # and the old aggregate engine is dragged off by the same bad click.
    _interframe, seg, transforms = _static_chain(400)
    rvec, tvec = _look_at((-8.0, 34.0, 20.0), (22.85, 34.0, 0.0))
    fp = field_points_3d()
    px = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    in_ids = [j for j in range(21) if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080]
    assert len(in_ids) >= 8
    near, far_id = in_ids[:7], in_ids[7]
    clicks = [Click(0, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080) for j in near]
    clicks.append(Click(200, far_id, float(px[far_id, 0] + 300.0) / 1920,
                        float(px[far_id, 1]) / 1080))  # corrupted, 300px off, 200 frames away
    out = poses_by_gated_propagation(
        clicks, transforms, seg, _K, (1920, 1080),
        max_reach=360, seed_size=6, gate_px=60.0, gap_dist=180, min_points=4, frames=[0])
    assert 0 in out
    assert out[0].n_points == 7  # 7 near kept; corrupted far rejected
    rec = cv2.projectPoints(fp, out[0].rvec, out[0].tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    gated_err = float(np.median(np.linalg.norm(px[in_ids] - rec[in_ids], axis=1)))
    assert gated_err < 1.0  # recovered ~= truth
    old = poses_by_click_propagation(
        clicks, transforms, seg, _K, (1920, 1080), window=360, min_points=4, frames=[0])
    rec_old = cv2.projectPoints(fp, old[0].rvec, old[0].tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    old_err = float(np.median(np.linalg.norm(px[in_ids] - rec_old[in_ids], axis=1)))
    assert old_err > gated_err  # the aggregate engine is corrupted; gating is better


def test_gated_propagation_stays_red_beyond_gap_dist() -> None:
    _interframe, seg, transforms = _static_chain(400)
    rvec, tvec = _look_at((-8.0, 34.0, 20.0), (22.85, 34.0, 0.0))
    fp = field_points_3d()
    px = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    in_ids = [j for j in range(21) if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080]
    clicks = [Click(300, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080) for j in in_ids[:7]]
    out = poses_by_gated_propagation(
        clicks, transforms, seg, _K, (1920, 1080),
        max_reach=360, seed_size=6, gate_px=60.0, gap_dist=180, min_points=4, frames=[0])
    assert 0 not in out  # nearest click 300 frames away (> gap_dist 180) -> red
    out2 = poses_by_gated_propagation(
        clicks, transforms, seg, _K, (1920, 1080),
        max_reach=360, seed_size=6, gate_px=60.0, gap_dist=320, min_points=4, frames=[0])
    assert 0 in out2  # a larger gap_dist recovers it -> gap_dist is the gate


def test_gated_propagation_accepts_consistent_far_clicks() -> None:
    _interframe, seg, transforms = _static_chain(200)
    rvec, tvec = _look_at((-8.0, 34.0, 20.0), (22.85, 34.0, 0.0))
    fp = field_points_3d()
    px = cv2.projectPoints(fp, rvec, tvec, _K, np.zeros(5))[0].reshape(-1, 2)
    in_ids = [j for j in range(21) if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080]
    assert len(in_ids) >= 10
    near, far = in_ids[:6], in_ids[6:10]
    clicks = [Click(0, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080) for j in near]
    clicks += [Click(100, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080) for j in far]
    out = poses_by_gated_propagation(
        clicks, transforms, seg, _K, (1920, 1080),
        max_reach=360, seed_size=6, gate_px=60.0, gap_dist=180, min_points=4, frames=[0])
    assert 0 in out
    assert out[0].n_points == 10  # consistent (clean-chain) far clicks accepted -> deeper fit
