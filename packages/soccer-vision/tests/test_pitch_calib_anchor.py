"""Phase 3a tests: shared propagation + calibration registration engines."""

from __future__ import annotations

import cv2
import numpy as np
from soccer_vision.calib.calibrate import homography_from_pose, pitch_homography
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.pitch.calib_anchor import (
    FramePose,
    _reproj_rms_px,
    calibrate_clicked_frames,
    frame_homography,
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
