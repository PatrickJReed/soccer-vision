"""Tests for the retained fixed-camera calib helpers.

The per-frame propagation engines (Engine A/B + gated) and the calib_compare harness were
retired in favour of the physical per-frame calibration (pitch.physical_calib); what remains
here is flag_outlier_clicks / _robust_sqpnp plus the pose -> homography / reprojection
helpers (and the shared manual_anchor click-propagation primitives).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np
from soccer_vision.calib.calibrate import homography_from_pose, pitch_homography
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.labeler.chain import normalize_homography
from soccer_vision.pitch.calib_anchor import (
    _reproj_rms_px,
    _robust_sqpnp,
    flag_outlier_clicks,
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
    clicks propagate consistently (unlike an identity chain over different views).
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


def test_flag_outlier_clicks_removes_and_flags_a_mislabel() -> None:
    # one clicked landmark corrupted at frame 4 -> flag_outlier_clicks removes it from
    # clean_clicks and records it; other frames are untouched. The clicks are projected
    # with _K, so the true intrinsics are passed directly (no separate focal recovery).
    poses, _interframe = _pan_sequence(9)
    clicks = _clicks_at(poses, [0, 4, 8])
    assert any(c.frame == 4 and c.kp_idx == 6 for c in clicks)
    clicks = [c if not (c.frame == 4 and c.kp_idx == 6)
              else Click(c.frame, c.kp_idx, c.x + 0.25, c.y) for c in clicks]
    clean, flagged = flag_outlier_clicks(clicks, _K, (1920, 1080), thr=40.0)
    assert flagged.get(4) == [6]                                  # mislabel flagged
    assert not any(c.frame == 4 and c.kp_idx == 6 for c in clean)  # removed from clean
    assert len(clean) == len(clicks) - 1                          # only that one dropped
