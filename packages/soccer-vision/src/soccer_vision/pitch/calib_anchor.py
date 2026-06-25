"""Calibration-based per-frame homography engines for a fixed-camera session.

Two engines turn a session's clicks + the registration chain into a calibrated
homography for every frame: A (propagate clicks, then refine_pose per frame) and B
(calibrate clicked frames, then propagate the pose via the chain's recovered camera
rotation). Both reuse the Phase 1/2 calib core, so each frame is a real camera pose
(no fold) solved against the field directly (no homography-chaining drift).

Internally full-pixel image space; emits the labeler's full-pixel image->pitch[0,1]
homography (the export format). Pure: no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.calibrate import (
    CalibError,
    calibrate_camera,
    homography_from_pose,
    pitch_homography,
    refine_pose,
)
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.calib.validate import fold_count
from soccer_vision.pitch.manual_anchor import Click, propagate_clicks


@dataclass(frozen=True, eq=False)
class FramePose:
    """A per-frame calibrated camera pose + quality."""

    rvec: NDArray[np.float64]
    tvec: NDArray[np.float64]
    residual_px: float   # reprojection RMS over the frame's obs (nan if none)
    n_points: int        # point landmarks used (0 if pose-propagated)
    fold_count: int      # landmarks projecting in-frame (slice size, ~6-12)


def frame_homography(
    k: NDArray[np.floating], rvec: NDArray[np.floating], tvec: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Full-pixel image -> pitch-[0,1]^2 homography (the labeler export format)."""
    h_pitch = pitch_homography(homography_from_pose(k, rvec, tvec))  # canon[0,1]^2 -> px
    return np.asarray(np.linalg.inv(h_pitch), dtype=np.float64)


def calibrate_clicked_frames(
    clicks: Sequence[Click],
    size: tuple[int, int],
    *,
    min_points: int = 6,
) -> tuple[NDArray[np.float64], dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]]:
    """Shared focal + per-clicked-frame pose from the DIRECTLY-clicked frames.

    Clicks are normalized [0,1]; converted to full pixel for the calib core. Raises
    CalibError if too few/degenerate clicked views.
    """
    w, h = size
    obs: dict[int, list[tuple[int, float, float]]] = {}
    for c in clicks:
        obs.setdefault(c.frame, []).append((c.kp_idx, c.x * w, c.y * h))
    result = calibrate_camera(obs, size, min_points=min_points)
    return result.K, result.poses


def _reproj_rms_px(
    k: NDArray[np.floating],
    rvec: NDArray[np.floating],
    tvec: NDArray[np.floating],
    point_obs: Sequence[tuple[int, float, float]],
) -> float:
    """Reprojection RMS (px) of point_obs under the pose; nan if no points."""
    if not point_obs:
        return float("nan")
    obj = field_points_3d()[[int(kp) for kp, _, _ in point_obs]].astype(np.float64)
    img = np.array([[x, y] for _, x, y in point_obs], dtype=np.float64)
    proj = cv2.projectPoints(
        obj, rvec, tvec, np.asarray(k, dtype=np.float64),
        np.zeros(5, dtype=np.float64))[0].reshape(-1, 2)
    d = proj - img
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _fold_for_pose(
    k: NDArray[np.floating],
    rvec: NDArray[np.floating],
    tvec: NDArray[np.floating],
    size: tuple[int, int],
) -> int:
    """fold_count for a pose's pitch homography."""
    return fold_count(pitch_homography(homography_from_pose(k, rvec, tvec)), size)


def _rotation_from_chain(
    g_px: NDArray[np.floating], k: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Relative camera rotation from an inter-frame PIXEL homography G = K R_rel K^-1.

    M = K^-1 G K equals R_rel up to scale; the nearest rotation (SVD U V^T, with the
    sign fix for a proper rotation) is returned, so a noisy / non-rotation transform
    degrades gracefully to the closest valid rotation.
    """
    k64 = np.asarray(k, dtype=np.float64)
    m = np.linalg.inv(k64) @ np.asarray(g_px, dtype=np.float64) @ k64
    u, _s, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u = u.copy()
        u[:, -1] *= -1
        r = u @ vt
    return np.asarray(r, dtype=np.float64)


def poses_by_pose_propagation(
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    k: NDArray[np.floating],
    clicked_poses: Mapping[int, tuple[NDArray[np.floating], NDArray[np.floating]]],
    size: tuple[int, int],
    *,
    frames: Sequence[int] | None = None,
) -> dict[int, FramePose]:
    """Engine B: propagate each clicked frame's pose to neighbours via the chain.

    For a target frame f, take the nearest clicked frame c in the same segment, the
    chain transform G_{c->f} = inv(M[f]) @ M[c] (converted normalized->pixel), recover
    the relative camera rotation, and compose it onto c's pose, keeping the (fixed)
    optical centre. A frame with no clicked frame in its segment is left uncovered.
    Returns {frame: FramePose} (n_points=0; residual_px=nan -- no obs at f).

    Precondition: clicked_poses keys are frames present in `transforms` (clicked
    frames are video frames, so the labeler always satisfies this).
    """
    w, h = size
    d_mat = np.diag([float(w), float(h), 1.0])
    d_inv = np.diag([1.0 / w, 1.0 / h, 1.0])
    k_arr = np.asarray(k, dtype=np.float64)
    clicked = sorted(clicked_poses)
    clicked_seg = {c: segment_of.get(c) for c in clicked}
    targets = sorted(transforms) if frames is None else [
        int(f) for f in frames if int(f) in transforms]
    out: dict[int, FramePose] = {}
    for f in targets:
        seg = segment_of.get(f)
        cands = [c for c in clicked if clicked_seg[c] == seg]
        if not cands:
            continue
        c = min(cands, key=lambda cc: abs(cc - f))
        rc, _ = cv2.Rodrigues(np.asarray(clicked_poses[c][0], dtype=np.float64))
        tc = np.asarray(clicked_poses[c][1], dtype=np.float64).reshape(3)
        if c == f:
            r_f, t_f = rc, tc.reshape(3, 1)
        else:
            g_norm = np.linalg.inv(np.asarray(transforms[f], dtype=np.float64)) \
                @ np.asarray(transforms[c], dtype=np.float64)
            g_px = d_mat @ g_norm @ d_inv
            r_rel = _rotation_from_chain(g_px, k_arr)
            r_f = r_rel @ rc
            center = -rc.T @ tc                 # fixed optical centre
            t_f = (-r_f @ center).reshape(3, 1)
        rvec_f, _ = cv2.Rodrigues(r_f)
        rvec_f_arr = np.asarray(rvec_f, dtype=np.float64)
        t_f_arr = np.asarray(t_f, dtype=np.float64)
        out[f] = FramePose(
            rvec=rvec_f_arr,
            tvec=t_f_arr,
            residual_px=float("nan"),
            n_points=0,
            fold_count=_fold_for_pose(k_arr, rvec_f_arr, t_f_arr, size),
        )
    return out


def poses_by_click_propagation(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    k: NDArray[np.floating],
    size: tuple[int, int],
    *,
    window: int,
    min_points: int = 4,
    line_obs: Mapping[int, Sequence[tuple[str, float, float]]] | None = None,
) -> dict[int, FramePose]:
    """Engine A: propagate clicks into each frame, then refine_pose (focal fixed).

    Per target frame: gather window-propagated point landmarks (and any supplied
    pixel-space line_obs for that frame), seed a pose with SQPNP on the propagated
    points, and refine with refine_pose. A frame with < min_points propagated points
    (SQPNP needs them) or a non-converging solve is left uncovered. line_obs are
    pre-propagated, pixel-space lines per frame (3a real run is point-only; line
    propagation is Phase 3b). Returns {frame: FramePose}.
    """
    w, h = size
    k_arr = np.asarray(k, dtype=np.float64)
    fp = field_points_3d()
    propagated = propagate_clicks(clicks, transforms, segment_of, window=window)
    out: dict[int, FramePose] = {}
    for f, kpmap in propagated.items():
        if len(kpmap) < min_points:
            continue
        idxs = sorted(kpmap)
        # propagated positions are normalized -> convert to pixel for the calib core
        point_obs = [(i, kpmap[i][0] * w, kpmap[i][1] * h) for i in idxs]
        lobs = list(line_obs.get(f, [])) if line_obs else []
        obj = fp[idxs].astype(np.float64)
        img = np.array([[x, y] for _, x, y in point_obs], dtype=np.float64)
        ok, rvec0, tvec0 = cv2.solvePnP(obj, img, k_arr, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        try:
            rvec, tvec = refine_pose(
                k_arr, np.asarray(rvec0, dtype=np.float64),
                np.asarray(tvec0, dtype=np.float64), point_obs, lobs)
        except CalibError:
            continue
        out[f] = FramePose(
            rvec=rvec,
            tvec=tvec,
            residual_px=_reproj_rms_px(k_arr, rvec, tvec, point_obs),
            n_points=len(idxs),
            fold_count=_fold_for_pose(k_arr, rvec, tvec, size),
        )
    return out
