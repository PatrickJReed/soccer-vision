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
from typing import Any

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
from soccer_vision.pitch.manual_anchor import (
    Click,
    propagate_clicks,
    propagate_clicks_with_distance,
)


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


def _robust_sqpnp(
    k: NDArray[np.floating],
    ids: list[int],
    img: NDArray[Any],
    *,
    thr: float = 40.0,
    min_points: int = 4,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int], list[int]] | None:
    """Planar-safe robust PnP by iterative drop-worst SQPNP.

    cv2.solvePnPRansac degenerates on the coplanar (Z=0) field, so instead: SQPNP the
    current point set, find the worst reprojection residual; if it exceeds `thr`, drop
    that point and refit; repeat until the worst residual is within `thr` or fewer than
    `min_points` remain. `ids` are landmark indices; `img` is the matching (N, 2) pixel
    array. Returns (rvec, tvec, inlier_ids, outlier_ids), or None if it can't converge.

    To avoid masking (a severe outlier pulling the fit so that clean points look bad),
    each drop step uses leave-one-out scoring: try removing each current candidate, fit
    SQPNP on the remainder, score by the max residual of the remainder, and permanently
    drop the candidate whose removal yields the best (lowest-max-residual) fit.
    """
    fp = field_points_3d()
    k_arr = np.asarray(k, dtype=np.float64)
    img_arr = np.asarray(img, dtype=np.float64)
    keep = list(range(len(ids)))
    while len(keep) >= min_points:
        oi = [ids[i] for i in keep]
        ok, rvec, tvec = cv2.solvePnP(
            fp[oi].astype(np.float64), img_arr[keep], k_arr, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        proj = cv2.projectPoints(fp[oi], rvec, tvec, k_arr, np.zeros(5))[0].reshape(-1, 2)
        res = np.linalg.norm(proj - img_arr[keep], axis=1)
        if float(res.max()) <= thr:
            kept = set(keep)
            inlier_ids = [ids[i] for i in keep]
            outlier_ids = [ids[i] for i in range(len(ids)) if i not in kept]
            return (np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64),
                    inlier_ids, outlier_ids)
        # Leave-one-out: drop the candidate whose removal produces the lowest max residual
        best_drop: int | None = None
        best_score = float("inf")
        for cand in range(len(keep)):
            trial = [keep[j] for j in range(len(keep)) if j != cand]
            if len(trial) < min_points:
                continue
            oi2 = [ids[i] for i in trial]
            ok2, rv2, tv2 = cv2.solvePnP(
                fp[oi2].astype(np.float64), img_arr[trial], k_arr, None,
                flags=cv2.SOLVEPNP_SQPNP)
            if not ok2:
                continue
            proj2 = cv2.projectPoints(fp[oi2], rv2, tv2, k_arr, np.zeros(5))[0].reshape(-1, 2)
            score = float(np.linalg.norm(proj2 - img_arr[trial], axis=1).max())
            if score < best_score:
                best_score = score
                best_drop = cand
        if best_drop is None:
            return None
        keep.pop(best_drop)
    return None


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
    frames: Sequence[int] | None = None,
) -> dict[int, FramePose]:
    """Engine A: propagate clicks into each frame, then SQPNP (focal fixed).

    Per target frame: gather window-propagated point landmarks, seed a pose with SQPNP
    on the propagated points, and (only if line_obs are present for that frame) refine
    with the line residuals (Phase-2 path). Outlier-click robustness is a separate
    clicked-frame preprocessing step (`flag_outlier_clicks`), so this engine stays the
    plain fast per-frame fit. `frames=` restricts the targets (windowed recompute).
    Returns {frame: FramePose}.
    """
    w, h = size
    k_arr = np.asarray(k, dtype=np.float64)
    fp = field_points_3d()
    propagated = propagate_clicks(clicks, transforms, segment_of, window=window, frames=frames)
    out: dict[int, FramePose] = {}
    for f, kpmap in propagated.items():
        if len(kpmap) < min_points:
            continue
        idxs = sorted(kpmap)
        point_obs = [(i, kpmap[i][0] * w, kpmap[i][1] * h) for i in idxs]
        lobs = list(line_obs.get(f, [])) if line_obs else []
        obj = fp[idxs].astype(np.float64)
        img = np.array([[x, y] for _, x, y in point_obs], dtype=np.float64)
        ok, rvec0, tvec0 = cv2.solvePnP(obj, img, k_arr, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        rvec = np.asarray(rvec0, dtype=np.float64)
        tvec = np.asarray(tvec0, dtype=np.float64)
        if lobs:
            try:
                rvec, tvec = refine_pose(k_arr, rvec, tvec, point_obs, lobs)
            except CalibError:
                pass  # keep the SQPNP pose
        out[f] = FramePose(
            rvec=rvec,
            tvec=tvec,
            residual_px=_reproj_rms_px(k_arr, rvec, tvec, point_obs),
            n_points=len(idxs),
            fold_count=_fold_for_pose(k_arr, rvec, tvec, size),
        )
    return out


def poses_by_gated_propagation(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    k: NDArray[np.floating],
    size: tuple[int, int],
    *,
    max_reach: int,
    seed_size: int = 6,
    gate_px: float = 60.0,
    gap_dist: int = 180,
    min_points: int = 4,
    line_obs: Mapping[int, Sequence[tuple[str, float, float]]] | None = None,
    frames: Sequence[int] | None = None,
) -> dict[int, FramePose]:
    """Reliability-aware Engine A. Per target frame: propagate landmarks (tracking each
    one's source frame-distance), red if the nearest is past `gap_dist`, seed an SQPNP pose
    from the nearest `seed_size` landmarks, accept farther landmarks only if they reproject
    within `gate_px` of the seed, then final SQPNP (+ refine_pose when line_obs present). A
    far click drifted across the chain can't corrupt a frame with good local clicks; true
    gaps stay red. Returns {frame: FramePose}.
    """
    w, h = size
    k_arr = np.asarray(k, dtype=np.float64)
    fp = field_points_3d()
    prop = propagate_clicks_with_distance(
        clicks, transforms, segment_of, window=max_reach, frames=frames)
    out: dict[int, FramePose] = {}
    for f, kpmap in prop.items():
        if len(kpmap) < min_points:
            continue
        if min(xyd[2] for xyd in kpmap.values()) > gap_dist:  # no reliable nearby anchor
            continue
        items = sorted(kpmap.items(), key=lambda kv: kv[1][2])  # ascending source distance
        kps = [kp for kp, _ in items]
        img = np.array([[v[0] * w, v[1] * h] for _, v in items], dtype=np.float64)
        n_seed = min(seed_size, len(kps))
        if n_seed < min_points:
            continue
        ok, rvec0, tvec0 = cv2.solvePnP(
            fp[kps[:n_seed]].astype(np.float64), img[:n_seed], k_arr, None,
            flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        proj = cv2.projectPoints(
            fp[kps].astype(np.float64), rvec0, tvec0, k_arr, np.zeros(5))[0].reshape(-1, 2)
        err = np.sqrt(np.sum((proj - img) ** 2, axis=1))  # NaN (behind-camera) -> gated out
        # items are distance-sorted, so i < n_seed is the seed (always kept); farther
        # landmarks join only if they reproject within gate_px of the seed pose.
        keep = [i for i in range(len(kps)) if i < n_seed or err[i] <= gate_px]
        if len(keep) < min_points:
            continue
        keep_kps = [kps[i] for i in keep]
        keep_img = img[keep]
        point_obs = [(keep_kps[i], float(keep_img[i, 0]), float(keep_img[i, 1]))
                     for i in range(len(keep_kps))]
        ok2, rvec2, tvec2 = cv2.solvePnP(
            fp[keep_kps].astype(np.float64), keep_img, k_arr, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok2:
            continue
        rvec = np.asarray(rvec2, dtype=np.float64)
        tvec = np.asarray(tvec2, dtype=np.float64)
        lobs = list(line_obs.get(f, [])) if line_obs else []
        if lobs:
            try:
                rvec, tvec = refine_pose(k_arr, rvec, tvec, point_obs, lobs)
            except CalibError:
                pass  # keep the SQPNP pose
        out[f] = FramePose(
            rvec=rvec,
            tvec=tvec,
            residual_px=_reproj_rms_px(k_arr, rvec, tvec, point_obs),
            n_points=len(keep_kps),
            fold_count=_fold_for_pose(k_arr, rvec, tvec, size),
        )
    return out


def flag_outlier_clicks(
    clicks: Sequence[Click],
    k: NDArray[np.floating],
    size: tuple[int, int],
    *,
    thr: float = 40.0,
    min_points: int = 4,
) -> tuple[list[Click], dict[int, list[int]]]:
    """Detect outlier clicks PER CLICKED FRAME (on that frame's own clicks).

    For each clicked frame with >= min_points clicks, run `_robust_sqpnp` on its own
    clicks (pixel = normalized x*w, y*h); any landmark whose reprojection exceeds `thr`
    is an outlier — removed from the returned `clean_clicks` and recorded in
    `flagged: {frame: [kp_idx]}`. Cheap (only the clicked frames). Robustness lives
    here, NOT in the per-propagated-frame fit (which would drop chain-noise everywhere).
    """
    w, h = size
    k_arr = np.asarray(k, dtype=np.float64)
    by_frame: dict[int, list[Click]] = {}
    for c in clicks:
        by_frame.setdefault(c.frame, []).append(c)
    clean: list[Click] = []
    flagged: dict[int, list[int]] = {}
    for f, cs in by_frame.items():
        if len(cs) < min_points:
            clean.extend(cs)
            continue
        ids = [c.kp_idx for c in cs]
        img = np.array([[c.x * w, c.y * h] for c in cs], dtype=np.float64)
        fit = _robust_sqpnp(k_arr, ids, img, thr=thr, min_points=min_points)
        if fit is None:
            clean.extend(cs)
            continue
        _rvec, _tvec, _inliers, outlier_ids = fit
        if outlier_ids:
            flagged[f] = outlier_ids
        bad = set(outlier_ids)
        clean.extend(c for c in cs if c.kp_idx not in bad)
    return clean, flagged
