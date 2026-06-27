"""Clicked-frame outlier flagging + small camera-pose helpers for a fixed-camera session.

The per-frame propagation engines (and the three-way `calib_compare` harness) were retired
once the field-anchored bundle solver (`pitch.global_calib.solve_bundle`) became the live
labeler path. What remains here is the robust per-clicked-frame outlier detector
(`flag_outlier_clicks` + its planar-safe `_robust_sqpnp`) and a few pose -> homography /
reprojection helpers. Internally full-pixel image space. Pure: no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.calibrate import homography_from_pose, pitch_homography
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.calib.validate import fold_count
from soccer_vision.pitch.manual_anchor import Click


def frame_homography(
    k: NDArray[np.floating], rvec: NDArray[np.floating], tvec: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Full-pixel image -> pitch-[0,1]^2 homography (the labeler export format)."""
    h_pitch = pitch_homography(homography_from_pose(k, rvec, tvec))  # canon[0,1]^2 -> px
    return np.asarray(np.linalg.inv(h_pitch), dtype=np.float64)


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
    here (a per-clicked-frame preprocessing step), not in the homography solve.
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
