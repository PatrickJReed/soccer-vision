"""Camera calibration against the known 9v9 field: shared focal + per-frame pose.

A per-frame homography H = K [r1 | r2 | t] comes from a PHYSICAL camera pose, so it
cannot fold the far field into view (the failure of the free per-frame homography);
and each frame is solved directly against the field, so there is no chained-
registration drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M, field_line_3d, field_points_3d


def homography_from_pose(
    k: NDArray[np.floating], rvec: NDArray[np.floating], tvec: NDArray[np.floating]
) -> NDArray[np.float64]:
    """World-metres (X, Y, Z=0) -> pixel homography for a camera (K, rvec, tvec)."""
    rmat, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    cols = np.column_stack([rmat[:, 0], rmat[:, 1], np.asarray(tvec, dtype=np.float64).ravel()])
    return np.asarray(np.asarray(k, dtype=np.float64) @ cols, dtype=np.float64)


def pitch_homography(h_world: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert a world-metres->pixel homography to canonical-[0,1]^2 -> pixel."""
    return np.asarray(np.asarray(h_world, dtype=np.float64) @ np.diag([WIDTH_M, LENGTH_M, 1.0]),
                      dtype=np.float64)


def line_residual(
    h_world: NDArray[np.floating],
    p1_3d: NDArray[np.floating],
    p2_3d: NDArray[np.floating],
    clicked_px: tuple[float, float],
) -> float:
    """Perpendicular pixel distance from clicked_px to the image line through the
    projections of the two world-metre line endpoints under h_world.

    p1_3d / p2_3d are 3D world-metre endpoints on the Z=0 ground plane: only X, Y
    are used (h_world is the ground-plane homography mapping (X, Y, 1) -> pixel),
    so any Z component is ignored.

    Returns 0.0 if the projected line is degenerate (endpoints coincide or project
    at infinity) so a least-squares residual vector keeps a constant length.
    """
    h = np.asarray(h_world, dtype=np.float64)

    def _project(p: NDArray[np.floating]) -> NDArray[np.float64] | None:
        v = h @ np.array([float(p[0]), float(p[1]), 1.0])
        if abs(v[2]) < 1e-9:
            return None
        return np.asarray(v[:2] / v[2], dtype=np.float64)

    a = _project(p1_3d)
    b = _project(p2_3d)
    if a is None or b is None:
        return 0.0
    ell = np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])
    norm = math.hypot(float(ell[0]), float(ell[1]))
    if norm < 1e-9:
        return 0.0
    u, v = clicked_px
    return float(abs(ell[0] * u + ell[1] * v + ell[2]) / norm)


class CalibError(Exception):
    """Calibration could not be solved (too few/degenerate views, implausible focal)."""


@dataclass
class CalibResult:
    K: NDArray[np.float64]                                  # shared 3x3 intrinsics
    poses: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]  # frame -> (rvec, tvec)
    rms_px: dict[int, float]                                # frame -> reprojection RMS (px)
    frames: list[int]
    n_excluded: int

    def homography(self, frame: int) -> NDArray[np.float64]:
        rvec, tvec = self.poses[frame]
        return homography_from_pose(self.K, rvec, tvec)

    def pitch_homography(self, frame: int) -> NDArray[np.float64]:
        return pitch_homography(self.homography(frame))


def _build_views(
    observations: dict[int, list[tuple[int, float, float]]], min_points: int
) -> tuple[list[int], list[NDArray[np.float32]], list[NDArray[np.float32]]]:
    fp = field_points_3d()
    frames: list[int] = []
    objp: list[NDArray[np.float32]] = []
    imgp: list[NDArray[np.float32]] = []
    for f in sorted(observations):
        seen: dict[int, tuple[float, float]] = {}
        for kp, x, y in observations[f]:
            seen[int(kp)] = (float(x), float(y))  # last wins on duplicates
        ids = sorted(seen)
        if len(ids) < min_points:
            continue
        frames.append(f)
        objp.append(fp[ids].astype(np.float32))
        imgp.append(np.array([seen[i] for i in ids], dtype=np.float32))
    return frames, objp, imgp


def _per_view_rms(
    objp: list[NDArray[np.float32]],
    imgp: list[NDArray[np.float32]],
    k: NDArray[np.float64],
    dist: NDArray[np.float64],
    rvecs: list[NDArray[np.float64]],
    tvecs: list[NDArray[np.float64]],
) -> list[float]:
    out: list[float] = []
    for o, im, rv, tv in zip(objp, imgp, rvecs, tvecs, strict=True):
        proj, _ = cv2.projectPoints(o, rv, tv, k, dist)
        d = proj.reshape(-1, 2) - im
        out.append(float(np.sqrt(np.mean(np.sum(d * d, axis=1)))))
    return out


_FLAGS = (
    cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
    | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
    | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
)


def calibrate_camera(
    observations: dict[int, list[tuple[int, float, float]]],
    frame_size: tuple[int, int],
    *,
    min_points: int = 6,
    focal_init: float | None = None,
    rms_reject_px: float = 50.0,
) -> CalibResult:
    """Shared-focal + per-frame-pose calibration against the 9v9 field.

    observations: {frame: [(kp_idx, x_px, y_px), ...]}. Estimates ONE focal across
    all frames (principal point fixed at centre, no distortion) + a per-frame pose;
    one outlier-view rejection pass on reprojection RMS.
    """
    w, h = frame_size
    frames, objp, imgp = _build_views(observations, min_points)
    sparse_dropped = len(observations) - len(frames)
    if len(frames) < 3:
        raise CalibError(
            f"need >= 3 calibratable views (>= {min_points} landmarks each); got {len(frames)}")

    f0 = float(focal_init if focal_init is not None else w)

    def _solve(
        op: list[NDArray[np.float32]], ip: list[NDArray[np.float32]]
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        list[NDArray[np.float64]],
        list[NDArray[np.float64]],
    ]:
        k0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]], dtype=np.float64)
        d0 = np.zeros(5, dtype=np.float64)
        _, k_out, dist_out, rvecs_out, tvecs_out = cv2.calibrateCamera(
            op, ip, (w, h), k0, d0, flags=_FLAGS
        )
        return (
            np.asarray(k_out, dtype=np.float64),
            np.asarray(dist_out, dtype=np.float64),
            [np.asarray(r, dtype=np.float64) for r in rvecs_out],
            [np.asarray(t, dtype=np.float64) for t in tvecs_out],
        )

    k, dist, rvecs, tvecs = _solve(objp, imgp)
    rms = _per_view_rms(objp, imgp, k, dist, rvecs, tvecs)

    keep = [i for i, r in enumerate(rms) if r <= rms_reject_px]
    n_excluded = sparse_dropped + (len(frames) - len(keep))
    if 0 < (len(frames) - len(keep)) <= len(frames) - 3:
        frames = [frames[i] for i in keep]
        objp = [objp[i] for i in keep]
        imgp = [imgp[i] for i in keep]
        k, dist, rvecs, tvecs = _solve(objp, imgp)
        rms = _per_view_rms(objp, imgp, k, dist, rvecs, tvecs)

    focal = float(k[0, 0])
    if not 0.1 * w < focal < 50 * w:
        raise CalibError(
            f"implausible focal {focal:.0f}px (frame width {w}); too few views or pose diversity")

    return CalibResult(
        K=np.asarray(k, dtype=np.float64),
        poses={f: (np.asarray(rvecs[i], np.float64), np.asarray(tvecs[i], np.float64))
               for i, f in enumerate(frames)},
        rms_px={f: rms[i] for i, f in enumerate(frames)},
        frames=frames,
        n_excluded=n_excluded,
    )


def refine_pose(
    k: NDArray[np.floating],
    rvec0: NDArray[np.floating],
    tvec0: NDArray[np.floating],
    point_obs: list[tuple[int, float, float]],
    line_obs: list[tuple[str, float, float]],
    *,
    min_constraints: int = 6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Refine a camera pose (rvec, tvec) against point + line observations.

    Minimizes, via scipy least-squares over the 6 pose DOF (focal `k` held fixed),
    residuals in PITCH space (canonical [0,1]^2): per point, the 2 components of
    (clicked-pixel mapped to pitch) minus its canonical landmark; per line click, the
    1 perpendicular pitch-distance of (clicked-pixel mapped to pitch) to its field line.

    Pitch space — NOT pixel space — is deliberate: a pixel line residual is the
    perpendicular pixel distance to the image line, which is extremely foreshortening-
    sensitive near the horizon (a ~7 ft real error can read as ~700 px). Those terms
    dominate a pixel least-squares and drag the pose off a good seed (observed: points
    4 ft -> 26 ft). Pitch units are uniform, so points and lines are comparably scaled
    and the far/near-horizon line clicks no longer destabilise the solve.

    Seeded at (rvec0, tvec0) — e.g. a solvePnP / `calibrate_camera` pose. Raises
    CalibError if the constraint count (`2*len(point_obs) + len(line_obs)`) is below
    `min_constraints` (a 6-DOF pose needs >= 6 residuals) or the solve does not converge.
    """
    n_constraints = 2 * len(point_obs) + len(line_obs)
    if n_constraints < min_constraints:
        raise CalibError(
            f"need >= {min_constraints} constraints to refine a 6-DOF pose; "
            f"got {n_constraints} ({len(point_obs)} points + {len(line_obs)} lines)")

    k_arr = np.asarray(k, dtype=np.float64)
    fp = field_points_3d()
    point_ids = [int(kp) for kp, _, _ in point_obs]
    if any(not 0 <= i < len(fp) for i in point_ids):
        raise CalibError(f"point landmark index out of range [0, {len(fp)}): {point_ids}")
    scale = np.array([WIDTH_M, LENGTH_M], dtype=np.float64)  # metres -> canonical [0,1]
    # clicked pixels as homogeneous rows, and pitch-space targets
    img_h = np.array([[x, y, 1.0] for _, x, y in point_obs], dtype=np.float64).reshape(-1, 3)
    pitch_targets = (fp[point_ids][:, :2] / scale) if point_ids else np.zeros((0, 2))
    # each line -> its normalized homogeneous line in canonical pitch coords + clicked pixel
    pitch_lines: list[tuple[NDArray[np.float64], float, float]] = []
    for lid, x, y in line_obs:
        p1, p2 = field_line_3d(lid)
        a = np.array([p1[0] / WIDTH_M, p1[1] / LENGTH_M, 1.0])
        b = np.array([p2[0] / WIDTH_M, p2[1] / LENGTH_M, 1.0])
        ell = np.cross(a, b)
        n = math.hypot(float(ell[0]), float(ell[1]))
        pitch_lines.append(((ell / n) if n > 1e-12 else ell, float(x), float(y)))
    n_res = 2 * len(point_ids) + len(line_obs)

    def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:].reshape(3, 1)
        h_pitch2px = pitch_homography(homography_from_pose(k_arr, rvec, tvec))  # pitch->px
        try:
            h_img2pitch = np.linalg.inv(h_pitch2px)  # px -> canonical pitch
        except np.linalg.LinAlgError:
            return np.full(n_res, 1e3)
        parts: list[NDArray[np.float64]] = []
        if len(point_ids):
            q = (h_img2pitch @ img_h.T).T
            q = q[:, :2] / q[:, 2:3]
            parts.append((q - pitch_targets).ravel())
        if pitch_lines:
            lr: list[float] = []
            for ell, x, y in pitch_lines:
                v = h_img2pitch @ np.array([x, y, 1.0])
                qx, qy = v[:2] / v[2]
                lr.append(float(ell[0] * qx + ell[1] * qy + ell[2]))
            parts.append(np.array(lr, dtype=np.float64))
        return np.concatenate(parts) if parts else np.zeros(0)

    x0 = np.concatenate(
        [np.asarray(rvec0, dtype=np.float64).ravel(), np.asarray(tvec0, dtype=np.float64).ravel()])
    sol = least_squares(residuals, x0, method="lm")
    if sol.status < 1 or not np.all(np.isfinite(sol.x)):
        raise CalibError(
            f"pose refinement did not converge (status {sol.status}); "
            "the refined pose is not returned — the caller must handle this")
    return sol.x[:3].reshape(3, 1).copy(), sol.x[3:].reshape(3, 1).copy()
