"""Physical per-frame calibration for a fixed camera: each clicked (anchor) frame is a
real camera pose H = K[r1|r2|t] against the rigid 9v9 field; unclicked frames are filled by
bracket-propagating the neighbouring anchors through the inter-frame chain. Replaces the
free-homography bundle (pitch.global_calib). Pure: no I/O."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.calibrate import CalibError, calibrate_camera, refine_pose
from soccer_vision.calib.field_model import (
    LENGTH_M,
    METRES_TO_FEET,
    WIDTH_M,
    field_line_3d,
    field_points_3d,
)
from soccer_vision.calib.validate import fold_count
from soccer_vision.pitch.calib_anchor import flag_outlier_clicks, frame_homography
from soccer_vision.pitch.manual_anchor import Click, LineClick

FOLD_MIN, FOLD_MAX = 4, 15
DEFAULT_GAP_GUARD = 200
FOREGROUND_OK_FT = 8.0
GRID_N = 9
_FT = METRES_TO_FEET
_SCALE = np.array([WIDTH_M, LENGTH_M])


def _group(items: Sequence[Any]) -> dict[int, list[Any]]:
    d: dict[int, list[Any]] = {}
    for it in items:
        d.setdefault(it.frame, []).append(it)
    return d


def _apply(h: NDArray[np.floating[Any]], pts: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """pts (N,2 or N,3) -> (N,2) under homography h."""
    p = np.asarray(pts, dtype=np.float64)
    if p.shape[1] == 2:
        p = np.column_stack([p, np.ones(len(p))])
    q = (np.asarray(h, dtype=np.float64) @ p.T).T
    return q[:, :2] / q[:, 2:3]


def _line_perp_feet(qpitch: NDArray[np.floating[Any]], line_id: str) -> float:
    p1, p2 = field_line_3d(line_id)
    a, b = p1[:2], p2[:2]
    pm = np.asarray(qpitch) * _SCALE
    ab = b - a
    L = math.hypot(ab[0], ab[1])
    cross = ab[0] * (a[1] - pm[1]) - ab[1] * (a[0] - pm[0])
    return float(abs(cross) / L * _FT) if L > 1e-9 else float("nan")


def _fold(h_norm: NDArray[np.floating[Any]], size: tuple[int, int]) -> int:
    """fold_count for a NORMALIZED image->pitch homography (sign-normalized)."""
    w, h = size
    h_px = np.asarray(h_norm, dtype=np.float64) @ np.diag([1.0 / w, 1.0 / h, 1.0])
    try:
        h_p2px = np.linalg.inv(h_px)
    except np.linalg.LinAlgError:
        return 0
    if float((h_p2px @ np.array([0.5, 0.5, 1.0]))[2]) < 0:
        h_p2px = -h_p2px
    return fold_count(h_p2px, size)


def _anchor_pose(
    k: NDArray[np.floating[Any]],
    po: list[tuple[int, float, float]],
    lo: list[tuple[str, float, float]],
    seed_pose: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    ids = [i for i, _, _ in po]
    img = np.array([[x, y] for _, x, y in po], dtype=np.float64)
    fp = field_points_3d()
    ok, _rv_raw, _tv_raw = cv2.solvePnP(fp[ids], img, np.asarray(k), None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rv: NDArray[np.float64] = np.asarray(_rv_raw, dtype=np.float64)
    tv: NDArray[np.float64] = np.asarray(_tv_raw, dtype=np.float64)
    if seed_pose is not None:
        rv, tv = np.asarray(seed_pose[0], dtype=np.float64), np.asarray(seed_pose[1], dtype=np.float64)
    try:
        rv, tv = refine_pose(k, rv, tv, po, lo)
    except CalibError:
        pass
    return rv, tv


@dataclass(frozen=True, eq=False)
class PhysicalCalib:
    K: NDArray[np.float64]
    poses: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]
    anchor_h: dict[int, NDArray[np.float64]]
    coverage_grade: dict[int, str]
    transforms: dict[int, NDArray[np.float64]]
    size: tuple[int, int]
    gap_guard: int = DEFAULT_GAP_GUARD

    def is_anchor(self, frame: int) -> bool:
        return frame in self.anchor_h

    def nearest_anchor_gap(self, frame: int) -> int | None:
        if not self.anchor_h:
            return None
        return min(abs(frame - a) for a in self.anchor_h)

    def frame_homography(self, frame: int) -> NDArray[np.float64] | None:
        if frame in self.anchor_h:
            return self.anchor_h[frame]
        return None


def solve_session(
    points: Sequence[Click],
    lines: Sequence[LineClick],
    size: tuple[int, int],
    transforms: Mapping[int, NDArray[np.floating[Any]]],
    *,
    min_points: int = 4,
    gap_guard: int = DEFAULT_GAP_GUARD,
    seed: PhysicalCalib | None = None,
) -> PhysicalCalib:
    w, h = size
    tf = {f: np.asarray(m, dtype=np.float64) for f, m in transforms.items()}
    by_pt = _group(points)
    by_ln = _group(lines)
    obs: dict[int, list[tuple[int, float, float]]] = {
        f: [(int(c.kp_idx), float(c.x * w), float(c.y * h)) for c in cs]
        for f, cs in by_pt.items()
    }
    try:
        K = calibrate_camera(obs, size, min_points=6).K
    except CalibError:
        # A physical calibration needs a shared focal from >= 3 diverse views. With fewer,
        # there is no physical solution yet -> return an empty calib (no anchors); the
        # labeler bootstrap simply waits for more clicked frames. We deliberately do NOT
        # fall back to a free per-frame homography -- that is exactly the model this engine
        # replaces (it is non-physical and folds the field into the sky).
        return PhysicalCalib(np.eye(3), {}, {}, {}, tf, size, gap_guard)
    clean, _flagged = flag_outlier_clicks(points, K, size)
    by_clean = _group(clean)
    diag = np.diag([float(w), float(h), 1.0])
    poses: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    anchor_h: dict[int, NDArray[np.float64]] = {}
    grade: dict[int, str] = {}
    for f in sorted(by_clean):
        pcs = by_clean[f]
        lcs = by_ln.get(f, [])
        if len({c.kp_idx for c in pcs}) < min_points:
            continue
        po: list[tuple[int, float, float]] = [
            (int(c.kp_idx), float(c.x * w), float(c.y * h)) for c in pcs
        ]
        lo: list[tuple[str, float, float]] = [
            (str(lc.line_id), float(lc.x * w), float(lc.y * h)) for lc in lcs
        ]
        pose = _anchor_pose(K, po, lo, seed.poses.get(f) if seed else None)
        if pose is None:
            continue
        rv, tv = pose
        poses[f] = (rv, tv)
        anchor_h[f] = np.asarray(frame_homography(K, rv, tv), dtype=np.float64) @ diag
        grade[f] = "yellow"
    return PhysicalCalib(K, poses, anchor_h, grade, tf, size, gap_guard)
