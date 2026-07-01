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
from soccer_vision.pitch.homography import HomographyError, fit_homography
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
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


def _foreground_errors(
    k: NDArray[np.floating[Any]],
    po: list[tuple[int, float, float]],
    line_clicks: Sequence[LineClick],
    size: tuple[int, int],
) -> list[float] | None:
    """Held-out near-touchline error (feet) for ONE frame: refit the pose WITHOUT its
    near-touchline clicks (points + any other lines only), then measure how far those
    near-touchline clicks land from the x=0 line. None if the frame has no near-touchline
    click (foreground unverifiable)."""
    if not any(lc.line_id == "near_touchline" for lc in line_clicks):
        return None
    w, h = size
    lo_fit = [(lc.line_id, lc.x * w, lc.y * h)
              for lc in line_clicks if lc.line_id != "near_touchline"]
    pose = _anchor_pose(k, po, lo_fit, None)
    if pose is None:
        return None
    rv, tv = pose
    h_norm = np.asarray(frame_homography(k, rv, tv), dtype=np.float64) @ np.diag(
        [float(w), float(h), 1.0])
    errs = [_line_perp_feet(_apply(h_norm, np.array([[lc.x, lc.y]]))[0], "near_touchline")
            for lc in line_clicks if lc.line_id == "near_touchline"]
    return errs or None


def _grade(
    k: NDArray[np.floating[Any]],
    po: list[tuple[int, float, float]],
    line_clicks: Sequence[LineClick],
    size: tuple[int, int],
) -> str:
    """green if the anchor's held-out near-touchline foreground error is within tolerance;
    yellow otherwise (including no near-touchline click -> foreground unverified)."""
    errs = _foreground_errors(k, po, line_clicks, size)
    if errs is None:
        return "yellow"
    return "green" if float(np.median(errs)) <= FOREGROUND_OK_FT else "yellow"


def _grid(n: int) -> NDArray[np.float64]:
    xs = np.linspace(0.0, 1.0, n)
    gx, gy = np.meshgrid(xs, xs)
    return np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])


def _shift_h(anchor_h: NDArray[np.floating[Any]], m_a: NDArray[np.floating[Any]],
             m_t: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    return np.asarray(
        np.asarray(anchor_h, dtype=np.float64)
        @ np.linalg.inv(np.asarray(m_a, dtype=np.float64))
        @ np.asarray(m_t, dtype=np.float64),
        dtype=np.float64,
    )


def _bracket_h(h_lo: NDArray[np.floating[Any]], h_hi: NDArray[np.floating[Any]],
               w: float) -> NDArray[np.float64]:
    g = _grid(GRID_N)
    blended = (1.0 - w) * _apply(h_lo, g) + w * _apply(h_hi, g)
    try:
        return np.asarray(fit_homography(g[:, :2], blended), dtype=np.float64)
    except HomographyError:
        return np.asarray(h_lo if w < 0.5 else h_hi, dtype=np.float64)


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
        if not self.anchor_h or frame not in self.transforms:
            return None
        gap = self.nearest_anchor_gap(frame)
        if gap is None or gap > self.gap_guard:
            return None
        anchors = sorted(self.anchor_h)
        lo = [a for a in anchors if a < frame]
        hi = [a for a in anchors if a > frame]
        if lo and hi and lo[-1] in self.transforms and hi[0] in self.transforms:
            a, b = lo[-1], hi[0]
            h_lo = _shift_h(self.anchor_h[a], self.transforms[a], self.transforms[frame])
            h_hi = _shift_h(self.anchor_h[b], self.transforms[b], self.transforms[frame])
            return _bracket_h(h_lo, h_hi, (frame - a) / (b - a))
        a = lo[-1] if lo else hi[0]
        if a not in self.transforms:
            return None
        return _shift_h(self.anchor_h[a], self.transforms[a], self.transforms[frame])

    def status(self, frame: int) -> str:
        """green = anchor that passed its own held-out foreground self-check and projects
        plausibly; yellow = anchor with unverified foreground OR a propagated (unclicked)
        frame within the gap guard; red = no homography (beyond gap / uncalibrated) or an
        implausible whole-field projection (fold_count out of range)."""
        h = self.frame_homography(frame)
        if h is None:
            return "red"
        if not FOLD_MIN <= _fold(h, self.size) <= FOLD_MAX:
            return "red"
        if frame in self.anchor_h:
            return self.coverage_grade.get(frame, "yellow")
        return "yellow"


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
        grade[f] = _grade(K, po, lcs, size)
    return PhysicalCalib(K, poses, anchor_h, grade, tf, size, gap_guard)


@dataclass(frozen=True)
class GateReport:
    """Held-out acceptance metrics (feet) for a session's physical calibration."""

    fg_median_ft: float   # near-touchline foreground (held out)
    fg_p90_ft: float
    fg_n: int
    prop_median_ft: float  # leave-one-anchor-out bracket propagation (within gap)
    prop_p90_ft: float
    prop_n: int
    passed_numeric: bool


def foreground_holdout(
    points: Sequence[Click],
    lines: Sequence[LineClick],
    size: tuple[int, int],
    *,
    min_points: int = 4,
) -> list[float]:
    """Per-anchor held-out near-touchline error (feet), pooled across all anchors that have
    a near-touchline click. Empty if the session can't calibrate a shared focal."""
    w, h = size
    by_pt = _group(points)
    by_ln = _group(lines)
    obs = {f: [(int(c.kp_idx), float(c.x * w), float(c.y * h)) for c in cs]
           for f, cs in by_pt.items()}
    try:
        k = calibrate_camera(obs, size, min_points=6).K
    except CalibError:
        return []
    errs: list[float] = []
    for f, pcs in by_pt.items():
        if len({c.kp_idx for c in pcs}) < min_points:
            continue
        po = [(int(c.kp_idx), float(c.x * w), float(c.y * h)) for c in pcs]
        fe = _foreground_errors(k, po, by_ln.get(f, []), size)
        if fe:
            errs.extend(fe)
    return errs


def propagation_holdout(
    points: Sequence[Click],
    lines: Sequence[LineClick],
    size: tuple[int, int],
    transforms: Mapping[int, NDArray[np.floating[Any]]],
    *,
    gap_guard: int = DEFAULT_GAP_GUARD,
    min_points: int = 4,
) -> list[float]:
    """Leave-one-anchor-out: for each anchor whose nearest OTHER anchor is within the gap
    guard, refit the session without it and bracket-predict its point clicks. Feet errors."""
    by_pt = _group(points)
    anchors = sorted(f for f in by_pt if len({c.kp_idx for c in by_pt[f]}) >= min_points)
    errs: list[float] = []
    for held in anchors:
        others = [a for a in anchors if a != held]
        if not others or min(abs(held - a) for a in others) > gap_guard:
            continue
        rest_p = [c for c in points if c.frame != held]
        rest_l = [lc for lc in lines if lc.frame != held]
        calib = solve_session(rest_p, rest_l, size, transforms, gap_guard=gap_guard)
        hmat = calib.frame_homography(held)
        if hmat is None:
            continue
        for c in by_pt[held]:
            q = _apply(hmat, np.array([[c.x, c.y]]))[0]
            disp = (q - PITCH_LANDMARKS[c.kp_idx]) * _SCALE
            errs.append(float(math.hypot(disp[0], disp[1]) * _FT))
    return errs


def evaluate_gate(
    points: Sequence[Click],
    lines: Sequence[LineClick],
    size: tuple[int, int],
    transforms: Mapping[int, NDArray[np.floating[Any]]],
    *,
    gap_guard: int = DEFAULT_GAP_GUARD,
) -> GateReport:
    """Numeric acceptance gate: foreground held-out (median <= 5, p90 <= 12 ft) AND
    leave-one-anchor-out propagation (median <= 5 ft)."""
    fg = foreground_holdout(points, lines, size)
    pr = propagation_holdout(points, lines, size, transforms, gap_guard=gap_guard)
    fg_med = float(np.median(fg)) if fg else float("inf")
    fg_p90 = float(np.percentile(fg, 90)) if fg else float("inf")
    pr_med = float(np.median(pr)) if pr else float("inf")
    pr_p90 = float(np.percentile(pr, 90)) if pr else float("inf")
    passed = fg_med <= 5.0 and fg_p90 <= 12.0 and pr_med <= 5.0
    return GateReport(fg_med, fg_p90, len(fg), pr_med, pr_p90, len(pr), passed)
