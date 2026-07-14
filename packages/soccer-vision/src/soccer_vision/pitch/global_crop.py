"""Global-crop calibration: ONE homography per registration segment + a 2-DOF
per-frame crop offset. Models the Trace virtual-PTZ exactly (each frame is a 2D
crop of one fixed view), so every click in a segment constrains the same global
homography and one click fully determines a clicked frame.

H_f = H_g @ T(d_f): pixel p in frame f sits at p + d_f on the segment's canvas
(the chain reference frame's coordinate system, normalized units). Offsets at
clicked frames come from clicks; unclicked frames use short-hop chain deltas
relative to their bracketing anchors — long chain compositions never enter.
Pure: no I/O. Spec: docs/superpowers/specs/2026-07-14-global-crop-calibration-design.md
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore[import-untyped]
from scipy.spatial import ConvexHull, QhullError  # type: ignore[import-untyped]

from soccer_vision.calib.field_model import FIELD_LINES, LENGTH_M, METRES_TO_FEET, WIDTH_M
from soccer_vision.pitch.homography import HomographyError, fit_homography
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click, LineClick

RANSAC_THRESH_PITCH = 0.012   # global-fit inlier gate, pitch units (~0.55-0.8 m per axis)
HULL_AREA_MIN = 0.02          # min spread (pitch-units^2) of fit landmarks
Y_SPAN_ONE_END = 0.5          # below this y-span the session saw ~one end -> cap yellow
# Grading: same conventions as physical_calib (tolerances, fold plausibility range,
# green radius, gap guard) — rationale documented there.
POINT_OK_FT = 6.0
LINE_OK_FT = 4.0
FOLD_MIN, FOLD_MAX = 4, 15
GREEN_RADIUS = 100
DEFAULT_GAP_GUARD = 200
# Export confidence tiers: green anchor, then a propagated ramp (max -> min with distance).
CONF_ANCHOR = 0.9
CONF_PROP_MAX, CONF_PROP_MIN = 0.8, 0.6
PRIOR_WEIGHT = 0.05           # weak chain prior for offset DOF the clicks don't constrain

_SCALE_M = np.array([WIDTH_M, LENGTH_M])
_FT = METRES_TO_FEET
# line_id -> (pitch axis index, constant): the five named lines are axis-aligned
_LINE_PITCH: dict[str, tuple[int, float]] = {
    "near_touchline": (0, 0.0),
    "far_touchline": (0, 1.0),
    "own_goal_line": (1, 0.0),
    "opp_goal_line": (1, 1.0),
    "midline": (1, 0.5),
}
# The named lines must stay in sync with the 3D field model's line set.
assert _LINE_PITCH.keys() == FIELD_LINES.keys()

PointObs = tuple[int, float, float]   # (kp_idx, x_norm, y_norm)
LineObs = tuple[str, float, float]    # (line_id, x_norm, y_norm)


def _translation(m: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Translation component of a (near-translation) chain transform."""
    a = np.asarray(m, dtype=np.float64)
    return np.asarray(a[:2, 2] / a[2, 2], dtype=np.float64)


def _t(d: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """The crop-offset translation factor T(d) in H_f = H_g @ T(d_f)."""
    return np.array([[1.0, 0.0, float(d[0])], [0.0, 1.0, float(d[1])], [0.0, 0.0, 1.0]])


def _apply(h: NDArray[np.floating[Any]], pts: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """pts (N,2) -> (N,2) under homography h (no w<=0 guard: solver-internal)."""
    p = np.column_stack([np.asarray(pts, np.float64), np.ones(len(pts))])
    q = (np.asarray(h, np.float64) @ p.T).T
    return np.asarray(q[:, :2] / q[:, 2:3], np.float64)


def _fit_h_g(
    canvas_pts: NDArray[np.floating[Any]], pitch_pts: NDArray[np.floating[Any]]
) -> NDArray[np.float64] | None:
    """RANSAC global fit canvas->pitch. None when the constraints are degenerate:
    < 4 distinct landmarks, or their pitch spread (hull area) is too small to pin a
    homography — the bootstrap-wait semantic (red, never a garbage fit)."""
    if len(canvas_pts) < 4:
        return None
    distinct = np.unique(np.round(np.asarray(pitch_pts, np.float64), 6), axis=0)
    if len(distinct) < 4:
        return None
    try:
        if ConvexHull(distinct).volume < HULL_AREA_MIN:  # 2-D: volume == area
            return None
    except QhullError:
        return None  # collinear
    try:
        h = fit_homography(
            np.asarray(canvas_pts, np.float64),
            np.asarray(pitch_pts, np.float64),
            ransac_thresh=RANSAC_THRESH_PITCH,
        )
    except HomographyError:
        return None
    return np.asarray(h, np.float64)


def _point_residuals_m(
    h_g: NDArray[np.floating[Any]], d: NDArray[np.floating[Any]], po: Sequence[PointObs]
) -> NDArray[np.float64]:
    """Per-point (2N,) metre residuals of clicks at offset d against their landmarks."""
    if not po:
        return np.zeros(0)
    pts = np.array([[x + d[0], y + d[1]] for _, x, y in po], np.float64)
    q = _apply(h_g, pts)
    lms = PITCH_LANDMARKS[[i for i, _, _ in po]]
    return np.asarray(((q - lms) * _SCALE_M).ravel(), np.float64)


def _line_residuals_m(
    h_g: NDArray[np.floating[Any]], d: NDArray[np.floating[Any]], lo: Sequence[LineObs]
) -> NDArray[np.float64]:
    """Per-line-click (N,) metre distances to the named (axis-aligned) pitch line."""
    if not lo:
        return np.zeros(0)
    pts = np.array([[x + d[0], y + d[1]] for _, x, y in lo], np.float64)
    q = _apply(h_g, pts)
    out: list[float] = []
    for (lid, _, _), qi in zip(lo, q, strict=True):
        ax, c = _LINE_PITCH[lid]
        out.append((float(qi[ax]) - c) * float(_SCALE_M[ax]))
    return np.array(out, np.float64)


def _line_pitch_axes(lo: Sequence[LineObs]) -> set[int]:
    """Which pitch axes (0=x, 1=y) the frame's line clicks constrain."""
    return {_LINE_PITCH[lid][0] for lid, _, _ in lo}


def _solve_offset(
    h_g: NDArray[np.floating[Any]],
    po: Sequence[PointObs],
    lo: Sequence[LineObs],
    d0: NDArray[np.floating[Any]],
    *,
    prior: bool = False,
) -> NDArray[np.float64]:
    """2-DOF canvas offset for one frame by robust least squares (metre residuals,
    soft_l1). `prior=True` adds a weak pull toward d0 — used when the frame's clicks
    constrain fewer than 2 DOF (e.g. a single one-axis line), so the unconstrained
    direction stays at the chain init instead of wandering."""
    d_init = np.asarray(d0, np.float64)

    def fun(d: NDArray[np.float64]) -> NDArray[np.float64]:
        parts = [_point_residuals_m(h_g, d, po), _line_residuals_m(h_g, d, lo)]
        if prior:
            # ~metres: canvas ≈ pitch scale is close enough for a weak tie-breaker
            parts.append((d - d_init) * _SCALE_M * PRIOR_WEIGHT)
        return np.concatenate(parts)

    res = least_squares(fun, d_init, method="trf", loss="soft_l1",
                        f_scale=0.5,  # metres (residual units): soft_l1 inlier scale
                        x_scale=1e-2)  # canvas units: typical offset correction
    return np.asarray(res.x, np.float64)


@dataclass(frozen=True, eq=False)
class SegmentSolve:
    """One segment's solved global model."""

    h_g: NDArray[np.float64]                    # canvas(norm) -> pitch[0,1]
    offsets: dict[int, NDArray[np.float64]]     # click-solved anchor offsets
    anchor_status: dict[int, str]               # "green" | "yellow" per anchor
    one_end_capped: bool                        # clicked landmarks span < Y_SPAN_ONE_END


@dataclass(frozen=True, eq=False)
class CropCalib:
    """Solved session. Same duck-typed surface LabelerState uses from PhysicalCalib."""

    segments: dict[int, SegmentSolve]
    transforms: dict[int, NDArray[np.float64]]
    segment_of: dict[int, int]
    size: tuple[int, int]
    gap_guard: int = DEFAULT_GAP_GUARD

    def _segment(self, frame: int) -> int:
        return self.segment_of.get(frame, 0)

    @property
    def anchor_h(self) -> dict[int, NDArray[np.float64]]:
        """Anchor frame -> normalized H (bootstrap/duck-type parity with PhysicalCalib)."""
        out: dict[int, NDArray[np.float64]] = {}
        for ss in self.segments.values():
            for f, d in ss.offsets.items():
                out[f] = np.asarray(ss.h_g @ _t(d), np.float64)
        return out

    def is_anchor(self, frame: int) -> bool:
        ss = self.segments.get(self._segment(frame))
        return ss is not None and frame in ss.offsets

    def _segment_anchors(self, frame: int) -> list[int]:
        ss = self.segments.get(self._segment(frame))
        if ss is None:
            return []
        return sorted(f for f in ss.offsets if f in self.transforms)

    def nearest_anchor_gap(self, frame: int) -> int | None:
        anchors = self._segment_anchors(frame)
        return min(abs(frame - a) for a in anchors) if anchors else None

    def used_anchors(self, frame: int) -> list[int]:
        """The bracket anchors an unclicked frame's offset actually interpolates from
        (F-C3: grading uses THESE, not the nearest green anywhere)."""
        anchors = self._segment_anchors(frame)
        if not anchors or frame not in self.transforms:
            return []
        if min(abs(frame - a) for a in anchors) > self.gap_guard:
            return []
        lo = [a for a in anchors if a < frame]
        hi = [a for a in anchors if a > frame]
        if lo and hi:
            return [lo[-1], hi[0]]
        return [lo[-1]] if lo else [hi[0]]

    def _offset_of(self, frame: int) -> NDArray[np.float64] | None:
        ss = self.segments.get(self._segment(frame))
        if ss is None:
            return None
        if frame in ss.offsets:
            return ss.offsets[frame]
        used = self.used_anchors(frame)
        if not used:
            return None
        t_f = _translation(self.transforms[frame])
        preds = [ss.offsets[a] + (t_f - _translation(self.transforms[a])) for a in used]
        if len(preds) == 1:
            return preds[0]
        a, b = used
        w = (frame - a) / (b - a)
        return np.asarray((1.0 - w) * preds[0] + w * preds[1], np.float64)

    def frame_homography(self, frame: int) -> NDArray[np.float64] | None:
        ss = self.segments.get(self._segment(frame))
        d = self._offset_of(frame)
        if ss is None or d is None:
            return None
        return np.asarray(ss.h_g @ _t(d), np.float64)


def _refine_h_g(
    h0: NDArray[np.float64],
    obs: Sequence[tuple[NDArray[np.float64], Sequence[PointObs], Sequence[LineObs]]],
) -> NDArray[np.float64]:
    """Refine the 8 free params of H_g against all well-constrained anchors' points
    AND line clicks (metre residuals, soft_l1). Falls back to h0 on non-finite output."""
    def fun(p: NDArray[np.float64]) -> NDArray[np.float64]:
        h = np.append(p, 1.0).reshape(3, 3)
        parts = [r for d, po, lo in obs
                 for r in (_point_residuals_m(h, d, po), _line_residuals_m(h, d, lo))]
        return np.concatenate(parts) if parts else np.zeros(1)

    p0 = (h0 / h0[2, 2]).ravel()[:8]
    res = least_squares(fun, p0, method="trf", loss="soft_l1", f_scale=0.5)  # metres
    h = np.append(res.x, 1.0).reshape(3, 3)
    return h if bool(np.isfinite(h).all()) else h0


def _grade_anchor(
    h_g: NDArray[np.float64], d: NDArray[np.float64],
    po: Sequence[PointObs], lo: Sequence[LineObs],
) -> str:
    """green iff the frame's own clicks fit in-sample within tolerance AND the clicks
    constrain both offset DOF (>=1 point, or lines spanning both pitch axes)."""
    full_dof = bool(po) or _line_pitch_axes(lo) == {0, 1}
    if not full_dof:
        return "yellow"
    pr = _point_residuals_m(h_g, d, po).reshape(-1, 2)
    pt_ft = [float(np.hypot(r[0], r[1]) * _FT) for r in pr]
    ln_ft = [abs(float(r)) * _FT for r in _line_residuals_m(h_g, d, lo)]
    if pt_ft and float(np.median(pt_ft)) > POINT_OK_FT:
        return "yellow"
    if ln_ft and float(np.median(ln_ft)) > LINE_OK_FT:
        return "yellow"
    return "green"


def solve_crop_session(
    points: Sequence[Click],
    lines: Sequence[LineClick],
    size: tuple[int, int],
    transforms: Mapping[int, NDArray[np.floating[Any]]],
    *,
    segment_of: Mapping[int, int] | None = None,
    gap_guard: int = DEFAULT_GAP_GUARD,
    rounds: int = 3,  # alternation converges ~0.6x/round; 3 bounds drifted-chain anchors < 1 ft
) -> CropCalib:
    """Solve every segment: RANSAC H_g from chain-initialized canvas correspondences,
    then alternate (offset re-solve per clicked frame) <-> (H_g refine with points+lines).
    Segments without >=4 spread landmarks stay unsolved (bootstrap waits)."""
    tf = {f: np.asarray(m, np.float64) for f, m in transforms.items()}
    seg_of: dict[int, int] = dict(segment_of) if segment_of is not None else {}
    by_pt: dict[int, list[Click]] = {}
    for c in points:
        by_pt.setdefault(c.frame, []).append(c)
    by_ln: dict[int, list[LineClick]] = {}
    for lc in lines:
        by_ln.setdefault(lc.frame, []).append(lc)

    segments: dict[int, SegmentSolve] = {}
    clicked_frames = sorted(set(by_pt) | set(by_ln))
    seg_ids = sorted({seg_of.get(f, 0) for f in clicked_frames})
    for seg_id in seg_ids:
        frames = [f for f in clicked_frames if seg_of.get(f, 0) == seg_id and f in tf]
        if not frames:
            continue
        d: dict[int, NDArray[np.float64]] = {f: _translation(tf[f]) for f in frames}
        po_of: dict[int, list[PointObs]] = {
            f: [(int(c.kp_idx), float(c.x), float(c.y)) for c in by_pt.get(f, [])]
            for f in frames}
        lo_of: dict[int, list[LineObs]] = {
            f: [(str(lc.line_id), float(lc.x), float(lc.y)) for lc in by_ln.get(f, [])]
            for f in frames}
        canvas = [[x + d[f][0], y + d[f][1]] for f in frames for _, x, y in po_of[f]]
        pitch = [PITCH_LANDMARKS[i] for f in frames for i, _, _ in po_of[f]]
        h_g = _fit_h_g(np.array(canvas, np.float64), np.array(pitch, np.float64)) \
            if canvas else None
        if h_g is None:
            continue
        for _ in range(rounds):
            for f in frames:
                full_dof = bool(po_of[f]) or _line_pitch_axes(lo_of[f]) == {0, 1}
                d[f] = _solve_offset(h_g, po_of[f], lo_of[f], d[f], prior=not full_dof)
            usable = [(d[f], po_of[f], lo_of[f]) for f in frames
                      if po_of[f] or _line_pitch_axes(lo_of[f]) == {0, 1}]
            if usable:
                h_g = _refine_h_g(h_g, usable)
        grade = {f: _grade_anchor(h_g, d[f], po_of[f], lo_of[f]) for f in frames}
        ys = [PITCH_LANDMARKS[i][1] for f in frames for i, _, _ in po_of[f]]
        capped = bool(ys) and (max(ys) - min(ys)) < Y_SPAN_ONE_END
        segments[seg_id] = SegmentSolve(
            h_g=np.asarray(h_g, np.float64), offsets=d, anchor_status=grade,
            one_end_capped=capped)
    return CropCalib(segments, tf, seg_of, size, gap_guard)
