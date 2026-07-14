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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore[import-untyped]
from scipy.spatial import ConvexHull, QhullError  # type: ignore[import-untyped]

from soccer_vision.calib.field_model import FIELD_LINES, LENGTH_M, METRES_TO_FEET, WIDTH_M
from soccer_vision.calib.validate import fold_count
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


def _full_dof(po: Sequence[PointObs], lo: Sequence[LineObs]) -> bool:
    """True when a frame's clicks constrain both offset DOF: any point gives two
    constraints; lines must span both pitch axes (independent-constraint count)."""
    return bool(po) or _line_pitch_axes(lo) == {0, 1}


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


def _pitch_to_px(
    h_norm: NDArray[np.floating[Any]], size: tuple[int, int]
) -> NDArray[np.float64] | None:
    """Sign-normalized pitch->pixel map from a NORMALIZED image->pitch homography
    (signed so the field centre projects with w > 0); None when singular."""
    w, h = size
    h_px = np.asarray(h_norm, np.float64) @ np.diag([1.0 / w, 1.0 / h, 1.0])
    try:
        p = np.linalg.inv(h_px)
    except np.linalg.LinAlgError:
        return None
    if float((p @ np.array([0.5, 0.5, 1.0]))[2]) < 0:
        p = -p
    return np.asarray(p, np.float64)


def _wsigns_ok(h_norm: NDArray[np.floating[Any]], size: tuple[int, int]) -> bool:
    """True iff ALL 21 canonical landmarks project with w > 0 (in front of the camera)
    under the sign-normalized pitch->pixel map. A pose whose far end flips behind the
    camera plane ("lines in the sky") fails this even when its near field looks fine."""
    p = _pitch_to_px(h_norm, size)
    if p is None:
        return False
    pts = np.column_stack([PITCH_LANDMARKS, np.ones(len(PITCH_LANDMARKS))])
    wz = (p @ pts.T).T[:, 2]
    return bool(np.all(wz > 1e-9))


def _fold_norm(h_norm: NDArray[np.floating[Any]], size: tuple[int, int]) -> int:
    """fold_count for a NORMALIZED image->pitch homography (sign-normalized)."""
    p = _pitch_to_px(h_norm, size)
    return fold_count(p, size) if p is not None else 0


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

    @cached_property
    def anchor_h(self) -> dict[int, NDArray[np.float64]]:
        """Anchor frame -> normalized H (bootstrap/duck-type parity with PhysicalCalib).

        Build once; do not call per frame."""
        out: dict[int, NDArray[np.float64]] = {}
        for ss in self.segments.values():
            for f, d in ss.offsets.items():
                out[f] = np.asarray(ss.h_g @ _t(d), np.float64)
        return out

    def is_anchor(self, frame: int) -> bool:
        ss = self.segments.get(self._segment(frame))
        return ss is not None and frame in ss.offsets

    @cached_property
    def _anchors_by_segment(self) -> dict[int, list[int]]:
        """Per-segment sorted anchor frames that have chain transforms (built once;
        the per-frame status path calls _segment_anchors 2-3x per frame)."""
        return {sid: sorted(f for f in ss.offsets if f in self.transforms)
                for sid, ss in self.segments.items()}

    def _segment_anchors(self, frame: int) -> list[int]:
        return self._anchors_by_segment.get(self._segment(frame), [])

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

    @cached_property
    def _wsigns_by_segment(self) -> dict[int, bool]:
        """Per-segment w-sign verdict, evaluated once at d = 0 (i.e. on h_g itself).

        Offset-invariant: for H_f = h_g @ T(d) the pitch->px map is
        diag(w, h, 1) @ T(-d) @ inv(h_g); T(-d) is affine (last row [0, 0, 1]) and
        the diag scales rows 1-2 only, so ROW 3 equals row 3 of inv(h_g) for every
        d. That row alone fixes each landmark's projected w sign AND the
        field-centre sign normalization, so the per-frame _wsigns_ok verdict is
        identical across a segment's frames (equivalence probed in
        test_wsign_cache_matches_per_frame)."""
        return {sid: _wsigns_ok(ss.h_g, self.size) for sid, ss in self.segments.items()}

    def status(self, frame: int) -> str:
        """green = click-solved anchor within in-sample tolerance, or a frame within
        GREEN_RADIUS whose USED bracket anchors are all green — in both cases the
        whole-field projection is physical (all-21 w>0, fold in band). yellow =
        propagated beyond radius / partially-constrained anchor / one-end-capped
        segment. red = no homography or an unphysical projection."""
        h = self.frame_homography(frame)
        if h is None:
            return "red"
        # w-sign gate via the offset-invariant per-segment cache: the pixel map is
        # now inverted once per status call (inside _fold_norm), not twice.
        if not self._wsigns_by_segment[self._segment(frame)]:
            return "red"
        if not FOLD_MIN <= _fold_norm(h, self.size) <= FOLD_MAX:
            return "red"
        ss = self.segments[self._segment(frame)]
        if ss.one_end_capped:
            return "yellow"
        if frame in ss.offsets:
            return ss.anchor_status.get(frame, "yellow")
        used = self.used_anchors(frame)
        if not used:  # unreachable: h is not None implies bracket anchors exist
            return "yellow"
        near = min(abs(frame - a) for a in used)
        if near <= GREEN_RADIUS and all(ss.anchor_status.get(a) == "green" for a in used):
            return "green"
        return "yellow"


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
    if not _full_dof(po, lo):
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
    then alternate (offset re-solve per clicked frame) <-> (H_g refine with points+lines),
    closing with a final offset sweep so stored (h_g, d) pairs are self-consistent.
    Segments without >=4 spread landmarks stay unsolved (bootstrap waits). Clicked
    frames absent from `transforms` are silently dropped (no canvas placement
    without a chain transform)."""
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
                d[f] = _solve_offset(h_g, po_of[f], lo_of[f], d[f],
                                     prior=not _full_dof(po_of[f], lo_of[f]))
            usable = [(d[f], po_of[f], lo_of[f]) for f in frames
                      if _full_dof(po_of[f], lo_of[f])]
            if usable:
                h_g = _refine_h_g(h_g, usable)
        # Closing sweep AFTER the last refine: offsets stay consistent with the
        # SHIPPED h_g (one click exactly determines its frame against it).
        for f in frames:
            d[f] = _solve_offset(h_g, po_of[f], lo_of[f], d[f],
                                 prior=not _full_dof(po_of[f], lo_of[f]))
        grade = {f: _grade_anchor(h_g, d[f], po_of[f], lo_of[f]) for f in frames}
        ys = [PITCH_LANDMARKS[i][1] for f in frames for i, _, _ in po_of[f]]
        capped = bool(ys) and (max(ys) - min(ys)) < Y_SPAN_ONE_END
        segments[seg_id] = SegmentSolve(
            h_g=np.asarray(h_g, np.float64), offsets=d, anchor_status=grade,
            one_end_capped=capped)
    return CropCalib(segments, tf, seg_of, size, gap_guard)


def frame_confidence(calib: Any, frame: int) -> float:
    """Honest export confidence for ANY engine exposing status/is_anchor/
    nearest_anchor_gap: 0.9 for a green anchor, 0.8 -> 0.6 ramp across GREEN_RADIUS
    for propagated green, 0.0 otherwise. Retires the constant-1.0 overclaim (F-C2)."""
    if calib.status(frame) != "green":
        return 0.0
    if calib.is_anchor(frame):
        return CONF_ANCHOR
    gap = calib.nearest_anchor_gap(frame) or 0
    return CONF_PROP_MAX - (CONF_PROP_MAX - CONF_PROP_MIN) * min(1.0, gap / GREEN_RADIUS)


_NEAR_TL_POINT_IDS = {0, 2}  # near-touchline endpoint landmarks (held out with its lines)


@dataclass(frozen=True)
class CropGateReport:
    """Held-out acceptance metrics (feet) for a session's crop calibration."""

    fg_median_ft: float
    fg_p90_ft: float
    fg_n: int
    prop_median_ft: float
    prop_p90_ft: float
    prop_n: int
    prop_by_end: dict[str, tuple[float, float, int]]  # end -> (median, p90, n)
    passed_numeric: bool


def foreground_holdout_crop(
    points: Sequence[Click], lines: Sequence[LineClick], size: tuple[int, int],
    transforms: Mapping[int, NDArray[np.floating[Any]]],
    *, segment_of: Mapping[int, int] | None = None,
    gap_guard: int = DEFAULT_GAP_GUARD,
) -> list[float]:
    """Remove ALL near-touchline evidence (its line clicks AND the point landmarks on
    it), re-solve, then measure the held line clicks' perpendicular error in feet."""
    held = [lc for lc in lines if lc.line_id == "near_touchline"]
    if not held:
        return []
    rest_l = [lc for lc in lines if lc.line_id != "near_touchline"]
    rest_p = [c for c in points if c.kp_idx not in _NEAR_TL_POINT_IDS]
    calib = solve_crop_session(rest_p, rest_l, size, transforms,
                               segment_of=segment_of, gap_guard=gap_guard)
    errs: list[float] = []
    for lc in held:
        h = calib.frame_homography(lc.frame)
        if h is None:
            continue
        q = _apply(h, np.array([[lc.x, lc.y]]))[0]
        errs.append(abs(float(q[0])) * float(_SCALE_M[0]) * _FT)
    return errs


def _end_of_frame(clicked: Sequence[Click]) -> str:
    ys = [float(PITCH_LANDMARKS[c.kp_idx][1]) for c in clicked]
    m = float(np.mean(ys))
    return "own" if m < 0.45 else "opp" if m > 0.55 else "both"


def propagation_holdout_crop(
    points: Sequence[Click], lines: Sequence[LineClick], size: tuple[int, int],
    transforms: Mapping[int, NDArray[np.floating[Any]]],
    *, segment_of: Mapping[int, int] | None = None,
    gap_guard: int = DEFAULT_GAP_GUARD,
) -> tuple[list[float], dict[str, list[float]]]:
    """Leave-one-anchor-out: drop each clicked frame entirely, re-solve, predict its
    point clicks. Returns (all errors ft, errors bucketed by which end the held frame
    saw) — the by-end buckets are the direct F-C1 measurement."""
    seg = dict(segment_of) if segment_of is not None else {}
    by_frame: dict[int, list[Click]] = {}
    for c in points:
        by_frame.setdefault(c.frame, []).append(c)
    errs: list[float] = []
    by_end: dict[str, list[float]] = {}
    for held, hcs in sorted(by_frame.items()):
        others = [f for f in by_frame if f != held and seg.get(f, 0) == seg.get(held, 0)]
        if not others or min(abs(held - f) for f in others) > gap_guard:
            continue
        calib = solve_crop_session(
            [c for c in points if c.frame != held],
            [lc for lc in lines if lc.frame != held],
            size, transforms, segment_of=segment_of, gap_guard=gap_guard)
        h = calib.frame_homography(held)
        if h is None:
            continue
        end = _end_of_frame(hcs)
        for c in hcs:
            q = _apply(h, np.array([[c.x, c.y]]))[0]
            e = float(np.linalg.norm((q - PITCH_LANDMARKS[c.kp_idx]) * _SCALE_M) * _FT)
            errs.append(e)
            by_end.setdefault(end, []).append(e)
    return errs, by_end


def evaluate_crop_gate(
    points: Sequence[Click], lines: Sequence[LineClick], size: tuple[int, int],
    transforms: Mapping[int, NDArray[np.floating[Any]]],
    *, segment_of: Mapping[int, int] | None = None,
    gap_guard: int = DEFAULT_GAP_GUARD,
) -> CropGateReport:
    """Numeric acceptance gate (same thresholds as the physical engine's):
    foreground med <= 5 ft & p90 <= 12 ft AND propagation med <= 5 ft."""
    fg = foreground_holdout_crop(points, lines, size, transforms,
                                 segment_of=segment_of, gap_guard=gap_guard)
    pr, by_end = propagation_holdout_crop(points, lines, size, transforms,
                                          segment_of=segment_of, gap_guard=gap_guard)

    def _stats(v: list[float]) -> tuple[float, float]:
        return ((float(np.median(v)), float(np.percentile(v, 90)))
                if v else (float("inf"), float("inf")))

    fg_med, fg_p90 = _stats(fg)
    pr_med, pr_p90 = _stats(pr)
    ends = {k: (*_stats(v), len(v)) for k, v in sorted(by_end.items())}
    passed = fg_med <= 5.0 and fg_p90 <= 12.0 and pr_med <= 5.0
    return CropGateReport(fg_med, fg_p90, len(fg), pr_med, pr_p90, len(pr), ends, passed)


class CropAssumptionReport(TypedDict):
    """Schema of crop_assumption_report (consumed as-is by the diagnostics CLI)."""

    n: int
    max_abs_rot_deg: float
    max_scale_dev: float
    max_perspective: float
    ok: bool


def crop_assumption_report(
    interframe: Mapping[int, NDArray[np.floating[Any]]], size: tuple[int, int]
) -> CropAssumptionReport:
    """Decompose NORMALIZED inter-frame pair transforms in PIXEL space and report how
    translation-pure they are. ok=False means the crop model is questionable for this
    clip (rotation/zoom/perspective present) — a loud warning, not a hard failure."""
    w, h = size
    s = np.diag([1.0 / w, 1.0 / h, 1.0])
    s_inv = np.diag([float(w), float(h), 1.0])
    rots: list[float] = []
    scales: list[float] = []
    persp: list[float] = []
    for m in interframe.values():
        g = s_inv @ np.asarray(m, np.float64) @ s     # back to pixel space
        g = g / g[2, 2]
        rots.append(math.degrees(math.atan2(g[1, 0], g[0, 0])))
        scales.append(math.sqrt(abs(float(np.linalg.det(g[:2, :2])))) - 1.0)
        persp.append(max(abs(float(g[2, 0])), abs(float(g[2, 1]))))
    rot = float(np.max(np.abs(rots))) if rots else 0.0
    scl = float(np.max(np.abs(scales))) if scales else 0.0
    per = float(np.max(persp)) if persp else 0.0
    return CropAssumptionReport(
        n=len(rots), max_abs_rot_deg=rot, max_scale_dev=scl, max_perspective=per,
        ok=bool(rot <= 0.2 and scl <= 0.005 and per <= 1e-5))


def implied_camera(
    h_g: NDArray[np.floating[Any]], size: tuple[int, int],
    *, pp_canvas: NDArray[np.floating[Any]],
) -> tuple[float, NDArray[np.float64]] | None:
    """REPORT-ONLY physical decomposition of H_g: assuming square pixels and principal
    point pp_canvas (normalized canvas units — use the mean frame centre), recover the
    focal (px) from the plane-homography orthonormality constraints and the camera
    centre (metres). None if the constraints are inconsistent (non-physical H_g):
    no positive f^2 candidate, or the two constraints disagreeing by more than 2x
    in focal (a big split usually means the pp assumption is off)."""
    w, h = size
    # pitch[0,1] -> canvas px, then pitch metres -> canvas px
    a = np.diag([float(w), float(h), 1.0]) @ np.linalg.inv(np.asarray(h_g, np.float64))
    a = a @ np.diag([1.0 / WIDTH_M, 1.0 / LENGTH_M, 1.0])
    cx, cy = float(pp_canvas[0]) * w, float(pp_canvas[1]) * h
    b = np.array([[1.0, 0, -cx], [0, 1.0, -cy], [0, 0, 1.0]]) @ a
    # Fix the homogeneous scale so the 1e-12 degeneracy guard below is scale-
    # meaningful; the f^2 ratios themselves are invariant to this normalization.
    b = b / float(np.linalg.norm(b))
    # Derivation: for the z=0 pitch plane, a ~ K [r1 r2 t]; removing the principal
    # point leaves b ~ diag(f, f, 1) [r1 r2 t] with r1.r2 = 0 and |r1| = |r2|.
    b1, b2 = b[:, 0], b[:, 1]
    # with D = diag(1/f^2, 1/f^2, 1): b1' D b2 = 0  and  |D^.5 b1| = |D^.5 b2|
    n1 = b1[0] * b2[0] + b1[1] * b2[1]
    d1 = -b1[2] * b2[2]
    n2 = (b1[0] ** 2 + b1[1] ** 2) - (b2[0] ** 2 + b2[1] ** 2)
    d2 = b2[2] ** 2 - b1[2] ** 2
    cands = [n / dd for n, dd in ((n1, d1), (n2, d2)) if abs(dd) > 1e-12 and n / dd > 0]
    if not cands:
        return None
    if len(cands) == 2 and max(cands) / min(cands) > 4.0:
        return None  # candidates disagree by > 2x in f: constraints inconsistent
    f2 = float(np.mean(cands))
    f_px = math.sqrt(f2)
    m = np.diag([1.0 / f_px, 1.0 / f_px, 1.0]) @ b
    # lam > 0 always: the decomposition is recovered up to overall-scale SIGN, so a
    # negative-scale h_g would flip the recovered camera-height sign (report-only).
    lam = 1.0 / float(np.linalg.norm(m[:, 0]))
    r1, r2, t = lam * m[:, 0], lam * m[:, 1], lam * m[:, 2]
    r3 = np.cross(r1, r2)
    rmat = np.column_stack([r1, r2, r3])
    c = -rmat.T @ t
    return f_px, np.asarray(c, np.float64)
