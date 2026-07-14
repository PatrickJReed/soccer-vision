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

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError  # type: ignore[import-untyped]

from soccer_vision.calib.field_model import FIELD_LINES, LENGTH_M, METRES_TO_FEET, WIDTH_M
from soccer_vision.pitch.homography import HomographyError, fit_homography
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

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
