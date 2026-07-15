"""Rasterize the known field structure through a trusted homography into a per-pixel
line-class mask — the auto-label core for the line-segmentation data engine.

Every drawn pixel is guaranteed physical, by three composed gates: polylines are
projected segment-wise through the pitch->pixel map with behind-camera clipping
(clipped_polyline); sub-segments whose paint width collapses below _MIN_PAINT_PX are
skipped (near-horizon scale collapse would put pixel centres kilometres off geometry
while keeping w > 0); and a final erase drops any raster spill past the horizon. A pose
whose far end flips behind the camera yields missing pixels, never wrapped ones.
Thickness is physically scaled (~0.12 m of paint) via the across-line projection scale.
Pure: no I/O.
Spec: docs/superpowers/specs/2026-07-14-line-mask-autolabel-design.md
"""
from __future__ import annotations

from itertools import pairwise
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M
from soccer_vision.pitch.spec import PitchSpec
from soccer_vision.viz.pitch_overlay import clipped_polyline

CLS_TOUCHLINE: Final = 1
CLS_GOAL_LINE: Final = 2
CLS_MIDLINE: Final = 3
CLS_BOX_LINE: Final = 4
CLS_CENTER_CIRCLE: Final = 5
LINE_CLASSES: Final[dict[int, str]] = {
    CLS_TOUCHLINE: "touchline",
    CLS_GOAL_LINE: "goal_line",
    CLS_MIDLINE: "midline",
    CLS_BOX_LINE: "box_line",
    CLS_CENTER_CIRCLE: "center_circle",
}
# BGR tint per class for contact sheets (background stays untinted)
_OVERLAY_BGR: Final[dict[int, tuple[int, int, int]]] = {
    CLS_TOUCHLINE: (0, 255, 255), CLS_GOAL_LINE: (255, 128, 0),
    CLS_MIDLINE: (0, 128, 255), CLS_BOX_LINE: (255, 0, 255),
    CLS_CENTER_CIRCLE: (0, 255, 0),
}
_SCALE_M = np.array([WIDTH_M, LENGTH_M])
_SEG_STEPS = 40      # subdivisions per polyline edge (curvature + partial clipping)
# Thickness clamps: floor 2 px so a line survives JPEG/resize augmentation (a 1-px
# line vanishes); cap 7 px so far-field lines can't bloat into blobs — labels must
# stay line-like.
_THICK_MIN, _THICK_MAX = 2, 7
# Skip sub-segments whose 0.12 m of paint is narrower than this many px measured
# ACROSS the drawn stroke in the image: 1 px would then span > 0.4 m of ground, so
# raster slack alone poisons the label. Near-horizon poses collapse the field into
# a sub-pixel band whose pixels map kilometres off geometry while keeping w > 0;
# the collapse is anisotropic, so the scale must be probed along the image normal
# of the drawn segment (a pitch-space perpendicular can project to a sane-looking
# length via the uncollapsed axis). Clean far-field geometry on the bake-off
# camera bottoms out at ~0.6 px, 2x above this gate.
_MIN_PAINT_PX = 0.3
_SHIFT = 4           # cv2 subpixel bits: int-truncated endpoints are ~1 px biased,
_SHIFT_MUL = 1 << _SHIFT   # which is 0.2 m at the far field edge — draw at 1/16 px


def _pitch_polylines(spec: PitchSpec) -> list[tuple[int, NDArray[np.float64]]]:
    """(class_id, (N,2) pitch-coord polyline) for the field structure. Draw order =
    list order; later classes win at intersections (midline/circle over touchlines)."""
    bl = spec.penalty_box_length_frac
    cx_l = 0.5 - spec.penalty_box_width_frac / 2.0
    cx_r = 0.5 + spec.penalty_box_width_frac / 2.0
    r_y = spec.center_circle_radius_frac                 # fraction of pitch LENGTH
    r_x = r_y * LENGTH_M / WIDTH_M                       # same metres, width units
    th = np.linspace(0.0, 2.0 * np.pi, 73)
    circle = np.column_stack([0.5 + r_x * np.cos(th), 0.5 + r_y * np.sin(th)])
    return [
        (CLS_TOUCHLINE, np.array([[0.0, 0.0], [0.0, 1.0]])),
        (CLS_TOUCHLINE, np.array([[1.0, 0.0], [1.0, 1.0]])),
        (CLS_GOAL_LINE, np.array([[0.0, 0.0], [1.0, 0.0]])),
        (CLS_GOAL_LINE, np.array([[0.0, 1.0], [1.0, 1.0]])),
        (CLS_BOX_LINE, np.array([[cx_l, 0.0], [cx_l, bl], [cx_r, bl], [cx_r, 0.0]])),
        (CLS_BOX_LINE, np.array([[cx_l, 1.0], [cx_l, 1.0 - bl], [cx_r, 1.0 - bl], [cx_r, 1.0]])),
        (CLS_MIDLINE, np.array([[0.0, 0.5], [1.0, 0.5]])),
        (CLS_CENTER_CIRCLE, circle),
    ]


def line_mask(
    h_img_to_pitch: NDArray[np.floating[Any]],
    size: tuple[int, int],
    *,
    spec: PitchSpec | None = None,
    paint_width_m: float = 0.12,
) -> NDArray[np.uint8]:
    """(H, W) uint8 class mask for a frame with FULL-PIXEL image->pitch homography."""
    w, h = size
    hm = np.asarray(h_img_to_pitch, np.float64)
    p = np.linalg.inv(hm)                                       # pitch -> pixel
    mask = np.zeros((h, w), np.uint8)
    for cls, poly in _pitch_polylines(spec or PitchSpec.standard_9v9()):
        for a, b in pairwise(poly):
            steps = np.linspace(0.0, 1.0, _SEG_STEPS + 1)
            samples = a[None, :] + steps[:, None] * (b - a)[None, :]
            for s0, s1 in pairwise(samples):
                # clipped_polyline is the physical/in-frame gate; the drawing
                # coordinates come from the same FLOAT projections because (a) the
                # circle's sub-segments are ~0.3 px, so int-quantized lengths blow
                # up the thickness ratio, and (b) int truncation shifts far-field
                # endpoints by up to a metre-scale pixel.
                # margin=200 (not the default 80): this gate is all-or-nothing per
                # sub-segment, so a sub-segment with one endpoint slightly
                # off-frame would drop its in-frame pixels too; sub-segments are
                # <~100 px long, so 200 keeps every partially-visible one.
                seg = clipped_polyline(p, np.array([s0, s1]), size=size, margin=200)
                if len(seg) != 2:
                    continue
                d_m = (s1 - s0) * _SCALE_M
                m_len = float(np.linalg.norm(d_m))
                if m_len <= 1e-9:
                    continue
                q0, q1 = (p @ np.array([s[0], s[1], 1.0]) for s in (s0, s1))
                a0 = (float(q0[0] / q0[2]), float(q0[1] / q0[2]))
                a1 = (float(q1[0] / q1[2]), float(q1[1] / q1[2]))
                seg_px = float(np.hypot(a1[0] - a0[0], a1[1] - a0[1]))
                if seg_px <= 1e-12:
                    continue                  # fully collapsed: no stroke direction
                # Paint width is measured ACROSS the drawn stroke: map the segment
                # midpoint and its 1-px image-normal neighbour back to pitch metres
                # (foreshortening makes the along-line px/m scale up to ~4x the
                # across-line one at the far field, so an along-line ratio would
                # over-thicken far lines by whole metres).
                mid_x = (a0[0] + a1[0]) / 2.0
                mid_y = (a0[1] + a1[1]) / 2.0
                nrm_x = (a0[1] - a1[1]) / seg_px
                nrm_y = (a1[0] - a0[0]) / seg_px
                gm = hm @ np.array([mid_x, mid_y, 1.0])
                gn = hm @ np.array([mid_x + nrm_x, mid_y + nrm_y, 1.0])
                if gm[2] <= 1e-9 or gn[2] <= 1e-9:
                    continue                  # 1 px crosses the horizon: degenerate
                m_per_px = float(np.linalg.norm(
                    (gn[:2] / gn[2] - gm[:2] / gm[2]) * _SCALE_M
                ))
                if m_per_px * _MIN_PAINT_PX > paint_width_m:
                    continue                  # paint < gate px: label would be poison
                t = min(max(round(paint_width_m / m_per_px), _THICK_MIN), _THICK_MAX)
                cv2.line(
                    mask,
                    (round(a0[0] * _SHIFT_MUL), round(a0[1] * _SHIFT_MUL)),
                    (round(a1[0] * _SHIFT_MUL), round(a1[1] * _SHIFT_MUL)),
                    int(cls), t, cv2.LINE_8, shift=_SHIFT,
                )
    # Physical guarantee: thick strokes and int-truncated endpoints can spill a few
    # pixels past the horizon (where image->pitch w flips sign); erase them so EVERY
    # drawn pixel maps in front of the camera.
    ys, xs = np.nonzero(mask)
    if len(xs):
        w_img = hm[2, 0] * xs + hm[2, 1] * ys + hm[2, 2]
        drop = w_img <= 1e-9
        if bool(drop.any()):
            mask[ys[drop], xs[drop]] = 0
    return mask


def mask_overlay(
    frame: NDArray[np.uint8], mask: NDArray[np.uint8], *, alpha: float = 0.6
) -> NDArray[np.uint8]:
    """Tint a frame with the mask's classes (contact-sheet rendering; humans assess)."""
    out = frame.copy()
    for cls, bgr in _OVERLAY_BGR.items():
        sel = mask == cls
        if sel.any():
            out[sel] = (np.asarray(bgr, np.float64) * alpha
                        + out[sel].astype(np.float64) * (1.0 - alpha)).astype(np.uint8)
    return out
