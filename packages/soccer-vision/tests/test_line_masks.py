"""Tests for pitch/line_masks.py — homography -> field-line class masks.

Geometry constants below are PitchSpec.standard_9v9()'s ACTUAL values (the dataclass
defaults: box length 0.187, box width 0.720, circle radius 0.106 — a fraction of
pitch LENGTH). 0.157/0.592/0.087 belong to fifa_11v11(), not this project's fields.
"""
import numpy as np
from numpy.typing import NDArray
from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M
from soccer_vision.pitch import line_masks as lm
from soccer_vision.pitch.calib_anchor import frame_homography

from tests.test_global_crop import K_TRUE, RVEC, SIZE, TVEC

W, H = SIZE
H_IMG2PITCH: NDArray[np.float64] = np.asarray(frame_homography(K_TRUE, RVEC, TVEC), np.float64)
PAINT = 0.12


def _pitch_of(mask: NDArray[np.uint8], cls: int, n: int = 200) -> NDArray[np.float64]:
    """Map up to n pixels of a class through the TRUE homography into pitch coords."""
    ys, xs = np.nonzero(mask == cls)
    assert len(xs) > 0, f"class {cls} drew no pixels"
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int)
    pts = np.column_stack([xs[idx], ys[idx], np.ones(len(idx))]).astype(np.float64)
    q = (H_IMG2PITCH @ pts.T).T
    assert np.all(q[:, 2] > 0), "mask pixel maps behind the camera"
    return np.asarray(q[:, :2] / q[:, 2:3], np.float64)


def _dist_m(cls: int, p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Metre distance from pitch points to the nearest geometry of a class."""
    x_m, y_m = p[:, 0] * WIDTH_M, p[:, 1] * LENGTH_M
    if cls == lm.CLS_TOUCHLINE:
        return np.minimum(np.abs(x_m), np.abs(x_m - WIDTH_M))
    if cls == lm.CLS_GOAL_LINE:
        return np.minimum(np.abs(y_m), np.abs(y_m - LENGTH_M))
    if cls == lm.CLS_MIDLINE:
        return np.abs(y_m - LENGTH_M / 2)
    if cls == lm.CLS_CENTER_CIRCLE:
        r_m = 0.106 * LENGTH_M
        d = np.hypot(x_m - WIDTH_M / 2, y_m - LENGTH_M / 2)
        return np.abs(d - r_m)
    raise AssertionError(cls)


def test_mask_pixels_lie_on_their_line() -> None:
    mask = lm.line_mask(H_IMG2PITCH, SIZE)
    assert mask.shape == (H, W) and mask.dtype == np.uint8
    for cls in (lm.CLS_TOUCHLINE, lm.CLS_GOAL_LINE, lm.CLS_MIDLINE, lm.CLS_CENTER_CIRCLE):
        d = _dist_m(cls, _pitch_of(mask, cls))
        # paint half-width + raster slack: every sampled pixel within ~2 paint widths
        assert float(np.max(d)) < PAINT * 2.5, f"class {cls}: max {np.max(d):.3f} m off"


def test_box_lines_lie_on_box_geometry() -> None:
    mask = lm.line_mask(H_IMG2PITCH, SIZE)
    p = _pitch_of(mask, lm.CLS_BOX_LINE)
    x_m, y_m = p[:, 0] * WIDTH_M, p[:, 1] * LENGTH_M
    bl = 0.187 * LENGTH_M
    cx_l, cx_r = (0.5 - 0.720 / 2) * WIDTH_M, (0.5 + 0.720 / 2) * WIDTH_M
    d_front = np.minimum(np.abs(y_m - bl), np.abs(y_m - (LENGTH_M - bl)))
    d_sides = np.minimum(np.abs(x_m - cx_l), np.abs(x_m - cx_r))
    assert float(np.max(np.minimum(d_front, d_sides))) < PAINT * 2.5


def test_sky_pose_draws_no_unphysical_pixels() -> None:
    """Corrupt the pose so part of the field flips behind the camera: every pixel the
    mask still draws must map with w > 0 (clipping worked); wrapped/mirrored line
    pixels are the failure mode this guards."""
    bad = H_IMG2PITCH.copy()
    bad[2, :] = np.array([bad[2, 0] * 0.9, bad[2, 1] - 300.0 / (H * bad[2, 2]), bad[2, 2]])
    mask = lm.line_mask(bad, SIZE)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return  # fully clipped is acceptable
    pts = np.column_stack([xs, ys, np.ones(len(xs))]).astype(np.float64)
    q = (bad @ pts.T).T
    assert np.all(q[:, 2] > 1e-9)


def test_thickness_scales_with_depth() -> None:
    """The midline spans depth in this side view: its near-field end must be drawn
    thicker (in px) than its far-field end. This fixture's camera puts the field
    LENGTH along image x, so the midline projects near-VERTICAL (cols ~957-961) and
    depth runs along image ROWS (near touchline = bottom row ~935, far = ~378);
    thickness is therefore measured per row, not per column."""
    mask = lm.line_mask(H_IMG2PITCH, SIZE)
    rows = np.nonzero((mask == lm.CLS_MIDLINE).any(axis=1))[0]
    assert len(rows) > 50
    lo, hi = rows[5], rows[-6]
    t_lo = int((mask[lo] == lm.CLS_MIDLINE).sum())
    t_hi = int((mask[hi] == lm.CLS_MIDLINE).sum())
    near, far = max(t_lo, t_hi), min(t_lo, t_hi)
    assert near >= far
    assert 2 <= far and near <= 9  # clamps honored (7 max + raster slack)


def test_overlay_shapes() -> None:
    frame = np.zeros((H, W, 3), np.uint8)
    mask = lm.line_mask(H_IMG2PITCH, SIZE)
    out = lm.mask_overlay(frame, mask)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert int((out != 0).sum()) > 0
