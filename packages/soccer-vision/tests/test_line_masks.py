"""Tests for pitch/line_masks.py — homography -> field-line class masks.

Geometry constants below are PitchSpec.standard_9v9()'s ACTUAL values (the dataclass
defaults: box length 0.187, box width 0.720, circle radius 0.106 — a fraction of
pitch LENGTH). 0.157/0.592/0.087 belong to fifa_11v11(), not this project's fields.
The oracles are deliberately hardcoded literals, NOT read from PitchSpec: importing
them would make the test circular (a wrong spec would silently validate itself).
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
    if cls == lm.CLS_BOX_LINE:
        bl = 0.187 * LENGTH_M
        cx_l, cx_r = (0.5 - 0.720 / 2) * WIDTH_M, (0.5 + 0.720 / 2) * WIDTH_M
        d_front = np.minimum(np.abs(y_m - bl), np.abs(y_m - (LENGTH_M - bl)))
        d_sides = np.minimum(np.abs(x_m - cx_l), np.abs(x_m - cx_r))
        return np.minimum(d_front, d_sides)
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
    d = _dist_m(lm.CLS_BOX_LINE, _pitch_of(mask, lm.CLS_BOX_LINE))
    assert float(np.max(d)) < PAINT * 2.5


def _assert_survivors_physical_and_on_geometry(
    mask: NDArray[np.uint8], bad: NDArray[np.float64]
) -> int:
    """Every surviving pixel must map with w > 0 AND lie near its class geometry
    under the corrupted map; returns the survivor count (0 = fully clipped)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0
    pts = np.column_stack([xs, ys, np.ones(len(xs))]).astype(np.float64)
    q = (bad @ pts.T).T
    assert np.all(q[:, 2] > 1e-9)
    p = np.asarray(q[:, :2] / q[:, 2:3], np.float64)
    labels = mask[ys, xs]
    for cls in np.unique(labels):
        d = _dist_m(int(cls), p[labels == cls])
        assert float(np.max(d)) < PAINT * 2.5, f"class {cls}: {np.max(d):.3f} m off"
    return len(xs)


def test_sky_pose_draws_no_unphysical_pixels() -> None:
    """Corrupt the pose so part of the field flips behind the camera. line_mask's
    composed guarantee (clipped_polyline endpoint clipping + the _MIN_PAINT_PX
    scale-collapse gate + the final w-erase) is: every pixel still drawn maps with
    w > 0 AND lies near its class geometry under the corrupted map. w > 0 alone
    would miss wrapped/smeared strokes that land in front of the camera but
    kilometres off geometry — that is what the paint-width gate exists for.

    Scenario A (harsh): collapses the whole field into a sub-pixel image band just
    in front of the horizon; pre-gate pixels there kept w > 0 while mapping up to
    1979.7 m off their line (measured), so the gate must leave a (near-)empty mask
    and whatever survives must satisfy both invariants.

    Scenario B (h21 negated, h22 = 0.12; tuned empirically): flips the entire far
    touchline side behind the camera (w = -5.06 at pitch (1, 0/0.5/1)) while the
    near side keeps sane scale (w = +11.5) — survivors are REQUIRED here, so the
    geometry invariant is exercised non-vacuously (~37k px, max 0.063 m today)."""
    bad_a = H_IMG2PITCH.copy()
    bad_a[2, :] = np.array(
        [bad_a[2, 0] * 0.9, bad_a[2, 1] - 300.0 / (H * bad_a[2, 2]), bad_a[2, 2]]
    )
    _assert_survivors_physical_and_on_geometry(lm.line_mask(bad_a, SIZE), bad_a)

    bad_b = H_IMG2PITCH.copy()
    bad_b[2, 1] = -bad_b[2, 1]
    bad_b[2, 2] = 0.12
    p_inv = np.linalg.inv(bad_b)
    w_far = [float((p_inv @ np.array([1.0, sy, 1.0]))[2]) for sy in (0.0, 0.5, 1.0)]
    assert max(w_far) < 0, "corruption premise lost: far side no longer flips"
    n = _assert_survivors_physical_and_on_geometry(lm.line_mask(bad_b, SIZE), bad_b)
    assert n > 0, "mild corruption must leave sane-scale survivors"


def test_thickness_scales_with_depth() -> None:
    """The midline spans depth in this side view: its near-field end must be drawn
    thicker (in px) than its far-field end. This fixture's camera puts the field
    LENGTH along image x, so the midline projects near-VERTICAL (cols ~957-961) and
    depth runs along image ROWS (near touchline = bottom row ~935, far = ~378);
    thickness is therefore measured per row, not per column."""
    mask = lm.line_mask(H_IMG2PITCH, SIZE)
    rows = np.nonzero((mask == lm.CLS_MIDLINE).any(axis=1))[0]
    assert len(rows) > 50
    top, bottom = rows[5], rows[-6]  # far-field end (small row) vs near-field end
    t_top = int((mask[top] == lm.CLS_MIDLINE).sum())
    t_bottom = int((mask[bottom] == lm.CLS_MIDLINE).sum())
    assert t_bottom > t_top  # near-field STRICTLY thicker (7 vs 3 px today)
    assert 2 <= t_top and t_bottom <= 9  # clamps honored (7 max + raster slack)


def test_global_sign_invariance() -> None:
    """H and -H are projectively identical; line_mask must not read the global sign.
    Field bug (oceanside v0): labeler ANCHOR frames export with h22 < 0 while
    bracket-PROPAGATED frames pass through findHomography (h22 = +1 convention) —
    the opposite global sign — so w>0 clipping emptied 64/70 masks."""
    a = lm.line_mask(H_IMG2PITCH, SIZE)
    b = lm.line_mask(-H_IMG2PITCH, SIZE)
    assert int((a > 0).sum()) > 0
    assert np.array_equal(a, b)


def test_overlay_shapes() -> None:
    frame = np.zeros((H, W, 3), np.uint8)
    mask = lm.line_mask(H_IMG2PITCH, SIZE)
    out = lm.mask_overlay(frame, mask)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert int((out != 0).sum()) > 0
