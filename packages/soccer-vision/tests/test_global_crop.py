"""Tests for pitch/global_crop.py — the exact virtual-PTZ crop model."""
import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M
from soccer_vision.pitch import global_crop as gc
from soccer_vision.pitch.calib_anchor import frame_homography
from soccer_vision.pitch.landmarks import NEAR_HALFWAY_IDX, PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click

SIZE = (1920, 1080)
W, H = SIZE
K_TRUE = np.array([[1460.0, 0, W / 2], [0, 1460.0, H / 2], [0, 0, 1.0]])
PAN_MARGIN = 0.1  # canvas units of pan pre-roll left of the field (applied twice)


def _sideline_pose() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One fixed sideline camera at midfield: 25 m out from the near touchline
    (x<0), 18 m up (world z points down, so above-ground is negative z), looking
    at the field centre with the image x-axis along the field length — so a
    canvas-x pan sweeps goal-to-goal. (test_physical_calib's behind-the-goal pose
    puts the field ends along canvas *y*, which would defeat this fixture's x-pan.)
    """
    cam = np.array([-25.0, LENGTH_M / 2.0, -18.0])
    fwd = np.array([WIDTH_M / 2.0, LENGTH_M / 2.0, 0.0]) - cam
    fwd /= np.linalg.norm(fwd)
    right = np.array([0.0, 1.0, 0.0])
    right -= right.dot(fwd) * fwd
    right /= np.linalg.norm(right)
    rot = np.stack([right, np.cross(fwd, right), fwd])  # rows = camera x/y/z axes
    rvec = np.asarray(cv2.Rodrigues(rot)[0], np.float64)
    tvec = np.asarray(-rot @ cam, np.float64).reshape(3, 1)
    return rvec, tvec


RVEC, TVEC = _sideline_pose()


def _h_g_true() -> NDArray[np.float64]:
    """Ground-truth canvas(norm)->pitch homography from the physical camera."""
    h_px = frame_homography(K_TRUE, RVEC, TVEC)          # full-pixel image -> pitch
    return np.asarray(h_px @ np.diag([float(W), float(H), 1.0]), np.float64)


def _canvas_of_landmarks(h_g: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalized canvas position of each of the 21 landmarks (inverse map)."""
    inv = np.linalg.inv(h_g)
    pts = np.column_stack([PITCH_LANDMARKS, np.ones(len(PITCH_LANDMARKS))])
    q = (inv @ pts.T).T
    return np.asarray(q[:, :2] / q[:, 2:3], np.float64)


class CropWorld:
    """Ground truth: h_g + per-frame offsets + generated clicks + exact chain."""

    def __init__(self, n_frames: int = 240, drift: float = 0.0) -> None:
        self.n_frames = n_frames
        self.h_g = _h_g_true()
        canvas = _canvas_of_landmarks(self.h_g)
        span = canvas[:, 0].max() - canvas[:, 0].min()
        # Pan sweeps the canvas x-range so both field ends are seen by some frame.
        # It starts 2*PAN_MARGIN left of the field and overshoots the right edge
        # (span * 1.2), so the last ~quarter of frames (>= ~175 of 240) see fewer
        # than 4 landmarks — later tasks should place anchors well inside the clip.
        x0 = canvas[:, 0].min() - 2.0 * PAN_MARGIN
        self.offsets = {
            f: np.array([x0 + span * 1.2 * f / (n_frames - 1), 0.0])
            for f in range(n_frames)
        }
        self.canvas = canvas
        # chain M[f] = T(d_f), optionally corrupted with linear drift in x
        self.transforms = {
            f: np.array([[1.0, 0.0, d[0] + drift * f], [0.0, 1.0, d[1]], [0.0, 0.0, 1.0]])
            for f, d in self.offsets.items()
        }
        # Self-check on EVERY construction: the pan really shows both field ends
        # (else the pose/pan constants are wrong). Valid for drift worlds too:
        # drift corrupts self.transforms only, never self.offsets (the truth).
        ends = [
            float(np.mean([PITCH_LANDMARKS[i][1] for i in self.visible(f)]))
            for f in range(n_frames)
            if len(self.visible(f)) >= 4
        ]
        assert min(ends) < 0.4 and max(ends) > 0.6

    def visible(self, frame: int) -> list[int]:
        d = self.offsets[frame]
        out = []
        for i, c in enumerate(self.canvas):
            x, y = c[0] - d[0], c[1] - d[1]
            # NEAR_HALFWAY_IDX sits under the real camera: never visible, never labeled.
            if 0.02 <= x <= 0.98 and 0.02 <= y <= 0.98 and i != NEAR_HALFWAY_IDX:
                out.append(i)
        return out

    def click(self, frame: int, kp_idx: int) -> Click:
        d = self.offsets[frame]
        c = self.canvas[kp_idx]
        return Click(frame=frame, kp_idx=kp_idx, x=float(c[0] - d[0]), y=float(c[1] - d[1]))

    def clicks_at(self, frame: int, ids: list[int] | None = None) -> list[Click]:
        ids = self.visible(frame) if ids is None else ids
        return [self.click(frame, i) for i in ids]


@pytest.fixture(scope="module")
def world() -> CropWorld:
    return CropWorld()  # the both-ends self-check runs inside __init__


def test_translation_and_compose_roundtrip(world: CropWorld) -> None:
    f = 30
    assert np.allclose(gc._translation(world.transforms[f]), world.offsets[f])
    # homogeneous transforms are defined up to scale: the /m[2,2] must normalize
    assert np.allclose(gc._translation(2.0 * world.transforms[f]), world.offsets[f])
    h_f = world.h_g @ gc._t(world.offsets[f])
    c = world.clicks_at(f)[0]
    q = gc._apply(h_f, np.array([[c.x, c.y]]))[0]
    assert np.linalg.norm(q - PITCH_LANDMARKS[c.kp_idx]) < 1e-9


def test_residuals_are_metres(world: CropWorld) -> None:
    f = 30
    po = [(c.kp_idx, c.x, c.y) for c in world.clicks_at(f)]
    r = gc._point_residuals_m(world.h_g, world.offsets[f], po)
    assert r.shape == (2 * len(po),)
    assert np.abs(r).max() < 1e-6  # exact clicks -> zero metre residual
    mx, my = world.canvas[4] - world.offsets[f]
    lo = [("midline", float(mx), float(my))]
    rl = gc._line_residuals_m(world.h_g, world.offsets[f], lo)
    assert rl.shape == (1,) and abs(rl[0]) < 1e-6  # landmark 4 lies ON the midline
    # Perturbed samples pin the metre CONVERSION and its SIGN (exact zeros can't).
    inv = np.linalg.inv(world.h_g)
    delta = 0.01  # pitch-x units past landmark 6, toward the far touchline
    p_pt = PITCH_LANDMARKS[6] + np.array([delta, 0.0])
    cx, cy = gc._apply(inv, p_pt[None, :])[0] - world.offsets[f]
    rp = gc._point_residuals_m(world.h_g, world.offsets[f], [(6, float(cx), float(cy))])
    assert abs(rp[0] - delta * WIDTH_M) < 1e-9 and abs(rp[1]) < 1e-9
    off = 0.02  # pitch-y units past the midline, toward the opp goal
    lx, ly = gc._apply(inv, np.array([[0.6, 0.5 + off]]))[0] - world.offsets[f]
    rl2 = gc._line_residuals_m(world.h_g, world.offsets[f], [("midline", float(lx), float(ly))])
    assert abs(rl2[0] - off * LENGTH_M) < 1e-9


def test_fit_h_g_recovers_truth(world: CropWorld) -> None:
    canvas_pts: list[list[float]] = []
    pitch_pts: list[NDArray[np.float64]] = []
    for f in range(0, world.n_frames, 20):
        for c in world.clicks_at(f):
            d = world.offsets[f]
            canvas_pts.append([c.x + d[0], c.y + d[1]])
            pitch_pts.append(PITCH_LANDMARKS[c.kp_idx])
    h = gc._fit_h_g(np.array(canvas_pts), np.array(pitch_pts))
    assert h is not None
    q = gc._apply(h, np.array(canvas_pts))
    assert np.abs((q - np.array(pitch_pts)) * gc._SCALE_M).max() < 0.05  # < 5 cm


def test_fit_h_g_rejects_outlier(world: CropWorld) -> None:
    canvas_pts: list[list[float]] = []
    pitch_pts: list[NDArray[np.float64]] = []
    for f in range(0, world.n_frames, 20):
        for c in world.clicks_at(f):
            d = world.offsets[f]
            canvas_pts.append([c.x + d[0], c.y + d[1]])
            pitch_pts.append(PITCH_LANDMARKS[c.kp_idx])
    canvas_pts[0] = [canvas_pts[0][0] + 0.2, canvas_pts[0][1] + 0.2]  # gross outlier
    h = gc._fit_h_g(np.array(canvas_pts), np.array(pitch_pts))
    assert h is not None
    q = gc._apply(h, np.array(canvas_pts[1:]))
    assert np.abs((q - np.array(pitch_pts[1:])) * gc._SCALE_M).max() < 0.05


def test_fit_h_g_degenerate_inputs() -> None:
    three = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.1]])
    assert gc._fit_h_g(three, three) is None                       # < 4 points
    col_c = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
    col_p = np.array([[0.0, 0.0], [0.0, 0.3], [0.0, 0.6], [0.0, 0.9]])
    assert gc._fit_h_g(col_c, col_p) is None                       # collinear landmarks
    tiny = np.array([[0.5, 0.5], [0.51, 0.5], [0.5, 0.51], [0.51, 0.51]])
    assert gc._fit_h_g(tiny, tiny * 0.01 + 0.5) is None            # hull too small
    # 5 rows but only 3 unique landmark targets: the distinct-count gate fires.
    rep_c = np.array([[0.1, 0.1], [0.8, 0.15], [0.45, 0.85], [0.2, 0.5], [0.7, 0.6]])
    rep_p = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0], [1.0, 0.0]])
    assert gc._fit_h_g(rep_c, rep_p) is None                       # < 4 distinct landmarks


def test_offset_from_single_click(world: CropWorld) -> None:
    f = 120
    c = world.clicks_at(f)[0]
    d = gc._solve_offset(world.h_g, [(c.kp_idx, c.x, c.y)], [], np.zeros(2))
    assert np.linalg.norm(d - world.offsets[f]) < 1e-4  # ONE click determines the frame


def test_offset_robust_to_one_bad_click(world: CropWorld) -> None:
    f = 120
    po = [(c.kp_idx, c.x, c.y) for c in world.clicks_at(f)]
    assert len(po) >= 4
    bad = (po[0][0], po[0][1] + 0.08, po[0][2] + 0.08)  # ~1.5 m wrong
    d = gc._solve_offset(world.h_g, [bad, *po[1:]], [], world.offsets[f] + 0.05)
    assert np.linalg.norm(d - world.offsets[f]) < 0.005  # soft_l1 downweights it


def test_offset_two_axis_lines_full_solve(world: CropWorld) -> None:
    f = 120
    d_true = world.offsets[f]
    lo = [
        ("midline", float(world.canvas[4][0] - d_true[0]), float(world.canvas[4][1] - d_true[1])),
        ("far_touchline", float(world.canvas[1][0] - d_true[0]), float(world.canvas[1][1] - d_true[1])),
    ]
    assert gc._offset_axes(lo) == {0, 1}
    d = gc._solve_offset(world.h_g, [], lo, d_true + np.array([0.03, -0.03]))
    assert np.linalg.norm(d - d_true) < 1e-3


def test_offset_one_axis_line_keeps_prior_direction(world: CropWorld) -> None:
    f = 120
    d_true = world.offsets[f]
    lo = [("midline", float(world.canvas[4][0] - d_true[0]), float(world.canvas[4][1] - d_true[1]))]
    assert gc._offset_axes(lo) == {1}
    d0 = d_true + np.array([0.05, 0.05])  # drifted chain init
    d = gc._solve_offset(world.h_g, [], lo, d0, prior=True)
    r = gc._line_residuals_m(world.h_g, d, lo)
    assert abs(r[0]) < 0.2                       # the line got satisfied (metres)
    # This fixture pans along canvas x (goal-to-goal), so the midline — pitch
    # axis 1 — pins canvas d[0]; canvas d[1] is the unconstrained direction.
    assert abs(d[1] - d0[1]) < 0.02              # unconstrained direction stayed near init
