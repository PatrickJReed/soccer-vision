"""Tests for pitch/global_crop.py — the exact virtual-PTZ crop model."""
import numpy as np
import pytest
from numpy.typing import NDArray
from soccer_vision.pitch import global_crop as gc
from soccer_vision.pitch.calib_anchor import frame_homography
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click

SIZE = (1920, 1080)
W, H = SIZE
K_TRUE = np.array([[1460.0, 0, W / 2], [0, 1460.0, H / 2], [0, 0, 1.0]])
# One fixed sideline camera at midfield (25 m out, 18 m up, image x along the
# field length) so a canvas-x pan sweeps goal-to-goal. Same synthetic-camera
# recipe as test_physical_calib; that file's behind-the-goal pose puts the field
# ends along canvas *y*, which would defeat this fixture's x-pan.
RVEC = np.array([[-0.9402], [-0.9402], [-1.3582]])
TVEC = np.array([[-34.25], [8.0452], [29.7368]])


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
        x0 = canvas[:, 0].min() - 0.1
        self.offsets = {
            f: np.array([x0 + span * 1.2 * f / (n_frames - 1) - 0.1, 0.0])
            for f in range(n_frames)
        }
        self.canvas = canvas
        # chain M[f] = T(d_f), optionally corrupted with linear drift in x
        self.transforms = {
            f: np.array([[1.0, 0.0, d[0] + drift * f], [0.0, 1.0, d[1]], [0.0, 0.0, 1.0]])
            for f, d in self.offsets.items()
        }

    def visible(self, frame: int) -> list[int]:
        d = self.offsets[frame]
        out = []
        for i, c in enumerate(self.canvas):
            x, y = c[0] - d[0], c[1] - d[1]
            if 0.02 <= x <= 0.98 and 0.02 <= y <= 0.98 and i != 5:
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
    w = CropWorld()
    ends = {f: np.mean([PITCH_LANDMARKS[i][1] for i in w.visible(f)])
            for f in range(w.n_frames) if len(w.visible(f)) >= 4}
    # self-check: the pan really shows both ends (else the fixture constants are wrong)
    assert min(ends.values()) < 0.4 and max(ends.values()) > 0.6
    return w


def test_translation_and_compose_roundtrip(world: CropWorld) -> None:
    f = 30
    assert np.allclose(gc._translation(world.transforms[f]), world.offsets[f])
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
