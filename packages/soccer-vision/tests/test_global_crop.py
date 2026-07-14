"""Tests for pitch/global_crop.py — the exact virtual-PTZ crop model."""
import dataclasses

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from soccer_vision.calib.field_model import LENGTH_M, WIDTH_M
from soccer_vision.pitch import global_crop as gc
from soccer_vision.pitch.calib_anchor import frame_homography
from soccer_vision.pitch.landmarks import NEAR_HALFWAY_IDX, PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click, LineClick

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
    # One midline line-obs alongside the points: exercises the mixed points+lines
    # residual concatenation in the solver (landmark 4 lies ON the midline).
    d_true = world.offsets[f]
    lo = [("midline", float(world.canvas[4][0] - d_true[0]), float(world.canvas[4][1] - d_true[1]))]
    d = gc._solve_offset(world.h_g, [bad, *po[1:]], lo, world.offsets[f] + 0.05)
    assert np.linalg.norm(d - world.offsets[f]) < 0.005  # soft_l1 downweights it


def test_offset_two_axis_lines_full_solve(world: CropWorld) -> None:
    f = 120
    d_true = world.offsets[f]
    lo = [
        ("midline", float(world.canvas[4][0] - d_true[0]), float(world.canvas[4][1] - d_true[1])),
        ("far_touchline", float(world.canvas[1][0] - d_true[0]), float(world.canvas[1][1] - d_true[1])),
    ]
    assert gc._line_pitch_axes(lo) == {0, 1}
    d = gc._solve_offset(world.h_g, [], lo, d_true + np.array([0.03, -0.03]))
    assert np.linalg.norm(d - d_true) < 1e-3


def test_offset_one_axis_line_keeps_prior_direction(world: CropWorld) -> None:
    f = 120
    d_true = world.offsets[f]
    lo = [("midline", float(world.canvas[4][0] - d_true[0]), float(world.canvas[4][1] - d_true[1]))]
    assert gc._line_pitch_axes(lo) == {1}
    d0 = d_true + np.array([0.05, 0.05])  # drifted chain init
    d = gc._solve_offset(world.h_g, [], lo, d0, prior=True)
    r = gc._line_residuals_m(world.h_g, d, lo)
    assert abs(r[0]) < 0.2                       # the line got satisfied (metres)
    # This fixture pans along canvas x (goal-to-goal), so the midline — pitch
    # axis 1 — pins canvas d[0]; canvas d[1] is the unconstrained direction.
    assert abs(d[1] - d0[1]) < 0.02              # unconstrained direction stayed near init


def _session(world: CropWorld, frames: list[int], *, drift: float = 0.0,
             lines: list[LineClick] | None = None,
             gap_guard: int = 200) -> gc.CropCalib:
    w = CropWorld(world.n_frames, drift=drift) if drift else world
    clicks = [c for f in frames for c in w.clicks_at(f)]
    return gc.solve_crop_session(clicks, lines or [], SIZE, w.transforms,
                                 gap_guard=gap_guard)


def _err_ft(calib: gc.CropCalib, world: CropWorld, frame: int, kp_idx: int) -> float:
    h = calib.frame_homography(frame)
    assert h is not None
    c = world.click(frame, kp_idx)
    q = gc._apply(h, np.array([[c.x, c.y]]))[0]
    return float(np.linalg.norm((q - PITCH_LANDMARKS[kp_idx]) * gc._SCALE_M) * gc._FT)


def test_solve_recovers_world(world: CropWorld) -> None:
    # Skip pan-overshoot frames with NO visible landmarks (fixture: last ~quarter):
    # they yield no clicks, so no implementation could make them anchors.
    frames = [f for f in range(0, world.n_frames, 20) if world.visible(f)]
    calib = _session(world, frames)
    for f in frames:
        assert calib.is_anchor(f)
        for i in world.visible(f):
            assert _err_ft(calib, world, f, i) < 1.0


def test_single_end_frame_places_far_end(world: CropWorld) -> None:
    """F-C1 regression — the point of this project. A frame clicked ONLY at one end
    must place the OTHER end's landmarks accurately, because H_g carries them.
    Runs on a DRIFTED chain so the test discriminates: on a drift-free chain the
    init offset already equals truth and an implementation that IGNORED the single
    click would still pass; with drift, ignoring it fails the 2 ft bound."""
    w = CropWorld(world.n_frames, drift=2e-4)
    frames = [f for f in range(0, w.n_frames, 20) if w.visible(f)]
    f_own = min(frames, key=lambda f: float(np.mean([PITCH_LANDMARKS[i][1] for i in w.visible(f)])))
    own_ids = [i for i in w.visible(f_own) if PITCH_LANDMARKS[i][1] < 0.5]
    assert len(own_ids) >= 1
    clicks = [c for f in frames if f != f_own for c in w.clicks_at(f)]
    clicks += w.clicks_at(f_own, own_ids[:1])   # ONE own-end click only
    calib = gc.solve_crop_session(clicks, [], SIZE, w.transforms)
    assert calib.is_anchor(f_own)
    # every landmark of the whole field projects to < 2 ft error under this frame's H
    for i in range(21):
        if i == NEAR_HALFWAY_IDX:
            continue
        assert _err_ft(calib, w, f_own, i) < 2.0


def test_chain_drift_corrected_at_anchors(world: CropWorld) -> None:
    """Corrupt the chain with linear drift; anchor frames must still solve to truth
    (clicks win), and unclicked frames' error stays bounded by SHORT-hop drift.
    Truth queries (visible/click) are drift-invariant — drift corrupts only the
    chain transforms — so `world` grounds the checks for the drifted session."""
    frames = [f for f in range(0, world.n_frames, 20) if world.visible(f)]
    calib = _session(world, frames, drift=2e-4)  # 0.048 norm ≈ 92 px over 240 frames
    for f in frames:
        for i in world.visible(f)[:3]:
            assert _err_ft(calib, world, f, i) < 1.0          # anchors: click-solved
    mid = frames[3] + 10                                       # 10-frame hop
    i = world.visible(mid)[0]
    assert _err_ft(calib, world, mid, i) < 3.0                 # short-hop bound, not 92 px


def test_segment_isolation(world: CropWorld) -> None:
    seg_of = {f: (0 if f < 120 else 1) for f in range(world.n_frames)}
    clicks = [c for f in (0, 40, 80) for c in world.clicks_at(f)]  # only segment 0 clicked
    calib = gc.solve_crop_session(clicks, [], SIZE, world.transforms, segment_of=seg_of)
    assert calib.frame_homography(60) is not None
    assert calib.frame_homography(150) is None                  # other segment: no H_g


def test_gap_guard(world: CropWorld) -> None:
    calib = _session(world, [0, 20], gap_guard=50)
    assert calib.frame_homography(30) is not None
    assert calib.frame_homography(200) is None                  # 180 > guard


def test_one_end_session_is_capped(world: CropWorld) -> None:
    own_frames = [f for f in range(world.n_frames)
                  if world.visible(f) and float(np.mean([PITCH_LANDMARKS[i][1]
                                                   for i in world.visible(f)])) < 0.35]
    clicks = [c for f in own_frames[:4] for c in world.clicks_at(f)]
    calib = gc.solve_crop_session(clicks, [], SIZE, world.transforms)
    seg = calib.segments[0]  # own-end landmarks have enough spread to solve H_g
    assert seg.one_end_capped
    # Negative: a session that clicked both ends is NOT capped.
    full = _session(world, [f for f in range(0, world.n_frames, 20) if world.visible(f)])
    assert not full.segments[0].one_end_capped


def test_anchor_grading(world: CropWorld) -> None:
    """anchor_status pins Task 5's status()/export gate: green for well-clicked
    anchors; yellow for under-constrained (single one-axis line) and for clicks
    scattered beyond POINT_OK_FT (median in-sample residual)."""
    frames = [f for f in range(0, world.n_frames, 20) if world.visible(f)]
    # (b) a frame with ONLY a midline click: one pitch axis -> not full DOF -> yellow
    f_line = 30
    d30 = world.offsets[f_line]
    line = LineClick(frame=f_line, line_id="midline",
                     x=float(world.canvas[4][0] - d30[0]),
                     y=float(world.canvas[4][1] - d30[1]))
    calib = _session(world, frames, lines=[line])
    status = calib.segments[0].anchor_status
    assert all(status[f] == "green" for f in frames)      # (a) well-clicked: green
    assert status[f_line] == "yellow"
    # (c) clicks scattered beyond POINT_OK_FT: alternating +-0.06 canvas so no
    # translation absorbs it -> median residual well over 6 ft -> yellow
    f_bad = 120

    def _corrupt(c: Click, k: int) -> Click:
        s = 0.06 if k % 2 == 0 else -0.06
        return Click(frame=c.frame, kp_idx=c.kp_idx, x=c.x + s, y=c.y + s)

    clicks = [c for f in frames if f != f_bad for c in world.clicks_at(f)]
    clicks += [_corrupt(c, k) for k, c in enumerate(world.clicks_at(f_bad))]
    calib2 = gc.solve_crop_session(clicks, [], SIZE, world.transforms)
    assert calib2.segments[0].anchor_status[f_bad] == "yellow"
    assert calib2.segments[0].anchor_status[frames[0]] == "green"


def test_too_few_landmarks_no_anchor(world: CropWorld) -> None:
    calib = gc.solve_crop_session(world.clicks_at(120)[:3], [], SIZE, world.transforms)
    assert calib.anchor_h == {}
    assert calib.frame_homography(120) is None


def test_status_green_anchor_and_propagated(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    calib = _session(world, frames)
    assert calib.status(frames[2]) == "green"
    assert calib.status(frames[2] + 5) == "green"     # within GREEN_RADIUS of used anchors
    assert calib.status(999_999) == "red"


def test_status_uses_bracket_anchors_not_nearest_green(world: CropWorld) -> None:
    """F-C3: a frame bracketed by a YELLOW anchor must not be green just because a
    green anchor sits nearby on the other side of it."""
    frames = list(range(0, world.n_frames, 20))
    clicks = [c for f in frames for c in world.clicks_at(f)]
    calib = gc.solve_crop_session(clicks, [], SIZE, world.transforms)
    f_mid = frames[3]
    ss = calib.segments[0]
    forced = dict(ss.anchor_status)
    forced[f_mid] = "yellow"
    calib2 = dataclasses.replace(
        calib, segments={0: dataclasses.replace(ss, anchor_status=forced)})
    probe = f_mid + 5   # bracketed by f_mid (yellow) and frames[4] (green)
    assert set(calib2.used_anchors(probe)) == {f_mid, frames[4]}
    assert calib2.status(probe) == "yellow"


def test_sky_pose_is_red(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    calib = _session(world, frames)
    ss = calib.segments[0]
    sky = ss.h_g.copy()
    # Corrupt row 2 at the h_g's NATURAL perspective scale (solved row2 ~ [0, 117, 1]):
    # -300 overwhelms the +117 y-term, so the w=0 horizon cuts the landmark set and
    # the far half flips behind the camera (7 of 21 landmarks w<0 on this geometry).
    sky[2, :] = np.array([0.9, -300.0, 1.0])
    calib2 = dataclasses.replace(calib, segments={0: dataclasses.replace(ss, h_g=sky)})
    f = frames[2]
    h = calib2.frame_homography(f)
    assert h is not None
    assert not gc._wsigns_ok(h, SIZE), "corruption no longer flips w-signs; strengthen row"
    assert calib2.status(f) == "red"


def _sky_calib(calib: gc.CropCalib) -> gc.CropCalib:
    """calib with segment 0's h_g corrupted to the sky pose (see test_sky_pose_is_red)."""
    ss = calib.segments[0]
    sky = ss.h_g.copy()
    sky[2, :] = np.array([0.9, -300.0, 1.0])
    return dataclasses.replace(calib, segments={0: dataclasses.replace(ss, h_g=sky)})


def test_green_implies_wsign_pass(world: CropWorld, monkeypatch: pytest.MonkeyPatch) -> None:
    """status() must red-gate failing w-signs INDEPENDENTLY of the fold gate.

    On this synthetic sky corruption _fold_norm happens to be 0 everywhere, so the
    fold band alone reds every frame and would mask a deleted w-sign gate. That is
    a fixture artifact, NOT a production guarantee: on REAL single-end sky poses
    the near field stays sane, so the fold count can sit inside [FOLD_MIN, FOLD_MAX]
    while the far end projects with w < 0 — the exact "lines in the sky" mechanism
    the w-sign gate exists for. The fold gate does NOT subsume it. To make this
    test falsify the gate, the sky leg opens the fold band wide so the w-sign gate
    is the ONLY possible red producer: deleting it turns these frames green."""
    frames = list(range(0, world.n_frames, 20))
    calib = _session(world, frames)
    for f in range(0, world.n_frames, 7):       # healthy leg: green implies w-signs pass
        if calib.status(f) == "green":
            h = calib.frame_homography(f)
            assert h is not None and gc._wsigns_ok(h, SIZE)
    monkeypatch.setattr(gc, "FOLD_MIN", 0)      # sky leg: take the fold gate out of play
    monkeypatch.setattr(gc, "FOLD_MAX", 100)
    sky = _sky_calib(calib)
    probed = 0
    for f in range(0, world.n_frames, 7):
        if sky.frame_homography(f) is not None:
            probed += 1
            assert sky.status(f) == "red"       # only the w-sign gate can red here
    assert probed > 0


def test_wsign_cache_matches_per_frame(world: CropWorld) -> None:
    """The per-segment cached w-sign verdict must equal the per-frame _wsigns_ok
    check for every frame (offset-invariance of the projective row) — probed on
    both the healthy calib (all pass) and the corrupted one (all fail)."""
    frames = list(range(0, world.n_frames, 20))
    calib = _session(world, frames)
    for c in (calib, _sky_calib(calib)):
        cached = c._wsigns_by_segment[0]
        for f in range(0, world.n_frames, 7):
            h = c.frame_homography(f)
            if h is not None:
                assert gc._wsigns_ok(h, SIZE) == cached


def test_one_end_capped_status_is_yellow(world: CropWorld) -> None:
    """A one-end-capped segment caps even perfect anchors at yellow (honesty rule)."""
    frames = list(range(0, world.n_frames, 20))
    calib = _session(world, frames)
    ss = calib.segments[0]
    capped = dataclasses.replace(
        calib, segments={0: dataclasses.replace(ss, one_end_capped=True)})
    assert capped.status(frames[2]) == "yellow"


def test_confidence_mapping(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    calib = _session(world, frames)
    f_anchor = frames[2]
    assert gc.frame_confidence(calib, f_anchor) == pytest.approx(0.9)
    c_near = gc.frame_confidence(calib, f_anchor + 2)
    c_far = gc.frame_confidence(calib, f_anchor + 9)
    assert 0.6 <= c_far < c_near <= 0.8
    assert gc.frame_confidence(calib, 999_999) == 0.0


def test_crop_assumption_report_pass_and_fail(world: CropWorld) -> None:
    pairs = {f: np.linalg.inv(world.transforms[f]) @ world.transforms[f + 1]
             for f in range(world.n_frames - 1)}
    rep = gc.crop_assumption_report(pairs, SIZE)
    assert rep["ok"] and rep["max_abs_rot_deg"] < 0.01 and abs(rep["max_scale_dev"]) < 1e-6
    # 0.05 rad in normalized coords ≈ 1.6° after the pixel-space aspect correction
    rot = np.array([[np.cos(0.05), -np.sin(0.05), 0], [np.sin(0.05), np.cos(0.05), 0], [0, 0, 1.0]])
    rep2 = gc.crop_assumption_report({0: rot}, SIZE)
    assert not rep2["ok"] and rep2["max_abs_rot_deg"] > 1.0


def _ntl_line_clicks(world: CropWorld, frame: int, n: int = 3) -> list[LineClick]:
    d = world.offsets[frame]
    out: list[LineClick] = []
    for yv in np.linspace(0.15, 0.85, n):
        pitch = np.array([[0.0, float(yv)]])                 # ON the near touchline
        cpt = gc._apply(np.linalg.inv(world.h_g), pitch)[0]  # canvas position
        out.append(LineClick(frame=frame, line_id="near_touchline",
                             x=float(cpt[0] - d[0]), y=float(cpt[1] - d[1])))
    return out


def test_foreground_holdout_crop(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    clicks = [c for f in frames for c in world.clicks_at(f)]
    lines = [lc for f in frames[:3] for lc in _ntl_line_clicks(world, f)]
    errs = gc.foreground_holdout_crop(clicks, lines, SIZE, world.transforms)
    assert len(errs) == 9
    assert float(np.median(errs)) < 2.0                       # feet, exact synthetic data


def test_propagation_holdout_split_by_end(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    clicks = [c for f in frames for c in world.clicks_at(f)]
    errs, by_end = gc.propagation_holdout_crop(clicks, [], SIZE, world.transforms)
    assert errs and float(np.median(errs)) < 2.0
    assert set(by_end) <= {"own", "opp", "both"}
    # Both ends must actually be measured (fixture: own=43/opp=53) — a split that
    # bucketed everything under one key would silently defeat the F-C1 evidence.
    assert {"own", "opp"} <= set(by_end)
    assert sum(len(v) for v in by_end.values()) == len(errs)


def test_evaluate_crop_gate_fails_without_foreground_evidence(world: CropWorld) -> None:
    """No near-touchline line clicks -> no foreground holdout -> the gate must FAIL
    (inf medians), never silently pass a session with zero foreground evidence."""
    frames = list(range(0, world.n_frames, 20))
    clicks = [c for f in frames for c in world.clicks_at(f)]
    rep = gc.evaluate_crop_gate(clicks, [], SIZE, world.transforms)
    assert rep.fg_n == 0 and not rep.passed_numeric


def test_evaluate_crop_gate_passes_on_clean_world(world: CropWorld) -> None:
    frames = list(range(0, world.n_frames, 20))
    clicks = [c for f in frames for c in world.clicks_at(f)]
    lines = [lc for f in frames[:3] for lc in _ntl_line_clicks(world, f)]
    rep = gc.evaluate_crop_gate(clicks, lines, SIZE, world.transforms)
    assert rep.passed_numeric
    assert rep.fg_n == 9 and rep.prop_n > 0
    assert rep.prop_by_end  # the F-C1 evidence table exists


def test_implied_camera_recovers_focal(world: CropWorld) -> None:
    # The symmetric midfield fixture pose makes the ORTHOGONALITY constraint
    # 0/0-degenerate (|d1| far below the 1e-12 guard after b-normalization), so this
    # test exercises the single-surviving-candidate branch — the guard is load-bearing
    # here: only the equal-norm constraint pins f.
    out = gc.implied_camera(world.h_g, SIZE, pp_canvas=np.array([0.5, 0.5]))
    assert out is not None
    f_px, c = out
    assert abs(f_px - 1460.0) / 1460.0 < 0.05      # within 5% of the true focal
    # camera centre in pitch metres: the fixture's true sideline camera (z up-negative)
    assert np.allclose(c, [-25.0, LENGTH_M / 2, -18.0], atol=0.5)
