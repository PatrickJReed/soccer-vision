import cv2
import numpy as np
from numpy.typing import NDArray
from soccer_vision.calib.field_model import LENGTH_M, field_points_3d
from soccer_vision.pitch.calib_anchor import frame_homography
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click, LineClick
from soccer_vision.pitch.physical_calib import (
    PhysicalCalib,
    evaluate_gate,
    foreground_holdout,
    solve_session,
)

SIZE = (1920, 1080)
IDS = [0, 1, 4, 9, 10, 13, 14]

# One shared camera (focal 1460) panning across the field: three DISTINCT poses. A physical
# calibration needs >= 3 diverse views to estimate the shared focal, so the test supplies that.
K_TRUE = np.array([[1460.0, 0, SIZE[0] / 2], [0, 1460.0, SIZE[1] / 2], [0, 0, 1.0]])
POSES: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {
    10: (np.array([[1.15], [-0.30], [0.02]]), np.array([[-20.0], [-3.0], [42.0]])),
    20: (np.array([[1.20], [0.00], [0.00]]), np.array([[-22.0], [-3.0], [40.0]])),
    30: (np.array([[1.18], [0.30], [-0.02]]), np.array([[-24.0], [-3.0], [41.0]])),
}


def _pose_clicks(frame: int, rvec: NDArray[np.float64], tvec: NDArray[np.float64]) -> list[Click]:
    fp = field_points_3d()
    img = cv2.projectPoints(fp[IDS], rvec, tvec, K_TRUE, None)[0].reshape(-1, 2)
    w, h = SIZE
    return [Click(frame=frame, kp_idx=i, x=float(x) / w, y=float(y) / h)
            for i, (x, y) in zip(IDS, img, strict=True)]


def test_solve_session_recovers_physical_anchors() -> None:
    clicks: list[Click] = []
    for f, (rv, tv) in POSES.items():
        clicks += _pose_clicks(f, rv, tv)
    transforms = {f: np.eye(3) for f in POSES}
    calib = solve_session(clicks, [], SIZE, transforms)

    for f in POSES:
        assert calib.is_anchor(f)
        H = calib.frame_homography(f)
        assert H is not None
        for c in (c for c in clicks if c.frame == f):
            q = H @ np.array([c.x, c.y, 1.0])
            q = q[:2] / q[2]
            # clicked landmark maps back to its canonical pitch position (<~1.4 ft)
            assert np.linalg.norm(q - PITCH_LANDMARKS[c.kp_idx]) < 0.02

    # a frame with no clicks is not an anchor; T1 has no propagation yet -> None
    assert not calib.is_anchor(15)
    assert calib.frame_homography(15) is None
    assert calib.frame_homography(999) is None


def test_too_few_views_returns_empty_not_free_homography() -> None:
    # One clicked frame cannot yield a shared focal -> physical-or-nothing (no anchors),
    # NOT a free-homography fallback.
    rv, tv = POSES[20]
    calib = solve_session(_pose_clicks(20, rv, tv), [], SIZE, {20: np.eye(3)})
    assert calib.anchor_h == {}
    assert calib.frame_homography(20) is None


def _trans(dx: float) -> NDArray[np.float64]:
    return np.array([[1.0, 0, dx], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float64)


def _act(H: NDArray[np.float64], pts: NDArray[np.float64]) -> NDArray[np.float64]:
    q = (H @ pts.T).T
    return np.asarray(q[:, :2] / q[:, 2:3], dtype=np.float64)


def _driftfree_calib(gap_guard: int = 200) -> tuple[PhysicalCalib, NDArray[np.float64], dict[int, NDArray[np.float64]]]:
    transforms = {f: _trans(-0.01 * f) for f in range(0, 410)}
    H0 = np.array([[0.5, 0.02, 0.10], [0.01, 0.4, 0.20], [0.0, 0.05, 1.0]])
    anchor_h = {0: H0, 20: H0 @ transforms[20]}  # chain-consistent anchors
    calib = PhysicalCalib(K=np.eye(3), poses={}, anchor_h=anchor_h, coverage_grade={},
                          transforms=transforms, size=SIZE, gap_guard=gap_guard)
    return calib, H0, transforms


def test_bracket_recovers_interior_on_driftfree_chain() -> None:
    calib, H0, T = _driftfree_calib()
    H10 = calib.frame_homography(10)          # bracketed by anchors 0 and 20
    assert H10 is not None
    expected = H0 @ T[10]
    pts = np.array([[0.2, 0.3, 1.0], [0.7, 0.6, 1.0], [0.5, 0.5, 1.0]])
    assert np.allclose(_act(H10, pts), _act(expected, pts), atol=1e-4)


def test_one_sided_shift_beyond_last_anchor() -> None:
    calib, H0, T = _driftfree_calib()
    H25 = calib.frame_homography(25)          # beyond anchor 20, within gap, one-sided
    assert H25 is not None
    expected = H0 @ T[25]
    pts = np.array([[0.2, 0.3, 1.0], [0.6, 0.4, 1.0]])
    assert np.allclose(_act(H25, pts), _act(expected, pts), atol=1e-6)


def test_gap_guard_returns_none_far_from_anchor() -> None:
    calib, _H0, _T = _driftfree_calib(gap_guard=200)
    assert calib.frame_homography(400) is None   # 380 > 200 from nearest anchor


def test_propagation_stays_within_segment() -> None:
    # Two registration segments (chain resets to identity at each segment start). A frame in
    # one segment must propagate ONLY from that segment's anchors — never bracket across a
    # segment break (which would compose two different reference frames -> garbage).
    transforms = {f: _trans(-0.01 * f) for f in range(0, 10)}          # seg 0
    transforms.update({f: _trans(-0.01 * (f - 10)) for f in range(10, 20)})  # seg 1 (M[10]=~I)
    transforms[25] = np.eye(3)                                          # seg 2, no anchors
    h0 = np.array([[0.5, 0.02, 0.10], [0.01, 0.4, 0.20], [0.0, 0.05, 1.0]])
    h1 = np.array([[0.4, -0.03, 0.30], [0.02, 0.55, 0.05], [0.0, -0.04, 1.0]])
    anchor_h = {0: h0, 5: h0 @ transforms[5], 10: h1, 15: h1 @ transforms[15]}
    segment_of = ({f: 0 for f in range(0, 10)} | {f: 1 for f in range(10, 20)} | {25: 2})
    calib = PhysicalCalib(K=np.eye(3), poses={}, anchor_h=anchor_h, coverage_grade={},
                          transforms=transforms, size=SIZE, gap_guard=200, segment_of=segment_of)
    pts = np.array([[0.2, 0.3, 1.0], [0.6, 0.4, 1.0]])

    # frame 8 (seg 0, no seg-0 anchor above it): one-sided shift from anchor 5 -> seg-0
    # geometry, NOT bracketed across to seg-1's anchor 10 (the old cross-segment bug).
    h8 = calib.frame_homography(8)
    assert h8 is not None
    assert np.allclose(_act(h8, pts), _act(h0 @ transforms[8], pts), atol=1e-6)
    assert not np.allclose(_act(h8, pts), _act(h1 @ transforms[8], pts), atol=1e-2)

    # frame 12 (seg 1): bracketed by seg-1 anchors 10 & 15 -> seg-1 geometry
    h12 = calib.frame_homography(12)
    assert h12 is not None
    assert np.allclose(_act(h12, pts), _act(h1 @ transforms[12], pts), atol=1e-4)

    # a segment with no anchors of its own -> no homography (never borrows another segment)
    assert calib.frame_homography(25) is None


# ---- T3: coverage grade + status ----
def _near_tl_clicks(frame: int, rvec: NDArray[np.float64], tvec: NDArray[np.float64],
                    n: int = 3) -> list[LineClick]:
    obj = np.array([[0.0, y, 0.0] for y in np.linspace(5.0, LENGTH_M - 5.0, n)])
    img = cv2.projectPoints(obj, rvec, tvec, K_TRUE, None)[0].reshape(-1, 2)
    w, h = SIZE
    return [LineClick(frame=frame, line_id="near_touchline", x=float(x) / w, y=float(y) / h)
            for x, y in img]


def _pose_h(focal: float, rvec: NDArray[np.float64], tvec: NDArray[np.float64]) -> NDArray[np.float64]:
    k = np.array([[focal, 0, SIZE[0] / 2], [0, focal, SIZE[1] / 2], [0, 0, 1.0]])
    diag = np.diag([float(SIZE[0]), float(SIZE[1]), 1.0])
    return np.asarray(frame_homography(k, rvec, tvec), dtype=np.float64) @ diag


def test_coverage_grade_green_with_near_touchline() -> None:
    pts: list[Click] = []
    lns: list[LineClick] = []
    for f, (rv, tv) in POSES.items():
        pts += _pose_clicks(f, rv, tv)
        lns += _near_tl_clicks(f, rv, tv)
    calib = solve_session(pts, lns, SIZE, {f: np.eye(3) for f in POSES})
    for f in POSES:
        assert calib.coverage_grade[f] == "green"   # foreground self-check passes


def test_coverage_grade_yellow_without_near_touchline() -> None:
    pts: list[Click] = []
    for f, (rv, tv) in POSES.items():
        pts += _pose_clicks(f, rv, tv)
    calib = solve_session(pts, [], SIZE, {f: np.eye(3) for f in POSES})
    for f in POSES:
        assert calib.coverage_grade[f] == "yellow"   # foreground unverified


def test_status_anchor_grade_and_fold() -> None:
    rv, tv = POSES[20]
    h_good = _pose_h(3000.0, rv, tv)   # narrow view, fold in [4,15]
    h_wide = _pose_h(1460.0, rv, tv)   # sees whole field, fold ~21 (out of range)

    def mk(anchor_h: dict[int, NDArray[np.float64]], grade: dict[int, str]) -> PhysicalCalib:
        return PhysicalCalib(K=np.eye(3), poses={}, anchor_h=anchor_h,
                             coverage_grade=grade, transforms={5: np.eye(3)}, size=SIZE)

    assert mk({5: h_good}, {5: "green"}).status(5) == "green"
    assert mk({5: h_good}, {5: "yellow"}).status(5) == "yellow"
    assert mk({5: h_wide}, {5: "green"}).status(5) == "red"   # implausible fold -> red


def test_status_propagated_yellow_and_gap_red() -> None:
    rv, tv = POSES[20]
    h0 = _pose_h(3000.0, rv, tv)
    transforms = {f: _trans(-0.002 * f) for f in range(0, 410)}
    anchor_h = {0: h0, 10: h0 @ transforms[10]}
    calib = PhysicalCalib(K=np.eye(3), poses={}, anchor_h=anchor_h,
                          coverage_grade={0: "green", 10: "green"},
                          transforms=transforms, size=SIZE, gap_guard=200)
    assert calib.status(5) == "yellow"    # propagated within gap, plausible fold
    assert calib.status(400) == "red"     # beyond gap -> no homography


# ---- T4: acceptance gate ----
# Five diverse anchors (a wider pan than POSES) so leave-one-anchor-out still leaves >= 3
# views for the shared-focal calibration. The chain is the TRUE inter-frame map
# M[f] = H_ref^-1 @ H_f, so chain-shift recovers a held frame exactly.
GATE_POSES: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {
    10: (np.array([[1.18], [-0.40], [0.0]]), np.array([[-19.0], [-3.0], [41.0]])),
    20: (np.array([[1.20], [-0.20], [0.0]]), np.array([[-21.0], [-3.0], [40.0]])),
    30: (np.array([[1.20], [0.00], [0.0]]), np.array([[-22.0], [-3.0], [40.0]])),
    40: (np.array([[1.20], [0.20], [0.0]]), np.array([[-23.0], [-3.0], [40.0]])),
    50: (np.array([[1.18], [0.40], [0.0]]), np.array([[-25.0], [-3.0], [41.0]])),
}


def _true_norm_h(rvec: NDArray[np.float64], tvec: NDArray[np.float64]) -> NDArray[np.float64]:
    diag = np.diag([float(SIZE[0]), float(SIZE[1]), 1.0])
    return np.asarray(frame_homography(K_TRUE, rvec, tvec), dtype=np.float64) @ diag


def _gate_fixture() -> tuple[list[Click], list[LineClick], dict[int, NDArray[np.float64]]]:
    pts: list[Click] = []
    lns: list[LineClick] = []
    for f, (rv, tv) in GATE_POSES.items():
        pts += _pose_clicks(f, rv, tv)
        lns += _near_tl_clicks(f, rv, tv)
    h_ref = _true_norm_h(*GATE_POSES[min(GATE_POSES)])
    transforms = {f: np.linalg.inv(h_ref) @ _true_norm_h(rv, tv)
                  for f, (rv, tv) in GATE_POSES.items()}
    return pts, lns, transforms


def test_evaluate_gate_passes_on_clean_session() -> None:
    pts, lns, transforms = _gate_fixture()
    rep = evaluate_gate(pts, lns, SIZE, transforms)
    assert rep.fg_n > 0 and rep.prop_n > 0
    assert rep.fg_median_ft <= 5.0 and rep.fg_p90_ft <= 12.0
    assert rep.prop_median_ft <= 5.0
    assert rep.passed_numeric


def test_foreground_holdout_counts() -> None:
    pts, lns, _transforms = _gate_fixture()
    errs = foreground_holdout(pts, lns, SIZE)
    assert len(errs) == len(GATE_POSES) * 3   # 3 near-touchline clicks per anchor


def test_gate_fails_without_foreground() -> None:
    pts, _lns, transforms = _gate_fixture()
    rep = evaluate_gate(pts, [], SIZE, transforms)   # no near-touchline -> unmeasurable
    assert rep.fg_n == 0
    assert not rep.passed_numeric


def test_grade_yellow_when_near_touchline_is_wrong() -> None:
    # The foreground hold-out is GENUINE: a deliberately-displaced near-touchline must not
    # pass (its endpoints are held out of the refit too), so the anchor stays yellow while
    # the honestly-clicked anchors go green.
    pts, lns, transforms = _gate_fixture()
    bad_frame = min(GATE_POSES)
    corrupted = [LineClick(lc.frame, lc.line_id, lc.x + 0.3, lc.y)
                 if lc.frame == bad_frame and lc.line_id == "near_touchline" else lc
                 for lc in lns]
    calib = solve_session(pts, corrupted, SIZE, transforms)
    assert calib.coverage_grade[bad_frame] == "yellow"          # wrong near-TL -> not green
    other = next(f for f in GATE_POSES if f != bad_frame)
    assert calib.coverage_grade[other] == "green"               # honest anchors still green
