"""Global-homography calibration for a fixed (Trace virtual-PTZ) camera.

The whole session is one canvas: every frame is a 2D crop of it, so there is ONE
image_reference -> pitch[0,1] homography per registration segment. Each click is
lifted to its segment's reference frame (via the cumulative transform reduced to a
2D translation) and ALL clicks are fit jointly, so own-end and opp-end clicks from
different frames constrain a single homography (no per-frame under-constraint, no
chained-error accumulation). Per-frame H_f = H_global @ T_f. Pure: no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from soccer_vision.calib.validate import fold_count
from soccer_vision.pitch.homography import HomographyError, fit_homography
from soccer_vision.pitch.landmarks import NEAR_HALFWAY_IDX, PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import Click

# Landmark index sets per field end (own goal at y=0, opp at y=1; midline y=0.5
# and the hidden under-camera point are excluded).
OWN_END_IDX: list[int] = [
    i for i in range(len(PITCH_LANDMARKS))
    if PITCH_LANDMARKS[i, 1] < 0.5 and i != NEAR_HALFWAY_IDX
]
OPP_END_IDX: list[int] = [
    i for i in range(len(PITCH_LANDMARKS))
    if PITCH_LANDMARKS[i, 1] > 0.5 and i != NEAR_HALFWAY_IDX
]

_CENTER = np.array([0.5, 0.5, 1.0])  # normalized image centre

FOLD_MIN: int = 4    # a real narrow Trace view shows ~6-12 of 21; <FOLD_MIN = sky/off-frame
FOLD_MAX: int = 15   # folding pulls the far field in toward 21


def _translation_of(m: NDArray[np.floating]) -> NDArray[np.float64]:
    """2D translation M induces at the image centre (normalized). M maps frame -> ref,
    so this is the offset to ADD to a frame click to reach the reference frame."""
    mapped = np.asarray(m, dtype=np.float64) @ _CENTER
    mapped = mapped[:2] / mapped[2]
    return np.asarray(mapped - _CENTER[:2], dtype=np.float64)


def _translation_matrix(offset: NDArray[np.floating]) -> NDArray[np.float64]:
    return np.array([[1.0, 0.0, float(offset[0])],
                     [0.0, 1.0, float(offset[1])],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _apply_h(h: NDArray[np.floating], pts: NDArray[np.floating]) -> NDArray[np.float64]:
    """Apply a 3x3 homography to (N,2) points -> (N,2)."""
    homog = np.column_stack([np.asarray(pts, dtype=np.float64), np.ones(len(pts))])
    out = (np.asarray(h, dtype=np.float64) @ homog.T).T
    return np.asarray(out[:, :2] / out[:, 2:3], dtype=np.float64)


@dataclass(frozen=True, eq=False)
class GlobalCalib:
    """One image_ref -> pitch[0,1] homography per segment + per-frame 2D offsets."""

    h_by_segment: dict[int, NDArray[np.float64]]   # segment -> H_global (norm ref -> pitch)
    offsets: dict[int, NDArray[np.float64]]        # frame -> (dx, dy) normalized
    segment_of: dict[int, int]
    rms_by_segment: dict[int, float]               # diagnostic: in-sample reproj RMS (norm px)
    n_by_segment: dict[int, int]                   # clicks used per segment

    def frame_homography(self, frame: int) -> NDArray[np.float64] | None:
        """Normalized image_frame -> pitch[0,1] homography, or None if uncalibrated."""
        seg = self.segment_of.get(frame)
        if seg is None or seg not in self.h_by_segment:
            return None
        off = self.offsets.get(frame)
        if off is None:
            return None
        return np.asarray(self.h_by_segment[seg] @ _translation_matrix(off), dtype=np.float64)


def solve_global(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    size: tuple[int, int],
    *,
    min_points: int = 4,
    outlier_thresh: float = 0.05,
) -> GlobalCalib:
    """Fit one image_ref -> pitch homography per segment from all clicks pooled.

    Each click is lifted to its segment's reference frame via the cumulative
    transform reduced to a 2D translation, then cv2.findHomography (RANSAC) fits
    the segment's homography over the union of its clicks (RANSAC rejects outliers).

    outlier_thresh is the RANSAC reprojection threshold in PITCH [0,1] units
    (~0.05 ≈ 3.4 m on a 68.5 m-wide pitch): large enough to keep legitimate clicks
    including mild chain-offset drift, small enough to reject a grossly-mislabeled
    click that would otherwise corrupt the whole segment's global homography. (The
    default findHomography threshold of 3.0 is in this same pitch space, so it would
    make every click an inlier — no rejection at all.) Task 6's cross-end validation
    is what tunes this value against real sessions.
    """
    offsets = {f: _translation_of(m) for f, m in transforms.items()}
    img_by_seg: dict[int, list[list[float]]] = {}
    pitch_by_seg: dict[int, list[NDArray[np.float64]]] = {}
    for c in clicks:
        if c.kp_idx == NEAR_HALFWAY_IDX:
            continue
        seg = segment_of.get(c.frame)
        off = offsets.get(c.frame)
        if seg is None or off is None:
            continue
        img_by_seg.setdefault(seg, []).append([c.x + float(off[0]), c.y + float(off[1])])
        pitch_by_seg.setdefault(seg, []).append(PITCH_LANDMARKS[c.kp_idx])

    h_by_seg: dict[int, NDArray[np.float64]] = {}
    rms_by_seg: dict[int, float] = {}
    n_by_seg: dict[int, int] = {}
    for seg, img_list in img_by_seg.items():
        if len(img_list) < min_points:
            continue
        img = np.asarray(img_list, dtype=np.float64)
        pitch = np.asarray(pitch_by_seg[seg], dtype=np.float64)
        try:
            h = fit_homography(img, pitch, ransac_thresh=outlier_thresh)
        except HomographyError:
            continue
        proj = _apply_h(h, img)
        h_by_seg[seg] = np.asarray(h, dtype=np.float64)
        rms_by_seg[seg] = float(np.sqrt(np.mean(np.sum((proj - pitch) ** 2, axis=1))))
        n_by_seg[seg] = len(img)

    return GlobalCalib(h_by_seg, offsets, dict(segment_of), rms_by_seg, n_by_seg)


# --- Field-anchored bundle adjustment -------------------------------------------------
# Model: H_f = H_g @ A_f @ M[f].  M[f] = transforms[f] is the (drift-prone) chain
# transform frame -> segment-reference; A_f is a per-clicked-frame 6-DOF AFFINE
# correction in REFERENCE space that absorbs the chain's long-span drift; H_g is one
# global reference -> pitch[0,1] homography per segment. H_g (8 dof) and every A_f are
# fit jointly to all of a segment's clicks (reprojection error in pitch units), with each
# A_f regularized toward identity. For an unclicked frame, A is linearly interpolated
# between the segment's bracketing clicked frames. This corrects the chain drift that the
# translation-only solve_global inherits over the wide own->opp pan (the case that fails
# the cross-end holdout). Validated prototype: in-sample ~2.6 ft, leave-one-out 4.58 ft.

_AFFINE_DOF = 6
_AFFINE_ID: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _affine_from(p: NDArray[np.floating]) -> NDArray[np.float64]:
    """6 affine params [a, b, tx, c, d, ty] -> 3x3 affine matrix."""
    return np.array([[p[0], p[1], p[2]],
                     [p[3], p[4], p[5]],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _interp_affine_params(
    a_params: Mapping[int, NDArray[np.floating]], frame: int
) -> NDArray[np.float64]:
    """Affine params at `frame`, linearly interpolated between bracketing clicked frames
    (nearest clicked frame if outside the clicked range; identity if there are none)."""
    if not a_params:
        return np.array(_AFFINE_ID, dtype=np.float64)
    if frame in a_params:
        return np.asarray(a_params[frame], dtype=np.float64)
    frames = sorted(a_params)
    lo = [g for g in frames if g < frame]
    hi = [g for g in frames if g > frame]
    if not lo:
        return np.asarray(a_params[hi[0]], dtype=np.float64)
    if not hi:
        return np.asarray(a_params[lo[-1]], dtype=np.float64)
    a, b = lo[-1], hi[0]
    t = (frame - a) / (b - a)
    return np.asarray(
        (1 - t) * np.asarray(a_params[a], dtype=np.float64)
        + t * np.asarray(a_params[b], dtype=np.float64),
        dtype=np.float64,
    )


@dataclass(frozen=True, eq=False)
class BundleCalib:
    """Field-anchored bundle result: per-segment global homography H_g + per-clicked-frame
    affine corrections A_f, plus the chain transforms needed to build a per-frame
    H_f = H_g @ A_interp @ M[frame]."""

    h_by_segment: dict[int, NDArray[np.float64]]                    # seg -> H_g (norm ref -> pitch)
    a_params_by_segment: dict[int, dict[int, NDArray[np.float64]]]  # seg -> {clicked frame -> 6 params}
    segment_of: dict[int, int]
    transforms: dict[int, NDArray[np.float64]]                      # frame -> M[frame] (frame -> ref)
    rms_by_segment: dict[int, float]                               # diagnostic: in-sample reproj RMS (pitch units)
    n_by_segment: dict[int, int]                                   # clicks used per segment

    def frame_homography(self, frame: int) -> NDArray[np.float64] | None:
        """Normalized image_frame -> pitch[0,1] homography H_g @ A_interp @ M[frame], or
        None if the frame's segment was not calibrated / it has no chain transform. A is
        linearly interpolated between the segment's bracketing clicked frames."""
        seg = self.segment_of.get(frame)
        if seg is None or seg not in self.h_by_segment:
            return None
        m = self.transforms.get(frame)
        if m is None:
            return None
        params = _interp_affine_params(self.a_params_by_segment.get(seg, {}), frame)
        a = _affine_from(params)
        return np.asarray(
            self.h_by_segment[seg] @ a @ np.asarray(m, dtype=np.float64), dtype=np.float64
        )


def _seed_hg(
    seg_clicks: Sequence[Click], transforms: Mapping[int, NDArray[np.floating]]
) -> NDArray[np.float64] | None:
    """Seed H_g: lift every click to reference space via its FULL chain transform M[f]
    (not just the translation) and fit one ref -> pitch homography (RANSAC, pitch-unit
    threshold). None if degenerate."""
    img: list[NDArray[np.float64]] = []
    pitch: list[NDArray[np.float64]] = []
    for c in seg_clicks:
        m = np.asarray(transforms[c.frame], dtype=np.float64)
        lifted = m @ np.array([c.x, c.y, 1.0])
        img.append(lifted[:2] / lifted[2])
        pitch.append(PITCH_LANDMARKS[c.kp_idx])
    try:
        h = fit_homography(
            np.asarray(img, dtype=np.float64), np.asarray(pitch, dtype=np.float64),
            ransac_thresh=0.05,
        )
    except HomographyError:
        return None
    return np.asarray(h, dtype=np.float64)


def _solve_segment(
    seg_clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    lam: float,
) -> tuple[NDArray[np.float64], dict[int, NDArray[np.float64]]] | None:
    """Fit one segment's H_g (8 dof) + a per-clicked-frame affine A_f, jointly minimizing
    click reprojection error (pitch units) + lam*(A_f - I). Returns (H_g, {frame: params})
    or None if the seed homography is degenerate. Ports the prototype's least_squares."""
    frames = sorted({c.frame for c in seg_clicks})
    fidx = {f: i for i, f in enumerate(frames)}
    hg0 = _seed_hg(seg_clicks, transforms)
    if hg0 is None:
        return None
    a_id = np.array(_AFFINE_ID, dtype=np.float64)
    x0 = np.concatenate([hg0.flatten()[:8], np.tile(a_id, len(frames))])

    def slot(x: NDArray[np.float64], f: int) -> NDArray[np.float64]:
        i = fidx[f]
        return np.asarray(x[8 + i * _AFFINE_DOF: 8 + (i + 1) * _AFFINE_DOF])

    def resid(x: NDArray[np.float64]) -> NDArray[np.float64]:
        hg = np.append(x[:8], 1.0).reshape(3, 3)
        r: list[float] = []
        for c in seg_clicks:
            hf = hg @ _affine_from(slot(x, c.frame)) @ np.asarray(
                transforms[c.frame], dtype=np.float64)
            q = hf @ np.array([c.x, c.y, 1.0])
            r.extend((q[:2] / q[2]) - PITCH_LANDMARKS[c.kp_idx])
        for f in frames:  # regularize each A toward identity
            r.extend(lam * (slot(x, f) - a_id))
        return np.asarray(r, dtype=np.float64)

    sol = least_squares(resid, x0, method="lm", max_nfev=4000)
    hg = np.asarray(np.append(sol.x[:8], 1.0).reshape(3, 3), dtype=np.float64)
    a_params = {f: np.asarray(slot(sol.x, f), dtype=np.float64).copy() for f in frames}
    return hg, a_params


def solve_bundle(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    size: tuple[int, int],
    *,
    lam: float = 0.01,
    min_points: int = 4,
) -> BundleCalib:
    """Field-anchored bundle adjustment, solved independently per registration segment.

    For each segment, fits one ref -> pitch homography H_g plus a 6-DOF affine A_f per
    clicked frame, minimizing click reprojection (pitch units) + lam*(A_f - I). The per-
    frame affine absorbs the chain's long-span drift that solve_global (translation-only)
    cannot, which is what lets the cross-end held-out accuracy pass. Segments with fewer
    than `min_points` clicks (or a degenerate seed) are skipped (no H). Pure: no I/O.
    """
    by_seg: dict[int, list[Click]] = {}
    for c in clicks:
        seg = segment_of.get(c.frame)
        if seg is None or c.frame not in transforms:
            continue
        by_seg.setdefault(seg, []).append(c)

    h_by_seg: dict[int, NDArray[np.float64]] = {}
    a_by_seg: dict[int, dict[int, NDArray[np.float64]]] = {}
    rms_by_seg: dict[int, float] = {}
    n_by_seg: dict[int, int] = {}
    for seg, seg_clicks in by_seg.items():
        if len(seg_clicks) < min_points:
            continue
        solved = _solve_segment(seg_clicks, transforms, lam)
        if solved is None:
            continue
        hg, a_params = solved
        h_by_seg[seg], a_by_seg[seg] = hg, a_params
        # In-sample reprojection RMS (pitch units) over the segment's clicks: predict
        # each click through H_g @ A @ M[frame] (A interpolated, exact at clicked frames).
        errs: list[NDArray[np.float64]] = []
        for c in seg_clicks:
            a = _affine_from(_interp_affine_params(a_params, c.frame))
            hf = hg @ a @ np.asarray(transforms[c.frame], dtype=np.float64)
            q = hf @ np.array([c.x, c.y, 1.0])
            errs.append((q[:2] / q[2]) - PITCH_LANDMARKS[c.kp_idx])
        rms_by_seg[seg] = float(np.sqrt(np.mean(np.sum(np.asarray(errs) ** 2, axis=1))))
        n_by_seg[seg] = len(seg_clicks)

    transforms_f64 = {f: np.asarray(m, dtype=np.float64) for f, m in transforms.items()}
    return BundleCalib(
        h_by_seg, a_by_seg, dict(segment_of), transforms_f64, rms_by_seg, n_by_seg)


def two_ended_segments(
    clicks: Sequence[Click], segment_of: Mapping[int, int]
) -> set[int]:
    """Segments whose clicks include BOTH an own-end and an opp-end landmark — i.e. the
    global homography is constrained across the whole field, so per-frame H is trustworthy."""
    own = set(OWN_END_IDX)
    opp = set(OPP_END_IDX)
    seen_own: set[int] = set()
    seen_opp: set[int] = set()
    for c in clicks:
        seg = segment_of.get(c.frame)
        if seg is None:
            continue
        if c.kp_idx in own:
            seen_own.add(seg)
        elif c.kp_idx in opp:
            seen_opp.add(seg)
    return seen_own & seen_opp


def fold_of_norm(h_norm: NDArray[np.floating] | None, size: tuple[int, int]) -> int:
    """fold_count for a NORMALIZED image->pitch homography (converts to the pitch->pixel
    form fold_count expects). 0 if h_norm is None or singular."""
    if h_norm is None:
        return 0
    w, h = size
    h_px = np.asarray(h_norm, dtype=np.float64) @ np.diag([1.0 / w, 1.0 / h, 1.0])
    try:
        h_pitch_to_px = np.linalg.inv(h_px)
    except np.linalg.LinAlgError:
        return 0
    return fold_count(h_pitch_to_px, size)


def frame_status(
    h_norm: NDArray[np.floating] | None,
    size: tuple[int, int],
    *,
    segment_two_ended: bool,
    fold_min: int = FOLD_MIN,
    fold_max: int = FOLD_MAX,
) -> str:
    """red = no H or implausible whole-field projection; green = plausible AND the
    segment saw both ends (homography constrained across the field); yellow = plausible
    but single-ended (honestly under-constrained)."""
    if h_norm is None:
        return "red"
    fold = fold_of_norm(h_norm, size)
    if not fold_min <= fold <= fold_max:
        return "red"
    return "green" if segment_two_ended else "yellow"


@dataclass(frozen=True)
class HoldoutReport:
    """Leave-one-frame-out cross-end accuracy in feet."""

    median_ft: float
    p90_ft: float
    own_median_ft: float | None
    opp_median_ft: float | None
    n: int


def cross_end_holdout(
    clicks: Sequence[Click],
    transforms: Mapping[int, NDArray[np.floating]],
    segment_of: Mapping[int, int],
    size: tuple[int, int],
    *,
    length_ft: float = 224.7,
    aspect_ratio: float = 1.5,
) -> HoldoutReport | None:
    """Leave-one-clicked-frame-out validation: for each clicked frame, refit the field-
    anchored bundle from the OTHER frames' clicks and measure how far (feet) this frame's
    clicks land from their canonical pitch coords, predicting the held frame via the
    bundle's interpolated per-frame affine (so the held frame is treated as an unclicked
    frame). Reports overall + per-end medians. None if there are <2 clicked frames. This
    is the acceptance bar (Task 6) and reflects the production solve_bundle path."""
    from soccer_vision.eval.pitch_metrics import displacement_to_feet

    own = set(OWN_END_IDX)
    clicked_frames = sorted({c.frame for c in clicks})
    if len(clicked_frames) < 2:
        return None

    feet: list[float] = []
    own_feet: list[float] = []
    opp_feet: list[float] = []
    for f in clicked_frames:
        rest = [c for c in clicks if c.frame != f]
        held = [c for c in clicks if c.frame == f and c.kp_idx != NEAR_HALFWAY_IDX]
        if not rest or not held:
            continue
        bc = solve_bundle(rest, transforms, segment_of, size)
        h = bc.frame_homography(f)
        if h is None:
            continue
        for c in held:
            pred = _apply_h(h, np.array([[c.x, c.y]]))[0]
            disp = pred - PITCH_LANDMARKS[c.kp_idx]
            ft = float(displacement_to_feet(disp, length_ft=length_ft, aspect_ratio=aspect_ratio))
            feet.append(ft)
            (own_feet if c.kp_idx in own else opp_feet).append(ft)

    if not feet:
        return None
    return HoldoutReport(
        median_ft=float(np.median(feet)),
        p90_ft=float(np.percentile(feet, 90)),
        own_median_ft=float(np.median(own_feet)) if own_feet else None,
        opp_median_ft=float(np.median(opp_feet)) if opp_feet else None,
        n=len(feet),
    )
