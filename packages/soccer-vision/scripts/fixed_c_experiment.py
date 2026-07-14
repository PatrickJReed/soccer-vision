"""Option-C experiment: predicted gate numbers for the FIXED-C rotation model on real
sessions. Stage 1 = the shipped physical solve (shared K, per-anchor 6-DOF). Stage 2 =
robust camera-centre estimate C_hat (component-wise median over well-constrained anchors),
then ROTATION-ONLY re-solve per anchor (3 DOF, tvec = -R @ C_hat). Evaluation mirrors the
shipped gates: foreground holdout (near-TL evidence removed) + leave-one-anchor-out bracket
propagation with a by-end split, plus far-end physicality (all-21 w-signs) and in-sample
cost of the constraint. Baseline physical numbers printed alongside.

Run: uv run python fixed_c_experiment.py <clip-stem> (training_clip | oceanside_clip)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from soccer_vision.calib.field_model import LENGTH_M, METRES_TO_FEET, WIDTH_M
from soccer_vision.labeler.chain import load_chain
from soccer_vision.labeler.state import clicks_from_sidecar, line_clicks_from_sidecar
from soccer_vision.pitch.calib_anchor import frame_homography
from soccer_vision.pitch.global_crop import _wsigns_ok
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS
from soccer_vision.pitch.manual_anchor import build_segments, cumulative_transforms
from soccer_vision.pitch.physical_calib import (
    _NEAR_TL_POINT_IDS,
    PhysicalCalib,
    _line_perp_feet,
    evaluate_gate,
    solve_session,
)

FT = METRES_TO_FEET
SCALE_M = np.array([WIDTH_M, LENGTH_M])
LINE_PITCH = {  # axis, const in pitch [0,1] units (same convention as global_crop)
    "near_touchline": (0, 0.0), "far_touchline": (0, 1.0),
    "own_goal_line": (1, 0.0), "opp_goal_line": (1, 1.0), "midline": (1, 0.5),
}


def apply_h(h, pts):
    p = np.column_stack([np.asarray(pts, float), np.ones(len(pts))])
    q = (np.asarray(h, float) @ p.T).T
    return q[:, :2] / q[:, 2:3]


def h_norm_of(K, rv, tv, size):
    w, h = size
    return np.asarray(frame_homography(K, rv, tv), float) @ np.diag([float(w), float(h), 1.0])


def centre_of(rv, tv):
    R, _ = cv2.Rodrigues(np.asarray(rv, float))
    return (-R.T @ np.asarray(tv, float).reshape(3)).reshape(3)


def solve_rotation_only(K, rv0, c_hat, po_px, lo_px, size):
    """3-DOF rotation refit with tvec = -R @ C_hat; metre residuals (points + lines)."""
    def resid(rvec):
        R, _ = cv2.Rodrigues(rvec)
        tv = (-R @ c_hat).reshape(3, 1)
        hn = h_norm_of(K, rvec.reshape(3, 1), tv, size)
        parts = []
        if po_px:
            pts = np.array([[x / size[0], y / size[1]] for _, x, y in po_px])
            q = apply_h(hn, pts)
            lms = PITCH_LANDMARKS[[i for i, _, _ in po_px]]
            parts.append(((q - lms) * SCALE_M).ravel())
        for lid, x, y in lo_px:
            q = apply_h(hn, np.array([[x / size[0], y / size[1]]]))[0]
            ax, cst = LINE_PITCH[lid]
            parts.append(np.array([(q[ax] - cst) * SCALE_M[ax]]))
        return np.concatenate(parts) if parts else np.zeros(1)

    res = least_squares(resid, np.asarray(rv0, float).reshape(3), method="trf",
                        loss="soft_l1", f_scale=0.5)
    rv = res.x.reshape(3, 1)
    R, _ = cv2.Rodrigues(rv)
    return rv, (-R @ c_hat).reshape(3, 1)


def stage2(K, calib, by_pt_px, by_ln_px, size, min_lms_for_c=6):
    """C_hat from stage-1 poses; rotation-only re-solve of every anchor."""
    centres = {f: centre_of(rv, tv) for f, (rv, tv) in calib.poses.items()}
    good = [f for f in centres if len({i for i, _, _ in by_pt_px.get(f, [])}) >= min_lms_for_c]
    used = good if len(good) >= 3 else sorted(centres)
    c_hat = np.median(np.array([centres[f] for f in used]), axis=0)
    spread = np.array([np.linalg.norm(centres[f] - c_hat) for f in used])
    poses2, anchor_h2 = {}, {}
    for f, (rv, tv) in calib.poses.items():
        rv2, tv2 = solve_rotation_only(K, rv, c_hat, by_pt_px.get(f, []), by_ln_px.get(f, []), size)
        poses2[f] = (rv2, tv2)
        anchor_h2[f] = h_norm_of(K, rv2, tv2, size)
    return c_hat, spread, poses2, anchor_h2


def in_sample_ft(K, poses, by_pt_px, size):
    out = {}
    for f, (rv, tv) in poses.items():
        po = by_pt_px.get(f, [])
        if not po:
            continue
        hn = h_norm_of(K, rv, tv, size)
        pts = np.array([[x / size[0], y / size[1]] for _, x, y in po])
        q = apply_h(hn, pts)
        lms = PITCH_LANDMARKS[[i for i, _, _ in po]]
        out[f] = float(np.median(np.linalg.norm((q - lms) * SCALE_M, axis=1) * FT))
    return out


def end_of(po_px):
    ys = [PITCH_LANDMARKS[i][1] for i, _, _ in po_px]
    m = float(np.mean(ys))
    return "own" if m < 0.45 else "opp" if m > 0.55 else "both"


def main(stem: str) -> None:
    home = Path.home() / "sv-labeler"
    chains = {"training_clip": "ef2546eaddd5e6fc", "oceanside_clip": "da63d2bb640cc974"}
    interframe, n_frames, size = load_chain(home / f".sv_labeler_cache/{chains[stem]}.npz")
    w, h = size
    points = clicks_from_sidecar(home / f".sv_labeler_cache/{stem}.clicks.json")
    lines = line_clicks_from_sidecar(home / f".sv_labeler_cache/{stem}.clicks.json")
    seg_of = build_segments(interframe, n_frames)
    transforms = cumulative_transforms(interframe, seg_of)
    by_pt_px, by_ln_px = {}, {}
    for c in points:
        by_pt_px.setdefault(c.frame, []).append((int(c.kp_idx), c.x * w, c.y * h))
    for lc in lines:
        by_ln_px.setdefault(lc.frame, []).append((str(lc.line_id), lc.x * w, lc.y * h))

    print(f"=== {stem}: {len(points)} pts / {len(by_pt_px)} frames, {len(lines)} lines ===")

    calib = solve_session(points, lines, size, transforms, segment_of=seg_of)
    K = calib.K
    print(f"stage-1 physical: {len(calib.poses)} anchors, K focal={K[0, 0]:.1f}px")

    c_hat, spread, poses2, anchor_h2 = stage2(K, calib, by_pt_px, by_ln_px, size)
    print(f"C_hat = ({c_hat[0]:.1f}, {c_hat[1]:.1f}, {c_hat[2]:.1f}) m;  centre spread over "
          f"used anchors: median={np.median(spread):.1f}m p90={np.percentile(spread, 90):.1f}m max={spread.max():.1f}m")

    is1 = in_sample_ft(K, calib.poses, by_pt_px, size)
    is2 = in_sample_ft(K, poses2, by_pt_px, size)
    common = sorted(set(is1) & set(is2))
    d_in = [is2[f] - is1[f] for f in common]
    print(f"in-sample anchor median ft: free={np.median([is1[f] for f in common]):.2f}  "
          f"fixed-C={np.median([is2[f] for f in common]):.2f}  (constraint cost: median +{np.median(d_in):.2f} ft, max +{max(d_in):.2f})")

    # far-end physicality of anchors: all-21 w-signs
    ws1 = sum(_wsigns_ok(h_norm_of(K, *calib.poses[f], size), size) for f in calib.poses)
    ws2 = sum(_wsigns_ok(anchor_h2[f], size) for f in poses2)
    print(f"anchors passing all-21 w-sign: free {ws1}/{len(calib.poses)}  fixed-C {ws2}/{len(poses2)}")

    # LOO propagation, fixed-C: hold each anchor out, rotation-only re-solve of the rest
    # (K kept; C_hat re-estimated from the rest), bracket-propagate via PhysicalCalib.
    anchors = sorted(calib.poses)
    errs, by_end = [], {}
    for held in anchors:
        others = [a for a in anchors if a != held and seg_of.get(a, 0) == seg_of.get(held, 0)]
        if not others or min(abs(held - a) for a in others) > 200:
            continue
        centres_rest = [centre_of(*calib.poses[a]) for a in others]
        c_rest = np.median(np.array(centres_rest), axis=0)
        poses_r, hs_r, grade_r = {}, {}, {}
        for a in others:
            rv2, tv2 = solve_rotation_only(K, calib.poses[a][0], c_rest,
                                           by_pt_px.get(a, []), by_ln_px.get(a, []), size)
            poses_r[a] = (rv2, tv2)
            hs_r[a] = h_norm_of(K, rv2, tv2, size)
            grade_r[a] = "green"
        pc = PhysicalCalib(K, poses_r, hs_r, grade_r, calib.transforms, size, 200, dict(seg_of))
        hm = pc.frame_homography(held)
        if hm is None:
            continue
        end = end_of(by_pt_px[held])
        for i, x, y in by_pt_px[held]:
            q = apply_h(hm, np.array([[x / w, y / h]]))[0]
            e = float(np.linalg.norm((q - PITCH_LANDMARKS[i]) * SCALE_M) * FT)
            errs.append(e)
            by_end.setdefault(end, []).append(e)
    if errs:
        print(f"\n[fixed-C] LOO propagation: median={np.median(errs):.2f} ft  "
              f"p90={np.percentile(errs, 90):.2f}  n={len(errs)}")
        for k, v in sorted(by_end.items()):
            print(f"    held frame saw {k:>4}: median={np.median(v):.2f}  p90={np.percentile(v, 90):.2f}  n={len(v)}")

    # foreground holdout, fixed-C: refit each near-TL frame without near-TL evidence
    fg = []
    for f in anchors:
        ntl = [(lid, x, y) for lid, x, y in by_ln_px.get(f, []) if lid == "near_touchline"]
        if not ntl:
            continue
        po_fit = [o for o in by_pt_px.get(f, []) if o[0] not in _NEAR_TL_POINT_IDS]
        lo_fit = [o for o in by_ln_px.get(f, []) if o[0] != "near_touchline"]
        if len(po_fit) < 3:
            continue
        rv2, tv2 = solve_rotation_only(K, calib.poses[f][0], c_hat, po_fit, lo_fit, size)
        hn = h_norm_of(K, rv2, tv2, size)
        for _, x, y in ntl:
            q = apply_h(hn, np.array([[x / w, y / h]]))[0]
            fg.append(_line_perp_feet(q, "near_touchline"))
    if fg:
        print(f"[fixed-C] foreground holdout: median={np.median(fg):.2f} ft  "
              f"p90={np.percentile(fg, 90):.2f}  n={len(fg)}")

    print("\n[baseline physical gate on the same session]")
    rep = evaluate_gate(points, lines, size, transforms, segment_of=seg_of)
    print(f"  foreground   median={rep.fg_median_ft:6.2f}  p90={rep.fg_p90_ft:6.2f}  n={rep.fg_n}")
    print(f"  propagation  median={rep.prop_median_ft:6.2f}  p90={rep.prop_p90_ft:6.2f}  n={rep.prop_n}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "oceanside_clip")
