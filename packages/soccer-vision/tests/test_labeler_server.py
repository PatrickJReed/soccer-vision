"""Tests for the labeler HTTP app via a threaded server on a loopback port."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import HTTPServer
from typing import Any
from urllib.error import HTTPError

import cv2
import numpy as np
from soccer_vision.calib.field_model import field_points_3d
from soccer_vision.labeler.chain import normalize_homography
from soccer_vision.labeler.server import make_handler
from soccer_vision.labeler.state import LabelerState
from soccer_vision.pitch.manual_anchor import Click

# Calibrated-engine camera intrinsics shared by tests that need real clicks.
_K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64)


def _look_at(
    eye: Any,
    target: Any,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    e, t, u = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    f = t - e
    f /= np.linalg.norm(f)
    r = np.cross(f, u)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    rvec, _ = cv2.Rodrigues(np.vstack([r, d, f]))
    return rvec, (-np.vstack([r, d, f]) @ e).reshape(3, 1)


def _calib_session(n: int = 9) -> tuple[dict[int, np.ndarray], list[Click]]:
    """Return interframe transforms + projected clicks for a pan session."""
    center = (-8.0, 34.0, 9.0)
    poses = {f: _look_at(center, (22.85, 34.0 + dy, 0.0))
             for f, dy in enumerate(np.linspace(-10, 10, n))}
    interframe: dict[int, np.ndarray] = {}
    for i in range(n - 1):
        ri, _ = cv2.Rodrigues(poses[i][0])
        rj, _ = cv2.Rodrigues(poses[i + 1][0])
        g = _K @ rj @ np.linalg.inv(ri) @ np.linalg.inv(_K)
        interframe[i] = normalize_homography(g, (1920, 1080))
    fp = field_points_3d()
    clicks: list[Click] = []
    for f in (0, 4, 8):
        px = cv2.projectPoints(fp, poses[f][0], poses[f][1], _K, np.zeros(5))[0].reshape(-1, 2)
        for j in range(21):
            if j != 5 and 0 < px[j, 0] < 1920 and 0 < px[j, 1] < 1080:
                clicks.append(Click(f, j, float(px[j, 0]) / 1920, float(px[j, 1]) / 1080))
    return interframe, clicks


def _serve() -> tuple[HTTPServer, LabelerState]:
    interframe = {i: np.eye(3) for i in range(5)}
    state = LabelerState(interframe=interframe, n_frames=6, size=(1920, 1080), window=10)

    def frame_jpeg(idx: int) -> bytes:
        return b"\xff\xd8stub-jpeg"

    handler = make_handler(state, frame_jpeg, landmark_names=["pitch"] * 21)
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, state


def _serve_calib() -> tuple[HTTPServer, LabelerState]:
    """Serve a session with calibratable interframe transforms."""
    interframe, _clicks = _calib_session(9)
    state = LabelerState(interframe=interframe, n_frames=9, size=(1920, 1080), window=360)

    def frame_jpeg(idx: int) -> bytes:
        return b"\xff\xd8stub-jpeg"

    handler = make_handler(state, frame_jpeg, landmark_names=["pitch"] * 21)
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, state


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        result: dict[str, Any] = json.loads(r.read())
        return result


def _get(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url) as r:
        result: dict[str, Any] = json.loads(r.read())
        return result


def test_click_then_state_reports_coverage() -> None:
    # Use a calibratable session: real projected clicks across 3 anchor frames.
    httpd, state = _serve_calib()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    _interframe, clicks = _calib_session(9)
    try:
        for c in clicks:
            _post(f"{base}/api/click",
                  {"frame": c.frame, "kp_idx": c.kp_idx, "x": c.x, "y": c.y})
        state.wait_idle(timeout=10)  # state reflects the background-settled frames
        resp = _get(f"{base}/api/state")
        assert 0.0 <= resp["coverage"] <= 1.0  # green coverage needs a planar-crop session
        assert len(resp["status_buckets"]) == 9
        assert resp["bucket_size"] == 1
        # clicks flow through to a per-frame homography reported via the API
        assert _get(f"{base}/api/frame_h/4")["h"] is not None
    finally:
        httpd.shutdown()


def test_frame_endpoint_returns_bytes() -> None:
    httpd, _ = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/frame/2") as r:
            assert r.read().startswith(b"\xff\xd8")
    finally:
        httpd.shutdown()


def test_frame_endpoint_ignores_cache_buster_query() -> None:
    # the browser appends ?t=<ms> to bust the cache; the index parse must ignore it.
    httpd, _ = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/frame/2?t=12345") as r:
            assert r.read().startswith(b"\xff\xd8")
    finally:
        httpd.shutdown()


def test_frame_h_includes_residual_and_n_points() -> None:
    # Use a calibratable session so the engine can produce valid homographies.
    httpd, state = _serve_calib()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    _interframe, clicks = _calib_session(9)
    try:
        for c in clicks:
            _post(f"{base}/api/click",
                  {"frame": c.frame, "kp_idx": c.kp_idx, "x": c.x, "y": c.y})
        state.wait_idle(timeout=10)  # propagated frames settle on the background worker
        fh = _get(f"{base}/api/frame_h/4")   # frame 4 is a clicked anchor
        assert fh["h"] is not None
        # residual is now the in-sample global-fit RMS in NORMALIZED px (~[0,1] space)
        assert fh["residual"] is not None and fh["residual"] < 1.0
        assert fh["n_points"] is not None and fh["n_points"] > 0
        assert _get(f"{base}/api/frame_h/2")["h"] is not None  # propagated frame
    finally:
        httpd.shutdown()


def test_nudge_endpoint_moves_click() -> None:
    httpd, state = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _post(f"{base}/api/click", {"frame": 0, "kp_idx": 0, "x": 0.2, "y": 0.3})
        out = _post(f"{base}/api/nudge", {"frame": 0, "kp_idx": 0, "x": 0.4, "y": 0.5})
        assert out["n_clicks"] == 1
        assert np.isclose(state.clicks[0].x, 0.4)
    finally:
        httpd.shutdown()


def test_nudge_endpoint_404_when_no_match() -> None:
    httpd, _ = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _post(f"{base}/api/nudge", {"frame": 9, "kp_idx": 9, "x": 0.1, "y": 0.1})
    except HTTPError as e:
        assert e.code == 404
    else:
        raise AssertionError("expected HTTP 404")
    finally:
        httpd.shutdown()


def test_clicks_endpoint_returns_session_clicks() -> None:
    httpd, _ = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _post(f"{base}/api/click", {"frame": 2, "kp_idx": 4, "x": 0.3, "y": 0.6})
        out = _get(f"{base}/api/clicks")
        assert out["clicks"] == [{"frame": 2, "kp_idx": 4, "x": 0.3, "y": 0.6}]
    finally:
        httpd.shutdown()


def test_make_handler_accepts_line_names_and_state_exposes_line_clicks() -> None:
    from soccer_vision.calib.field_model import FIELD_LINES
    from soccer_vision.labeler.state import LabelerState
    # self-contained: an empty chain is fine — add_line_click on an uncalibrated state
    # just stores the click (no refit), which is what we assert.
    st = LabelerState(interframe={}, n_frames=5, size=(1920, 1080), window=360)
    st.add_line_click(2, "midline", 0.5, 0.5)
    assert st.line_clicks[0].line_id == "midline"
    handler_cls = make_handler(st, lambda i: b"", [f"kp{i}" for i in range(21)],
                               line_names=sorted(FIELD_LINES))
    assert handler_cls is not None  # make_handler accepts the line_names kwarg


def test_line_click_endpoint_stores_and_returns_state() -> None:
    httpd, state = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        out = _post(f"{base}/api/line_click",
                    {"frame": 2, "line_id": "midline", "x": 0.5, "y": 0.5})
        assert "line_names" in out  # the response is the state payload
        assert len(state.line_clicks) == 1
        assert state.line_clicks[0].line_id == "midline"
    finally:
        httpd.shutdown()


def test_line_click_endpoint_rejects_unknown_line_id() -> None:
    httpd, state = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _post(f"{base}/api/line_click",
              {"frame": 2, "line_id": "not_a_line", "x": 0.5, "y": 0.5})
    except HTTPError as e:
        assert e.code == 400
    else:
        raise AssertionError("expected HTTP 400")
    finally:
        httpd.shutdown()
    assert state.line_clicks == []  # nothing stored on a rejected line_id


def test_state_payload_includes_pending() -> None:
    httpd, _ = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        out = _get(f"{base}/api/state")
        assert "pending" in out
        assert isinstance(out["pending"], int)
    finally:
        httpd.shutdown()


def test_state_payload_includes_residual_threshold() -> None:
    # The frontend colours the per-frame residual readout against the server's
    # threshold instead of a hard-coded 0.05, so /api/state must expose it.
    httpd, state = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        out = _get(f"{base}/api/state")
        assert "residual_px_threshold" in out
        assert out["residual_px_threshold"] == state.residual_px_threshold
    finally:
        httpd.shutdown()


def test_clicks_endpoint_returns_both_point_and_line_clicks() -> None:
    httpd, _ = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _post(f"{base}/api/click", {"frame": 2, "kp_idx": 4, "x": 0.3, "y": 0.6})
        _post(f"{base}/api/line_click",
              {"frame": 3, "line_id": "near_touchline", "x": 0.1, "y": 0.9})
        out = _get(f"{base}/api/clicks")
        assert out["clicks"] == [{"frame": 2, "kp_idx": 4, "x": 0.3, "y": 0.6}]
        assert out["line_clicks"] == [
            {"frame": 3, "line_id": "near_touchline", "x": 0.1, "y": 0.9}]
    finally:
        httpd.shutdown()
