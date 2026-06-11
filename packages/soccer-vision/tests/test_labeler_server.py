"""Tests for the labeler HTTP app via a threaded server on a loopback port."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import HTTPServer
from typing import Any

import numpy as np
from soccer_vision.labeler.server import make_handler
from soccer_vision.labeler.state import LabelerState
from soccer_vision.pitch.landmarks import PITCH_LANDMARKS


def _serve() -> tuple[HTTPServer, LabelerState]:
    interframe = {i: np.eye(3) for i in range(5)}
    state = LabelerState(interframe=interframe, n_frames=6, size=(1920, 1080), window=10)

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
    httpd, _ = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        for f, idx in enumerate([0, 3, 6, 11, 16, 19]):
            px, py = PITCH_LANDMARKS[idx] * 1000.0
            _post(f"{base}/api/click",
                  {"frame": f, "kp_idx": int(idx), "x": float(px), "y": float(py)})
        state = _get(f"{base}/api/state")
        assert state["coverage"] > 0.0
        assert len(state["status_buckets"]) == 6
        assert state["bucket_size"] == 1
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
    httpd, _ = _serve()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        for f, idx in enumerate([0, 3, 6, 11, 16, 19]):
            px, py = PITCH_LANDMARKS[idx] * 1000.0
            _post(f"{base}/api/click",
                  {"frame": f, "kp_idx": int(idx), "x": float(px), "y": float(py)})
        fh = _get(f"{base}/api/frame_h/3")
        assert fh["h"] is not None
        assert fh["residual"] is not None and fh["residual"] < 0.05
        assert fh["n_points"] == 6
        assert _get(f"{base}/api/frame_h/0")["h"] is not None
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
