"""Stdlib HTTP app for the manual anchor labeler.

make_handler() builds a BaseHTTPRequestHandler bound to a LabelerState and a
frame-bytes provider, so it is testable without a real video. run() wires a real
video (chain precompute + JPEG frames) and serves the static UI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from soccer_vision.labeler.state import LabelerState

_STATIC = Path(__file__).parent / "static"


def make_handler(
    state: LabelerState,
    frame_jpeg: Callable[[int], bytes],
    landmark_names: list[str],
    *,
    landmark_xy: list[list[float]] | None = None,
    export_dir: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class closed over the session state.

    landmark_xy is the canonical [x, y] of each keypoint in pitch [0,1]^2, sent to
    the frontend so it can draw the reprojected pitch overlay.
    """
    xy: list[list[float]] = landmark_xy or []

    class LabelerHandler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: dict[str, Any], code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _state_payload(self) -> dict[str, Any]:
            return {
                "n_frames": state.n_frames,
                "coverage": state.coverage(),
                "status": state.status_list(),
                "n_clicks": len(state.clicks),
                "landmark_names": landmark_names,
                "landmark_xy": xy,
            }

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, (_STATIC / "index.html").read_bytes(), "text/html")
            elif self.path == "/app.js":
                self._send(200, (_STATIC / "app.js").read_bytes(),
                           "application/javascript")
            elif self.path == "/api/state":
                self._json(self._state_payload())
            elif self.path.startswith("/api/frame_h/"):
                idx = int(self.path.rsplit("/", 1)[1])
                fit = state.frame_homography(idx)
                self._json({"h": None if fit is None
                            else [float(v) for v in np.asarray(fit.H).reshape(9)]})
            elif self.path.startswith("/api/frame/"):
                idx = int(self.path.rsplit("/", 1)[1])
                self._send(200, frame_jpeg(idx), "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            payload: dict[str, Any] = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/click":
                state.add_click(int(payload["frame"]), int(payload["kp_idx"]),
                                float(payload["x"]), float(payload["y"]))
                self._json(self._state_payload())
            elif self.path == "/api/undo":
                state.remove_last()
                self._json(self._state_payload())
            elif self.path == "/api/export":
                state.export(export_dir or Path.cwd())
                self._json({"exported_to": str(export_dir or Path.cwd())})
            else:
                self._send(404, b"not found", "text/plain")

    return LabelerHandler


def run(
    video_path: Path,
    *,
    port: int = 8000,
    downscale_display: float = 0.5,
    export_dir: Path | None = None,
) -> None:  # pragma: no cover - launches a blocking server
    """Precompute the chain, open the video, and serve the labeler UI."""
    import cv2

    from soccer_vision.labeler.chain import compute_chain
    from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

    interframe, n_frames, _ = compute_chain(video_path)
    state = LabelerState(interframe=interframe, n_frames=n_frames)
    cap = cv2.VideoCapture(str(video_path))

    def frame_jpeg(idx: int) -> bytes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return b""
        small = cv2.resize(frame, None, fx=downscale_display, fy=downscale_display)
        ok2, buf = cv2.imencode(".jpg", small)
        return buf.tobytes() if ok2 else b""

    names = [f"kp{i}" for i in range(len(PITCH_LANDMARKS))]
    xy = [[float(x), float(y)] for x, y in PITCH_LANDMARKS]
    handler = make_handler(state, frame_jpeg, names, landmark_xy=xy, export_dir=export_dir)
    httpd = HTTPServer(("127.0.0.1", port), handler)
    print(f"Labeler running at http://127.0.0.1:{port}  (video: {video_path})")
    httpd.serve_forever()
