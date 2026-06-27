"""Stdlib HTTP app for the manual anchor labeler.

make_handler() builds a BaseHTTPRequestHandler bound to a LabelerState and a
frame-bytes provider, so it is testable without a real video. run() wires a real
video (chain precompute + JPEG frames) and serves the static UI.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from soccer_vision.labeler.state import LabelerState

_STATIC = Path(__file__).parent / "static"


def make_frame_jpeg(
    cap: Any, downscale_display: float = 0.5, *, cache_size: int = 64
) -> Callable[[int], bytes]:
    """JPEG-bytes provider for a shared VideoCapture, with an LRU cache + sequential
    fast-path. Assumes ALL-INTRA clips (the documented Trace operating mode), so a
    per-frame seek is cheap and decode order is stable: when the requested index is the
    next one in order we read() without an expensive cap.set() seek, and repeat requests
    for the same index are served from the LRU cache (no re-decode)."""
    import cv2

    last_pos = {"i": -1}

    def _decode(idx: int) -> bytes:
        if idx != last_pos["i"] + 1:          # only seek when not reading the next frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        last_pos["i"] = idx
        if not ok:
            return b""
        small = cv2.resize(frame, None, fx=downscale_display, fy=downscale_display)
        ok2, buf = cv2.imencode(".jpg", small)
        return bytes(buf.tobytes()) if ok2 else b""

    return functools.lru_cache(maxsize=cache_size)(_decode)


def make_handler(
    state: LabelerState,
    frame_jpeg: Callable[[int], bytes],
    landmark_names: list[str],
    *,
    landmark_xy: list[list[float]] | None = None,
    line_names: list[str] | None = None,
    export_dir: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class closed over the session state.

    landmark_xy is the canonical [x, y] of each keypoint in pitch [0,1]^2, sent to
    the frontend so it can draw the reprojected pitch overlay.
    """
    xy: list[list[float]] = landmark_xy or []
    lines: list[str] = line_names or []

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
            buckets, bucket_size = state.status_buckets()
            return {
                "n_frames": state.n_frames,
                "coverage": state.coverage(),
                "status_buckets": buckets,
                "bucket_size": bucket_size,
                "n_clicks": len(state.clicks),
                "landmark_names": landmark_names,
                "landmark_xy": xy,
                "line_names": lines,
                "pending": state.pending(),
                "residual_px_threshold": state.residual_px_threshold,
            }

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]  # drop cache-buster / query string
            if path in ("/", "/index.html"):
                self._send(200, (_STATIC / "index.html").read_bytes(), "text/html")
            elif path == "/app.js":
                self._send(200, (_STATIC / "app.js").read_bytes(),
                           "application/javascript")
            elif path == "/api/state":
                self._json(self._state_payload())
            elif path == "/api/clicks":
                self._json({
                    "clicks": [
                        {"frame": c.frame, "kp_idx": c.kp_idx, "x": c.x, "y": c.y}
                        for c in state.clicks
                    ],
                    "line_clicks": [
                        {"frame": lc.frame, "line_id": lc.line_id, "x": lc.x, "y": lc.y}
                        for lc in state.line_clicks
                    ],
                })
            elif path.startswith("/api/frame_h/"):
                idx = int(path.rsplit("/", 1)[1])
                fit = state.frame_homography(idx)
                self._json({
                    "h": None if fit is None
                    else [float(v) for v in np.asarray(fit.H).reshape(9)],
                    "residual": None if fit is None else fit.residual,
                    "n_points": None if fit is None else fit.n_points,
                    "outliers": list(state._outliers.get(idx, [])),  # flagged mislabel kp_idx
                })
            elif path.startswith("/api/frame/"):
                idx = int(path.rsplit("/", 1)[1])
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
            elif self.path == "/api/nudge":
                found = state.nudge_click(
                    int(payload["frame"]), int(payload["kp_idx"]),
                    float(payload["x"]), float(payload["y"]),
                )
                if found:
                    self._json(self._state_payload())
                else:
                    self._json({"error": "no click at frame/kp_idx"}, code=404)
            elif self.path == "/api/recalibrate":
                ok = state.recalibrate()
                self._json({"recalibrated": ok, **self._state_payload()})
            elif self.path == "/api/line_click":
                from soccer_vision.calib.field_model import FIELD_LINES
                line_id = str(payload["line_id"])
                if line_id not in FIELD_LINES:
                    self._json({"error": f"unknown line_id {line_id!r}"}, code=400)
                    return
                state.add_line_click(int(payload["frame"]), line_id,
                                     float(payload["x"]), float(payload["y"]))
                self._json(self._state_payload())
            else:
                self._send(404, b"not found", "text/plain")

    return LabelerHandler


def run(
    video_path: Path,
    *,
    port: int = 8000,
    downscale_display: float = 0.5,
    export_dir: Path | None = None,
    resume: Path | None = None,
    workers: int | None = None,
) -> None:  # pragma: no cover - launches a blocking server
    """Precompute the chain, open the video, and serve the labeler UI.

    resume: a previously exported keypoints.parquet — its clicks are loaded
    into the session (converted back from full-pixel to normalized coords).
    workers: parallel workers for chain precompute (default: cores-1).
    """
    import cv2

    from soccer_vision.calib.field_model import FIELD_LINES
    from soccer_vision.labeler.chain import compute_chain
    from soccer_vision.labeler.state import (
        clicks_from_keypoints_parquet,
        clicks_from_sidecar,
        line_clicks_from_parquet,
        line_clicks_from_sidecar,
    )
    from soccer_vision.pitch.landmarks import LANDMARK_NAMES, PITCH_LANDMARKS

    interframe, n_frames, size = compute_chain(video_path, workers=workers)
    cache_dir = Path(video_path).parent / ".sv_labeler_cache"
    sidecar = cache_dir / f"{Path(video_path).stem}.clicks.json"
    state = LabelerState(
        interframe=interframe, n_frames=n_frames, size=size,
        autosave_path=sidecar,
    )
    if resume is not None:
        if sidecar.exists():
            backup = sidecar.parent / (sidecar.name + ".bak")
            sidecar.replace(backup)
            print(f"existing autosave backed up to {backup}")
        state.add_clicks(clicks_from_keypoints_parquet(resume, size))
        print(f"resumed {len(state.clicks)} clicks from {resume}")
    elif sidecar.exists():
        state.add_clicks(clicks_from_sidecar(sidecar))
        print(f"restored {len(state.clicks)} clicks from autosave {sidecar}")
    if resume is not None:
        lc_path = Path(resume).parent / "line_clicks.parquet"
        if lc_path.exists():
            state.add_line_clicks(line_clicks_from_parquet(lc_path, size))
    elif sidecar.exists():
        state.add_line_clicks(line_clicks_from_sidecar(sidecar))
    cap = cv2.VideoCapture(str(video_path))
    frame_jpeg = make_frame_jpeg(cap, downscale_display)

    names = list(LANDMARK_NAMES)
    xy = [[float(x), float(y)] for x, y in PITCH_LANDMARKS]
    handler = make_handler(state, frame_jpeg, names, landmark_xy=xy,
                           line_names=sorted(FIELD_LINES), export_dir=export_dir)
    httpd = HTTPServer(("127.0.0.1", port), handler)
    print(f"Labeler running at http://127.0.0.1:{port}  (video: {video_path})")
    try:
        httpd.serve_forever()
    finally:
        cap.release()
        state.stop_worker()
