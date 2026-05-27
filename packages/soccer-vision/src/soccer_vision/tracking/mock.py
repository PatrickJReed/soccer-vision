"""MockBackend: a deterministic stand-in TrackingBackend used in tests.

Emits 16 outfield players (8 own + 8 opp) in fixed grid positions for every
frame of the video. No detection model required. Used to verify that
downstream code is decoupled from any specific upstream tracker.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from soccer_vision.io.schema import validate_trajectories


class MockBackend:
    """Deterministic fixture backend; produces 16 well-spaced detections per frame."""

    def __init__(self) -> None:
        self.name = "mock"
        self.version = "0.1.0"

    def process(self, video_path: Path) -> pd.DataFrame:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Two 4x2 grids of players, left half = own, right half = opp
        rows: list[dict[str, float | int | str]] = []
        for frame in range(n_frames):
            for team, x_offset, track_id_offset in [("own", w * 0.25, 0), ("opp", w * 0.75, 100)]:
                for i in range(8):
                    col, row = i % 4, i // 4
                    x = x_offset + (col - 1.5) * (w * 0.05)
                    y = h * 0.3 + row * (h * 0.3)
                    rows.append({
                        "frame": frame,
                        "t_seconds": frame / fps,
                        "track_id": track_id_offset + i,
                        "x_px": float(x),
                        "y_px": float(y),
                        "bbox_x1": float(x - 10),
                        "bbox_y1": float(y - 20),
                        "bbox_x2": float(x + 10),
                        "bbox_y2": float(y + 20),
                        "class": "player",
                        "team": team,
                        "conf": 1.0,
                    })

        df = pd.DataFrame(rows)
        df = df.astype({"frame": "int64", "track_id": "int64"})
        validate_trajectories(df)
        return df
