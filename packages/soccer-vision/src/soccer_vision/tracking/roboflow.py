"""RoboflowBackend: TrackingBackend adapter wrapping roboflow/sports.

Heavy dependencies (ultralytics, supervision, sports, torch, gdown) are
imported lazily inside process() so this module loads fine in CI without
the 'roboflow' optional extra installed.

Install extras:
    uv pip install -e "packages/soccer-vision[roboflow]"
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from soccer_vision.io.schema import validate_trajectories

if TYPE_CHECKING:
    # Only imported for type-checking; never executed at runtime at module level.
    pass

# ---------------------------------------------------------------------------
# Weight registry
# ---------------------------------------------------------------------------

WEIGHTS: Final[dict[str, tuple[str, str]]] = {
    "ball":   ("1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V", "football-ball-detection.pt"),
    "player": ("17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q", "football-player-detection.pt"),
    # pitch model deferred to Plan B Phase 3 (homography / field calibration)
}

DEFAULT_CACHE_DIR: Final[Path] = Path.home() / ".cache" / "soccer_vision" / "weights"

# Synthetic ball track IDs use negative space to avoid collisions with ByteTrack IDs.
# ByteTrack tracker_id values are always non-negative, so negative IDs are collision-proof.
_BALL_TRACK_ID_BASE: Final = -1_000_000

# ---------------------------------------------------------------------------
# Schema-aware empty-DataFrame helper
# ---------------------------------------------------------------------------

_SCHEMA_DTYPES: Final[dict[str, str]] = {
    "frame":    "int64",
    "t_seconds": "float64",
    "track_id": "int64",
    "x_px":     "float64",
    "y_px":     "float64",
    "bbox_x1":  "float64",
    "bbox_y1":  "float64",
    "bbox_x2":  "float64",
    "bbox_y2":  "float64",
    "class":    "object",
    "team":     "object",
    "conf":     "float64",
}

# Class IDs from the roboflow football models
_CLS_BALL      = 0
_CLS_GK        = 1
_CLS_PLAYER    = 2
_CLS_REFEREE   = 3

_CLS_NAME: Final[dict[int, str]] = {
    _CLS_BALL:    "ball",
    _CLS_GK:      "goalkeeper",
    _CLS_PLAYER:  "player",
    _CLS_REFEREE: "referee",
}

# Cluster index → team label (by convention; caller may flip if needed)
_CLUSTER_TEAM: Final[dict[int, str]] = {0: "own", 1: "opp"}


def _empty_df() -> pd.DataFrame:
    """Return a zero-row DataFrame that passes validate_trajectories()."""
    return pd.DataFrame({col: pd.Series(dtype=t) for col, t in _SCHEMA_DTYPES.items()})


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class RoboflowBackend:
    """TrackingBackend adapter for the roboflow/sports football pipeline.

    Parameters
    ----------
    device:
        Torch device string ("cpu", "cuda", "mps", …).  When *None* (default),
        auto-detected inside process() to avoid importing torch at module load.
    weights_cache_dir:
        Directory where YOLO .pt weight files are cached after first download.
        Defaults to ~/.cache/soccer_vision/weights/.
    ball_weights_path:
        Optional path to a fine-tuned ball detector weights file (.pt).
        When provided, this model is used instead of the default ball detector.
        If the path does not exist, raises FileNotFoundError.
    """

    name: Final = "roboflow-sports"
    version: Final = "main@2026-05-28"

    def __init__(
        self,
        device: str | None = None,
        weights_cache_dir: Path | None = None,
        ball_weights_path: Path | None = None,
    ) -> None:
        self._device_override = device
        self._weights_dir = Path(weights_cache_dir) if weights_cache_dir else DEFAULT_CACHE_DIR
        if ball_weights_path is not None and not ball_weights_path.exists():
            raise FileNotFoundError(f"ball_weights_path does not exist: {ball_weights_path}")
        self.ball_weights_path: Path | None = ball_weights_path

    # ------------------------------------------------------------------
    # Weight download helper (lazy gdown import)
    # ------------------------------------------------------------------

    def _download_weights(self) -> dict[str, Path]:
        """Ensure all three .pt weight files are in the cache dir.

        Returns a mapping of role → local Path.
        """
        try:
            import gdown  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "gdown is required to download model weights. "
                "Install the roboflow extras: "
                'uv pip install -e "packages/soccer-vision[roboflow]"'
            ) from exc

        self._weights_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for role, (gdrive_id, filename) in WEIGHTS.items():
            dest = self._weights_dir / filename
            if not dest.exists():
                url = f"https://drive.google.com/uc?id={gdrive_id}"
                gdown.download(url, str(dest), quiet=False)
            paths[role] = dest
        return paths

    # ------------------------------------------------------------------
    # Main processing entry-point
    # ------------------------------------------------------------------

    def process(self, video_path: Path) -> pd.DataFrame:
        """Run roboflow/sports detection + tracking on *video_path*.

        Returns a DataFrame validated against validate_trajectories().
        An empty (but schema-conformant) DataFrame is returned when the
        video contains no detections (e.g. a blank test clip).
        """
        # ---- lazy heavy imports ----------------------------------------
        try:
            import cv2
            import supervision as sv  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
            from sports.common.team import TeamClassifier  # type: ignore[import-not-found]
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                f"RoboflowBackend.process() requires the 'roboflow' optional extra. "
                f"Install with: uv pip install -e 'packages/soccer-vision[roboflow]'\n"
                f"Missing: {exc}"
            ) from exc

        # ---- device selection ------------------------------------------
        if self._device_override is not None:
            device = self._device_override
        elif torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # ---- download / locate weights ---------------------------------
        weight_paths = self._download_weights()

        # ---- load models -----------------------------------------------
        player_model = YOLO(str(weight_paths["player"]))
        ball_weights = self.ball_weights_path or weight_paths["ball"]
        ball_model   = YOLO(str(ball_weights))

        # ---- video metadata --------------------------------------------
        cap = cv2.VideoCapture(str(video_path))
        fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # ---- Pass 1: collect player crops for TeamClassifier ----------
        crops: list[object] = []
        for frame_idx, frame in enumerate(
            sv.get_video_frames_generator(source_path=str(video_path))
        ):
            if frame_idx % 60 != 0:
                continue
            result = player_model(frame, device=device, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(result)
            # Keep only player/GK class IDs for team classification
            player_mask = (dets.class_id == _CLS_PLAYER) | (dets.class_id == _CLS_GK)
            player_dets = dets[player_mask]
            for xyxy in player_dets.xyxy:
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size > 0:
                    crops.append(crop)

        # Fit TeamClassifier (needs at least 2 crops; skip if blank video)
        team_classifier: TeamClassifier | None = None
        if len(crops) >= 2:
            team_classifier = TeamClassifier(device=device)
            team_classifier.fit(crops)

        # ---- Pass 2: full detection, tracking, team prediction --------
        tracker = sv.ByteTrack()
        rows: list[dict[str, float | int | str]] = []

        for frame_idx, frame in enumerate(
            sv.get_video_frames_generator(source_path=str(video_path))
        ):
            t_sec = frame_idx / fps

            # --- player / GK / referee detection + tracking ---
            p_result = player_model(frame, device=device, verbose=False)[0]
            p_dets = sv.Detections.from_ultralytics(p_result)

            # Run ByteTrack on all non-ball detections
            tracked = tracker.update_with_detections(p_dets)

            for j in range(len(tracked)):
                cls_id = int(tracked.class_id[j])
                if cls_id not in _CLS_NAME:
                    continue
                cls_name = _CLS_NAME[cls_id]

                x1, y1, x2, y2 = tracked.xyxy[j]
                x_px = (x1 + x2) / 2.0   # bottom-center x for player/GK/ref
                y_px = float(y2)          # foot point y

                conf_val = float(tracked.confidence[j]) if tracked.confidence is not None else 0.5
                track_id = int(tracked.tracker_id[j]) if tracked.tracker_id is not None else 0

                # Team assignment
                if cls_name == "referee":
                    team = "ref"
                elif cls_name in ("player", "goalkeeper") and team_classifier is not None:
                    crop = frame[max(0, int(y1)):max(0, int(y2)), max(0, int(x1)):max(0, int(x2))]
                    if crop.size > 0:
                        cluster = int(team_classifier.predict([crop])[0])
                        team = _CLUSTER_TEAM.get(cluster, "unknown")
                    else:
                        team = "unknown"
                else:
                    team = "unknown"

                rows.append({
                    "frame":    frame_idx,
                    "t_seconds": t_sec,
                    "track_id": track_id,
                    "x_px":     float(x_px),
                    "y_px":     float(y_px),
                    "bbox_x1":  float(x1),
                    "bbox_y1":  float(y1),
                    "bbox_x2":  float(x2),
                    "bbox_y2":  float(y2),
                    "class":    cls_name,
                    "team":     team,
                    "conf":     conf_val,
                })

            # --- ball detection (separate model, synthetic track IDs) ---
            b_result = ball_model(frame, device=device, verbose=False)[0]
            b_dets = sv.Detections.from_ultralytics(b_result)

            for j in range(len(b_dets)):
                x1, y1, x2, y2 = b_dets.xyxy[j]
                x_px = (x1 + x2) / 2.0
                y_px = (y1 + y2) / 2.0  # center for ball

                conf_val = float(b_dets.confidence[j]) if b_dets.confidence is not None else 0.5
                # Synthetic track ID: negative space, collision-proof with ByteTrack IDs
                synthetic_id = _BALL_TRACK_ID_BASE - frame_idx

                rows.append({
                    "frame":    frame_idx,
                    "t_seconds": t_sec,
                    "track_id": synthetic_id,
                    "x_px":     float(x_px),
                    "y_px":     float(y_px),
                    "bbox_x1":  float(x1),
                    "bbox_y1":  float(y1),
                    "bbox_x2":  float(x2),
                    "bbox_y2":  float(y2),
                    "class":    "ball",
                    "team":     "unknown",
                    "conf":     conf_val,
                })

        # ---- Build and validate DataFrame ------------------------------
        if not rows:
            df = _empty_df()
        else:
            df = pd.DataFrame(rows)
            df = df.astype({
                "frame":    "int64",
                "track_id": "int64",
                "x_px":     "float64",
                "y_px":     "float64",
                "bbox_x1":  "float64",
                "bbox_y1":  "float64",
                "bbox_x2":  "float64",
                "bbox_y2":  "float64",
                "conf":     "float64",
                "t_seconds": "float64",
            })

        validate_trajectories(df)
        return df
