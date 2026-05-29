"""RoboflowBackend: TrackingBackend adapter wrapping roboflow/sports.

Heavy dependencies (ultralytics, supervision, sports, torch, gdown) are
imported lazily inside process() so this module loads fine in CI without
the 'roboflow' optional extra installed.

Install extras:
    uv pip install -e "packages/soccer-vision[roboflow]"
"""

from __future__ import annotations

import urllib.request
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

# Direct download URL for the fine-tuned ball detector (Phase 2), published as
# a GitHub release asset. The asset filename must match the tail of this URL.
BALL_V1_URL: Final = (
    "https://github.com/PatrickJReed/soccer-vision/releases/download/"
    "ball-v1/ball_yolov8_v1.pt"
)

# Model weights registry. Each entry: (kind, locator, filename).
#   kind="gdrive" -> locator is a Google Drive file ID (fetched via gdown)
#   kind="url"    -> locator is a direct HTTPS download URL (fetched via urllib)
# The ball role defaults to the fine-tuned detector. The roboflow baseline ball
# model it replaced lived at gdrive id 1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V.
WEIGHTS: Final[dict[str, tuple[str, str, str]]] = {
    "ball":   ("url",    BALL_V1_URL, "ball_yolov8_v1.pt"),
    "player": ("gdrive", "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q", "football-player-detection.pt"),
    "pitch":  ("gdrive", "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf", "football-pitch-detection.pt"),
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


# Position columns linearly interpolated when bridging a ball-trajectory gap.
_BALL_INTERP_COLS: Final = ("x_px", "y_px", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")


def interpolate_ball_gaps(
    df: pd.DataFrame,
    fps: float,
    max_gap_frames: int = 15,
) -> pd.DataFrame:
    """Bridge short gaps in the ball trajectory with interpolated rows.

    The roboflow ball detector assigns a fresh synthetic ``track_id`` per frame
    and bypasses ByteTrack, so any frame the detector misses leaves a hole in
    the ball trajectory. Downstream possession/phase logic needs a continuous
    ball signal, so we linearly interpolate position across runs of at most
    ``max_gap_frames`` consecutive missing frames.

    Interpolated rows carry ``conf == 0.0`` so they stay distinguishable from
    real detections (which run at a conf floor > 0). Gaps longer than
    ``max_gap_frames`` are left untouched — those are sustained losses (fast
    pans, long occlusions) that interpolation must not paper over.

    Non-ball rows pass through unchanged. The result is sorted by frame and
    remains schema-conformant. When a frame holds multiple ball detections,
    the highest-conf one anchors the interpolation; original rows are never
    modified or dropped.
    """
    if max_gap_frames < 1 or df.empty:
        return df

    ball = df[df["class"] == "ball"]
    if ball["frame"].nunique() < 2:
        return df

    # One anchor position per frame: the highest-conf detection that frame.
    anchors = (
        ball.sort_values("conf")
        .drop_duplicates("frame", keep="last")
        .sort_values("frame")
        .reset_index(drop=True)
    )
    frames = anchors["frame"].to_numpy()

    new_rows: list[dict[str, float | int | str]] = []
    for i in range(len(frames) - 1):
        f0, f1 = int(frames[i]), int(frames[i + 1])
        n_missing = f1 - f0 - 1
        if n_missing < 1 or n_missing > max_gap_frames:
            continue
        a, b = anchors.iloc[i], anchors.iloc[i + 1]
        for fm in range(f0 + 1, f1):
            alpha = (fm - f0) / (f1 - f0)
            row: dict[str, float | int | str] = {
                "frame":     fm,
                "t_seconds": fm / fps,
                "track_id":  _BALL_TRACK_ID_BASE - fm,
                "class":     "ball",
                "team":      "unknown",
                "conf":      0.0,
            }
            for col in _BALL_INTERP_COLS:
                row[col] = float(a[col] + alpha * (b[col] - a[col]))
            new_rows.append(row)

    if not new_rows:
        return df

    filled = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    filled = filled.astype(_SCHEMA_DTYPES)
    return filled.sort_values(["frame", "track_id"]).reset_index(drop=True)


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
    ball_imgsz:
        Inference resolution for the ball model. Defaults to 1280 to match the
        fine-tune training resolution — the ball is a tiny object, and running
        inference at ultralytics' 640 default downscales it below the
        detection floor.
    ball_conf:
        Confidence threshold for the ball model. Defaults to 0.05: the
        fine-tuned detector's operating point sits well below the ultralytics
        0.25 default, and the gap is bridged by interpolation + downstream
        tracking rather than a high threshold.
    ball_max_gap_frames:
        Maximum run of consecutive missed frames to bridge by linear
        interpolation of the ball position. Defaults to 15 (~0.5s at 30fps).
        Longer gaps are left as holes (sustained losses, not flicker).
    detect_pitch:
        When True, download and load the pitch keypoint detection model.
        Required to call process_with_pitch(). Defaults to False to avoid
        downloading the extra weights file unless pitch detection is needed.
    """

    name: Final = "roboflow-sports"
    version: Final = "main@2026-05-28"

    def __init__(
        self,
        device: str | None = None,
        weights_cache_dir: Path | None = None,
        ball_weights_path: Path | None = None,
        ball_imgsz: int = 1280,
        ball_conf: float = 0.05,
        ball_max_gap_frames: int = 15,
        detect_pitch: bool = False,
    ) -> None:
        self._device_override = device
        self._weights_dir = Path(weights_cache_dir) if weights_cache_dir else DEFAULT_CACHE_DIR
        if ball_weights_path is not None and not ball_weights_path.exists():
            raise FileNotFoundError(f"ball_weights_path does not exist: {ball_weights_path}")
        self.ball_weights_path: Path | None = ball_weights_path
        self.ball_imgsz: int = ball_imgsz
        self.ball_conf: float = ball_conf
        self.ball_max_gap_frames: int = ball_max_gap_frames
        self.detect_pitch: bool = detect_pitch

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
        for role, (kind, locator, filename) in WEIGHTS.items():
            if role == "pitch" and not self.detect_pitch:
                continue
            dest = self._weights_dir / filename
            if not dest.exists():
                if kind == "gdrive":
                    gdown.download(f"https://drive.google.com/uc?id={locator}", str(dest), quiet=False)
                else:  # direct HTTPS URL (e.g. GitHub release asset)
                    urllib.request.urlretrieve(locator, str(dest))
            paths[role] = dest
        return paths

    # ------------------------------------------------------------------
    # Empty-keypoints-DataFrame helper
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_kp_df() -> pd.DataFrame:
        """Return a zero-row keypoints DataFrame with the canonical schema."""
        return pd.DataFrame(
            columns=["frame", "kp_idx", "x_px", "y_px", "conf"]
        ).astype({
            "frame":  "int64",
            "kp_idx": "int64",
            "x_px":   "float64",
            "y_px":   "float64",
            "conf":   "float64",
        })

    # ------------------------------------------------------------------
    # Core pipeline (shared by process() and process_with_pitch())
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        video_path: Path,
        emit_keypoints: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run detection + tracking on *video_path*.

        Parameters
        ----------
        video_path:
            Path to the input video file.
        emit_keypoints:
            When True, also run the pitch keypoint model per frame and
            accumulate results into the returned keypoints DataFrame.
            Requires self.detect_pitch=True (weights must be downloaded).

        Returns
        -------
        (trajectories_df, keypoints_df)
            trajectories_df is validated against validate_trajectories().
            keypoints_df has columns: frame, kp_idx, x_px, y_px, conf.
            When emit_keypoints is False, keypoints_df is empty but
            schema-conformant.
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
                f"RoboflowBackend requires the 'roboflow' optional extra. "
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

        pitch_model = None
        if emit_keypoints:
            pitch_model = YOLO(str(weight_paths["pitch"])).to(device=device)

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
        kp_records: list[dict[str, float | int]] = []

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
            # imgsz/conf default to the ball model's training resolution and a
            # low operating point; missed frames are bridged downstream by
            # interpolate_ball_gaps() rather than a high threshold.
            b_result = ball_model(
                frame,
                imgsz=self.ball_imgsz,
                conf=self.ball_conf,
                device=device,
                verbose=False,
            )[0]
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

            # --- pitch keypoint detection (only when emit_keypoints=True) ---
            if emit_keypoints and pitch_model is not None:
                pk_result = pitch_model(frame, imgsz=1280, verbose=False)[0]
                kp = sv.KeyPoints.from_ultralytics(pk_result)
                # kp.xy is shape (N_instances, N_keypoints, 2)
                # kp.confidence is (N_instances, N_keypoints) or None
                for inst_idx in range(len(kp)):
                    for kp_idx in range(kp.xy.shape[1]):
                        x, y = kp.xy[inst_idx, kp_idx]
                        c = (
                            float(kp.confidence[inst_idx, kp_idx])
                            if kp.confidence is not None
                            else 0.5
                        )
                        kp_records.append({
                            "frame":  frame_idx,
                            "kp_idx": kp_idx,
                            "x_px":   float(x),
                            "y_px":   float(y),
                            "conf":   c,
                        })

        # ---- Build and validate trajectories DataFrame -----------------
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

        # Bridge short ball-detection gaps so downstream possession/phase logic
        # gets a continuous ball signal (the ball bypasses ByteTrack above).
        df = interpolate_ball_gaps(df, fps=fps, max_gap_frames=self.ball_max_gap_frames)

        validate_trajectories(df)

        # ---- Build keypoints DataFrame ---------------------------------
        if not kp_records:
            kp_df = self._empty_kp_df()
        else:
            kp_df = pd.DataFrame(kp_records).astype({
                "frame":  "int64",
                "kp_idx": "int64",
                "x_px":   "float64",
                "y_px":   "float64",
                "conf":   "float64",
            })

        return df, kp_df

    # ------------------------------------------------------------------
    # Public processing entry-points
    # ------------------------------------------------------------------

    def process(self, video_path: Path) -> pd.DataFrame:
        """Run roboflow/sports detection + tracking on *video_path*.

        Returns a DataFrame validated against validate_trajectories().
        An empty (but schema-conformant) DataFrame is returned when the
        video contains no detections (e.g. a blank test clip).
        """
        df, _ = self._run_pipeline(video_path, emit_keypoints=False)
        return df

    def process_with_pitch(self, video_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run detection + tracking and pitch keypoint detection on *video_path*.

        Requires that the backend was constructed with detect_pitch=True.

        Returns
        -------
        (trajectories_df, keypoints_df)
            trajectories_df: validated against validate_trajectories().
            keypoints_df: per-frame pitch keypoints with columns
                frame, kp_idx, x_px, y_px, conf.
        """
        if not self.detect_pitch:
            raise ValueError(
                "process_with_pitch() requires detect_pitch=True. "
                "Re-create the backend with RoboflowBackend(detect_pitch=True)."
            )
        return self._run_pipeline(video_path, emit_keypoints=True)
