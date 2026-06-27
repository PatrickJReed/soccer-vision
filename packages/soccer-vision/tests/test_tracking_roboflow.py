"""Tests for the RoboflowBackend adapter.

Protocol conformance is tested unconditionally. The heavy detection path
requires roboflow/sports + ultralytics installed (the 'roboflow' optional
extra); those tests are skipped if the extras aren't available.
"""

from __future__ import annotations

import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from soccer_vision.io.schema import validate_trajectories
from soccer_vision.tracking.base import TrackingBackend
from soccer_vision.tracking.roboflow import RoboflowBackend

try:
    import sports  # type: ignore[import-not-found]  # noqa: F401
    import ultralytics  # type: ignore[import-not-found]  # noqa: F401
    _HEAVY_AVAILABLE = True
except ImportError:
    _HEAVY_AVAILABLE = False


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    """30-frame, 320x240, single-colour video — exercises the codec path only."""
    out = tmp_path / "tiny.mp4"
    fps = 30
    w, h = 320, 240
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    for _ in range(30):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()
    return out


def test_adapter_satisfies_protocol() -> None:
    backend = RoboflowBackend()
    assert isinstance(backend, TrackingBackend)
    assert backend.name == "roboflow-sports"
    assert backend.version


def test_adapter_module_import_does_not_require_heavy_deps() -> None:
    """Importing the adapter module must not fail when ultralytics/sports aren't installed."""
    import soccer_vision.tracking.roboflow as _  # noqa: F401


@pytest.mark.skipif(not _HEAVY_AVAILABLE, reason="roboflow extras not installed")
def test_adapter_returns_schema_conformant_df(tiny_video: Path) -> None:
    """Heavy path: full process() on a blank video; empty df is acceptable."""
    backend = RoboflowBackend()
    df = backend.process(tiny_video)
    validate_trajectories(df)


def test_default_detect_pitch_false() -> None:
    backend = RoboflowBackend()
    assert backend.detect_pitch is False


@pytest.mark.skipif(not _HEAVY_AVAILABLE, reason="roboflow extras not installed")
def test_adapter_with_pitch_returns_keypoints(tiny_video: Path) -> None:
    """When detect_pitch=True, process_with_pitch() returns (df, keypoints_df)."""
    from soccer_vision.io.schema import validate_trajectories
    backend = RoboflowBackend(detect_pitch=True)
    df, kp_df = backend.process_with_pitch(tiny_video)
    validate_trajectories(df)
    assert {"frame", "kp_idx", "x_px", "y_px", "conf"}.issubset(kp_df.columns)


def test_pitch_weights_use_release_url() -> None:
    from soccer_vision.tracking.roboflow import PITCH_V1_URL, WEIGHTS

    kind, locator, filename = WEIGHTS["pitch"]
    assert kind == "url"
    assert locator == PITCH_V1_URL
    assert filename == "pitch_yolov8_v1.pt"
    assert PITCH_V1_URL.endswith("pitch_yolov8_v1.pt")
    assert "releases/download/pitch-v1/" in PITCH_V1_URL


def test_pitch_weights_path_override_missing_raises(tmp_path: Path) -> None:
    from soccer_vision.tracking.roboflow import RoboflowBackend

    missing = tmp_path / "nope.pt"
    try:
        RoboflowBackend(pitch_weights_path=missing)
    except FileNotFoundError as e:
        assert "pitch_weights_path" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_pitch_weights_path_override_accepted(tmp_path: Path) -> None:
    from soccer_vision.tracking.roboflow import RoboflowBackend

    w = tmp_path / "custom_pitch.pt"
    w.write_bytes(b"stub")
    backend = RoboflowBackend(pitch_weights_path=w, detect_pitch=True)
    assert backend.pitch_weights_path == w


def test_download_weights_skips_pitch_when_override_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local pitch override must not trigger a fetch of the (unpublished,
    404-ing) canonical pitch asset — only player/ball get downloaded."""
    import sys
    import urllib.request

    import soccer_vision.tracking.roboflow as rf

    fetched: list[str] = []

    def _fake_urlretrieve(url: str, dest: str) -> None:
        fetched.append(f"url:{url}")
        Path(dest).write_bytes(b"stub")

    class _FakeGdown:
        @staticmethod
        def download(url: str, dest: str, quiet: bool = False) -> None:
            fetched.append(f"gdrive:{url}")
            Path(dest).write_bytes(b"stub")

    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_urlretrieve)
    monkeypatch.setitem(sys.modules, "gdown", _FakeGdown)

    pitch = tmp_path / "custom_pitch.pt"
    pitch.write_bytes(b"stub")
    backend = rf.RoboflowBackend(
        pitch_weights_path=pitch, detect_pitch=True, weights_cache_dir=tmp_path
    )
    paths = backend._download_weights()

    assert "pitch" not in paths  # skipped entirely
    assert "player" in paths and "ball" in paths
    assert not any("pitch" in f for f in fetched)  # no pitch URL/gdrive hit


def test_version_is_not_stale_date_marker() -> None:
    """The stale 'main@YYYY-MM-DD' value is not a real upstream ref; version must be
    a git SHA, a vX.Y[.Z] tag, or the explicit UNPINNED sentinel (pinned by a human)."""
    v = RoboflowBackend.version
    assert not re.fullmatch(r"main@\d{4}-\d{2}-\d{2}", v), (
        "version is still the stale date marker; pin to a real roboflow/sports SHA or tag"
    )
    assert re.fullmatch(r"[0-9a-f]{7,40}|v\d+\.\d+(\.\d+)?|UNPINNED", v), (
        f"version {v!r} must be a SHA, a vX.Y tag, or the UNPINNED sentinel"
    )


@dataclass
class _Canned:
    """Handle returned by _install_canned: live counters/captures for assertions."""

    decode_count: int = 0
    bytetrack_kwargs: dict[str, Any] = field(default_factory=dict)


def _install_canned(
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_frames: int = 5,
    n_balls: int = 1,
    ball_confs: tuple[float, ...] = (0.9,),
) -> _Canned:
    """Inject fake supervision/ultralytics/torch/sports modules into sys.modules so the
    lazily-imported heavy path of _run_pipeline runs with no GPU/weights. Player model
    emits player(cls2)/GK(cls1)/referee(cls3); ball model emits n_balls; pitch model emits
    1 instance x 3 keypoints. Returns a handle counting video decodes + ByteTrack kwargs."""
    handle = _Canned()
    height, width = 720, 1280

    class FakeDetections:
        def __init__(self, xyxy: Any, class_id: Any,
                     confidence: Any = None, tracker_id: Any = None) -> None:
            self.xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
            self.class_id = np.asarray(class_id, dtype=np.int64)
            self.confidence = None if confidence is None else np.asarray(confidence, dtype=np.float64)
            self.tracker_id = None if tracker_id is None else np.asarray(tracker_id, dtype=np.int64)

        def __len__(self) -> int:
            return len(self.xyxy)

        def __getitem__(self, mask: Any) -> FakeDetections:
            m = np.asarray(mask)
            return FakeDetections(
                self.xyxy[m], self.class_id[m],
                None if self.confidence is None else self.confidence[m],
                None if self.tracker_id is None else self.tracker_id[m],
            )

        @classmethod
        def from_ultralytics(cls, result: Any) -> FakeDetections:
            return result.detections  # type: ignore[no-any-return]

    class FakeKeyPoints:
        def __init__(self, xy: Any, confidence: Any = None) -> None:
            self.xy = np.asarray(xy, dtype=np.float64)
            self.confidence = None if confidence is None else np.asarray(confidence, dtype=np.float64)

        def __len__(self) -> int:
            return int(self.xy.shape[0])

        @classmethod
        def from_ultralytics(cls, result: Any) -> FakeKeyPoints:
            return result.keypoints  # type: ignore[no-any-return]

    class FakeByteTrack:
        def __init__(self, **kwargs: Any) -> None:
            handle.bytetrack_kwargs = dict(kwargs)

        def update_with_detections(self, dets: FakeDetections) -> FakeDetections:
            n = len(dets)
            return FakeDetections(
                dets.xyxy, dets.class_id, dets.confidence,
                tracker_id=np.arange(1, n + 1, dtype=np.int64),
            )

    def get_video_frames_generator(source_path: str) -> Any:
        handle.decode_count += 1

        def _gen() -> Any:
            for _ in range(n_frames):
                yield np.zeros((height, width, 3), dtype=np.uint8)

        return _gen()

    class _PlayerResult:
        detections = FakeDetections(
            xyxy=[[100, 200, 140, 300], [400, 200, 440, 300], [700, 200, 720, 300]],
            class_id=[2, 1, 3], confidence=[0.8, 0.7, 0.6])

    class _BallResult:
        detections = FakeDetections(
            xyxy=[[600 + 10 * i, 350, 610 + 10 * i, 360] for i in range(n_balls)],
            class_id=[0] * n_balls, confidence=list(ball_confs))

    class _PitchResult:
        keypoints = FakeKeyPoints(xy=[[[10, 20], [30, 40], [50, 60]]],
                                  confidence=[[0.9, 0.8, 0.7]])

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            p = str(path).upper()
            self._role = "ball" if "BALL" in p else "pitch" if "PITCH" in p else "player"

        def to(self, device: str | None = None) -> FakeYOLO:
            return self

        def __call__(self, frame: Any, **kwargs: Any) -> list[Any]:
            if self._role == "ball":
                return [_BallResult()]
            if self._role == "pitch":
                return [_PitchResult()]
            return [_PlayerResult()]

    class FakeTeamClassifier:
        def __init__(self, device: str | None = None) -> None:
            pass

        def fit(self, crops: Any) -> None:
            pass

        def predict(self, crops: Any) -> list[int]:
            return [0] * len(crops)  # cluster 0 -> "own" via _CLUSTER_TEAM

    def _module(name: str, **attrs: Any) -> types.ModuleType:
        # Non-literal attr key -> no ruff B010; setattr (not `mod.x =`) -> no mypy attr-defined.
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        return mod

    team_mod = _module("sports.common.team", TeamClassifier=FakeTeamClassifier)
    common_mod = _module("sports.common", team=team_mod)
    sports_mod = _module("sports", common=common_mod)
    modules: list[tuple[str, types.ModuleType]] = [
        ("supervision", _module(
            "supervision", Detections=FakeDetections, KeyPoints=FakeKeyPoints,
            ByteTrack=FakeByteTrack, get_video_frames_generator=get_video_frames_generator)),
        ("ultralytics", _module("ultralytics", YOLO=FakeYOLO)),
        ("torch", _module(
            "torch",
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False)))),
        ("sports", sports_mod),
        ("sports.common", common_mod),
        ("sports.common.team", team_mod),
    ]
    for name, mod in modules:
        monkeypatch.setitem(sys.modules, name, mod)
    return handle


def _run_canned(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path, **kw: Any
) -> tuple[Any, Any, _Canned, RoboflowBackend]:
    handle = _install_canned(monkeypatch, **kw)
    backend = RoboflowBackend(device="cpu", detect_pitch=True)
    monkeypatch.setattr(
        backend, "_download_weights",
        lambda: {"player": Path("PLAYER.pt"), "ball": Path("BALL.pt"), "pitch": Path("PITCH.pt")},
    )
    df, kp_df = backend._run_pipeline(tiny_video, emit_keypoints=True)
    return df, kp_df, handle, backend


def test_run_pipeline_core_with_canned_models(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path
) -> None:
    df, kp_df, _handle, _backend = _run_canned(monkeypatch, tiny_video)
    validate_trajectories(df)

    f0 = df[df["frame"] == 0]
    player = f0[f0["class"] == "player"].iloc[0]
    gk = f0[f0["class"] == "goalkeeper"].iloc[0]
    ball = f0[f0["class"] == "ball"].iloc[0]
    ref = f0[f0["class"] == "referee"].iloc[0]

    # foot point: x = bbox centre, y = bbox bottom
    assert player["x_px"] == 120.0 and player["y_px"] == 300.0
    assert gk["x_px"] == 420.0 and gk["y_px"] == 300.0
    # ball centre
    assert ball["x_px"] == 605.0 and ball["y_px"] == 355.0
    # class mapping (_CLS_NAME) + referee team from class
    assert ref["team"] == "ref"
    # per-track team-map applied to player/GK (cluster 0 -> own)
    assert player["team"] == "own" and gk["team"] == "own"
    # synthetic ball id = _BALL_TRACK_ID_BASE - frame_idx
    from soccer_vision.tracking.roboflow import _BALL_TRACK_ID_BASE
    assert int(ball["track_id"]) == _BALL_TRACK_ID_BASE - 0

    # keypoint reshape: 1 instance x 3 keypoints per frame
    kp0 = kp_df[kp_df["frame"] == 0].sort_values("kp_idx")
    assert list(kp0["kp_idx"]) == [0, 1, 2]
    assert list(kp0["x_px"]) == [10.0, 30.0, 50.0]
    assert list(kp0["y_px"]) == [20.0, 40.0, 60.0]
    assert kp0.iloc[0]["conf"] == 0.9


def test_tracker_kwargs_default_is_empty(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path
) -> None:
    _df, _kp, handle, _b = _run_canned(monkeypatch, tiny_video)
    assert handle.bytetrack_kwargs == {}  # default: today's bare sv.ByteTrack()


def test_tracker_kwargs_forwarded_to_bytetrack(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path
) -> None:
    handle = _install_canned(monkeypatch)
    backend = RoboflowBackend(
        device="cpu", detect_pitch=True,
        tracker_kwargs={"track_activation_threshold": 0.3, "lost_track_buffer": 60},
    )
    monkeypatch.setattr(
        backend, "_download_weights",
        lambda: {"player": Path("PLAYER.pt"), "ball": Path("BALL.pt"), "pitch": Path("PITCH.pt")},
    )
    backend._run_pipeline(tiny_video, emit_keypoints=True)
    assert handle.bytetrack_kwargs == {"track_activation_threshold": 0.3, "lost_track_buffer": 60}


def test_run_pipeline_decodes_video_once(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path
) -> None:
    _df, _kp, handle, _b = _run_canned(monkeypatch, tiny_video)
    assert handle.decode_count == 1  # no separate TeamClassifier-fit pass


def test_single_pass_still_classifies_teams(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path
) -> None:
    df, _kp, _h, _b = _run_canned(monkeypatch, tiny_video)
    players = df[df["class"].isin(["player", "goalkeeper"])]
    assert (players["team"] == "own").all()  # fit-after-pass still labels every track


def test_one_highest_conf_ball_per_frame(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path
) -> None:
    df, _kp, _h, _b = _run_canned(monkeypatch, tiny_video, n_balls=2, ball_confs=(0.3, 0.9))
    from soccer_vision.tracking.roboflow import _BALL_TRACK_ID_BASE
    balls = df[df["class"] == "ball"]
    per_frame = balls.groupby("frame").size()
    assert (per_frame == 1).all()  # exactly one ball row per frame
    b0 = balls[balls["frame"] == 0].iloc[0]
    assert b0["conf"] == 0.9                      # the higher-conf detection won
    assert b0["x_px"] == 615.0                    # second box centre (610..620)
    assert int(b0["track_id"]) == _BALL_TRACK_ID_BASE - 0  # unique synthetic id
