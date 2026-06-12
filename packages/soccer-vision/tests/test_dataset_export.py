"""Tests for the YOLO-pose training-dataset exporter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from soccer_vision.dataset_export import build_data_yaml, export_game, main, select_frames
from soccer_vision.pipeline import homographies_to_parquet
from soccer_vision.pitch.landmarks import FLIP_IDX
from soccer_vision.pitch.propagation import HomographyEntry


def test_select_frames_stride_and_temporal_split() -> None:
    covered = list(range(0, 100))
    train, val = select_frames(covered, stride=5, val_frac=0.10)
    assert train + val == list(range(0, 100, 5))   # stride-5, order preserved
    assert len(val) == 2                            # 20 sampled * 0.10 = 2
    assert val == [90, 95]                          # temporally LAST
    assert set(train).isdisjoint(val)


def test_select_frames_small_input() -> None:
    train, val = select_frames([3, 9], stride=1, val_frac=0.10)
    assert train == [3, 9]
    assert val == []                                # floor(2*0.1)=0


def test_select_frames_empty() -> None:
    assert select_frames([], stride=5, val_frac=0.1) == ([], [])


def test_build_data_yaml_contents(tmp_path: Path) -> None:
    build_data_yaml(tmp_path)
    text = (tmp_path / "data.yaml").read_text()
    # Parse with a minimal key: value parser (pyyaml not available in dev env)
    cfg: dict[str, object] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, raw = line.partition(":")
        val = raw.strip()
        # Parse lists: [a, b, c]
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            items = [x.strip() for x in inner.split(",")]
            try:
                cfg[key.strip()] = [int(x) for x in items]
            except ValueError:
                cfg[key.strip()] = [x.strip('"').strip("'") for x in items]
        else:
            try:
                cfg[key.strip()] = int(val)
            except ValueError:
                cfg[key.strip()] = val.strip('"').strip("'")
    assert cfg["nc"] == 1
    assert cfg["names"] == ["pitch"]
    assert cfg["kpt_shape"] == [21, 3]
    assert cfg["flip_idx"] == list(FLIP_IDX)
    assert cfg["train"] == "images/train"
    assert cfg["val"] == "images/val"
    assert cfg["path"] == "."


_W, _H, _N = 320, 240, 60
# full-pixel -> pitch: x_pitch = x_px / W, y_pitch = y_px / H
_H_PX = np.diag([1.0 / _W, 1.0 / _H, 1.0])


def _write_video(path: Path) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (_W, _H))  # type: ignore[attr-defined]
    for i in range(_N):
        frame = np.full((_H, _W, 3), (40, 120, 40), dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        vw.write(frame)
    vw.release()


def _write_homs(path: Path, frames: range) -> None:
    homographies_to_parquet(
        {f: HomographyEntry(_H_PX, "manual", 1.0) for f in frames}, path
    )


def test_export_game_end_to_end(tmp_path: Path) -> None:
    video = tmp_path / "g1.mp4"
    _write_video(video)
    homs = tmp_path / "h1.parquet"
    _write_homs(homs, range(_N))
    out = tmp_path / "ds"

    stats = export_game(video, homs, out, "g1", stride=5, min_confidence=0.5,
                        val_frac=0.10, qa_seed=0)

    train_imgs = sorted((out / "images/train").glob("g1_f*.jpg"))
    val_imgs = sorted((out / "images/val").glob("g1_f*.jpg"))
    assert len(train_imgs) + len(val_imgs) == 12          # 60/5
    assert len(val_imgs) == 1                              # floor(12*0.1)
    for img in train_imgs + val_imgs:
        lbl = out / "labels" / img.parent.name / (img.stem + ".txt")
        assert lbl.exists()
        tokens = lbl.read_text().split()
        assert len(tokens) == 1 + 4 + 21 * 3
    assert (out / "qa_g1.jpg").exists()
    assert stats["n_train"] == 11 and stats["n_val"] == 1
    assert len(stats["per_landmark"]) == 21
    assert stats["per_landmark"][5] == 0                   # never-visible idx 5


def test_export_game_wrong_video_raises(tmp_path: Path) -> None:
    video = tmp_path / "g1.mp4"
    _write_video(video)                                    # 60 frames
    homs = tmp_path / "h1.parquet"
    _write_homs(homs, range(400, 410))                     # beyond the video
    try:
        export_game(video, homs, tmp_path / "ds", "g1", stride=5,
                    min_confidence=0.5, val_frac=0.1, qa_seed=0)
    except ValueError as e:
        assert "wrong video" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_main_duplicate_stems_rejected(tmp_path: Path) -> None:
    video = tmp_path / "g1.mp4"
    _write_video(video)
    homs = tmp_path / "h1.parquet"
    _write_homs(homs, range(_N))
    argv = ["--game", str(video), str(homs), "--game", str(video), str(homs),
            "--out-dir", str(tmp_path / "ds")]
    try:
        main(argv)
    except ValueError as e:
        assert "duplicate" in str(e).lower()
    else:
        raise AssertionError("expected rejection of duplicate game stems")


def test_main_end_to_end_with_zip(tmp_path: Path) -> None:
    video = tmp_path / "g1.mp4"
    _write_video(video)
    homs = tmp_path / "h1.parquet"
    _write_homs(homs, range(_N))
    out = tmp_path / "ds"
    main(["--game", str(video), str(homs), "--out-dir", str(out), "--zip"])
    assert (out / "data.yaml").exists()
    assert (tmp_path / "ds.zip").exists()


def test_qa_sheet_dots_survive_downscale(tmp_path: Path) -> None:
    # dots drawn on a hi-res frame must remain detectable after the sheet's
    # 480px-cell downscale (the acceptance run caught them vanishing).
    from soccer_vision.dataset_export import _draw_keypoints, _write_qa_sheet

    frame = np.full((1080, 1920, 3), (40, 120, 40), dtype=np.uint8)
    kpts = np.zeros((21, 3))
    kpts[0] = [960.0, 540.0, 2.0]
    kpts[3] = [400.0, 300.0, 2.0]
    cell = _draw_keypoints(frame, kpts)
    _write_qa_sheet([cell], tmp_path / "qa.jpg")
    sheet = cv2.imread(str(tmp_path / "qa.jpg"))
    assert sheet is not None, "qa.jpg was not written"
    mask = (np.abs(sheet.astype(int) - np.array([60, 220, 120])).sum(axis=2) < 90)
    assert int(mask.sum()) > 10   # green markers detectable post-downscale
