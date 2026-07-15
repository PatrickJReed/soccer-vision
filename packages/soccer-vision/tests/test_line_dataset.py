"""Tests for line_dataset.py — selection, generation, manifest idempotency."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from soccer_vision import line_dataset as ld
from soccer_vision.pipeline import homographies_to_parquet
from soccer_vision.pitch.propagation import HomographyEntry

_W, _H = 320, 240
# A benign synthetic homography for a 320x240 frame: image -> pitch, roughly the
# whole pitch visible (scale px to [0,1] with mild perspective).
_H_IMG2PITCH = np.array([[1.0 / _W, 0.0, 0.0], [0.0, 1.0 / _H, 0.0], [0.0, 1e-4, 1.0]])


def _video(path: Path, n_frames: int = 90) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (_W, _H))  # type: ignore[attr-defined]
    for i in range(n_frames):
        frame = np.full((_H, _W, 3), (i * 3) % 255, np.uint8)
        vw.write(frame)
    vw.release()
    return path


def _session(tmp_path: Path, frames: list[int], *, source: str = "registered",
             conf: float = 0.9) -> Path:
    session = tmp_path / "session"
    session.mkdir(exist_ok=True)
    entries = {f: HomographyEntry(_H_IMG2PITCH, source, conf) for f in frames}
    homographies_to_parquet(entries, session / "homographies.parquet")
    return session


def test_select_frames_gates_and_strides() -> None:
    entries = {f: HomographyEntry(_H_IMG2PITCH, "registered", 0.9) for f in range(90)}
    # Gated entries sit ON stride multiples so the gate is load-bearing (a deleted
    # gate changes the output), not shadowed by the stride.
    entries[30] = HomographyEntry(_H_IMG2PITCH, "registered", 0.3)   # below gate
    entries[60] = HomographyEntry(_H_IMG2PITCH, "none", 0.9)         # bad source
    sel = ld.select_frames(entries, fps=30.0, stride_s=1.0, per_view_cap=120,
                           min_confidence=0.6, view_of=None)
    assert sel == [0]  # 30 gated by confidence, 60 gated by source
    sel2 = ld.select_frames(entries, fps=30.0, stride_s=0.5, per_view_cap=120,
                            min_confidence=0.6, view_of=None)
    assert sel2 == [0, 15, 45, 75]  # stride 15, minus the gated 30 and 60


def test_select_frames_per_view_cap() -> None:
    entries = {f: HomographyEntry(_H_IMG2PITCH, "manual", 0.9) for f in range(0, 300, 3)}
    view_of = {f: (0 if f < 150 else 1) for f in range(300)}
    sel = ld.select_frames(entries, fps=30.0, stride_s=0.1, per_view_cap=5,
                           min_confidence=0.6, view_of=view_of)
    per_view = {0: 0, 1: 0}
    for f in sel:
        per_view[view_of[f]] += 1
    assert per_view[0] == 5 and per_view[1] == 5


def test_build_writes_pairs_manifest_and_stats(tmp_path: Path) -> None:
    video = _video(tmp_path / "game.mp4")
    session = _session(tmp_path, list(range(90)))
    out = tmp_path / "dataset"
    stats = ld.build_game(video, session / "homographies.parquet", out,
                          game_id="g1", field_id="fieldA", stride_s=1.0)
    assert stats.n_written == 3 and stats.n_undecodable == 0
    man = pd.read_parquet(out / "manifest.parquet")
    assert len(man) == 3
    assert set(man.columns) >= {"game_id", "field_id", "view_id", "frame", "source",
                                "confidence", "image", "mask"}
    row = man.iloc[0]
    img = cv2.imread(str(out / row["image"]))
    msk = cv2.imread(str(out / row["mask"]), cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (_H, _W, 3)
    assert msk is not None and msk.shape == (_H, _W)
    assert set(np.unique(msk)) <= {0, 1, 2, 3, 4, 5}
    assert (msk > 0).sum() > 0                       # lines actually drawn
    # PNG losslessness: reloaded mask equals the in-memory rasterization exactly
    from soccer_vision.pitch.line_masks import line_mask
    assert np.array_equal(msk, line_mask(_H_IMG2PITCH, (_W, _H)))


def test_rerun_is_idempotent_per_game(tmp_path: Path) -> None:
    video = _video(tmp_path / "game.mp4")
    session = _session(tmp_path, list(range(90)))
    out = tmp_path / "dataset"
    ld.build_game(video, session / "homographies.parquet", out, game_id="g1",
                  field_id="fieldA", stride_s=1.0)
    ld.build_game(video, session / "homographies.parquet", out, game_id="g1",
                  field_id="fieldA", stride_s=1.0)                    # re-run: replaces
    other = ld.build_game(video, session / "homographies.parquet", out, game_id="g2",
                          field_id="fieldB", stride_s=1.0)
    man = pd.read_parquet(out / "manifest.parquet")
    assert len(man) == 6 and set(man["game_id"]) == {"g1", "g2"}
    assert other.n_written == 3


def test_unopenable_video_raises(tmp_path: Path) -> None:
    session = _session(tmp_path, [0, 30])
    with pytest.raises(ValueError, match="cannot open video"):
        ld.build_game(tmp_path / "missing.mp4", session / "homographies.parquet",
                      tmp_path / "dataset", game_id="g1", field_id="fieldA")


def test_undecodable_frames_are_dropped_and_counted(tmp_path: Path) -> None:
    video = _video(tmp_path / "game.mp4", n_frames=60)
    session = _session(tmp_path, [0, 30, 300])       # 300 is beyond the video
    out = tmp_path / "dataset"
    stats = ld.build_game(video, session / "homographies.parquet", out, game_id="g1",
                          field_id="fieldA", stride_s=1.0)
    assert stats.n_written == 2 and stats.n_undecodable == 1
    man = pd.read_parquet(out / "manifest.parquet")
    assert sorted(man["frame"]) == [0, 30]


def test_load_games_registry(tmp_path: Path) -> None:
    reg = tmp_path / "games.toml"
    reg.write_text(
        '[g1]\nfield = "fieldA"\nvideo = "v1.mp4"\nsession = "s1"\n'
        '[g2]\nfield = "fieldB"\nvideo = "v2.mp4"\nsession = "s2"\n')
    games = ld.load_games(reg)
    assert set(games) == {"g1", "g2"}
    assert games["g1"].field == "fieldA"
    assert games["g1"].video == (tmp_path / "v1.mp4")     # relative to the registry
    assert games["g2"].session == (tmp_path / "s2")


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _video(tmp_path / "game.mp4")
    session = _session(tmp_path, list(range(90)))
    (tmp_path / "games.toml").write_text(
        f'[g1]\nfield = "fieldA"\nvideo = "game.mp4"\nsession = "{session.name}"\n'
        '[missing]\nfield = "fieldB"\nvideo = "nope.mp4"\nsession = "nope"\n')
    out = tmp_path / "dataset"
    ld.main(["--games", str(tmp_path / "games.toml"), "--out", str(out)])
    text = capsys.readouterr().out
    assert "g1" in text and "SKIP" in text            # missing inputs skipped loudly
    stats = out / "dataset_stats.json"
    assert stats.exists()
    payload = json.loads(stats.read_text())
    assert "g1" in payload["games"] and payload["config"]["stride_s"] == 1.0
    man = pd.read_parquet(out / "manifest.parquet")
    assert set(man["game_id"]) == {"g1"}


def test_stats_file_merges_across_reruns(tmp_path: Path) -> None:
    _video(tmp_path / "game.mp4")
    session = _session(tmp_path, list(range(90)))
    (tmp_path / "games.toml").write_text(
        f'[g1]\nfield = "fieldA"\nvideo = "game.mp4"\nsession = "{session.name}"\n'
        f'[g2]\nfield = "fieldB"\nvideo = "game.mp4"\nsession = "{session.name}"\n')
    out = tmp_path / "dataset"
    ld.main(["--games", str(tmp_path / "games.toml"), "--out", str(out)])
    ld.main(["--games", str(tmp_path / "games.toml"), "--out", str(out), "--game", "g1"])
    payload = json.loads((out / "dataset_stats.json").read_text())
    # g1-only re-run must not discard g2's stats (parallel to per-game manifest merge)
    assert set(payload["games"]) == {"g1", "g2"}
    assert payload["games"]["g2"]["field_id"] == "fieldB"


def test_cli_game_filter_and_sparse_warning(tmp_path: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    _video(tmp_path / "game.mp4")
    # sparse-anchor session: trusted frames all OFF stride multiples -> n_selected == 0
    session = _session(tmp_path, [7, 41, 73])
    (tmp_path / "games.toml").write_text(
        f'[g1]\nfield = "fieldA"\nvideo = "game.mp4"\nsession = "{session.name}"\n')
    out = tmp_path / "dataset"
    ld.main(["--games", str(tmp_path / "games.toml"), "--out", str(out), "--game", "g1"])
    text = capsys.readouterr().out
    assert "WARNING" in text and "sparse" in text.lower()
