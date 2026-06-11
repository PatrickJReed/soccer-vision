"""Tests for the registration-chain cache round-trip + normalization."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from soccer_vision.labeler.chain import (
    compute_chain,
    denormalize_homography,
    load_chain,
    normalize_homography,
    save_chain,
)


def test_normalize_homography_translation() -> None:
    # a full-res px translation of +10 in x on a width-100 frame is +0.1 normalized.
    g = np.eye(3)
    g[0, 2] = 10.0
    gn = normalize_homography(g, (100, 50))
    out = gn @ np.array([0.2, 0.0, 1.0])
    out = out[:2] / out[2]
    assert np.allclose(out, [0.3, 0.0])


def test_chain_cache_round_trip(tmp_path: Path) -> None:
    interframe = {0: np.eye(3), 2: np.diag([1.0, 2.0, 1.0])}
    path = tmp_path / "chain.npz"
    save_chain(path, interframe, n_frames=4, size=(1920, 1080))
    loaded = load_chain(path)
    assert loaded is not None
    loaded_if, n_frames, size = loaded
    assert n_frames == 4
    assert size == (1920, 1080)
    assert set(loaded_if) == {0, 2}
    assert np.allclose(loaded_if[2], np.diag([1.0, 2.0, 1.0]))


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert load_chain(tmp_path / "nope.npz") is None


def test_denormalize_homography_inverts_normalize() -> None:
    # a normalized-space identity maps normalized->normalized; denormalized it
    # should map full-pixel (W,H) to normalized (1,1).
    hn = np.eye(3)
    hp = denormalize_homography(hn, (200, 100))
    out = hp @ np.array([200.0, 100.0, 1.0])
    out = out[:2] / out[2]
    assert np.allclose(out, [1.0, 1.0])


def _write_pan_video(path: Path, n: int = 40) -> None:
    """Synthetic textured video with a slow pan so registration succeeds."""
    rng = np.random.default_rng(0)
    big = (rng.random((300, 500, 3)) * 255).astype(np.uint8)
    big = cv2.GaussianBlur(big, (5, 5), 0).astype(np.uint8)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))  # type: ignore[attr-defined]
    for i in range(n):
        x = 2 * i  # 2px pan per frame
        vw.write(big[20:260, x:x + 320])
    vw.release()


def test_parallel_chain_equals_serial(tmp_path: Path) -> None:
    video = tmp_path / "pan.mp4"
    _write_pan_video(video)
    serial_if, n1, size1 = compute_chain(video, cache_dir=tmp_path / "c1", workers=1)
    par_if, n2, size2 = compute_chain(video, cache_dir=tmp_path / "c2", workers=3)
    assert (n1, size1) == (n2, size2)
    assert set(serial_if) == set(par_if)
    for k in serial_if:
        assert np.allclose(serial_if[k], par_if[k], atol=1e-9)
