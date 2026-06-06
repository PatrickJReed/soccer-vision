"""Tests for the registration-chain cache round-trip + normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from soccer_vision.labeler.chain import load_chain, normalize_homography, save_chain


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
