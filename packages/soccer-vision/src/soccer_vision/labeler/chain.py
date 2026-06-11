"""Compute the inter-frame registration chain over a video, with an on-disk cache.

Thin wrapper over pitch.propagation.compute_interframe_homographies (which the
propagation tests already cover) plus a deterministic .npz cache so reopening a
video is instant.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.pitch.propagation import compute_interframe_homographies


def normalize_homography(
    g: NDArray[np.floating],
    size: tuple[int, int],
) -> NDArray[np.float64]:
    """Rescale a full-res px->px homography to normalized [0,1] image coords.

    G_norm = S @ G @ inv(S), S = diag(1/W, 1/H, 1). This lets clicks (sent as
    normalized canvas fractions) compose with the inter-frame chain consistently.
    """
    w, h = size
    s = np.diag([1.0 / w, 1.0 / h, 1.0])
    s_inv = np.diag([float(w), float(h), 1.0])
    return (s @ np.asarray(g, dtype=np.float64) @ s_inv).astype(np.float64)


def denormalize_homography(
    h_norm: NDArray[np.floating], size: tuple[int, int]
) -> NDArray[np.float64]:
    """Convert a normalized-image->pitch homography to full-pixel->pitch.

    H_pixel = H_norm @ diag(1/W, 1/H, 1), since normalized = diag(1/W,1/H,1) @ pixel.
    """
    w, h = size
    s = np.diag([1.0 / w, 1.0 / h, 1.0])
    return (np.asarray(h_norm, dtype=np.float64) @ s).astype(np.float64)


def save_chain(
    path: Path,
    interframe: Mapping[int, NDArray[np.floating]],
    n_frames: int,
    size: tuple[int, int],
) -> None:
    """Persist {i: 3x3} + n_frames + (w, h) to a single .npz."""
    flat = {f"H{i}": np.asarray(H, dtype=np.float64) for i, H in interframe.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        keys=np.array(sorted(interframe), dtype=np.int64),
        n_frames=np.array(n_frames),
        size=np.array(size, dtype=np.int64),
        **flat,  # type: ignore[arg-type]
    )


def load_chain(
    path: Path,
) -> tuple[dict[int, NDArray[np.float64]], int, tuple[int, int]] | None:
    """Inverse of save_chain; None if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p)
    interframe = {
        int(k): np.asarray(data[f"H{int(k)}"], dtype=np.float64)
        for k in data["keys"]
    }
    size = (int(data["size"][0]), int(data["size"][1]))
    return interframe, int(data["n_frames"]), size


def _video_hash(video_path: Path) -> str:
    """Stable hash of a video from its path, size, and mtime (cheap, no full read)."""
    st = Path(video_path).stat()
    key = f"{Path(video_path).resolve()}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _chain_worker(
    args: tuple[str, int, int, float],
) -> dict[int, NDArray[np.float64]]:
    """Register pairs [start, end) of one video chunk (runs in a subprocess)."""
    video_path, start, end, downscale = args
    cap = cv2.VideoCapture(video_path)
    pos = 0

    def read_frame(idx: int) -> NDArray[np.uint8] | None:
        nonlocal pos
        if idx < pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            pos = idx
        while pos < idx:
            if not cap.grab():
                return None
            pos += 1
        ok, frame = cap.read()
        pos += 1
        return frame if ok else None  # type: ignore[return-value]

    boxes = pd.DataFrame(
        columns=["frame", "class", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    )
    try:
        result: dict[int, NDArray[np.floating]] = compute_interframe_homographies(
            read_frame, set(range(start, end)), boxes, downscale=downscale
        )
        return {k: np.asarray(v, dtype=np.float64) for k, v in result.items()}
    finally:
        cap.release()


def compute_chain(
    video_path: Path,
    *,
    cache_dir: Path | None = None,
    downscale: float = 1.0,
    player_boxes: pd.DataFrame | None = None,
    workers: int | None = None,
) -> tuple[dict[int, NDArray[np.float64]], int, tuple[int, int]]:
    """Inter-frame chain for the whole video (cached). Returns (interframe, n, (w,h)).

    workers: parallel chunked registration; None = cpu_count()-1; forced to 1
    when player_boxes is given (masking data is not carried into workers).
    """
    cache_dir = Path(cache_dir or (Path(video_path).parent / ".sv_labeler_cache"))
    cache_path = cache_dir / f"{_video_hash(video_path)}.npz"
    cached = load_chain(cache_path)
    if cached is not None:
        return cached

    # Probe the video for metadata, then release.
    probe = cv2.VideoCapture(str(video_path))
    n_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    probe.release()

    n_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    if player_boxes is not None:
        n_workers = 1

    interframe_px: dict[int, NDArray[np.floating]] = {}

    if n_workers <= 1:
        # Serial path — identical to original code.
        cap = cv2.VideoCapture(str(video_path))
        pos = 0

        def read_frame(idx: int) -> NDArray[np.uint8] | None:
            nonlocal pos
            if idx < pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                pos = idx
            while pos < idx:
                if not cap.grab():
                    return None
                pos += 1
            ok, frame = cap.read()
            pos += 1
            return frame if ok else None  # type: ignore[return-value]

        boxes = player_boxes if player_boxes is not None else pd.DataFrame(
            columns=["frame", "class", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        )
        needed = set(range(n_frames - 1))
        try:
            interframe_px = compute_interframe_homographies(
                read_frame, needed, boxes, downscale=downscale
            )
        finally:
            cap.release()
    else:
        n_pairs = n_frames - 1
        bounds = np.linspace(0, n_pairs, n_workers + 1).astype(int)
        jobs = [
            (str(video_path), int(bounds[i]), int(bounds[i + 1]), downscale)
            for i in range(n_workers)
            if bounds[i] < bounds[i + 1]
        ]
        with Pool(processes=len(jobs)) as pool:
            for part in pool.imap_unordered(_chain_worker, jobs):
                interframe_px.update(part)
                print(f"chain: {len(interframe_px)}/{n_pairs} pairs registered")

    interframe = {
        i: normalize_homography(g, (width, height)) for i, g in interframe_px.items()
    }
    save_chain(cache_path, interframe, n_frames, (width, height))
    return interframe, n_frames, (width, height)
