"""Auto-label dataset builder: trusted homographies + video -> (JPEG, line-mask PNG)
pairs with a split-agnostic manifest (game/field/view identifiers) — the training-data
engine for the per-frame line-segmentation model.

Sampling discipline: confidence-gated (>= min_confidence, accepted sources only),
strided to ~stride_s seconds, capped per view (near-duplicate control). Frames decode in
ONE ascending sequential pass, streamed one frame at a time (never per-frame seek, never
a full-selection buffer); a decode failure drops that frame AND every selected frame
after it (EOF semantics, not per-frame skip), all counted undecodable, never silently
written. Masks are lossless PNG. Re-running a game REPLACES its manifest rows
(idempotent per game).
Spec: docs/superpowers/specs/2026-07-14-line-mask-autolabel-design.md
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from soccer_vision.pipeline import homographies_from_parquet
from soccer_vision.pitch.line_masks import LINE_CLASSES, line_mask, mask_overlay
from soccer_vision.pitch.propagation import HomographyEntry
from soccer_vision.pitch.spec import PitchSpec

ACCEPTED_SOURCES = ("rep", "registered", "manual")
_MANIFEST_COLS = ["game_id", "field_id", "view_id", "frame", "source", "confidence",
                  "image", "mask"]


@dataclass(frozen=True)
class GameStats:
    game_id: str
    n_candidates: int      # frames passing the trust gate, before sampling
    n_selected: int        # after stride + per-view cap
    n_written: int
    n_undecodable: int
    class_pixel_frac: dict[str, float]


def _trusted(e: HomographyEntry, min_confidence: float) -> bool:
    """The trust gate: accepted provenance AND confidence at/above the floor."""
    return e.source in ACCEPTED_SOURCES and e.confidence >= min_confidence


def select_frames(
    entries: Mapping[int, HomographyEntry],
    *,
    fps: float,
    stride_s: float,
    per_view_cap: int,
    min_confidence: float,
    view_of: Mapping[int, int] | None,
) -> list[int]:
    """Trust-gate, stride to ~stride_s seconds, cap per view. Deterministic.

    Stride anchoring (f % stride == 0) assumes DENSE frame coverage (registered/
    propagated parquets); a sparse-anchor parquet (e.g. a handful of manual reps at
    arbitrary frames) will select ~nothing — the CLI warns when n_selected == 0
    while n_candidates > 0.
    """
    trusted = sorted(f for f, e in entries.items() if _trusted(e, min_confidence))
    stride = max(1, round(fps * stride_s))
    strided = [f for f in trusted if f % stride == 0]
    if view_of is None:
        return strided
    taken: dict[int, int] = {}
    out: list[int] = []
    for f in strided:
        v = view_of.get(f, -1)
        if taken.get(v, 0) < per_view_cap:
            taken[v] = taken.get(v, 0) + 1
            out.append(f)
    return out


def _decode_selected(
    video_path: Path, frames: list[int]
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """Stream (frame, image) pairs in ONE ascending sequential pass.

    One frame in memory at a time — a full game's selection materialized would be
    tens of GB. Stops at the first grab/read failure (EOF semantics): that frame
    and every later one are dropped; the caller counts undecodables as
    len(selected) - pairs consumed.
    """
    want = sorted(frames)
    cap = cv2.VideoCapture(str(video_path))
    pos = 0
    try:
        for f in want:
            while pos < f:
                if not cap.grab():
                    return
                pos += 1
            ok, img = cap.read()
            if not ok:
                return
            pos += 1
            yield f, img  # type: ignore[misc]  # cv2 MatLike -> uint8 image
    finally:
        cap.release()


def _merge_manifest(out_dir: Path, game_id: str, rows: list[dict[str, object]]) -> None:
    """Idempotent per game: replace the game's rows, keep every other game's."""
    path = out_dir / "manifest.parquet"
    new = pd.DataFrame(rows, columns=_MANIFEST_COLS)
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[old["game_id"] != game_id], new], ignore_index=True)
    new.to_parquet(path, index=False)


def build_game(
    video_path: str | Path,
    homographies_path: str | Path,
    out_dir: str | Path,
    *,
    game_id: str,
    field_id: str,
    view_of: Mapping[int, int] | None = None,
    stride_s: float = 1.0,
    per_view_cap: int = 120,
    min_confidence: float = 0.6,
    jpeg_quality: int = 90,
    spec: PitchSpec | None = None,
    contact_sheet: bool = True,
) -> GameStats:
    """Generate one game's (image, mask) pairs into out_dir + merge its manifest rows."""
    video_path, out_dir = Path(video_path), Path(out_dir)
    entries = homographies_from_parquet(Path(homographies_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    trusted = [f for f, e in entries.items() if _trusted(e, min_confidence)]
    selected = select_frames(entries, fps=fps, stride_s=stride_s,
                             per_view_cap=per_view_cap, min_confidence=min_confidence,
                             view_of=view_of)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    pix_counts = dict.fromkeys(LINE_CLASSES.values(), 0)
    total_px = 0
    # Contact-sheet frames are picked evenly across the SELECTION (streaming decode:
    # the written set isn't known up front); this drifts from even-across-written
    # only when tail frames are undecodable — acceptable.
    if selected:
        idx = np.linspace(0, len(selected) - 1, min(16, len(selected))).astype(int)
        sheet_frames = {selected[i] for i in idx}
    else:
        sheet_frames = set()
    overlays: list[NDArray[np.uint8]] = []
    for f, img in _decode_selected(video_path, selected):
        e = entries[f]
        mask = line_mask(e.H, size, spec=spec)
        img_rel = f"images/{game_id}_{f:06d}.jpg"
        msk_rel = f"masks/{game_id}_{f:06d}.png"
        cv2.imwrite(str(out_dir / img_rel), img,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        cv2.imwrite(str(out_dir / msk_rel), mask)
        for cls, name in LINE_CLASSES.items():
            pix_counts[name] += int((mask == cls).sum())
        total_px += mask.size
        rows.append({"game_id": game_id, "field_id": field_id,
                     "view_id": int(view_of.get(f, -1)) if view_of else -1,
                     "frame": int(f), "source": e.source,
                     "confidence": float(e.confidence),
                     "image": img_rel, "mask": msk_rel})
        if contact_sheet and f in sheet_frames:
            tile: NDArray[np.uint8] = cv2.resize(  # type: ignore[assignment]
                mask_overlay(img, mask), (0, 0), fx=0.25, fy=0.25)
            overlays.append(tile)
    n_undec = len(selected) - len(rows)
    _merge_manifest(out_dir, game_id, rows)
    if contact_sheet and overlays:
        cols = 4
        rws = -(-len(overlays) // cols)
        th, tw = overlays[0].shape[:2]
        sheet = np.zeros((rws * th, cols * tw, 3), np.uint8)
        for i, o in enumerate(overlays):
            r, c = divmod(i, cols)
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = o
        cv2.imwrite(str(out_dir / f"contact_{game_id}.jpg"), sheet)
    frac = {k: (v / total_px if total_px else 0.0) for k, v in pix_counts.items()}
    return GameStats(game_id, len(trusted), len(selected), len(rows), n_undec, frac)
