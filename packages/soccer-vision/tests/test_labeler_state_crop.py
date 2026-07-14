"""Crop-engine wiring through LabelerState: bootstrap, dirty scoping, honest export."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from soccer_vision.labeler.state import LabelerState

from tests.test_global_crop import SIZE, CropWorld


def _interframe(world: CropWorld) -> dict[int, NDArray[np.float64]]:
    # cumulative_transforms convention: interframe[f] maps frame f -> f+1, and
    # M[f] = M[f-1] @ inv(interframe[f-1]) with M[0] = I. With interframe[f] =
    # T(d_f) @ inv(T(d_{f+1})) the reconstructed canvas is world's shifted by the
    # constant d_0 — an origin choice H_g absorbs, so the solve is unaffected.
    return {f: np.asarray(world.transforms[f] @ np.linalg.inv(world.transforms[f + 1]),
                          np.float64)
            for f in range(world.n_frames - 1)}


def _state(world: CropWorld, tmp_path: Path) -> LabelerState:
    return LabelerState(interframe=_interframe(world), n_frames=world.n_frames, size=SIZE,
                        autosave_path=tmp_path / "clicks.json", engine="crop")


def test_crop_bootstrap_single_frame(tmp_path: Path) -> None:
    """The crop engine bootstraps from ONE well-spread frame (physical needed 3)."""
    world = CropWorld()
    st = _state(world, tmp_path)
    try:
        f = 120
        ids = world.visible(f)
        assert len(ids) >= 4
        st.add_clicks(world.clicks_at(f, ids))
        st.wait_idle()
        cf = st.frame_homography(f)
        assert cf is not None and cf.is_anchor
    finally:
        st.stop_worker()


def test_export_honest_confidence_and_gate_json(tmp_path: Path) -> None:
    world = CropWorld()
    st = _state(world, tmp_path)
    try:
        for f in range(0, world.n_frames, 20):
            st.add_clicks(world.clicks_at(f))
        st.wait_idle()
        out = tmp_path / "out"
        st.export(out)
    finally:
        st.stop_worker()
    df = pd.read_parquet(out / "homographies.parquet")
    assert not df.empty
    assert (df["confidence"] <= 0.9).all() and (df["confidence"] >= 0.6).all()
    assert not (df["confidence"] == 1.0).any()          # the F-C2 overclaim is retired
    gate = json.loads((out / "calib_gate.json").read_text())
    assert gate["engine"] == "crop"
    assert "prop_by_end" in gate and isinstance(gate["passed_numeric"], bool)
    assert gate["n_anchors"] >= 1


def test_point_click_dirty_scoped_to_segment(tmp_path: Path) -> None:
    """LabelerState builds segment_of from interframe gaps: dropping key 119 breaks
    the chain at frame 120. In crop mode a point click re-solves only its own
    segment's H_g (no shared focal), so a click at 150 must dirty only frames >= 120.
    Both segments get >= 4-landmark bulk frames so both bootstrap (fixture pan
    overshoot leaves frames >= ~175 with < 4 visible landmarks — avoid those)."""
    world = CropWorld()
    interframe = {f: m for f, m in _interframe(world).items() if f != 119}
    st = LabelerState(interframe=interframe, n_frames=world.n_frames, size=SIZE,
                      autosave_path=tmp_path / "c.json", engine="crop")
    try:
        for f in (0, 40, 80, 140, 160):
            clicks = world.clicks_at(f)
            assert len(clicks) >= 4
            st.add_clicks(clicks)
        st.wait_idle()
        marked: list[int] = []
        st._worker.mark_dirty = lambda frames: marked.extend(frames)  # type: ignore[method-assign]
        c = world.clicks_at(150)[0]
        st.add_click(150, c.kp_idx, c.x, c.y)
        assert marked and min(marked) >= 120        # only the second segment re-solves
    finally:
        st.stop_worker()
