"""Tests for the per-track team-classification helpers in RoboflowBackend.

These are module-level pure functions (the heavy roboflow deps are imported
lazily inside _run_pipeline), so they're testable without the roboflow extra.
"""

from __future__ import annotations

from collections.abc import Sequence

from soccer_vision.tracking.roboflow import _classify_teams_per_track, _keep_top_crop

_CLUSTER_TEAM = {0: "own", 1: "opp"}


def test_keep_top_crop_keeps_largest_by_area() -> None:
    tc: dict[int, list[tuple[float, object]]] = {}
    for area in [10, 50, 5, 80, 30]:
        _keep_top_crop(tc, track_id=1, crop=f"c{area}", area=float(area), limit=3)
    kept = sorted(a for a, _ in tc[1])
    assert kept == [30, 50, 80]  # the three largest survive


def test_keep_top_crop_separates_tracks() -> None:
    tc: dict[int, list[tuple[float, object]]] = {}
    _keep_top_crop(tc, 1, "a", 10.0)
    _keep_top_crop(tc, 2, "b", 20.0)
    assert set(tc) == {1, 2}
    assert len(tc[1]) == 1 and len(tc[2]) == 1


def test_classify_modal_team_per_track_and_ties() -> None:
    # Fake predict returns each crop (an int cluster) unchanged.
    def predict(crops: list[object]) -> Sequence[int]:
        return [int(c) for c in crops]  # type: ignore[call-overload]

    track_crops: dict[int, list[object]] = {
        7: [0, 0, 1],  # mode own
        9: [1, 1, 1],  # opp
        3: [0, 1],     # tie -> unknown
    }
    out = _classify_teams_per_track(track_crops, predict, _CLUSTER_TEAM)
    assert out == {7: "own", 9: "opp", 3: "unknown"}


def test_classify_uses_single_batched_call() -> None:
    calls: list[int] = []

    def predict(crops: list[object]) -> Sequence[int]:
        calls.append(len(crops))
        return [0] * len(crops)

    track_crops: dict[int, list[object]] = {1: [0, 0], 2: [0], 3: [0, 0, 0]}
    _classify_teams_per_track(track_crops, predict, _CLUSTER_TEAM)
    assert calls == [6]  # one predict() over all six crops, not per-crop


def test_classify_unknown_cluster_maps_to_unknown() -> None:
    def predict(crops: list[object]) -> Sequence[int]:
        return [9] * len(crops)  # cluster 9 absent from the map

    out = _classify_teams_per_track({5: [0, 0]}, predict, _CLUSTER_TEAM)
    assert out == {5: "unknown"}


def test_classify_empty_returns_empty() -> None:
    assert _classify_teams_per_track({}, lambda c: [], _CLUSTER_TEAM) == {}
