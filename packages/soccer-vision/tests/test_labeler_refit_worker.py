"""RefitWorker unit tests: deterministic via wait_idle(), no sleeps, no calibration."""

from __future__ import annotations

import threading
from collections.abc import Callable

from soccer_vision.labeler.refit_worker import RefitWorker


def test_marks_get_computed_and_applied() -> None:
    applied: dict[int, int] = {}

    def compute(frames: list[int], is_cancelled: Callable[[], bool]) -> dict[int, int]:
        return {f: f * 10 for f in frames}

    def apply(results: dict[int, int]) -> None:
        applied.update(results)

    w = RefitWorker(compute, apply)
    w.start()
    try:
        w.mark_dirty([1, 2, 3])
        w.wait_idle(timeout=5)
        assert applied == {1: 10, 2: 20, 3: 30}
        assert w.pending() == 0
    finally:
        w.stop()


def test_overlapping_marks_union() -> None:
    seen_batches: list[list[int]] = []

    def compute(frames: list[int], is_cancelled: Callable[[], bool]) -> dict[int, int]:
        seen_batches.append(sorted(frames))
        return {f: f for f in frames}

    w: RefitWorker[int] = RefitWorker(compute, lambda results: None)
    w.start()
    try:
        w.mark_dirty([1, 2])
        w.mark_dirty([2, 3])
        w.wait_idle(timeout=5)
        # every requested frame was computed at least once (union, nothing dropped)
        computed = {f for batch in seen_batches for f in batch}
        assert computed == {1, 2, 3}
    finally:
        w.stop()


def test_cancellation_requeues_and_eventually_applies() -> None:
    # First compute blocks until a second mark_dirty bumps the revision, so the
    # first pass is cancelled; the worker must re-run and eventually apply.
    bumped = threading.Event()
    applied: dict[int, int] = {}
    calls = {"n": 0}

    def compute(frames: list[int], is_cancelled: Callable[[], bool]) -> dict[int, int] | None:
        calls["n"] += 1
        if calls["n"] == 1:
            bumped.wait(timeout=5)      # let the test add a newer mark first
            if is_cancelled():
                return None             # superseded -> discard
        return {f: f for f in frames}

    def apply(results: dict[int, int]) -> None:
        applied.update(results)

    w = RefitWorker(compute, apply)
    w.start()
    try:
        w.mark_dirty([1])
        w.mark_dirty([2])              # bumps revision while pass 1 is inside compute
        bumped.set()
        w.wait_idle(timeout=5)
        assert applied == {1: 1, 2: 2}  # nothing lost despite the cancel
    finally:
        w.stop()


def test_stop_is_idempotent_and_joins() -> None:
    w: RefitWorker[int] = RefitWorker(
        lambda frames, c: {}, lambda r: None
    )
    w.start()
    w.stop()
    w.stop()  # no error on double stop
    assert w.pending() == 0
