"""A generic background "recompute a dirty set" worker.

Calibration-agnostic: owns one daemon thread, a dirty frame-set, and a monotonic
revision counter. On each edit the owner calls mark_dirty(frames); the worker
snapshots the set, runs the compute callback OFF the lock (so callers/readers are
not blocked), and merges via the apply callback. A newer mark_dirty bumps the
revision, so an in-flight pass can cancel itself (is_cancelled()) and be re-run with
the unioned set — rapid edits coalesce instead of piling up. wait_idle() blocks until
the set is drained (used by tests and by export).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T")
# Compute returns {frame: result} for the requested frames, or None if it detected
# cancellation mid-pass and the partial result must be discarded.


class RefitWorker(Generic[T]):
    def __init__(
        self,
        compute: Callable[[list[int], Callable[[], bool]], dict[int, T] | None],
        apply: Callable[[dict[int, T]], None],
    ) -> None:
        self._compute = compute
        self._apply = apply
        self._cv = threading.Condition()
        self._dirty: set[int] = set()
        self._revision = 0
        self._inflight = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()

    def start(self) -> None:
        with self._cv:  # keep _running access consistent with the rest of the class
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="refit-worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            if not self._running:
                return
            self._running = False
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def mark_dirty(self, frames: Iterable[int]) -> None:
        batch = list(frames)
        if not batch:
            return
        with self._cv:
            self._dirty.update(batch)
            self._revision += 1
            self._idle.clear()
            self._cv.notify_all()

    def pending(self) -> int:
        with self._cv:
            return len(self._dirty) + self._inflight

    def wait_idle(self, timeout: float | None = None) -> None:
        self._idle.wait(timeout)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._running and not self._dirty:
                    self._idle.set()
                    self._cv.wait()
                if not self._running:
                    return
                frames = sorted(self._dirty)
                self._dirty.clear()
                rev = self._revision
                self._inflight = len(frames)

            def is_cancelled(_r: int = rev) -> bool:
                with self._cv:
                    return self._revision != _r

            results = self._compute(frames, is_cancelled)

            if results is None:
                # Superseded: put the frames back so they are recomputed with the
                # newer edit's set; do not apply the discarded partial.
                with self._cv:
                    self._dirty.update(frames)
                    self._inflight = 0
                continue

            self._apply(results)
            with self._cv:
                self._inflight = 0
                if not self._dirty:
                    self._idle.set()
