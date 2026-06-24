"""Synthetic-ground-truth tests for the pitch-model eval metrics."""

from __future__ import annotations

import numpy as np
from soccer_vision.eval.pitch_metrics import DEFAULT_PITCH_LENGTH_FT, canonical_to_feet


def test_canonical_to_feet_scalar() -> None:
    # both pitch axes are fractions of length, so a 0.1 canonical distance is
    # 0.1 * length_ft feet.
    assert canonical_to_feet(0.1) == DEFAULT_PITCH_LENGTH_FT * 0.1


def test_canonical_to_feet_array() -> None:
    out = canonical_to_feet(np.array([0.0, 0.5, 1.0]), length_ft=200.0)
    assert np.allclose(out, [0.0, 100.0, 200.0])
