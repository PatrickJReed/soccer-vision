"""Pure metrics for scoring the pitch-keypoint model against labeler ground truth.

Everything here is numpy in / dataclass out, no I/O — so the eval logic itself is
unit-testable (the lesson of anchor_cov, which shipped untested and gave a false
pass). Errors are reported in real-world FEET: both pitch axes are fractions of
pitch length, so a canonical Euclidean distance scales by one constant.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Nominal US Soccer 9v9 pitch length: ~68.5 m. Youth fields vary; this is a fixed
# nominal scale so feet errors are interpretable and comparable across retrains.
DEFAULT_PITCH_LENGTH_FT: float = 224.7
HIDDEN_IDX: int = 5  # under-camera landmark, never ground-truth-visible


def canonical_to_feet(
    distance: float | NDArray[np.floating],
    length_ft: float = DEFAULT_PITCH_LENGTH_FT,
) -> float | NDArray[np.floating]:
    """Convert a canonical-pitch distance (fraction of length) to feet."""
    return distance * length_ft
