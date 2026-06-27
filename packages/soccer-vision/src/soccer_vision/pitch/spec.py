"""PitchSpec — dimensionless pitch proportions for soccer-vision metrics.

All distances expressed as fractions of pitch length (the canonical unit).
This avoids per-game field-size calibration while keeping metrics comparable.

`length_norm_xy` is the single source of truth for aspect-corrected pitch
distance: x is a fraction of WIDTH and y of LENGTH (§3), so dividing x by
aspect_ratio puts both axes on the same physical scale. hygiene/core.py uses
this convention inline; possession (and SP5's §6 radius metrics) import it here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

PitchCoord = float | NDArray[np.float64]


@dataclass(frozen=True)
class PitchSpec:
    """Dimensionless pitch description.

    Defaults are US Soccer 9v9 mid-range (~68.5 m x 45.7 m).
    """

    aspect_ratio: float = 1.5
    n_outfield_per_team: int = 8
    penalty_box_length_frac: float = 0.187
    penalty_box_width_frac: float = 0.720
    center_circle_radius_frac: float = 0.106
    coverage_cell_frac: float = 0.011
    goal_width_frac: float = 0.140

    @classmethod
    def standard_9v9(cls) -> PitchSpec:
        return cls()

    @classmethod
    def fifa_11v11(cls) -> PitchSpec:
        return cls(
            aspect_ratio=1.54,
            n_outfield_per_team=10,
            penalty_box_length_frac=0.157,
            penalty_box_width_frac=0.592,
            center_circle_radius_frac=0.087,
            coverage_cell_frac=0.0095,
            goal_width_frac=0.108,
        )


def length_norm_xy(
    x_pitch: PitchCoord, y_pitch: PitchCoord, spec: PitchSpec
) -> tuple[PitchCoord, PitchCoord]:
    """Return (x / aspect_ratio, y): coords in isotropic pitch-LENGTH units.

    x is a fraction of pitch WIDTH and y a fraction of pitch LENGTH (master spec
    §3); dividing x by aspect_ratio puts both axes on the same physical scale so
    that Euclidean distances are isotropic (a circle, not an ellipse). Accepts
    scalars or numpy arrays. This is the one convention `hygiene/core.py`
    applies inline and SP5's §6 metrics will reuse.
    """
    return x_pitch / spec.aspect_ratio, y_pitch
