"""PitchSpec — dimensionless pitch proportions for soccer-vision metrics.

All distances expressed as fractions of pitch length (the canonical unit).
This avoids per-game field-size calibration while keeping metrics comparable.
"""

from __future__ import annotations

from dataclasses import dataclass


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
