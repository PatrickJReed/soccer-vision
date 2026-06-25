"""The 9v9 pitch as a known rigid 3D model (planar, Z=0) in metres.

Camera calibration solves the physical camera against THIS fixed structure, so a
real camera pose can't fold the far field into view the way a free homography can.
Dimensions are the nominal US Soccer 9v9 mid-range from PitchSpec's docstring.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from soccer_vision.pitch.landmarks import PITCH_LANDMARKS

LENGTH_M: float = 68.5  # goal-to-goal (the canonical y axis)
WIDTH_M: float = 45.7   # touchline-to-touchline (the canonical x axis)
METRES_TO_FEET: float = 3.28084


def field_points_3d() -> NDArray[np.float64]:
    """The 21 canonical landmarks as real-world metres on the Z=0 plane.

    PITCH_LANDMARKS col 0 is a fraction of WIDTH, col 1 a fraction of LENGTH.
    """
    pts = np.zeros((len(PITCH_LANDMARKS), 3), dtype=np.float64)
    pts[:, 0] = PITCH_LANDMARKS[:, 0] * WIDTH_M
    pts[:, 1] = PITCH_LANDMARKS[:, 1] * LENGTH_M
    return pts
