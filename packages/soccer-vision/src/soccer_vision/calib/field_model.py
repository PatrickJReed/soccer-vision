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


# Named pitch lines as pairs of landmark indices whose 3D positions are the line's
# two endpoints. These cover the regions with no clickable point landmark: the near
# touchline (x=0, under the camera) and the FULL midline (halfway_near/idx 5 is
# hidden, so only halfway_far/idx 4 is a clickable point — but the line is known).
FIELD_LINES: dict[str, tuple[int, int]] = {
    "near_touchline": (0, 2),  # corner_own_left -> corner_opp_left (x=0)
    "far_touchline": (1, 3),   # corner_own_right -> corner_opp_right (x=WIDTH_M)
    "own_goal_line": (0, 1),   # corner_own_left -> corner_own_right (y=0)
    "opp_goal_line": (2, 3),   # corner_opp_left -> corner_opp_right (y=LENGTH_M)
    "midline": (5, 4),         # halfway_near -> halfway_far (y=LENGTH_M/2)
}


def field_line_3d(line_id: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The two 3D metre endpoints of a named field line (on the Z=0 plane)."""
    if line_id not in FIELD_LINES:
        raise KeyError(f"unknown line {line_id!r}; valid: {sorted(FIELD_LINES)}")
    i, j = FIELD_LINES[line_id]
    fp = field_points_3d()
    return fp[i], fp[j]
