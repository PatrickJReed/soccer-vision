"""Per-frame homography fitting between image-plane points and pitch coordinates."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


class HomographyError(ValueError):
    """Raised when homography fitting fails (e.g., too few points)."""


def fit_homography(
    image_points: NDArray[np.floating],
    pitch_points: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Fit a 3x3 homography mapping image_points -> pitch_points.

    Parameters
    ----------
    image_points
        Nx2 array of pixel coords from the source frame.
    pitch_points
        Nx2 array of corresponding canonical-pitch coords (in [0, 1]^2 typically).

    Returns
    -------
    H
        3x3 homography matrix such that ``(x, y, 1) -> H @ (x_img, y_img, 1)`` and the
        result is normalized so the last component is 1.

    Raises
    ------
    HomographyError
        If fewer than 4 points are provided, shapes mismatch, or RANSAC fails.
    """
    if image_points.shape != pitch_points.shape:
        raise HomographyError(
            f"image_points and pitch_points must have the same number of rows; "
            f"got {image_points.shape} vs {pitch_points.shape}"
        )
    if image_points.shape[0] < 4:
        raise HomographyError(
            f"Need at least 4 corresponding points; got {image_points.shape[0]}"
        )
    H, _ = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC)
    if H is None:
        raise HomographyError("cv2.findHomography returned None — degenerate input")
    return H.astype(np.float64)
