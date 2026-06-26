"""Per-frame homography fitting between image-plane points and pitch coordinates."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np
from numpy.typing import NDArray


class HomographyError(ValueError):
    """Raised when homography fitting fails (e.g., too few points)."""


def fit_homography(
    image_points: NDArray[np.floating],
    pitch_points: NDArray[np.floating],
    *,
    ransac_thresh: float | None = None,
) -> NDArray[np.floating]:
    """Fit a 3x3 homography mapping image_points -> pitch_points.

    Parameters
    ----------
    image_points
        Nx2 array of pixel coords from the source frame.
    pitch_points
        Nx2 array of corresponding canonical-pitch coords (in [0, 1]^2 typically).
    ransac_thresh
        RANSAC reprojection threshold measured in the DESTINATION (pitch) space.
        When None (default) the call is identical to before — cv2.findHomography uses
        its built-in default of 3.0, which in pitch [0,1] units makes every point an
        inlier (no rejection). Set it to a small pitch-unit value to actually reject
        gross outliers.

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
    if ransac_thresh is None:
        H, _ = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC)
    else:
        H, _ = cv2.findHomography(
            image_points, pitch_points, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
        )
    if H is None:
        raise HomographyError("cv2.findHomography returned None — degenerate input")
    return H.astype(np.float64)


def smooth_homographies(
    homographies: Mapping[int, NDArray[np.floating]],
    alpha: float = 0.5,
) -> dict[int, NDArray[np.floating]]:
    """Exponential moving average over frame-indexed homographies.

    H_smoothed[t] = alpha * H[t] + (1 - alpha) * H_smoothed[t-1].
    Missing frames between known ones carry forward the previous smoothed H.

    Parameters
    ----------
    homographies
        Dict mapping frame_idx → 3x3 H. May be sparse.
    alpha
        Smoothing factor in [0, 1]. 1.0 = no smoothing; 0.0 = fully carry first.

    Returns
    -------
    smoothed
        Dict over all frames from min to max of input, fully filled (no gaps).
    """
    if not homographies:
        return {}
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    frames_sorted = sorted(homographies.keys())
    f_min, f_max = frames_sorted[0], frames_sorted[-1]
    smoothed: dict[int, NDArray[np.floating]] = {}
    prev: NDArray[np.floating] | None = None
    for fi in range(f_min, f_max + 1):
        H_obs = homographies.get(fi)
        if H_obs is not None:
            if prev is None:
                smoothed[fi] = H_obs.copy()
            else:
                smoothed[fi] = alpha * H_obs + (1.0 - alpha) * prev
        else:
            assert prev is not None, "first frame must have an observation"
            smoothed[fi] = prev.copy()
        prev = smoothed[fi]
    return smoothed
