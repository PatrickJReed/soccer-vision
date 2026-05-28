"""PitchMapper — applies per-frame homography H_t to detection rows."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class PitchMapper:
    """Stateless utility for mapping pixel detections through per-frame homographies."""

    def transform(
        self,
        detections: pd.DataFrame,
        homographies: Mapping[int, NDArray[np.floating]],
    ) -> pd.DataFrame:
        """Append x_pitch, y_pitch columns by applying homographies[frame] to (x_px, y_px).

        Frames absent from `homographies` produce NaN pitch coords for their rows.
        Returns a new DataFrame; does not mutate input.
        """
        out = detections.copy()
        x_pitch = np.full(len(out), np.nan)
        y_pitch = np.full(len(out), np.nan)
        for frame_idx, group in out.groupby("frame", sort=False):
            H = homographies.get(frame_idx)  # type: ignore[arg-type]
            if H is None:
                continue
            pts = np.column_stack([
                group["x_px"].to_numpy(),
                group["y_px"].to_numpy(),
                np.ones(len(group)),
            ])
            mapped = (H @ pts.T).T
            mapped /= mapped[:, 2:3]
            x_pitch[group.index] = mapped[:, 0]
            y_pitch[group.index] = mapped[:, 1]
        out["x_pitch"] = x_pitch
        out["y_pitch"] = y_pitch
        return out
