"""Display first frame of bakeoff_clip.mp4; user clicks 4 pitch corners.

Outputs pixel coordinates of (top-left, top-right, bottom-left, bottom-right)
pitch corners as JSON. Pitch corners = visible field-marking corners, not
camera image corners.

Usage:
    uv run python scripts/annotate_corners.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

CLIP = Path(__file__).parent.parent / "data" / "bakeoff_clip.mp4"
OUT = Path(__file__).parent.parent / "data" / "bakeoff_clip_corners.json"
LABELS = ["top_left", "top_right", "bottom_left", "bottom_right"]


def main() -> None:
    cap = cv2.VideoCapture(str(CLIP))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame of {CLIP}")

    clicks: list[tuple[int, int]] = []

    def on_click(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            cv2.circle(frame, (x, y), 8, (0, 255, 0), 2)
            cv2.putText(
                frame, LABELS[len(clicks) - 1], (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            cv2.imshow("Click 4 pitch corners (TL, TR, BL, BR)", frame)

    cv2.imshow("Click 4 pitch corners (TL, TR, BL, BR)", frame)
    cv2.setMouseCallback("Click 4 pitch corners (TL, TR, BL, BR)", on_click)

    while len(clicks) < 4:
        if cv2.waitKey(20) & 0xFF == 27:  # Esc
            break
    cv2.destroyAllWindows()

    if len(clicks) != 4:
        raise SystemExit("Did not click 4 corners. Aborting.")

    out = {label: {"x": x, "y": y} for label, (x, y) in zip(LABELS, clicks, strict=True)}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {OUT}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
