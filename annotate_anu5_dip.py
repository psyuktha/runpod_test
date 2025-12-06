"""
Draw bounding boxes and pin labels on `square/anu5_07_edges.png` for a 14-pin DIP
(7 pins on top, 7 on bottom) using provided coordinates.

Run:
  python annotate_anu5_dip.py \
    --input square/anu5_07_edges.png \
    --output debug/anu5_07_edges_bbox.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2


# Provided bounding boxes (x1, y1, x2, y2) with labels
PIN_BOXES: List[Tuple[str, Tuple[int, int, int, int]]] = [
    ("Pin 7", (801, 804, 835, 875)),
    ("Pin 6", (801, 690, 835, 761)),
    ("Pin 5", (801, 574, 835, 645)),
    ("Pin 4", (801, 458, 832, 528)),
    ("Pin 3", (801, 343, 828, 414)),
    ("Pin 2", (800, 227, 831, 298)),
    ("Pin 1", (800, 112, 830, 183)),
    ("Pin 8", (163, 804, 228, 876)),
    ("Pin 9", (163, 690, 228, 761)),
    ("Pin 10", (165, 575, 229, 646)),
    ("Pin 11", (163, 458, 228, 529)),
    ("Pin 12", (162, 345, 227, 414)),
    ("Pin 13", (162, 228, 225, 300)),
    ("Pin 14", (160, 113, 227, 186)),
]

PALETTE = [
    (0, 215, 255),   # gold
    (255, 144, 30),  # orange
    (255, 105, 180), # pink
    (72, 249, 239),  # cyan
]


def annotate(image_path: Path, output_path: Path) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")

    annotated = img.copy()
    for idx, (label, (x1, y1, x2, y2)) in enumerate(PIN_BOXES):
        color = PALETTE[idx % len(PALETTE)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate 14 DIP pins on anu5 image.")
    parser.add_argument("--input", type=Path, default=Path("square/anu5_07_edges.png"))
    parser.add_argument("--output", type=Path, default=Path("debug/anu5_07_edges_bbox.png"))
    args = parser.parse_args()

    annotate(args.input, args.output)


if __name__ == "__main__":
    main()

