#!/usr/bin/env python3
"""
Compute the area (pixel count) of a specific connected component label
in the outline image. Labels are 1-based in the order discovered by a
4-neighborhood flood fill on a binarized (white>128) mask.

Usage:
    python area_of_label.py bi_full/bilateral_canny_anu2.jpeg --label 60
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def load_binary_mask(path: Path, threshold: int = 128) -> np.ndarray:
    """Load image, convert to grayscale, binarize: white -> 1, black -> 0."""
    arr = np.array(Image.open(path).convert("L"))
    return (arr > threshold).astype(np.uint8)


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int, List[int]]:
    """4-neighborhood connected components; returns labels, count, areas."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    areas: List[int] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or labels[y, x] != 0:
                continue
            label += 1
            q = deque([(y, x)])
            labels[y, x] = label
            area = 0
            while q:
                cy, cx = q.popleft()
                area += 1
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx] == 1
                        and labels[ny, nx] == 0
                    ):
                        labels[ny, nx] = label
                        q.append((ny, nx))
            areas.append(area)
    return labels, label, areas


def main() -> None:
    parser = argparse.ArgumentParser(description="Get area of a specific component label.")
    parser.add_argument("image", type=Path, help="Path to the image file")
    parser.add_argument("--label", type=int, required=True, help="1-based component label to query")
    parser.add_argument("--threshold", type=int, default=128, help="Binarization threshold (white > threshold)")
    args = parser.parse_args()

    mask = load_binary_mask(args.image, args.threshold)
    _, num_labels, areas = label_components(mask)

    if args.label < 1 or args.label > num_labels:
        raise SystemExit(f"Label {args.label} is out of range (1..{num_labels})")

    area = areas[args.label - 1]
    print(f"Label: {args.label}")
    print(f"Area: {area} pixels")
    print(f"Total labels: {num_labels}")


if __name__ == "__main__":
    main()

