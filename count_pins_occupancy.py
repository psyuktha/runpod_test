#!/usr/bin/env python3
"""
Detect components, classify to nearest package side, and report occupancy per side.

Defaults follow the current tuning used in count_pins.py:
- min_area: 200
- max_area: 500
- min_side: 3

Usage (from repo root):
  python count_pins_occupancy.py bi_full/bilateral_canny_anu2.jpeg --out-image bi_full/anu2_side_occupancy.png
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_binary_mask(path: Path, threshold: int = 128) -> np.ndarray:
    """Load image, convert to grayscale, binarize: white -> 1, black -> 0."""
    arr = np.array(Image.open(path).convert("L"))
    return (arr > threshold).astype(np.uint8)


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int, List[int], List[Tuple[int, int, int, int]]]:
    """4-neighborhood connected components; returns labels, count, areas, bboxes."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    areas: List[int] = []
    bboxes: List[Tuple[int, int, int, int]] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or labels[y, x] != 0:
                continue
            label += 1
            q = deque([(y, x)])
            labels[y, x] = label
            area = 0
            min_x = max_x = x
            min_y = max_y = y
            while q:
                cy, cx = q.popleft()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
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
            bboxes.append((min_x, min_y, max_x, max_y))
    return labels, label, areas, bboxes


def _side_for_pin(pin_bbox: Tuple[int, int, int, int], package_bbox: Tuple[int, int, int, int]) -> str:
    """Classify to the nearest package edge using center-to-edge distance."""
    px_min, py_min, px_max, py_max = package_bbox
    cx = (pin_bbox[0] + pin_bbox[2]) / 2
    cy = (pin_bbox[1] + pin_bbox[3]) / 2

    d_top = abs(cy - py_min)
    d_bottom = abs(py_max - cy)
    d_left = abs(cx - px_min)
    d_right = abs(px_max - cx)

    distances = {
        "top": d_top,
        "bottom": d_bottom,
        "left": d_left,
        "right": d_right,
    }
    return min(distances.items(), key=lambda kv: (kv[1], kv[0]))[0]


def analyze(
    path: Path,
    min_area: int = 200,
    max_area: int = 500,
    min_side: int = 3,
    threshold: int = 128,
) -> Tuple[
    Dict[str, int],
    Dict[str, int],
    Dict[str, List[int]],
    List[Tuple[int, int, int, int]],
    List[int],
    Tuple[int, int, int, int],
]:
    mask = load_binary_mask(path, threshold)
    _, num_labels, areas, bboxes = label_components(mask)

    # Drop largest two (outline and void)
    largest_indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)[:2]
    package_bbox = bboxes[largest_indices[0]] if largest_indices else (0, 0, 0, 0)
    package_area = areas[largest_indices[0]] if largest_indices else 0

    auto_max_area = max_area
    if package_area > 0:
        auto_max_area = min(max_area, int(package_area * 0.05))

    yellow_indices = [
        i
        for i, (bb, area) in enumerate(zip(bboxes, areas))
        if i not in largest_indices
        and (bb[2] - bb[0] + 1) >= min_side
        and (bb[3] - bb[1] + 1) >= min_side
        and min_area <= area <= auto_max_area
    ]
    yellow_labels = [i + 1 for i in yellow_indices]
    yellow_bboxes = [bboxes[i] for i in yellow_indices]

    side_counts: Dict[str, int] = {"top": 0, "bottom": 0, "left": 0, "right": 0, "unknown": 0}
    side_areas: Dict[str, int] = {"top": 0, "bottom": 0, "left": 0, "right": 0, "unknown": 0}
    side_labels: Dict[str, List[int]] = {k: [] for k in side_counts.keys()}

    for lbl, bb in zip(yellow_labels, yellow_bboxes):
        side = _side_for_pin(bb, package_bbox)
        side_counts[side] += 1
        side_areas[side] += areas[lbl - 1]
        side_labels[side].append(lbl)

    return side_counts, side_areas, side_labels, yellow_bboxes, yellow_labels, package_bbox


def draw_overlay(
    image_path: Path,
    out_path: Path,
    yellow_bboxes: List[Tuple[int, int, int, int]],
    yellow_labels: List[int],
    package_bbox: Tuple[int, int, int, int],
) -> None:
    base = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default()
    side_colors = {
        "top": (0, 255, 0),
        "bottom": (0, 128, 255),
        "left": (255, 165, 0),
        "right": (255, 0, 255),
        "unknown": (128, 128, 128),
    }

    # Compute a bounding box that encloses all left-side components (if any).
    left_boxes = []
    for (min_x, min_y, max_x, max_y) in yellow_bboxes:
        side = _side_for_pin((min_x, min_y, max_x, max_y), package_bbox)
        if side == "left":
            left_boxes.append((min_x, min_y, max_x, max_y))

    if left_boxes:
        lx_min = min(b[0] for b in left_boxes)
        ly_min = min(b[1] for b in left_boxes)
        lx_max = max(b[2] for b in left_boxes)
        ly_max = max(b[3] for b in left_boxes)
        draw.rectangle([(lx_min, ly_min), (lx_max, ly_max)], outline=(255, 215, 0), width=3)  # gold
        draw.text((lx_min, max(ly_min - 14, 0)), "LEFT BOUNDARY", fill=(255, 215, 0), font=font)

    for lbl, (min_x, min_y, max_x, max_y) in zip(yellow_labels, yellow_bboxes):
        side = _side_for_pin((min_x, min_y, max_x, max_y), package_bbox)
        color = side_colors.get(side, (255, 255, 0))
        draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=color, width=2)
        draw.text((min_x, max(min_y - 10, 0)), str(lbl), fill=color, font=font)
    base.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Side occupancy of filtered yellow boxes.")
    parser.add_argument("image", type=Path, help="Path to the image file")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum area for a component")
    parser.add_argument("--max-area", type=int, default=500, help="Maximum area for a component")
    parser.add_argument("--min-side", type=int, default=3, help="Minimum bbox side (pixels)")
    parser.add_argument("--threshold", type=int, default=128, help="Binarization threshold (white > threshold)")
    parser.add_argument("--out-image", type=Path, help="Optional path to save overlay image")
    args = parser.parse_args()

    side_counts, side_areas, side_labels, yellow_bboxes, yellow_labels, package_bbox = analyze(
        args.image, args.min_area, args.max_area, args.min_side, args.threshold
    )

    print(f"Per-side counts: {side_counts}")
    print(f"Per-side areas (pixels): {side_areas}")
    print(f"Per-side labels: {side_labels}")

    if args.out_image:
        draw_overlay(args.image, args.out_image, yellow_bboxes, yellow_labels, package_bbox)
        print(f"Saved overlay to: {args.out_image}")


if __name__ == "__main__":
    main()

