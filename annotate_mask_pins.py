"""
Annotate pins and mask the center of `debug/006_07_edges.png`.

Usage:
    python annotate_mask_pins.py \
        --input debug/006_07_edges.png \
        --output debug/006_07_edges_masked.png

The script:
- detects slender pin contours around the package outline,
- masks the package center to hide silkscreen text,
- labels pins clockwise starting from the top edge.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
Pin = Tuple[float, float, float]  # (x, y, area)


def find_pin_centers(img: np.ndarray, min_area: float, max_area: float = 5000.0) -> List[Pin]:
    """Return ordered pins (x, y, area), clockwise starting at top."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Emphasize edges and thin strokes.
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0
    min_radius = 0.35 * min(w, h)  # Exclude the center of the chip.

    candidates: List[Pin] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < min_area or area > max_area:
            continue

        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect < 1.5 or aspect > 10.0:
            continue

        px, py = x + bw / 2.0, y + bh / 2.0
        if math.hypot(px - cx, py - cy) < min_radius:
            continue

        candidates.append((px, py, float(area)))

    # Order pins clockwise starting from the top (12 o'clock).
    def angle_key(pt: Point) -> float:
        ang = math.degrees(math.atan2(pt[1] - cy, pt[0] - cx))
        return (ang - 90.0) % 360.0

    candidates.sort(key=angle_key)

    # De-duplicate nearby detections that come from double edges.
    unique: List[Pin] = []
    for pt in candidates:
        if all(math.hypot(pt[0] - up[0], pt[1] - up[1]) > 8.0 for up in unique):
            unique.append(pt)

    return unique


def mask_center(img: np.ndarray, ratio: float = 0.55) -> np.ndarray:
    """Return a copy of the image with the center masked out."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    half_w = int(w * ratio / 2)
    half_h = int(h * ratio / 2)
    masked = img.copy()
    top_left = (cx - half_w, cy - half_h)
    bottom_right = (cx + half_w, cy + half_h)
    cv2.rectangle(masked, top_left, bottom_right, (0, 0, 0), thickness=-1)
    return masked


def annotate(img: np.ndarray, pins: List[Pin]) -> np.ndarray:
    """Draw pin indices on the image."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    palette = [
        (0, 215, 255),   # Gold
        (255, 144, 30),  # Orange
        (255, 105, 180), # Pink
        (72, 249, 239),  # Cyan
    ]

    annotated = img.copy()
    for idx, (px, py, _) in enumerate(pins, start=1):
        vec = np.array([px - cx, py - cy], dtype=float)
        norm = np.linalg.norm(vec) or 1.0
        offset = vec / norm * 12.0
        label_pos = (int(px + offset[0] - 6), int(py + offset[1] + 4))

        color = palette[(idx - 1) % len(palette)]
        cv2.circle(annotated, (int(px), int(py)), 6, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(
            annotated,
            str(idx),
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            str(idx),
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return annotated


def count_pins_by_side(pins: List[Pin], cx: float, cy: float) -> dict:
    """Count pins per side using dominant axis (QFP style)."""
    counts = {"top": 0, "right": 0, "bottom": 0, "left": 0}
    for px, py, _ in pins:
        dx, dy = px - cx, py - cy
        if abs(dx) > abs(dy):
            side = "right" if dx > 0 else "left"
        else:
            side = "bottom" if dy > 0 else "top"
        counts[side] += 1
    return counts


def pin_side(px: float, py: float, cx: float, cy: float) -> str:
    """Return side label for a pin based on dominant axis."""
    dx, dy = px - cx, py - cy
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "bottom" if dy > 0 else "top"


def side_regularity(pins: List[Pin], side: str, cx: float, cy: float) -> Tuple[int, float]:
    """
    Return (count, regularity_score) for a side.
    Score increases with count and uniform spacing (low coefficient of variation).
    """
    axis_vals: List[float] = []
    for px, py, _ in pins:
        if pin_side(px, py, cx, cy) != side:
            continue
        axis_vals.append(px if side in ("top", "bottom") else py)

    axis_vals.sort()
    count = len(axis_vals)
    if count < 2:
        return count, 0.0

    diffs = np.diff(axis_vals)
    mean = float(np.mean(diffs))
    std = float(np.std(diffs))
    cv = std / (mean + 1e-6)
    score = count * (1.0 / (1.0 + cv))
    return count, score


def run(input_path: Path, output_path: Path, mask_ratio: float, min_area: float) -> None:
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {input_path}")

    pins = find_pin_centers(img, min_area=min_area)
    masked = mask_center(img, ratio=mask_ratio)
    result = annotate(masked, pins)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    h, w = img.shape[:2]
    counts = count_pins_by_side(pins, w / 2.0, h / 2.0)

    cx, cy = w / 2.0, h / 2.0
    sides = ["top", "right", "bottom", "left"]
    side_metrics = {}
    for side in sides:
        cnt, score = side_regularity(pins, side, cx, cy)
        side_metrics[side] = (cnt)
    print("______",side_metrics)
    # best_side = max(sides, key=lambda s: (side_metrics[s][1], side_metrics[s][0]))
    
    from collections import Counter
    side_counter = Counter(side_metrics.values())
    best_side = side_counter.most_common(1)[0][0]
    print("______",side_counter.most_common(1))
    estimated_total = best_side * 4
    print(f"Detected pins: {len(pins)}")
    print(
        f"Per side -> top: {counts['top']}, right: {counts['right']}, "
        f"bottom: {counts['bottom']}, left: {counts['left']}"
    )
    print("Regularity (count, score):")
    # for side in sides:
    #     cnt, score = side_metrics[side]
    #     print(f"  {side}: count={cnt}, score={score:.3f}")
    print(f"Best side: {best_side} -> estimated total pins = {estimated_total}")
    print("Pin labels (idx: x, y, side, area):")
    for idx, (px, py, area) in enumerate(pins, start=1):
        side = pin_side(px, py, w / 2.0, h / 2.0)
        print(f"  {idx}: {int(px)}, {int(py)}, {side}, area={area:.1f}")
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mask center and annotate pins.")
    parser.add_argument("--input", type=Path, default=Path("debug/006_07_edges.png"))
    parser.add_argument("--output", type=Path, default=Path("debug/006_07_edges_masked.png"))
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.55,
        help="Fraction of width/height to cover in the center (0-1).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=250.0,
        help="Minimum bounding-box area for a contour to count as a pin.",
    )
    args = parser.parse_args()

    if not (0.1 <= args.mask_ratio <= 0.9):
        raise SystemExit("--mask-ratio should be between 0.1 and 0.9")

    if args.min_area <= 0:
        raise SystemExit("--min-area must be positive")

    run(args.input, args.output, args.mask_ratio, args.min_area)


if __name__ == "__main__":
    main()
