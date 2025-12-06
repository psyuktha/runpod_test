"""
Annotate pins on square/anu5_07_edges.png, mask center, and count pins.

Run:
  python annotate_anu5_square.py \
    --input square/anu5_07_edges.png \
    --output debug/anu5_07_edges_masked.png \
    --mask-ratio 0.55 \
    --min-area 250
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterable

import cv2
import numpy as np

@dataclass(frozen=True)
class Pin:
    cx: float
    cy: float
    w: float
    h: float
    area: float


def _chip_bbox(binary: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    points = cv2.findNonZero(binary)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return x, y, w, h


def _dedupe(pins: Iterable[Pin], min_dist: float = 8.0) -> List[Pin]:
    unique: List[Pin] = []
    for pt in pins:
        if all(math.hypot(pt.cx - up.cx, pt.cy - up.cy) > min_dist for up in unique):
            unique.append(pt)
    return unique


def find_pin_centers(img: np.ndarray, min_area: float, max_area: float = 6000.0) -> List[Pin]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Keep only the top and bottom bands; drop center text/noise.
    h, w = gray.shape
    band_mask = np.zeros_like(binary)
    top_band_h = int(h * 0.28)
    bot_band_start = int(h * 0.72)
    band_mask[:top_band_h, :] = 255
    band_mask[bot_band_start:, :] = 255
    binary = cv2.bitwise_and(binary, band_mask)

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)

    chip_box = _chip_bbox(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx, cy = w / 2.0, h / 2.0
    min_radius = 0.25 * min(w, h)
    x_band_min, x_band_max = 0.0, float(w)
    if chip_box:
        bx, by, bw, bh = chip_box
        pad_x = 0.05 * bw
        x_band_min = max(0.0, bx - pad_x)
        x_band_max = min(float(w), bx + bw + pad_x)

    candidates: List[Pin] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect < 0.6 or aspect > 3.0:
            continue
        if bh < 8:
            continue
        px, py = x + bw / 2.0, y + bh / 2.0
        if math.hypot(px - cx, py - cy) < min_radius:
            continue
        if px < x_band_min or px > x_band_max:
            continue
        candidates.append(Pin(px, py, float(bw), float(bh), float(area)))

    def angle_key(pt: Pin) -> float:
        ang = math.degrees(math.atan2(pt.cy - cy, pt.cx - cx))
        return (ang - 90.0) % 360.0

    deduped = _dedupe(sorted(candidates, key=angle_key), min_dist=12.0)
    return deduped


def mask_center(img: np.ndarray, ratio: float) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    half_w = int(w * ratio / 2)
    half_h = int(h * ratio / 2)
    masked = img.copy()
    cv2.rectangle(masked, (cx - half_w, cy - half_h), (cx + half_w, cy + half_h), (0, 0, 0), -1)
    return masked


def pin_side(px: float, py: float, cx: float, cy: float) -> str:
    dx, dy = px - cx, py - cy
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "bottom" if dy > 0 else "top"


def count_pins_by_side(pins: List[Pin], cx: float, cy: float) -> Dict[str, int]:
    counts = {"top": 0, "right": 0, "bottom": 0, "left": 0}
    for pin in pins:
        counts[pin_side(pin.cx, pin.cy, cx, cy)] += 1
    return counts


def side_regularity(pins: List[Pin], side: str, cx: float, cy: float) -> Tuple[int, float]:
    vals: List[float] = []
    for pin in pins:
        if pin_side(pin.cx, pin.cy, cx, cy) != side:
            continue
        vals.append(pin.cx if side in ("top", "bottom") else pin.cy)
    vals.sort()
    count = len(vals)
    if count < 2:
        return count, 0.0
    diffs = np.diff(vals)
    mean = float(np.mean(diffs))
    std = float(np.std(diffs))
    cv = std / (mean + 1e-6)
    score = count * (1.0 / (1.0 + cv))
    return count, score


def annotate(img: np.ndarray, pins: List[Pin]) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    palette = [(0, 215, 255), (255, 144, 30), (255, 105, 180), (72, 249, 239)]
    box_color = (255, 0, 255)

    out = img.copy()
    def side_key(pin: Pin) -> Tuple[int, float]:
        side = pin_side(pin.cx, pin.cy, cx, cy)
        order = {"top": 0, "right": 1, "bottom": 2, "left": 3}[side]
        along = pin.cx if side in ("top", "bottom") else pin.cy
        return order, along

    sorted_pins = sorted(pins, key=side_key)

    for idx, pin in enumerate(sorted_pins, start=1):
        px, py = pin.cx, pin.cy
        vec = np.array([px - cx, py - cy], dtype=float)
        norm = np.linalg.norm(vec) or 1.0
        offset = vec / norm * 12.0
        label_pos = (int(px + offset[0] - 6), int(py + offset[1] + 4))
        color = palette[(idx - 1) % len(palette)]
        x1 = int(px - pin.w / 2.0) - 2
        y1 = int(py - pin.h / 2.0) - 2
        x2 = int(px + pin.w / 2.0) + 2
        y2 = int(py + pin.h / 2.0) + 2
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
        cv2.circle(out, (int(px), int(py)), 5, color, 2, cv2.LINE_AA)
        cv2.putText(out, str(idx), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(idx), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def run(input_path: Path, output_path: Path, mask_ratio: float, min_area: float) -> None:
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {input_path}")

    masked = mask_center(img, mask_ratio)
    pins = find_pin_centers(masked, min_area=min_area)
    result = annotate(masked, pins)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)

    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    counts = count_pins_by_side(pins, cx, cy)
    sides = ["top", "right", "bottom", "left"]
    metrics = {s: side_regularity(pins, s, cx, cy) for s in sides}
    best = max(sides, key=lambda s: (metrics[s][1], metrics[s][0]))
    estimated_total = metrics[best][0] * 4

    print(f"Detected pins: {len(pins)}")
    print(f"Per side -> top: {counts['top']}, right: {counts['right']}, bottom: {counts['bottom']}, left: {counts['left']}")
    print("Regularity (count, score):")
    for s in sides:
        c, sc = metrics[s]
        print(f"  {s}: count={c}, score={sc:.3f}")
    print(f"Best side: {best} -> estimated total pins = {estimated_total}")
    print("Pin labels (idx: x, y, side, area):")
    for idx, pin in enumerate(pins, start=1):
        side = pin_side(pin.cx, pin.cy, cx, cy)
        print(f"  {idx}: {int(pin.cx)}, {int(pin.cy)}, {side}, area={pin.area:.1f}, box=({pin.w}x{pin.h})")
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate and count pins for anu5 square edge image.")
    parser.add_argument("--input", type=Path, default=Path("square/anu5_07_edges.png"))
    parser.add_argument("--output", type=Path, default=Path("debug/anu5_07_edges_masked.png"))
    parser.add_argument("--mask-ratio", type=float, default=0.55, help="Fraction of width/height to mask in center.")
    parser.add_argument("--min-area", type=float, default=80.0, help="Minimum bounding-box area to accept a pin.")
    args = parser.parse_args()

    if not (0.1 <= args.mask_ratio <= 0.9):
        raise SystemExit("--mask-ratio should be between 0.1 and 0.9")
    if args.min_area <= 0:
        raise SystemExit("--min-area must be positive")

    run(args.input, args.output, args.mask_ratio, args.min_area)


if __name__ == "__main__":
    main()
