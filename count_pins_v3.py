#!/usr/bin/env python3
"""
Pin counting with stricter pin-shape filtering:
- Drops the two largest components (outline + void).
- Keeps only components that (a) fit area limits, (b) have width/height near
  the median size of candidates, (c) have reasonable fill ratio, and (d) lie
  on one of the four sides.
- Chooses the most regular side (spacing + straightness) and applies symmetry:
  symmetric_estimate = best_side_count * sides_with_pins.

Overlay (if --out-image) draws only yellow boxes for the kept pins.

Usage (from repo root):
  python count_pins_v3.py bi_full/bilateral_canny_anu2.jpeg --min-area 3 --max-area 70 --out-image bi_full/anu2_boxes_v3.png
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def load_binary_mask(path: Path, threshold: int = 128) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr > threshold).astype(np.uint8)


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int, List[int], List[Tuple[int, int, int, int]]]:
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


def _side_for_pin(
    pin_bbox: Tuple[int, int, int, int],
    package_bbox: Tuple[int, int, int, int],
    margin_ratio: float = 0.2,
) -> str:
    px_min, py_min, px_max, py_max = package_bbox
    pw = px_max - px_min + 1
    ph = py_max - py_min + 1
    cx = (pin_bbox[0] + pin_bbox[2]) / 2
    cy = (pin_bbox[1] + pin_bbox[3]) / 2
    x0 = px_min + margin_ratio * pw
    x1 = px_max - margin_ratio * pw
    y0 = py_min + margin_ratio * ph
    y1 = py_max - margin_ratio * ph
    if cy <= y0:
        return "top"
    if cy >= y1:
        return "bottom"
    if cx <= x0:
        return "left"
    if cx >= x1:
        return "right"
    return "unknown"


def _regularity_score(side: str, centers: List[Tuple[float, float]], package_bbox: Tuple[int, int, int, int]) -> float:
    if len(centers) < 2:
        return float("inf")
    px_min, py_min, px_max, py_max = package_bbox
    pw = max(px_max - px_min, 1)
    ph = max(py_max - py_min, 1)
    xs = np.array([c[0] for c in centers], dtype=float)
    ys = np.array([c[1] for c in centers], dtype=float)
    if side in ("top", "bottom"):
        xs_sorted = np.sort(xs)
        spacings = np.diff(xs_sorted)
        spacing_cv = float(np.std(spacings) / np.mean(spacings)) if np.mean(spacings) > 0 else float("inf")
        line_std_norm = float(np.std(ys) / ph)
    else:
        ys_sorted = np.sort(ys)
        spacings = np.diff(ys_sorted)
        spacing_cv = float(np.std(spacings) / np.mean(spacings)) if np.mean(spacings) > 0 else float("inf")
        line_std_norm = float(np.std(xs) / pw)
    return spacing_cv + line_std_norm


def filter_pins(
    areas: List[int],
    bboxes: List[Tuple[int, int, int, int]],
    largest_indices: List[int],
    min_area: int,
    max_area: int,
    package_area: int,
) -> Tuple[List[int], List[Tuple[int, int, int, int]]]:
    # Auto max area capped to 5% of package.
    auto_max_area = max_area
    if package_area > 0:
        auto_max_area = min(max_area, int(package_area * 0.05))

    # Initial candidates by area.
    cand_labels = [
        idx
        for idx, a in enumerate(areas, start=1)
        if (idx - 1) not in largest_indices and min_area <= a <= auto_max_area
    ]
    cand_bboxes = [bboxes[idx - 1] for idx in cand_labels]

    if not cand_labels:
        return [], []

    # Size-based filtering: keep near median width/height.
    widths = np.array([bb[2] - bb[0] + 1 for bb in cand_bboxes], dtype=float)
    heights = np.array([bb[3] - bb[1] + 1 for bb in cand_bboxes], dtype=float)
    med_w = float(np.median(widths))
    med_h = float(np.median(heights))
    low_w, high_w = med_w * 0.5, med_w * 1.5
    low_h, high_h = med_h * 0.5, med_h * 1.5

    # Fill ratio filtering.
    fill_ratios = np.array(
        [areas[idx - 1] / ((widths[i]) * (heights[i])) for i, idx in enumerate(cand_labels)],
        dtype=float,
    )

    filtered_labels: List[int] = []
    filtered_bboxes: List[Tuple[int, int, int, int]] = []
    for i, (lbl, bb) in enumerate(zip(cand_labels, cand_bboxes)):
        w = widths[i]
        h = heights[i]
        fr = fill_ratios[i]
        if not (low_w <= w <= high_w and low_h <= h <= high_h):
            continue
        if not (0.01 <= fr <= 0.6):
            continue
        filtered_labels.append(lbl)
        filtered_bboxes.append(bb)

    # Fallback: if nothing survived, return area-only candidates.
    if not filtered_labels:
        return cand_labels, cand_bboxes
    return filtered_labels, filtered_bboxes


def analyze(
    path: Path, min_area: int = 3, max_area: int = 70
) -> Tuple[
    Dict[str, int],
    Dict[str, List[int]],
    Dict[str, float],
    str,
    int,
    List[Tuple[int, int, int, int]],
    List[Tuple[int, int, int, int]],
    Tuple[int, int, int, int],
]:
    mask = load_binary_mask(path)
    _, num_labels, areas, bboxes = label_components(mask)

    # Exclude largest (outline) and second-largest (void).
    largest_indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)[:2]
    package_bbox = bboxes[largest_indices[0]] if largest_indices else (0, 0, 0, 0)
    package_area = areas[largest_indices[0]] if largest_indices else 0

    pin_labels, pin_bboxes = filter_pins(areas, bboxes, largest_indices, min_area, max_area, package_area)

    side_counts: Dict[str, int] = {"top": 0, "bottom": 0, "left": 0, "right": 0, "unknown": 0}
    side_labels: Dict[str, List[int]] = {"top": [], "bottom": [], "left": [], "right": [], "unknown": []}
    side_centers: Dict[str, List[Tuple[float, float]]] = {k: [] for k in side_counts.keys()}
    for lbl, bb in zip(pin_labels, pin_bboxes):
        side = _side_for_pin(bb, package_bbox)
        cx = (bb[0] + bb[2]) / 2
        cy = (bb[1] + bb[3]) / 2
        side_counts[side] += 1
        side_labels[side].append(lbl)
        side_centers[side].append((cx, cy))

    side_scores: Dict[str, float] = {}
    for s in ("top", "bottom", "left", "right"):
        side_scores[s] = _regularity_score(s, side_centers[s], package_bbox)

    sides_present = [s for s in ("top", "bottom", "left", "right") if side_counts[s] > 0]
    best_side = None
    if sides_present:
        best_side = min(
            sides_present,
            key=lambda s: (side_scores[s], -side_counts[s], s),
        )
    best_side_count = side_counts.get(best_side, 0) if best_side else 0
    symmetric_estimate = best_side_count * len(sides_present) if sides_present else 0

    return side_counts, side_labels, side_scores, best_side or "unknown", symmetric_estimate, pin_bboxes, package_bbox


def draw_overlay(
    image_path: Path,
    out_path: Path,
    pin_bboxes: List[Tuple[int, int, int, int]],
) -> None:
    base = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    for (min_x, min_y, max_x, max_y) in pin_bboxes:
        draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=(255, 255, 0), width=2)
    base.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pin count with shape-consistency filtering.")
    parser.add_argument("image", type=Path, help="Path to the image file")
    parser.add_argument("--min-area", type=int, default=3, help="Minimum area for a pin component")
    parser.add_argument("--max-area", type=int, default=70, help="Maximum area for a pin component")
    parser.add_argument("--out-image", type=Path, help="Optional path to save overlay image (yellow boxes for kept pins)")
    args = parser.parse_args()

    (
        side_counts,
        side_labels,
        side_scores,
        best_side,
        symmetric_estimate,
        pin_bboxes,
        package_bbox,
    ) = analyze(args.image, args.min_area, args.max_area)

    print(f"Per-side pin counts (filtered): {side_counts}")
    print(f"Per-side labels: {side_labels}")
    print(f"Per-side regularity scores (lower=more regular): {side_scores}")
    print(f"Best side by regularity: {best_side} with count {side_counts.get(best_side, 0)}")
    print(f"Symmetric pin estimate (best side count * sides with pins): {symmetric_estimate}")
    if args.out_image:
        draw_overlay(args.image, args.out_image, pin_bboxes)
        print(f"Saved overlay to {args.out_image}")


if __name__ == "__main__":
    main()

