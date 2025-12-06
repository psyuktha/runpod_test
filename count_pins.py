#!/usr/bin/env python3
"""
Count pin-like shapes in a monochrome outline image.

The script loads an image, binarizes it (white > 128), runs a simple
4-neighborhood connected-component labeling, and counts components whose
areas fall within a configurable range (defaults tuned for the provided
outline image).

Usage:
    python count_pins.py /path/to/image.jpeg [--min-area 40] [--max-area 4000]
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _regularity_score(side: str, centers: List[Tuple[float, float]], package_bbox: Tuple[int, int, int, int]) -> float:
    """Lower is better: combines spacing uniformity and straightness along a side."""
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


def load_binary_mask(path: Path, threshold: int = 128) -> np.ndarray:
    """Load image, convert to grayscale, and binarize (white -> 1, black -> 0)."""
    img = Image.open(path).convert("L")
    arr = np.array(img)
    mask = (arr > threshold).astype(np.uint8)
    return mask


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int, List[int], List[Tuple[int, int, int, int]]]:
    """Label connected components (4-neighborhood) and collect bounding boxes."""
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
                # update bounds
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
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


def _select_package_bbox(areas: List[int], bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Pick the largest-area component as the package outline bbox."""
    if not areas:
        return (0, 0, 0, 0)
    largest_idx = int(np.argmax(areas))
    return bboxes[largest_idx]


def _side_for_pin(
    pin_bbox: Tuple[int, int, int, int],
    package_bbox: Tuple[int, int, int, int],
) -> str:
    """
    Classify to an edge using bbox center vs an inset package box:
    - If center is beyond inset band on an axis, pick that side.
    - Otherwise fall back to nearest edge distance.
    """
    px_min, py_min, px_max, py_max = package_bbox
    pw = px_max - px_min + 1
    ph = py_max - py_min + 1
    cx = (pin_bbox[0] + pin_bbox[2]) / 2
    cy = (pin_bbox[1] + pin_bbox[3]) / 2

    margin_x = 0.05 * pw
    margin_y = 0.05 * ph
    if cx <= px_min + margin_x:
        return "left"
    if cx >= px_max - margin_x:
        return "right"
    if cy <= py_min + margin_y:
        return "top"
    if cy >= py_max - margin_y:
        return "bottom"

    d_top = abs(cy - py_min)
    d_bottom = abs(py_max - cy)
    d_left = abs(cx - px_min)
    d_right = abs(px_max - cx)
    distances = {"top": d_top, "bottom": d_bottom, "left": d_left, "right": d_right}
    return min(distances.items(), key=lambda kv: (kv[1], kv[0]))[0]


def count_pins(
    path: Path, min_area: int = 200, max_area: int = 500, min_side: int = 3
) -> Tuple[
    int,
    List[int],
    List[int],
    List[int],
    List[Tuple[int, int, int, int]],
    List[Tuple[int, int, int, int]],
    Tuple[int, int, int, int],
    Dict[str, int],
    int,
    Dict[str, List[int]],
    Dict[str, int],
    Dict[str, List[int]],
    int,
    List[Tuple[int, int, int, int]],
    Dict[str, float],
    str,
    List[int],
]:
    """Return pin count data plus per-side counts and symmetric estimate."""
    mask = load_binary_mask(path)
    _, num_labels, areas, bboxes = label_components(mask)

    package_bbox = _select_package_bbox(areas, bboxes)

    # Treat largest (outline) and second-largest (center void) as non-pins.
    largest_indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)[:2]
    package_bbox = bboxes[largest_indices[0]] if largest_indices else (0, 0, 0, 0)
    package_area = areas[largest_indices[0]] if largest_indices else 0

    # Auto cap pin size: pins should be much smaller than the package.
    auto_max_area = max_area
    if package_area > 0:
        auto_max_area = min(max_area, int(package_area * 0.05))

    # Pins = small components (small yellow boxes), excluding the largest pieces.
    pin_labels = []
    pin_areas = []
    pin_bboxes = []
    for idx, a in enumerate(areas, start=1):
        if (idx - 1) in largest_indices:
            continue
        if not (min_area <= a <= auto_max_area):
            continue
        bb = bboxes[idx - 1]
        w = bb[2] - bb[0] + 1
        h = bb[3] - bb[1] + 1
        if w < min_side or h < min_side:
            continue
        pin_labels.append(idx)
        pin_areas.append(a)
        pin_bboxes.append(bb)

    # Count pins per side (only boundary pins considered).
    side_counts: Dict[str, int] = {"top": 0, "bottom": 0, "left": 0, "right": 0, "unknown": 0}
    side_labels: Dict[str, List[int]] = {"top": [], "bottom": [], "left": [], "right": [], "unknown": []}
    for lbl, pb in zip(pin_labels, pin_bboxes):
        if areas[lbl - 1] <= 200:
            continue
        side = _side_for_pin(pb, package_bbox)
        side_counts[side] = side_counts.get(side, 0) + 1
        side_labels.setdefault(side, []).append(lbl)
        print(f"Side {side} count: {side_counts[side]}")

    # Count ALL yellow boxes (all components except largest/center) per side.
    yellow_indices = [
        i
        for i, bb in enumerate(bboxes)
        if i not in largest_indices
        and (bb[2] - bb[0] + 1) >= min_side
        and (bb[3] - bb[1] + 1) >= min_side
    ]
    yellow_labels = [i + 1 for i in yellow_indices]
    yellow_bboxes = [bboxes[i] for i in yellow_indices]
    yellow_side_counts: Dict[str, int] = {"top": 0, "bottom": 0, "left": 0, "right": 0, "unknown": 0}
    yellow_side_labels: Dict[str, List[int]] = {"top": [], "bottom": [], "left": [], "right": [], "unknown": []}
    yellow_side_centers: Dict[str, List[Tuple[float, float]]] = {k: [] for k in yellow_side_counts.keys()}
    for lbl, bb in zip(yellow_labels, yellow_bboxes):
        if areas[lbl - 1] <= 200:
            continue
        side = _side_for_pin(bb, package_bbox)
        yellow_side_counts[side] = yellow_side_counts.get(side, 0) + 1
        yellow_side_labels.setdefault(side, []).append(lbl)
        cx = (bb[0] + bb[2]) / 2
        cy = (bb[1] + bb[3]) / 2
        yellow_side_centers[side].append((cx, cy))
    yellow_sides_with = sum(1 for s in ("top", "bottom", "left", "right") if yellow_side_counts[s] > 0)
    yellow_side_scores: Dict[str, float] = {s: _regularity_score(s, yellow_side_centers[s], package_bbox) for s in ("top", "bottom", "left", "right")}
    best_yellow_side = (
        min(
            [s for s in ("top", "bottom", "left", "right") if yellow_side_counts[s] > 0],
            key=lambda s: (yellow_side_scores[s], -yellow_side_counts[s], s),
        )
        if yellow_sides_with
        else None
    )
    yellow_symmetric_count = (
        yellow_side_counts.get(best_yellow_side, 0) * yellow_sides_with if best_yellow_side else 0
    )
    # Use the regularity-chosen yellow side for symmetric estimate.
    symmetric_pin_count = yellow_symmetric_count

    return (
        num_labels,
        areas,
        pin_areas,
        pin_labels,
        bboxes,
        pin_bboxes,
        package_bbox,
        side_counts,
        symmetric_pin_count,
        side_labels,
        yellow_side_counts,
        yellow_side_labels,
        yellow_symmetric_count,
        yellow_bboxes,
        yellow_side_scores,
        best_yellow_side or "unknown",
        yellow_labels,
        yellow_side_centers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Count pin-like shapes in an outline image.")
    parser.add_argument("image", type=Path, help="Path to the image file")
    # parser.add_argument("--min-area", type=int, default=3, help="Minimum area for a pin component")
    # parser.add_argument("--max-area", type=int, default=70, help="Maximum area for a pin component")
    parser.add_argument("--out-image", type=Path, help="Optional path to save image with bounding boxes drawn")
    args = parser.parse_args()

    (
        num_labels,
        areas,
        pin_areas,
        pin_labels,
        bboxes,
        pin_bboxes,
        package_bbox,
        side_counts,
        symmetric_pin_count,
        side_labels,
        yellow_side_counts,
        yellow_side_labels,
        yellow_symmetric_count,
        yellow_bboxes,
        yellow_side_scores,
        best_yellow_side,
        yellow_labels,
        yellow_side_centers,
    ) = count_pins(args.image)

    print(f"All yellow boxes per side: top={yellow_side_counts['top']}, bottom={yellow_side_counts['bottom']}, left={yellow_side_counts['left']}, right={yellow_side_counts['right']}, unknown={yellow_side_counts['unknown']}")
    print(f"All yellow boxes labels: top={yellow_side_labels['top']}, bottom={yellow_side_labels['bottom']}, left={yellow_side_labels['left']}, right={yellow_side_labels['right']}, unknown={yellow_side_labels['unknown']}")
    print(f"Regularity scores (yellow boxes): {yellow_side_scores}")
    print(f"Chosen side by regularity (yellow boxes): {best_yellow_side}")
    print(f"Symmetric estimate from yellow boxes (regularity-chosen side): {yellow_symmetric_count}")
    for side in ("top", "bottom", "left", "right", "unknown"):
        print(f"Side {side} count: {yellow_side_counts[side]}")
    if pin_labels:
        print(f"Pin candidate labels: {pin_labels}")
    else:
        print("Pin candidate labels: []")
    if pin_areas:
        print(f"Pin area stats -> min: {min(pin_areas)}, max: {max(pin_areas)}, count: {len(pin_areas)}")
    else:
        print("No pin candidates detected; adjust area thresholds.")
    # Spacing diagnostics: list spacing ranges per side.
    for s in ("top", "bottom", "left", "right"):
        centers = yellow_side_centers.get(s, [])
        if len(centers) < 2:
            print(f"{s} spacings: n<{2}")
            continue
        if s in ("top", "bottom"):
            coords = sorted(c[0] for c in centers)
        else:
            coords = sorted(c[1] for c in centers)
        spacings = [coords[i + 1] - coords[i] for i in range(len(coords) - 1)]
        print(f"{s} spacings -> min:{min(spacings):.2f} max:{max(spacings):.2f} mean:{(sum(spacings)/len(spacings)):.2f}")

    # Draw bounding boxes if requested.
    if args.out_image:
        # Load original in RGB for drawing
        base = Image.open(args.image).convert("RGB")
        draw = ImageDraw.Draw(base)
        font = ImageFont.load_default()
        # Draw all filtered yellow boxes and label with their component id.
        for lbl, (min_x, min_y, max_x, max_y) in zip(yellow_labels, yellow_bboxes):
            draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=(255, 255, 0), width=2)
            draw.text((min_x, max(min_y - 10, 0)), str(lbl), fill=(255, 255, 0), font=font)
        base.save(args.out_image)
        print(f"Saved bounding-box image to: {args.out_image}")


if __name__ == "__main__":
    main()

