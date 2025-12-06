#!/usr/bin/env python3
"""
IC Pin Detection with Center Masking and Symmetric Counting.

This script:
1. Masks the center region of the IC to exclude logo/text
2. Detects pins using connected component analysis
3. Divides pins into left, right, top, bottom regions
4. Counts pins per region with proper thresholds
5. Uses symmetry to estimate total pin count
"""

import argparse
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_and_mask_center(path: Path, threshold: int = 128, center_mask_ratio: float = 0.4) -> np.ndarray:
    """
    Load image, convert to grayscale, binarize, and mask the center region.

    Args:
        path: Path to input image
        threshold: Binarization threshold (white > threshold = 1)
        center_mask_ratio: Ratio of image to mask in center (0.4 = 40% of width/height)

    Returns:
        Binary mask with center region zeroed out
    """
    img = Image.open(path).convert("L")
    arr = np.array(img)
    mask = (arr > threshold).astype(np.uint8)

    # Mask the center region to exclude logo/text
    h, w = mask.shape
    center_x, center_y = w // 2, h // 2
    mask_w = int(w * center_mask_ratio)
    mask_h = int(h * center_mask_ratio)

    y_start = max(0, center_y - mask_h // 2)
    y_end = min(h, center_y + mask_h // 2)
    x_start = max(0, center_x - mask_w // 2)
    x_end = min(w, center_x + mask_w // 2)

    mask[y_start:y_end, x_start:x_end] = 0

    print(f"Image size: {w}x{h}")
    print(f"Masked center region: ({x_start},{y_start}) to ({x_end},{y_end})")

    return mask


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int, List[int], List[Tuple[int, int, int, int]]]:
    """
    Label connected components using 4-neighborhood connectivity.

    Returns:
        labels: Array with component labels
        num_labels: Total number of components found
        areas: List of component areas
        bboxes: List of bounding boxes (min_x, min_y, max_x, max_y)
    """
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
                    if (0 <= ny < h and 0 <= nx < w and
                        mask[ny, nx] == 1 and labels[ny, nx] == 0):
                        labels[ny, nx] = label
                        q.append((ny, nx))

            areas.append(area)
            bboxes.append((min_x, min_y, max_x, max_y))

    return labels, label, areas, bboxes


def get_package_bbox(areas: List[int], bboxes: List[Tuple[int, int, int, int]],
                     mask_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Find the IC package bounding box by computing the overall extent of all components.
    This is more robust than just taking the largest component.
    """
    if not areas:
        return (0, 0, 0, 0)

    # Find components that are likely part of the IC (not tiny noise)
    # Use top 80% of components by area
    min_component_area = 100
    valid_bboxes = [bbox for area, bbox in zip(areas, bboxes) if area >= min_component_area]

    if not valid_bboxes:
        # Fallback to image dimensions
        h, w = mask_shape
        margin = int(0.05 * min(w, h))
        return (margin, margin, w - margin, h - margin)

    # Compute bounding box of all valid components
    all_x_min = min(bbox[0] for bbox in valid_bboxes)
    all_y_min = min(bbox[1] for bbox in valid_bboxes)
    all_x_max = max(bbox[2] for bbox in valid_bboxes)
    all_y_max = max(bbox[3] for bbox in valid_bboxes)

    # Add small margin
    margin = 5
    return (all_x_min - margin, all_y_min - margin,
            all_x_max + margin, all_y_max + margin)


def classify_pin_side(pin_bbox: Tuple[int, int, int, int],
                      package_bbox: Tuple[int, int, int, int],
                      margin_ratio: float = 0.25) -> str:
    """
    Classify pin to a side (top, bottom, left, right) based on its center position.

    Uses margin-based classification: if pin center is within margin_ratio of an edge,
    it belongs to that edge.
    """
    px_min, py_min, px_max, py_max = package_bbox
    pw = px_max - px_min + 1
    ph = py_max - py_min + 1

    # Pin center
    cx = (pin_bbox[0] + pin_bbox[2]) / 2
    cy = (pin_bbox[1] + pin_bbox[3]) / 2

    margin_x = margin_ratio * pw
    margin_y = margin_ratio * ph

    # Classify based on which margin zone the pin falls into
    # Prioritize horizontal (left/right) for corners
    in_left = cx <= px_min + margin_x
    in_right = cx >= px_max - margin_x
    in_top = cy <= py_min + margin_y
    in_bottom = cy >= py_max - margin_y

    # Check if pin is in margin zones
    if in_left and not (in_top or in_bottom):
        return "left"
    if in_right and not (in_top or in_bottom):
        return "right"
    if in_top:
        return "top"
    if in_bottom:
        return "bottom"
    if in_left:
        return "left"
    if in_right:
        return "right"

    # Fallback to nearest edge
    d_top = abs(cy - py_min)
    d_bottom = abs(py_max - cy)
    d_left = abs(cx - px_min)
    d_right = abs(px_max - cx)
    distances = {"top": d_top, "bottom": d_bottom, "left": d_left, "right": d_right}

    return min(distances.items(), key=lambda kv: kv[1])[0]


def calculate_spacing_uniformity(centers: List[Tuple[float, float]], side: str) -> float:
    """
    Calculate spacing uniformity score for pins on a side.
    Lower score = more uniform spacing.

    Returns coefficient of variation of spacings.
    """
    if len(centers) < 2:
        return float("inf")

    # Extract relevant coordinate (x for top/bottom, y for left/right)
    if side in ("top", "bottom"):
        coords = sorted([c[0] for c in centers])
    else:
        coords = sorted([c[1] for c in centers])

    # Calculate spacings between consecutive pins
    spacings = np.diff(coords)

    if len(spacings) == 0 or np.mean(spacings) == 0:
        return float("inf")

    # Coefficient of variation: std/mean
    cv = float(np.std(spacings) / np.mean(spacings))

    return cv


def is_near_edge(pin_bbox: Tuple[int, int, int, int],
                 package_bbox: Tuple[int, int, int, int],
                 edge_threshold: float = 0.20) -> bool:
    """
    Check if a pin is near any edge of the package.
    Returns True if pin is within edge_threshold of any package edge.
    """
    px_min, py_min, px_max, py_max = package_bbox
    pw = px_max - px_min + 1
    ph = py_max - py_min + 1

    cx = (pin_bbox[0] + pin_bbox[2]) / 2
    cy = (pin_bbox[1] + pin_bbox[3]) / 2

    dist_to_left = abs(cx - px_min) / pw
    dist_to_right = abs(cx - px_max) / pw
    dist_to_top = abs(cy - py_min) / ph
    dist_to_bottom = abs(cy - py_max) / ph

    min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)

    return min_dist <= edge_threshold


def detect_pins_with_regions(
    path: Path,
    min_area: int = 150,
    max_area: int = 600,
    min_side: int = 3,
    center_mask_ratio: float = 0.0,
    threshold: int = 128,
) -> Dict:
    """
    Detect IC pins, classify by region, and calculate symmetric count.

    Returns dictionary with:
        - pin_count: Total detected pins
        - side_counts: Dict with counts per side
        - side_centers: Dict with pin centers per side
        - side_bboxes: Dict with bboxes per side
        - symmetric_count: Estimated total using best side
        - best_side: Side with most uniform spacing
        - package_bbox: IC package bounding box
        - all_bboxes: All pin bounding boxes
    """
    # Load image without center masking (we'll filter by edge proximity instead)
    img = Image.open(path).convert("L")
    arr = np.array(img)
    mask = (arr > threshold).astype(np.uint8)

    print(f"Image size: {mask.shape[1]}x{mask.shape[0]}")

    # Find connected components
    labels, num_labels, areas, bboxes = label_components(mask)
    print(f"Found {num_labels} components")

    # Get package outline (overall extent of all components)
    package_bbox = get_package_bbox(areas, bboxes, mask.shape)
    print(f"Package bbox: {package_bbox}")

    # Exclude largest components - package outline and center void
    # Sort by area and exclude top 5% as these are likely package/text, not pins
    sorted_by_area = sorted(enumerate(areas), key=lambda x: x[1], reverse=True)
    num_to_exclude = max(3, int(len(areas) * 0.05))  # Exclude top 5% or at least 3
    largest_indices = set(idx for idx, _ in sorted_by_area[:num_to_exclude])

    print(f"Excluding {len(largest_indices)} largest components as non-pins")
    for idx, area in sorted_by_area[:num_to_exclude]:
        print(f"  Excluded: idx={idx}, area={area}, bbox={bboxes[idx]}")

    # Filter for pin-sized components near edges
    pin_data = []
    filtered_counts = {"size": 0, "edge": 0, "excluded": 0}

    for idx, (area, bbox) in enumerate(zip(areas, bboxes)):
        if idx in largest_indices:
            filtered_counts["excluded"] += 1
            continue

        w = bbox[2] - bbox[0] + 1
        h = bbox[3] - bbox[1] + 1

        # Filter by area and minimum dimensions
        if not (min_area <= area <= max_area and w >= min_side and h >= min_side):
            filtered_counts["size"] += 1
            continue

        # Additional filter: pin must be near an edge
        if not is_near_edge(bbox, package_bbox, edge_threshold=0.22):
            filtered_counts["edge"] += 1
            continue

        pin_data.append({
            'idx': idx,
            'area': area,
            'bbox': bbox,
            'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        })

    print(f"Filtered out: {filtered_counts['excluded']} (excluded), {filtered_counts['size']} (size), {filtered_counts['edge']} (edge)")

    print(f"Detected {len(pin_data)} pin candidates")

    # Classify pins by side
    side_counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    side_centers = {"top": [], "bottom": [], "left": [], "right": []}
    side_bboxes = {"top": [], "bottom": [], "left": [], "right": []}

    for pin in pin_data:
        side = classify_pin_side(pin['bbox'], package_bbox)
        side_counts[side] += 1
        side_centers[side].append(pin['center'])
        side_bboxes[side].append(pin['bbox'])

    print(f"\nPin counts per side:")
    for side in ["top", "bottom", "left", "right"]:
        print(f"  {side.capitalize()}: {side_counts[side]}")

    # Calculate spacing uniformity for each side
    spacing_scores = {}
    for side in ["top", "bottom", "left", "right"]:
        if side_counts[side] > 0:
            score = calculate_spacing_uniformity(side_centers[side], side)
            spacing_scores[side] = score
            print(f"  {side.capitalize()} spacing uniformity score: {score:.4f}")

    # Find best side (most uniform spacing)
    valid_sides = [s for s in ["top", "bottom", "left", "right"] if side_counts[s] > 0]
    if valid_sides:
        best_side = min(valid_sides, key=lambda s: spacing_scores[s])
        num_sides_with_pins = len(valid_sides)
        symmetric_count = side_counts[best_side] * num_sides_with_pins

        print(f"\nBest side (most uniform): {best_side}")
        print(f"Pins on best side: {side_counts[best_side]}")
        print(f"Number of sides with pins: {num_sides_with_pins}")
        print(f"Symmetric pin count estimate: {symmetric_count}")
    else:
        best_side = None
        symmetric_count = 0
        print("\nNo valid sides detected")

    return {
        'pin_count': len(pin_data),
        'side_counts': side_counts,
        'side_centers': side_centers,
        'side_bboxes': side_bboxes,
        'symmetric_count': symmetric_count,
        'best_side': best_side,
        'package_bbox': package_bbox,
        'all_bboxes': [p['bbox'] for p in pin_data],
        'spacing_scores': spacing_scores
    }


def draw_results(image_path: Path, results: Dict, output_path: Path):
    """Draw bounding boxes on image and save result."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw package outline in cyan
    px_min, py_min, px_max, py_max = results['package_bbox']
    draw.rectangle([(px_min, py_min), (px_max, py_max)], outline=(0, 255, 255), width=4)

    # Color map for sides
    side_colors = {
        "top": (255, 0, 0),      # Red
        "bottom": (0, 255, 0),    # Green
        "left": (255, 255, 0),    # Yellow
        "right": (255, 0, 255)    # Magenta
    }

    # Draw pins by side
    for side in ["top", "bottom", "left", "right"]:
        color = side_colors[side]
        for idx, bbox in enumerate(results['side_bboxes'][side]):
            min_x, min_y, max_x, max_y = bbox
            draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=color, width=3)
            draw.text((min_x, max(min_y - 12, 0)), f"{side[0]}{idx+1}", fill=color, font=ImageFont.load_default())

    # Add legend text
    draw.text((10, 10), f"Total pins: {results['pin_count']}", fill=(255, 255, 255))
    draw.text((10, 30), f"Symmetric estimate: {results['symmetric_count']}", fill=(255, 255, 255))
    draw.text((10, 50), f"Best side: {results['best_side']}", fill=(255, 255, 255))

    y_offset = 80
    for side in ["top", "bottom", "left", "right"]:
        color = side_colors[side]
        text = f"{side.capitalize()}: {results['side_counts'][side]}"
        draw.text((10, y_offset), text, fill=color)
        y_offset += 20

    img.save(output_path)
    print(f"\nSaved annotated image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect IC pins with edge-based filtering and symmetric counting")
    parser.add_argument("image", type=Path, help="Path to IC image")
    parser.add_argument("--min-area", type=int, default=120, help="Minimum pin area (default: 120)")
    parser.add_argument("--max-area", type=int, default=800, help="Maximum pin area (default: 800)")
    parser.add_argument("--min-side", type=int, default=3, help="Minimum bbox side length (default: 3)")
    parser.add_argument("--threshold", type=int, default=128, help="Binarization threshold (default: 128)")
    parser.add_argument("--output", type=Path, help="Output image path with bounding boxes")

    args = parser.parse_args()

    # Detect pins
    results = detect_pins_with_regions(
        args.image,
        min_area=args.min_area,
        max_area=args.max_area,
        min_side=args.min_side,
        threshold=args.threshold
    )

    # Draw results if output specified
    if args.output:
        draw_results(args.image, results, args.output)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Detected pins: {results['pin_count']}")
    print(f"Symmetric count estimate: {results['symmetric_count']}")
    print(f"Based on best side: {results['best_side']}")
    print("="*60)


if __name__ == "__main__":
    main()
