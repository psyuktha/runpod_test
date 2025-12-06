#!/usr/bin/env python3
"""
IC Pin Counter using Contour Detection and Morphological Operations

This approach:
1. Uses morphological erosion to separate touching pins
2. Finds contours instead of connected components
3. Filters pins by size, aspect ratio, and edge proximity
4. Uses symmetric counting based on the most reliable side
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import cv2


def preprocess_image(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess image."""
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    # Binarize
    _, binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to separate touching components
    # First erode to break connections
    kernel_erode = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel_erode, iterations=2)

    # Then dilate slightly to restore pin size
    kernel_dilate = np.ones((2, 2), np.uint8)
    processed = cv2.dilate(eroded, kernel_dilate, iterations=1)

    return img_array, processed


def get_package_bounds(contours: List) -> Tuple[int, int, int, int]:
    """Calculate overall package bounding box from all contours."""
    all_points = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        all_points.extend([(x, y), (x+w, y+h)])

    if not all_points:
        return (0, 0, 0, 0)

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    return (min(xs), min(ys), max(xs), max(ys))


def filter_pin_contours(
    contours: List,
    package_bbox: Tuple[int, int, int, int],
    min_area: int = 80,
    max_area: int = 800,
    min_aspect: float = 0.3,
    max_aspect: float = 3.5
) -> List[Dict]:
    """
    Filter contours to find pin candidates based on size, shape, and position.
    """
    px_min, py_min, px_max, py_max = package_bbox
    pw = px_max - px_min
    ph = py_max - py_min

    edge_margin = 0.22  # Pins must be within 22% of an edge

    pin_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter by area
        if not (min_area <= area <= max_area):
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter by minimum size
        if w < 3 or h < 3:
            continue

        # Filter by aspect ratio (pins shouldn't be too elongated)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        if not (min_aspect <= aspect_ratio <= max_aspect):
            continue

        # Calculate center
        cx = x + w / 2
        cy = y + h / 2

        # Check if near any edge
        dist_left = abs(cx - px_min) / pw
        dist_right = abs(cx - px_max) / pw
        dist_top = abs(cy - py_min) / ph
        dist_bottom = abs(cy - py_max) / ph

        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        if min_dist > edge_margin:
            continue

        pin_contours.append({
            'contour': cnt,
            'area': area,
            'bbox': (x, y, x+w, y+h),
            'center': (cx, cy),
            'aspect_ratio': aspect_ratio
        })

    return pin_contours


def classify_pin_side(pin_center: Tuple[float, float],
                      package_bbox: Tuple[int, int, int, int]) -> str:
    """Classify pin to a side based on proximity."""
    px_min, py_min, px_max, py_max = package_bbox
    cx, cy = pin_center

    # Calculate distance to each edge
    d_left = abs(cx - px_min)
    d_right = abs(cx - px_max)
    d_top = abs(cy - py_min)
    d_bottom = abs(cy - py_max)

    # Find closest edge
    distances = {
        'left': d_left,
        'right': d_right,
        'top': d_top,
        'bottom': d_bottom
    }

    return min(distances.items(), key=lambda x: x[1])[0]


def calculate_spacing_uniformity(centers: List[Tuple[float, float]], side: str) -> float:
    """Calculate coefficient of variation of pin spacings."""
    if len(centers) < 2:
        return float('inf')

    # Extract relevant coordinate
    if side in ('top', 'bottom'):
        coords = sorted([c[0] for c in centers])
    else:
        coords = sorted([c[1] for c in centers])

    # Calculate spacings
    spacings = np.diff(coords)

    if len(spacings) == 0 or np.mean(spacings) == 0:
        return float('inf')

    # Coefficient of variation
    cv = np.std(spacings) / np.mean(spacings)

    return float(cv)


def count_pins(image_path: Path, min_area: int = 80, max_area: int = 800) -> Dict:
    """
    Main pin counting function.

    Returns dict with pin counts, side information, and symmetric estimate.
    """
    print(f"Processing: {image_path}")

    # Preprocess
    original, binary = preprocess_image(image_path)
    print(f"Image size: {binary.shape[1]}x{binary.shape[0]}")

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")

    # Get package bounds
    package_bbox = get_package_bounds(contours)
    print(f"Package bbox: {package_bbox}")

    # Filter for pin contours
    pin_data = filter_pin_contours(contours, package_bbox, min_area, max_area)
    print(f"Detected {len(pin_data)} pin candidates")

    # Classify pins by side
    side_counts = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    side_centers = {'top': [], 'bottom': [], 'left': [], 'right': []}
    side_bboxes = {'top': [], 'bottom': [], 'left': [], 'right': []}

    for pin in pin_data:
        side = classify_pin_side(pin['center'], package_bbox)
        side_counts[side] += 1
        side_centers[side].append(pin['center'])
        side_bboxes[side].append(pin['bbox'])

    print(f"\nPins per side:")
    for side in ['top', 'bottom', 'left', 'right']:
        print(f"  {side.capitalize()}: {side_counts[side]}")

    # Calculate spacing uniformity
    spacing_scores = {}
    for side in ['top', 'bottom', 'left', 'right']:
        if side_counts[side] >= 2:
            score = calculate_spacing_uniformity(side_centers[side], side)
            spacing_scores[side] = score
            print(f"  {side.capitalize()} spacing score: {score:.4f}")

    # Find best side for symmetric counting
    valid_sides = [s for s in ['top', 'bottom', 'left', 'right']
                   if side_counts[s] >= 8]  # Require at least 8 pins for reliability

    if valid_sides:
        best_side = min(valid_sides, key=lambda s: spacing_scores.get(s, float('inf')))
        num_sides = sum(1 for s in ['top', 'bottom', 'left', 'right'] if side_counts[s] > 0)
        symmetric_count = side_counts[best_side] * num_sides

        print(f"\nBest side (most uniform with >= 8 pins): {best_side}")
        print(f"Pins on best side: {side_counts[best_side]}")
        print(f"Sides with pins: {num_sides}")
        print(f"Symmetric estimate: {symmetric_count}")
    else:
        # Fallback: use average of all sides
        valid_counts = [c for c in side_counts.values() if c > 0]
        if valid_counts:
            avg_count = int(np.mean(valid_counts))
            num_sides = len(valid_counts)
            symmetric_count = avg_count * num_sides
            best_side = 'average'

            print(f"\nNo side has >= 8 pins, using average")
            print(f"Average pins per side: {avg_count}")
            print(f"Symmetric estimate: {symmetric_count}")
        else:
            symmetric_count = 0
            best_side = None

    return {
        'total_detected': len(pin_data),
        'side_counts': side_counts,
        'side_centers': side_centers,
        'side_bboxes': side_bboxes,
        'symmetric_count': symmetric_count,
        'best_side': best_side,
        'package_bbox': package_bbox,
        'spacing_scores': spacing_scores
    }


def draw_results(image_path: Path, results: Dict, output_path: Path):
    """Draw annotated image with bounding boxes."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw package bbox
    px_min, py_min, px_max, py_max = results['package_bbox']
    draw.rectangle([(px_min, py_min), (px_max, py_max)],
                   outline=(0, 255, 255), width=3)

    # Side colors
    colors = {
        'top': (255, 0, 0),
        'bottom': (0, 255, 0),
        'left': (255, 255, 0),
        'right': (255, 0, 255)
    }

    # Draw pins
    for side in ['top', 'bottom', 'left', 'right']:
        color = colors[side]
        for bbox in results['side_bboxes'][side]:
            x1, y1, x2, y2 = bbox
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

    # Add text
    y = 10
    draw.text((10, y), f"Total detected: {results['total_detected']}",
              fill=(255, 255, 255))
    y += 20
    draw.text((10, y), f"Symmetric estimate: {results['symmetric_count']}",
              fill=(255, 255, 255))
    y += 20
    draw.text((10, y), f"Best side: {results['best_side']}",
              fill=(255, 255, 255))
    y += 30

    for side in ['top', 'bottom', 'left', 'right']:
        draw.text((10, y), f"{side.capitalize()}: {results['side_counts'][side]}",
                  fill=colors[side])
        y += 20

    img.save(output_path)
    print(f"\nSaved annotated image: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Count IC pins using contour detection"
    )
    parser.add_argument("image", type=Path, help="Path to IC image")
    parser.add_argument("--min-area", type=int, default=80,
                        help="Minimum pin area (default: 80)")
    parser.add_argument("--max-area", type=int, default=800,
                        help="Maximum pin area (default: 800)")
    parser.add_argument("--output", type=Path,
                        help="Output annotated image path")

    args = parser.parse_args()

    # Count pins
    results = count_pins(args.image, args.min_area, args.max_area)

    # Draw results
    if args.output:
        draw_results(args.image, results, args.output)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Detected pins: {results['total_detected']}")
    print(f"Symmetric count estimate: {results['symmetric_count']}")
    print(f"Based on: {results['best_side']}")
    print("="*60)


if __name__ == "__main__":
    main()
