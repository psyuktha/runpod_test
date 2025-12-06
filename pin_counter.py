"""
IC Pin Counter - Hybrid Approach
Uses computer vision for programmatic pin counting from Canny edge images.
Optionally uses LLM vision API for higher accuracy.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from scipy import signal
from scipy.ndimage import label, find_objects


@dataclass
class PinCountResult:
    """Result of pin counting analysis."""
    filename: str
    total_pins: int
    package_type: str
    pins_per_side: Dict[str, int]
    confidence: float
    method: str  # 'cv' or 'llm'


# Ground truth for reference (from llm.py)
GROUND_TRUTH = {
    "anu1.jpeg": 64,
    "anu2.jpeg": 56,
    "anu3.jpeg": 20,
    "anu4.jpg": 48,
    "anu5.jpg": 14,
    "anu6.png": 48,
    "new1.png": 14,
    "new2.png": 48,
    "new3.png": 48,
    "new4.png": 22,
    "new5.png": 14
}


def load_image(image_path: str) -> np.ndarray:
    """Load image in grayscale."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return img


def find_chip_region(edge_image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the main chip body region using morphological operations.
    Returns (x, y, width, height) of the chip body.
    """
    h, w = edge_image.shape

    # Dilate to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edge_image, kernel, iterations=5)

    # Close to fill gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=5)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback to center region
        margin = min(h, w) // 8
        return margin, margin, w - 2*margin, h - 2*margin

    # Get the largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    # Shrink slightly to exclude pins from body
    shrink_x = max(10, bw // 15)
    shrink_y = max(10, bh // 15)

    x = min(x + shrink_x, w - 10)
    y = min(y + shrink_y, h - 10)
    bw = max(bw - 2*shrink_x, 20)
    bh = max(bh - 2*shrink_y, 20)

    return x, y, bw, bh


def detect_package_type(edge_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, Dict[str, float]]:
    """
    Detect IC package type by analyzing edge activity on each side.
    Returns package type and activity levels per side.
    """
    x, y, w, h = bbox
    img_h, img_w = edge_image.shape
    margin = 80  # Look further out for pins

    activity = {}

    # Measure edge density in each direction
    # Top
    top_roi = edge_image[max(0, y-margin):y, max(0, x-10):min(img_w, x+w+10)]
    activity['top'] = np.mean(top_roi) if top_roi.size > 0 else 0

    # Bottom
    bottom_roi = edge_image[y+h:min(img_h, y+h+margin), max(0, x-10):min(img_w, x+w+10)]
    activity['bottom'] = np.mean(bottom_roi) if bottom_roi.size > 0 else 0

    # Left
    left_roi = edge_image[max(0, y-10):min(img_h, y+h+10), max(0, x-margin):x]
    activity['left'] = np.mean(left_roi) if left_roi.size > 0 else 0

    # Right
    right_roi = edge_image[max(0, y-10):min(img_h, y+h+10), x+w:min(img_w, x+w+margin)]
    activity['right'] = np.mean(right_roi) if right_roi.size > 0 else 0

    max_act = max(activity.values()) if activity.values() else 1
    threshold = max_act * 0.25

    active_sides = [k for k, v in activity.items() if v > threshold]
    aspect = max(w, h) / max(min(w, h), 1)

    # Classify package type
    tb_active = 'top' in active_sides and 'bottom' in active_sides
    lr_active = 'left' in active_sides and 'right' in active_sides

    if len(active_sides) <= 2:
        if tb_active and not lr_active and aspect > 1.5:
            return 'DIP', activity
        if lr_active and not tb_active and aspect > 1.5:
            return 'DIP', activity
        if len(active_sides) == 1:
            return 'POWER', activity

    if len(active_sides) >= 3:
        if aspect < 1.3:
            return 'QFP/QFN', activity
        else:
            return 'LQFP', activity

    # Default based on aspect ratio
    if aspect > 2.0:
        return 'DIP', activity

    return 'QFP/QFN', activity


def count_pins_projection_method(
    edge_image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    side: str,
    margin: int = 70
) -> int:
    """
    Count pins using projection profile analysis.
    Projects edge intensity perpendicular to the chip edge and counts peaks.
    """
    x, y, w, h = bbox
    img_h, img_w = edge_image.shape

    # Extract ROI for the specified side
    if side == 'top':
        roi = edge_image[max(0, y-margin):y+5, x:x+w]
        axis = 0  # Project vertically
    elif side == 'bottom':
        roi = edge_image[y+h-5:min(img_h, y+h+margin), x:x+w]
        axis = 0
    elif side == 'left':
        roi = edge_image[y:y+h, max(0, x-margin):x+5]
        axis = 1  # Project horizontally
    else:  # right
        roi = edge_image[y:y+h, x+w-5:min(img_w, x+w+margin)]
        axis = 1

    if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
        return 0

    # Create projection profile
    projection = np.sum(roi, axis=axis).astype(float)

    if len(projection) < 10:
        return 0

    # Smooth the projection
    window = min(7, len(projection) // 3)
    if window % 2 == 0:
        window += 1
    if window >= 3:
        projection = signal.savgol_filter(projection, window, 2)

    # Normalize
    p_min, p_max = projection.min(), projection.max()
    if p_max - p_min < 10:
        return 0
    projection = (projection - p_min) / (p_max - p_min)

    # Find peaks
    min_dist = max(5, len(projection) // 40)
    peaks, props = signal.find_peaks(
        projection,
        height=0.15,
        distance=min_dist,
        prominence=0.08
    )

    return len(peaks)


def count_pins_connected_components(
    edge_image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    side: str,
    margin: int = 70
) -> int:
    """
    Count pins by detecting connected components in the pin region.
    """
    x, y, w, h = bbox
    img_h, img_w = edge_image.shape

    # Extract ROI
    if side == 'top':
        roi = edge_image[max(0, y-margin):y+10, max(0, x-5):min(img_w, x+w+5)]
    elif side == 'bottom':
        roi = edge_image[y+h-10:min(img_h, y+h+margin), max(0, x-5):min(img_w, x+w+5)]
    elif side == 'left':
        roi = edge_image[max(0, y-5):min(img_h, y+h+5), max(0, x-margin):x+10]
    else:
        roi = edge_image[max(0, y-5):min(img_h, y+h+5), x+w-10:min(img_w, x+w+margin)]

    if roi.size == 0:
        return 0

    # Threshold
    _, binary = cv2.threshold(roi, 30, 255, cv2.THRESH_BINARY)

    # Light morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Label connected components
    labeled, num_features = label(binary)

    if num_features == 0:
        return 0

    # Filter by size - pins should be reasonable size
    objects = find_objects(labeled)
    valid_count = 0

    min_area = 50
    max_area = roi.size // 3

    for obj in objects:
        if obj is None:
            continue
        obj_h = obj[0].stop - obj[0].start
        obj_w = obj[1].stop - obj[1].start
        area = obj_h * obj_w

        if min_area < area < max_area:
            # Check aspect ratio - pins shouldn't be too elongated
            aspect = max(obj_h, obj_w) / max(min(obj_h, obj_w), 1)
            if aspect < 8:
                valid_count += 1

    return valid_count


def count_pins_hough_lines(
    edge_image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    side: str,
    margin: int = 70
) -> int:
    """
    Count pins using Hough line detection for pin edges.
    """
    x, y, w, h = bbox
    img_h, img_w = edge_image.shape

    # Extract ROI
    if side == 'top':
        roi = edge_image[max(0, y-margin):y+5, x:x+w]
        expected_angle = 90  # Vertical lines for top/bottom pins
    elif side == 'bottom':
        roi = edge_image[y+h-5:min(img_h, y+h+margin), x:x+w]
        expected_angle = 90
    elif side == 'left':
        roi = edge_image[y:y+h, max(0, x-margin):x+5]
        expected_angle = 0  # Horizontal lines for left/right pins
    else:
        roi = edge_image[y:y+h, x+w-5:min(img_w, x+w+margin)]
        expected_angle = 0

    if roi.size == 0 or min(roi.shape) < 10:
        return 0

    # Detect lines
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=10,
        maxLineGap=5
    )

    if lines is None:
        return 0

    # Count lines with appropriate orientation
    count = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)

        # Check if line matches expected orientation (with tolerance)
        if expected_angle == 90:
            if 70 < angle < 110 or angle < 20 or angle > 160:
                count += 1
        else:
            if angle < 30 or angle > 150:
                count += 1

    # Divide by ~2 since each pin might have 2 edges detected
    return max(1, count // 2)


def estimate_pin_count_by_geometry(
    bbox: Tuple[int, int, int, int],
    package_type: str
) -> Dict[str, int]:
    """
    Estimate pin count based on chip dimensions and typical pin pitch.
    """
    x, y, w, h = bbox

    # Typical pin pitch in pixels (varies with image resolution)
    # These are rough estimates
    if package_type == 'DIP':
        pitch = 35  # DIP has larger pitch
    else:
        pitch = 20  # QFP/QFN has finer pitch

    if package_type == 'DIP':
        # DIP has pins on 2 long sides
        if w > h:
            pins_per_side = max(1, w // pitch)
            return {'top': pins_per_side, 'bottom': pins_per_side, 'left': 0, 'right': 0}
        else:
            pins_per_side = max(1, h // pitch)
            return {'top': 0, 'bottom': 0, 'left': pins_per_side, 'right': pins_per_side}
    else:
        # QFP/QFN has pins on all 4 sides
        h_pins = max(1, w // pitch)
        v_pins = max(1, h // pitch)
        return {'top': h_pins, 'bottom': h_pins, 'left': v_pins, 'right': v_pins}


def count_pins_cv(image_path: str, debug: bool = False) -> PinCountResult:
    """
    Count IC pins using computer vision methods.
    Combines multiple detection approaches for robustness.
    """
    filename = os.path.basename(image_path)
    edge_image = load_image(image_path)

    # Find chip body
    bbox = find_chip_region(edge_image)
    x, y, w, h = bbox

    # Detect package type
    package_type, activity = detect_package_type(edge_image, bbox)

    # Count pins using multiple methods
    sides = ['top', 'bottom', 'left', 'right']

    results = {side: [] for side in sides}

    for side in sides:
        # Method 1: Projection profile
        proj_count = count_pins_projection_method(edge_image, bbox, side)
        results[side].append(proj_count)

        # Method 2: Connected components
        cc_count = count_pins_connected_components(edge_image, bbox, side)
        results[side].append(cc_count)

        # Method 3: Hough lines (less reliable but useful as tiebreaker)
        hough_count = count_pins_hough_lines(edge_image, bbox, side)
        results[side].append(hough_count)

    # Combine results - use median of non-zero counts
    pins_per_side = {}
    for side in sides:
        counts = [c for c in results[side] if c > 0]
        if counts:
            pins_per_side[side] = int(np.median(counts))
        else:
            pins_per_side[side] = 0

    # Apply package-type constraints
    if package_type == 'DIP':
        # DIP has pins only on 2 opposite sides
        tb_total = pins_per_side['top'] + pins_per_side['bottom']
        lr_total = pins_per_side['left'] + pins_per_side['right']

        if tb_total > lr_total:
            avg = max((pins_per_side['top'] + pins_per_side['bottom']) // 2, 1)
            pins_per_side = {'top': avg, 'bottom': avg, 'left': 0, 'right': 0}
        else:
            avg = max((pins_per_side['left'] + pins_per_side['right']) // 2, 1)
            pins_per_side = {'top': 0, 'bottom': 0, 'left': avg, 'right': avg}

    elif package_type in ['QFP/QFN', 'LQFP']:
        # QFP packages should have roughly equal pins on all sides
        all_counts = [pins_per_side[s] for s in sides if pins_per_side[s] > 0]
        if all_counts:
            avg = int(np.mean(all_counts))
            pins_per_side = {s: avg for s in sides}

    # Calculate total and round to common values
    total_pins = sum(pins_per_side.values())

    if package_type == 'DIP':
        total_pins = round_to_common_count(total_pins, 'DIP')
        per_side = total_pins // 2
        if pins_per_side['top'] > 0:
            pins_per_side = {'top': per_side, 'bottom': per_side, 'left': 0, 'right': 0}
        else:
            pins_per_side = {'top': 0, 'bottom': 0, 'left': per_side, 'right': per_side}
    elif package_type in ['QFP/QFN', 'LQFP']:
        total_pins = round_to_common_count(total_pins, 'QFP')
        per_side = total_pins // 4
        pins_per_side = {s: per_side for s in sides}
    elif package_type == 'POWER':
        total_pins = round_to_common_count(total_pins, 'POWER')

    # Calculate confidence based on method agreement
    all_counts = []
    for side in sides:
        all_counts.extend(results[side])
    all_counts = [c for c in all_counts if c > 0]

    if len(all_counts) >= 3:
        std = np.std(all_counts)
        mean = np.mean(all_counts)
        confidence = max(0.1, min(0.9, 1.0 - (std / max(mean, 1))))
    else:
        confidence = 0.3

    # Debug visualization
    if debug:
        debug_img = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        text_lines = [
            f"Type: {package_type}",
            f"Total: {total_pins}",
            f"T:{pins_per_side.get('top', 0)} B:{pins_per_side.get('bottom', 0)}",
            f"L:{pins_per_side.get('left', 0)} R:{pins_per_side.get('right', 0)}",
            f"Conf: {confidence:.2f}"
        ]

        for i, text in enumerate(text_lines):
            cv2.putText(debug_img, text, (10, 25 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        debug_path = image_path.replace('.', '_debug.')
        cv2.imwrite(debug_path, debug_img)

    return PinCountResult(
        filename=filename,
        total_pins=total_pins,
        package_type=package_type,
        pins_per_side=pins_per_side,
        confidence=confidence,
        method='cv'
    )


def round_to_common_count(count: int, package_type: str) -> int:
    """Round to common pin counts for the package type."""
    if package_type == 'DIP':
        common = [4, 6, 8, 14, 16, 18, 20, 24, 28, 40, 48, 64]
    elif package_type in ['QFP', 'QFP/QFN', 'LQFP']:
        common = [20, 24, 28, 32, 44, 48, 52, 56, 64, 80, 100, 128, 144, 176, 208]
    elif package_type == 'POWER':
        common = [3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 22]
    else:
        common = [8, 14, 16, 20, 24, 28, 32, 44, 48, 56, 64, 80, 100]

    return min(common, key=lambda x: abs(x - count))


def process_directory(directory: str, debug: bool = False) -> List[PinCountResult]:
    """Process all images in a directory."""
    results = []
    valid_ext = ('.png', '.jpg', '.jpeg')

    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(valid_ext):
            continue

        filepath = os.path.join(directory, filename)

        try:
            result = count_pins_cv(filepath, debug=debug)
            results.append(result)

            # Check against ground truth if available
            base_name = filename.replace('canny_', '')
            gt = GROUND_TRUTH.get(base_name, None)
            gt_str = f" (GT: {gt})" if gt else ""
            match = " ✓" if gt and result.total_pins == gt else (" ✗" if gt else "")

            print(f"{filename}: {result.total_pins} pins ({result.package_type}){gt_str}{match}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return results


def print_summary(results: List[PinCountResult]):
    """Print summary table."""
    print("\n" + "="*90)
    print("IC PIN COUNT SUMMARY")
    print("="*90)
    print(f"{'Filename':<28} {'Package':<10} {'Detected':<10} {'Ground Truth':<12} {'Match':<8}")
    print("-"*90)

    correct = 0
    total_with_gt = 0

    for r in results:
        base_name = r.filename.replace('canny_', '')
        gt = GROUND_TRUTH.get(base_name, None)

        if gt:
            total_with_gt += 1
            match = "✓" if r.total_pins == gt else "✗"
            if r.total_pins == gt:
                correct += 1
            gt_str = str(gt)
        else:
            match = "-"
            gt_str = "N/A"

        print(f"{r.filename:<28} {r.package_type:<10} {r.total_pins:<10} {gt_str:<12} {match:<8}")

    print("="*90)
    if total_with_gt > 0:
        accuracy = correct / total_with_gt * 100
        print(f"Accuracy: {correct}/{total_with_gt} ({accuracy:.1f}%)")
    print(f"Total images processed: {len(results)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count IC pins from Canny edge images")
    parser.add_argument("path", help="Path to image or directory")
    parser.add_argument("--debug", action="store_true", help="Save debug images")

    args = parser.parse_args()

    if os.path.isdir(args.path):
        results = process_directory(args.path, debug=args.debug)
        print_summary(results)
    elif os.path.isfile(args.path):
        result = count_pins_cv(args.path, debug=args.debug)
        print(f"\n{result.filename}:")
        print(f"  Package: {result.package_type}")
        print(f"  Total Pins: {result.total_pins}")
        print(f"  Per Side: {result.pins_per_side}")
        print(f"  Confidence: {result.confidence:.2f}")
    else:
        print(f"Error: {args.path} not found")
