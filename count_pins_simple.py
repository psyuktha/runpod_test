"""
Count IC pins using edge detection and symmetry.

Usage:
    python count_pins_simple.py square/chetan_07_edges.png
"""

import sys
import cv2
import numpy as np


def count_pins(image_path: str, output_path: str = None):
    """
    Count IC pins. Uses symmetry: finds the best-detected side and
    multiplies by number of pin sides (2 for DIP, 4 for QFP).
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Mask the center part (IC body) to avoid false detections
    # Keep only top and bottom bands where pins are located
    mask_top = int(h * 0.35)      # Top 35% for top pins
    mask_bottom = int(h * 0.65)   # Bottom 35% for bottom pins

    masked_gray = gray.copy()
    masked_gray[mask_top:mask_bottom, :] = 0  # Black out center

    # Very permissive threshold
    _, binary = cv2.threshold(masked_gray, 5, 255, cv2.THRESH_BINARY)

    # Find contours only in pin regions
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Estimate IC body center
    body_center_y = h / 2.0

    # Collect all potential pin contours
    print(f"\nImage size: {w}x{h}")
    print(f"Center masked: y={mask_top} to y={mask_bottom}")
    print(f"Total contours found: {len(contours)}")

    all_candidates = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        cy = y + bh / 2.0

        # Permissive area filter
        if area < 100 or area > 10000:
            continue

        # Aspect ratio (pins are usually wider than tall or squarish)
        aspect = bw / max(bh, 1)
        if aspect < 0.4 or aspect > 4.0:
            continue

        # Determine side based on y position relative to image center
        side = "top" if cy < body_center_y else "bottom"
        all_candidates.append((x, y, bw, bh, area, side))

    print(f"Candidates after basic filters: {len(all_candidates)}")

    # Separate by side
    top_raw = [(x, y, bw, bh, a) for x, y, bw, bh, a, s in all_candidates if s == "top"]
    bottom_raw = [(x, y, bw, bh, a) for x, y, bw, bh, a, s in all_candidates if s == "bottom"]

    print(f"Top candidates: {len(top_raw)}")
    print(f"Bottom candidates: {len(bottom_raw)}")

    # Print all candidate areas for debugging
    print(f"\nTop candidate areas: {sorted([a for _, _, _, _, a in top_raw])}")
    print(f"Bottom candidate areas: {sorted([a for _, _, _, _, a in bottom_raw])}")

    def filter_by_horizontal_line(pins, y_tolerance=30):
        """Keep only pins aligned on the same horizontal line (median y)."""
        if len(pins) < 2:
            return pins
        
        # Find median y-center
        y_centers = [p[1] + p[3] / 2.0 for p in pins]
        median_y = float(np.median(y_centers))
        
        # Keep only pins within tolerance of median
        aligned = [p for p in pins if abs((p[1] + p[3] / 2.0) - median_y) < y_tolerance]
        return aligned

    def dedupe_by_x(pins, min_dist=30):
        """De-duplicate pins that are too close horizontally."""
        if not pins:
            return []
        pins = sorted(pins, key=lambda p: p[0])  # sort by x
        result = [pins[0]]
        for p in pins[1:]:
            last_cx = result[-1][0] + result[-1][2] / 2.0
            this_cx = p[0] + p[2] / 2.0
            if abs(this_cx - last_cx) > min_dist:
                result.append(p)
            else:
                # Keep the one with larger area
                if p[4] > result[-1][4]:
                    result[-1] = p
        return result

    # First filter by horizontal alignment, then dedupe
    top_aligned = filter_by_horizontal_line(top_raw)
    bottom_aligned = filter_by_horizontal_line(bottom_raw)
    
    print(f"\nAfter horizontal alignment filter:")
    print(f"  Top aligned: {len(top_aligned)} (was {len(top_raw)})")
    print(f"  Bottom aligned: {len(bottom_aligned)} (was {len(bottom_raw)})")

    top_pins = dedupe_by_x(top_aligned)
    bottom_pins = dedupe_by_x(bottom_aligned)

    # Use symmetry: take the max of top/bottom and assume IC is symmetric
    detected_top = len(top_pins)
    detected_bottom = len(bottom_pins)
    best_side_count = max(detected_top, detected_bottom)

    # For DIP package (pins on 2 sides), total = best_side * 2
    estimated_total = best_side_count * 2

    # Annotate and save output
    out = img.copy()
    color = (255, 0, 255)  # Magenta

    # Draw top pins
    for idx, (x, y, bw, bh, area) in enumerate(sorted(top_pins, key=lambda p: p[0]), start=1):
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(out, str(idx), (x + 2, y + bh + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Draw bottom pins (continue numbering)
    start_idx = best_side_count + 1
    for idx, (x, y, bw, bh, area) in enumerate(sorted(bottom_pins, key=lambda p: p[0]), start=start_idx):
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(out, str(idx), (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if output_path:
        cv2.imwrite(output_path, out)
        print(f"\nSaved annotated image to {output_path}")

    # Print results with areas
    print(f"\n=== IC Pin Count ===")
    print(f"Top row detected:    {detected_top} pins")
    for idx, (x, y, bw, bh, area) in enumerate(sorted(top_pins, key=lambda p: p[0]), start=1):
        print(f"  Pin {idx}: x={x}, y={y}, w={bw}, h={bh}, area={area}")

    print(f"\nBottom row detected: {detected_bottom} pins")
    for idx, (x, y, bw, bh, area) in enumerate(sorted(bottom_pins, key=lambda p: p[0]), start=1):
        print(f"  Pin {idx}: x={x}, y={y}, w={bw}, h={bh}, area={area}")

    print(f"\nBest side count: {best_side_count} pins")
    print(f"\nUsing symmetry (DIP package with 2 sides):")
    print(f"  Estimated total = {best_side_count} x 2 = {estimated_total} pins")

    return estimated_total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_pins_simple.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "debug/pins_detected.png"

    count_pins(image_path, output_path)
