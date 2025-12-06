"""
IC Pin Counter - Remove largest rectangle (IC body) and count smaller rectangles (pins)
Supports DIP (horizontal/vertical) and QFP packages with pins on any/all sides.
"""

import cv2
import numpy as np
from scipy import signal

def count_ic_pins(image_path: str, debug: bool = True):
    """
    Count IC pins by analyzing the Canny edge image structure.
    Detects pins on all four sides (top, bottom, left, right).
    """
    # Load the Canny edge image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    print(f"Image loaded: {img.shape}")
    img_h, img_w = img.shape

    # Create a copy for visualization
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Threshold to ensure binary
    _, binary = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # Step 1: Find IC body boundaries using horizontal and vertical projections
    h_proj = np.sum(binary, axis=1).astype(float)  # Sum along rows
    v_proj = np.sum(binary, axis=0).astype(float)  # Sum along columns

    # Find the main IC body region using thresholds on projections
    h_threshold = np.max(h_proj) * 0.15
    v_threshold = np.max(v_proj) * 0.15

    # Find rows/cols with significant content
    active_rows = np.where(h_proj > h_threshold)[0]
    active_cols = np.where(v_proj > v_threshold)[0]

    if len(active_rows) == 0 or len(active_cols) == 0:
        print("No significant features found!")
        return 0

    # Get the bounding box of all active regions
    top_edge = active_rows[0]
    bottom_edge = active_rows[-1]
    left_edge = active_cols[0]
    right_edge = active_cols[-1]

    print(f"Feature bounds: top={top_edge}, bottom={bottom_edge}, left={left_edge}, right={right_edge}")

    # Calculate aspect ratio to help determine package type
    feature_width = right_edge - left_edge
    feature_height = bottom_edge - top_edge
    aspect_ratio = feature_width / max(feature_height, 1)
    print(f"Aspect ratio (W/H): {aspect_ratio:.2f}")

    # Step 2: Find the IC body (main rectangle) vs pins
    # Smooth the projections
    window = 21
    h_proj_smooth = np.convolve(h_proj, np.ones(window)/window, mode='same')
    v_proj_smooth = np.convolve(v_proj, np.ones(window)/window, mode='same')

    # Find IC body boundaries (where projection is above 50% of max)
    h_body_threshold = np.max(h_proj_smooth) * 0.5
    v_body_threshold = np.max(v_proj_smooth) * 0.5

    body_rows = np.where(h_proj_smooth > h_body_threshold)[0]
    body_cols = np.where(v_proj_smooth > v_body_threshold)[0]

    if len(body_rows) > 10:
        ic_body_top = body_rows[0]
        ic_body_bottom = body_rows[-1]
    else:
        range_h = bottom_edge - top_edge
        ic_body_top = top_edge + int(range_h * 0.2)
        ic_body_bottom = bottom_edge - int(range_h * 0.2)

    if len(body_cols) > 10:
        ic_body_left = body_cols[0]
        ic_body_right = body_cols[-1]
    else:
        range_w = right_edge - left_edge
        ic_body_left = left_edge + int(range_w * 0.2)
        ic_body_right = right_edge - int(range_w * 0.2)

    # Calculate IC body dimensions
    ic_width = ic_body_right - ic_body_left
    ic_height = ic_body_bottom - ic_body_top
    ic_aspect = ic_width / max(ic_height, 1)

    print(f"\nIC body region: top={ic_body_top}, bottom={ic_body_bottom}, left={ic_body_left}, right={ic_body_right}")
    print(f"IC body aspect ratio: {ic_aspect:.2f}")

    # Draw IC body region in red
    cv2.rectangle(vis_img, (ic_body_left, ic_body_top),
                  (ic_body_right, ic_body_bottom), (0, 0, 255), 2)

    # Step 3: Extract pin regions on ALL FOUR sides
    pin_margin = 80

    # Define ROIs for each side - extend OUTSIDE the IC body only
    rois = {}
    roi_coords = {}

    # Top pins region (above IC body)
    top_roi_start = max(0, top_edge)
    top_roi_end = ic_body_top + 10  # Just slightly into the body
    rois['top'] = binary[top_roi_start:top_roi_end, ic_body_left:ic_body_right]
    roi_coords['top'] = (ic_body_left, top_roi_start, ic_body_right, top_roi_end)

    # Bottom pins region (below IC body)
    bottom_roi_start = ic_body_bottom - 10
    bottom_roi_end = min(img_h, bottom_edge)
    rois['bottom'] = binary[bottom_roi_start:bottom_roi_end, ic_body_left:ic_body_right]
    roi_coords['bottom'] = (ic_body_left, bottom_roi_start, ic_body_right, bottom_roi_end)

    # Left pins region (left of IC body)
    left_roi_start = max(0, left_edge)
    left_roi_end = ic_body_left + 10
    rois['left'] = binary[ic_body_top:ic_body_bottom, left_roi_start:left_roi_end]
    roi_coords['left'] = (left_roi_start, ic_body_top, left_roi_end, ic_body_bottom)

    # Right pins region (right of IC body)
    right_roi_start = ic_body_right - 10
    right_roi_end = min(img_w, right_edge)
    rois['right'] = binary[ic_body_top:ic_body_bottom, right_roi_start:right_roi_end]
    roi_coords['right'] = (right_roi_start, ic_body_top, right_roi_end, ic_body_bottom)

    for side, roi in rois.items():
        print(f"{side.capitalize()} ROI shape: {roi.shape}")

    def count_pins_in_region(roi, side_name, is_horizontal=True, is_qfp=False):
        """
        Count pins using projection profile analysis.
        is_horizontal: True for top/bottom (pins arranged horizontally)
                      False for left/right (pins arranged vertically)
        is_qfp: True if likely QFP package (finer pin pitch)
        """
        if roi.size == 0:
            return 0

        # Check minimum size for the pin arrangement direction
        if is_horizontal:
            if roi.shape[1] < 50:  # Need enough width for horizontal pins
                return 0
        else:
            if roi.shape[0] < 50:  # Need enough height for vertical pins
                return 0

        # Project along the appropriate axis
        if is_horizontal:
            # For top/bottom: project vertically to get horizontal profile
            projection = np.sum(roi, axis=0).astype(float)
        else:
            # For left/right: project horizontally to get vertical profile
            projection = np.sum(roi, axis=1).astype(float)

        if len(projection) < 30:
            return 0

        # Smooth the projection - use smaller window for QFP (finer pitch)
        if is_qfp:
            window = min(7, len(projection) // 20)
        else:
            window = min(15, len(projection) // 10)
        if window < 3:
            window = 3
        if window % 2 == 0:
            window += 1

        projection_smooth = np.convolve(projection, np.ones(window)/window, mode='same')

        # Normalize
        p_min, p_max = np.min(projection_smooth), np.max(projection_smooth)
        if p_max - p_min < 50:  # Not enough contrast - likely no pins
            return 0
        projection_norm = (projection_smooth - p_min) / (p_max - p_min)

        # Calculate expected minimum distance between pins
        # QFP has finer pitch (more pins), DIP has coarser pitch
        if is_qfp:
            expected_pins = 16  # QFP can have many more pins per side
            min_distance = max(8, len(projection) // (expected_pins + 4))
        else:
            expected_pins = 8
            min_distance = max(15, len(projection) // (expected_pins + 2))

        # Use scipy peak finding with adjusted parameters
        try:
            peaks, properties = signal.find_peaks(
                projection_norm,
                height=0.15 if is_qfp else 0.25,
                distance=min_distance,
                prominence=0.05 if is_qfp else 0.1
            )
            count = len(peaks)
        except:
            # Fallback: manual peak detection
            peaks = []
            threshold = 0.2 if is_qfp else 0.3
            for i in range(1, len(projection_norm) - 1):
                if projection_norm[i] > threshold:
                    if projection_norm[i] >= projection_norm[i-1] and projection_norm[i] >= projection_norm[i+1]:
                        if not peaks or (i - peaks[-1]) >= min_distance:
                            peaks.append(i)
            count = len(peaks)

        return count

    # Step 4: Determine package type based on aspect ratio FIRST
    # DIP packages have aspect ratio > 1.5 (horizontal) or < 0.67 (vertical)
    # QFP packages are more square (aspect ratio 0.8 - 1.2)

    # Pre-determine likely package type from aspect ratio
    if ic_aspect > 1.8:
        likely_package = "DIP_HORIZONTAL"
    elif ic_aspect < 0.55:
        likely_package = "DIP_VERTICAL"
    else:
        likely_package = "QFP"

    print(f"\nLikely package type (from aspect ratio): {likely_package}")

    is_qfp = (likely_package == "QFP")

    # Count pins on each side with appropriate parameters
    pin_counts = {}
    pin_counts['top'] = count_pins_in_region(rois['top'], "Top", is_horizontal=True, is_qfp=is_qfp)
    pin_counts['bottom'] = count_pins_in_region(rois['bottom'], "Bottom", is_horizontal=True, is_qfp=is_qfp)
    pin_counts['left'] = count_pins_in_region(rois['left'], "Left", is_horizontal=False, is_qfp=is_qfp)
    pin_counts['right'] = count_pins_in_region(rois['right'], "Right", is_horizontal=False, is_qfp=is_qfp)

    print(f"\nRaw pin counts:")
    for side, count in pin_counts.items():
        print(f"  {side.capitalize()}: {count}")

    # SIMPLE APPROACH: Find max pins on one side, multiply by number of sides
    threshold_pins = 2  # Minimum pins to consider a side active

    # Determine which sides have pins
    if likely_package == "DIP_HORIZONTAL":
        # DIP horizontal: pins on top and bottom only
        package_type = "DIP (Horizontal)"
        max_pins_per_side = max(pin_counts['top'], pin_counts['bottom'])
        num_sides = 2
        pin_counts['top'] = pin_counts['bottom'] = max_pins_per_side
        pin_counts['left'] = pin_counts['right'] = 0

    elif likely_package == "DIP_VERTICAL":
        # DIP vertical: pins on left and right only
        package_type = "DIP (Vertical)"
        max_pins_per_side = max(pin_counts['left'], pin_counts['right'])
        num_sides = 2
        pin_counts['left'] = pin_counts['right'] = max_pins_per_side
        pin_counts['top'] = pin_counts['bottom'] = 0

    else:
        # QFP: pins on all 4 sides
        # Find the maximum count from any side
        max_pins_per_side = max(pin_counts.values())

        # Check how many sides actually have significant pins
        active_sides = [side for side, count in pin_counts.items() if count >= threshold_pins]

        if len(active_sides) >= 3:
            # QFP with pins on all 4 sides
            package_type = "QFP/QFN (Quad Flat Package)"
            num_sides = 4
            pin_counts = {side: max_pins_per_side for side in pin_counts}
        elif len(active_sides) == 2:
            # Could be DIP
            if 'top' in active_sides or 'bottom' in active_sides:
                package_type = "DIP (Horizontal)"
                num_sides = 2
                pin_counts['top'] = pin_counts['bottom'] = max_pins_per_side
                pin_counts['left'] = pin_counts['right'] = 0
            else:
                package_type = "DIP (Vertical)"
                num_sides = 2
                pin_counts['left'] = pin_counts['right'] = max_pins_per_side
                pin_counts['top'] = pin_counts['bottom'] = 0
        else:
            package_type = "Unknown"
            num_sides = len(active_sides) if active_sides else 1

    # Calculate total: max pins per side × number of sides
    total_pins = max_pins_per_side * num_sides

    print(f"\nMax pins per side: {max_pins_per_side}")
    print(f"Number of sides with pins: {num_sides}")
    print(f"Calculated total: {max_pins_per_side} × {num_sides} = {total_pins}")

    # Round to common pin counts
    if "QFP" in package_type:
        common_counts = [20, 32, 44, 48, 52, 56, 64, 80, 100, 128, 144, 176, 208]
    else:
        common_counts = [4, 6, 8, 14, 16, 18, 20, 22, 24, 28, 40, 48, 56, 64]

    nearest = min(common_counts, key=lambda x: abs(x - total_pins))

    print(f"\n{'='*50}")
    print(f"PIN COUNT RESULTS")
    print(f"{'='*50}")
    print(f"Package type: {package_type}")
    print(f"Pins per side:")
    print(f"  Top:    {pin_counts['top']}")
    print(f"  Bottom: {pin_counts['bottom']}")
    print(f"  Left:   {pin_counts['left']}")
    print(f"  Right:  {pin_counts['right']}")
    print(f"Raw total: {total_pins}")
    print(f"Nearest standard count: {nearest}")
    print(f"{'='*50}")

    if debug:
        import os
        base_name = os.path.splitext(image_path)[0]

        # Draw pin regions for active sides
        colors = {'top': (255, 0, 0), 'bottom': (255, 0, 0),
                  'left': (0, 255, 255), 'right': (0, 255, 255)}

        for side, coords in roi_coords.items():
            if pin_counts[side] > 0:
                x1, y1, x2, y2 = coords
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), colors[side], 2)

        # Add text
        cv2.putText(vis_img, f"Total Pins: {nearest}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_img, f"T:{pin_counts['top']} B:{pin_counts['bottom']} L:{pin_counts['left']} R:{pin_counts['right']}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_img, f"Package: {package_type}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(f"{base_name}_result.jpg", vis_img)
        print(f"\nDebug image saved: {base_name}_result.jpg")

    return nearest


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "/Users/rishichirchi/Development/runpod_test/yuktha/canny_anu5.jpg"

    pin_count = count_ic_pins(image_path, debug=True)
    print(f"\n*** FINAL PIN COUNT: {pin_count} ***")
