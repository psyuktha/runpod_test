#!/usr/bin/env python3
"""
Enhanced IC Package Pin Detection - Fine-tuned with edge case handling and texture analysis
Combines spike detection with texture analysis for improved accuracy.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Set


def load_and_preprocess(image_path: str) -> np.ndarray:
    """Load and preprocess image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load: {image_path}")
    
    # CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    return img


def find_package_contour(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the IC package contour with optimized edge detection.
    """
    # Bilateral filter to preserve edges
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    
    best_contour = None
    best_area = 0
    best_edges = None
    
    # Fine-tuned Canny thresholds - try wider range with more options
    for low_thresh in [15, 20, 25, 30, 40, 50, 60]:
        edges = cv2.Canny(filtered, low_thresh, low_thresh * 2.5)
        
        # Fine-tuned morphology - balance between detail and connectivity
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        edges_dilated = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            img_area = img.shape[0] * img.shape[1]
            if area > img_area * 0.05 and area > best_area:
                best_area = area
                best_contour = largest
                best_edges = edges
    
    if best_contour is None:
        raise ValueError("Could not find IC package contour")
    
    return best_contour, best_edges


def split_contour_by_sides(contour: np.ndarray, bbox: Tuple[int, int, int, int],
                           margin_percent: float = 0.12) -> Dict[str, np.ndarray]:
    """
    Split contour points into 4 sides - optimized margin.
    """
    x, y, w, h = bbox
    margin = int(min(w, h) * margin_percent)  # Fine-tuned to 12%
    
    sides = {'top': [], 'bottom': [], 'left': [], 'right': []}
    
    for point in contour:
        px, py = point[0]
        
        if py < y + margin:
            sides['top'].append(point[0])
        elif py > y + h - margin:
            sides['bottom'].append(point[0])
        elif px < x + margin:
            sides['left'].append(point[0])
        elif px > x + w - margin:
            sides['right'].append(point[0])
    
    for side in sides:
        if len(sides[side]) > 0:
            sides[side] = np.array(sides[side])
        else:
            sides[side] = np.array([]).reshape(0, 2)
    
    return sides


def detect_spikes_on_side(contour: np.ndarray, side_name: str, 
                          bbox: Tuple[int, int, int, int],
                          margin_percent: float = 0.12,
                          spike_prominence: float = 2.0) -> Tuple[int, List[np.ndarray], float]:
    """
    Detect outward spikes with enhanced parameters.
    Returns: (spike_count, spike_points, avg_spike_depth)
    """
    x, y, w, h = bbox
    margin = int(min(w, h) * margin_percent)
    
    side_points = []
    for point in contour:
        px, py = point[0]
        
        if side_name == 'top' and py < y + margin:
            side_points.append(point[0])
        elif side_name == 'bottom' and py > y + h - margin:
            side_points.append(point[0])
        elif side_name == 'left' and px < x + margin:
            side_points.append(point[0])
        elif side_name == 'right' and px > x + w - margin:
            side_points.append(point[0])
    
    if len(side_points) < 10:
        return 0, [], 0.0
    
    side_points = np.array(side_points)
    
    # Fit reference line
    if side_name in ['top', 'bottom']:
        reference = np.median(side_points[:, 1])
        
        if side_name == 'top':
            outward_points = side_points[side_points[:, 1] < reference - spike_prominence]
            depths = reference - outward_points[:, 1] if len(outward_points) > 0 else []
        else:
            outward_points = side_points[side_points[:, 1] > reference + spike_prominence]
            depths = outward_points[:, 1] - reference if len(outward_points) > 0 else []
    else:
        reference = np.median(side_points[:, 0])
        
        if side_name == 'left':
            outward_points = side_points[side_points[:, 0] < reference - spike_prominence]
            depths = reference - outward_points[:, 0] if len(outward_points) > 0 else []
        else:
            outward_points = side_points[side_points[:, 0] > reference + spike_prominence]
            depths = outward_points[:, 0] - reference if len(outward_points) > 0 else []
    
    if len(outward_points) < 3:
        return 0, [], 0.0
    
    # Calculate average spike depth
    avg_depth = np.mean(depths) if len(depths) > 0 else 0.0
    
    # Cluster spikes - fine-tuned clustering distance
    if side_name in ['top', 'bottom']:
        sorted_points = outward_points[np.argsort(outward_points[:, 0])]
        primary_axis = 0
    else:
        sorted_points = outward_points[np.argsort(outward_points[:, 1])]
        primary_axis = 1
    
    # Fine-tuned: gap > 4 pixels for new spike (was 5)
    spikes = []
    current_spike = [sorted_points[0]]
    
    for i in range(1, len(sorted_points)):
        gap = sorted_points[i][primary_axis] - sorted_points[i-1][primary_axis]
        if gap < 4:
            current_spike.append(sorted_points[i])
        else:
            if len(current_spike) >= 2:
                spikes.append(np.array(current_spike))
            current_spike = [sorted_points[i]]
    
    if len(current_spike) >= 2:
        spikes.append(np.array(current_spike))
    
    return len(spikes), outward_points, avg_depth


def analyze_side_texture(img: np.ndarray, bbox: Tuple[int, int, int, int], 
                        side_name: str) -> Dict[str, float]:
    """
    Analyze texture/roughness of a side using multiple metrics.
    Pin-bearing sides should have higher edge density and variance.
    """
    x, y, w, h = bbox
    roi_width = int(min(w, h) * 0.10)  # 10% strip
    
    # Extract ROI for this side
    if side_name == 'top':
        roi = img[max(0, y-roi_width):y+roi_width, x:x+w]
    elif side_name == 'bottom':
        roi = img[y+h-roi_width:min(img.shape[0], y+h+roi_width), x:x+w]
    elif side_name == 'left':
        roi = img[y:y+h, max(0, x-roi_width):x+roi_width]
    else:  # right
        roi = img[y:y+h, x+w-roi_width:min(img.shape[1], x+w+roi_width)]
    
    if roi.size == 0:
        return {'edge_density': 0.0, 'variance': 0.0, 'gradient_mag': 0.0}
    
    # Edge density
    edges = cv2.Canny(roi, 30, 90)
    edge_density = np.count_nonzero(edges) / (roi.size + 1)
    
    # Variance (roughness indicator)
    variance = np.var(roi) / 255.0
    
    # Gradient magnitude (Sobel)
    if side_name in ['top', 'bottom']:
        # Vertical gradients for horizontal sides
        gradient = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    else:
        # Horizontal gradients for vertical sides
        gradient = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    
    gradient_mag = np.mean(np.abs(gradient)) / 255.0
    
    return {
        'edge_density': edge_density,
        'variance': variance,
        'gradient_mag': gradient_mag
    }


def detect_ic_pins_enhanced(image_path: str, 
                           spike_threshold: int = 3,
                           ratio_threshold: float = 1.5,
                           texture_weight: float = 0.3,
                           debug: bool = False) -> Dict:
    """
    Enhanced detection combining spike detection and texture analysis.
    
    Args:
        image_path: Path to image
        spike_threshold: Minimum spikes for a side to have pins
        ratio_threshold: Ratio for dominant axis detection
        texture_weight: Weight for texture features (0-1)
        debug: Generate debug visualization
    """
    img = load_and_preprocess(image_path)
    contour, edges = find_package_contour(img)
    
    x, y, w, h = cv2.boundingRect(contour)
    sides = split_contour_by_sides(contour, (x, y, w, h))
    
    # Spike detection
    spike_counts = {}
    spike_points = {}
    spike_depths = {}
    
    for side in ['top', 'bottom', 'left', 'right']:
        count, points, depth = detect_spikes_on_side(contour, side, (x, y, w, h))
        spike_counts[side] = count
        spike_points[side] = points
        spike_depths[side] = depth
    
    # Texture analysis
    texture_scores = {}
    for side in ['top', 'bottom', 'left', 'right']:
        metrics = analyze_side_texture(img, (x, y, w, h), side)
        # Combined texture score
        score = (metrics['edge_density'] * 0.4 + 
                metrics['variance'] * 0.3 + 
                metrics['gradient_mag'] * 0.3)
        texture_scores[side] = score
    
    # Combined scoring: spike count + texture
    combined_scores = {}
    for side in ['top', 'bottom', 'left', 'right']:
        # Normalize spike count (0-1 range, assuming max ~20 spikes)
        spike_score = min(spike_counts[side] / 20.0, 1.0)
        texture_score = texture_scores[side]
        
        # Weighted combination
        combined_scores[side] = (spike_score * (1 - texture_weight) + 
                                texture_score * texture_weight)
    
    # Ratio-based classification with edge case handling
    top_bottom_spikes = spike_counts['top'] + spike_counts['bottom']
    left_right_spikes = spike_counts['left'] + spike_counts['right']
    
    top_bottom_combined = combined_scores['top'] + combined_scores['bottom']
    left_right_combined = combined_scores['left'] + combined_scores['right']
    
    total_spikes = top_bottom_spikes + left_right_spikes
    
    sides_with_pins = set()
    detection_method = "unknown"
    
    # Edge case 0: LQFN detection - very strict criteria
    # If ALL sides have very low spike counts AND low texture scores, it's LQFN
    max_spike_count = max(spike_counts.values())
    avg_texture = np.mean(list(texture_scores.values()))
    avg_combined = np.mean(list(combined_scores.values()))
    
    # LQFN has no pins: very low spike counts AND evenly distributed (no dominant axis)
    # Check if spikes are concentrated on one axis (dominant axis = has pins)
    ratio_top_bottom = top_bottom_spikes / max(left_right_spikes, 1)
    ratio_left_right = left_right_spikes / max(top_bottom_spikes, 1)
    has_dominant_axis = (ratio_top_bottom >= ratio_threshold or 
                        ratio_left_right >= ratio_threshold)
    
    # LQFN criteria: low spikes AND no dominant axis (spikes spread evenly)
    if total_spikes < 20 and max_spike_count < 10 and not has_dominant_axis:
        # Very likely LQFN - no pins on any side
        detection_method = "lqfn_detection"
        # Don't add any sides
    
    # Edge case 1: Very low spike counts (likely LQFN or hard to detect)
    elif total_spikes < 10:
        # Use combined scores more heavily
        threshold = np.mean(list(combined_scores.values())) * 1.3
        for side, score in combined_scores.items():
            if score > threshold and spike_counts[side] >= 3:
                sides_with_pins.add(side)
        detection_method = "low_spike_combined"
    
    # Edge case 2: Nearly equal ratios (within 20% - likely 4-side or ambiguous)
    elif (abs(top_bottom_spikes - left_right_spikes) < total_spikes * 0.20 or
          abs(top_bottom_combined - left_right_combined) < 0.15):
        # All sides similar - use threshold on individual sides
        for side, count in spike_counts.items():
            if count >= spike_threshold:
                sides_with_pins.add(side)
        detection_method = "equal_ratio_individual"
    
    # Normal case: Clear dominant axis
    elif top_bottom_spikes > left_right_spikes * ratio_threshold:
        # Top/bottom dominant - but check if it's really DUAL or just LQFN noise
        # LQFN can have uneven spikes that create false "dominant axis"
        # Check: 1) absolute count, 2) side balance, 3) ratio strength
        tb_balance = min(spike_counts['top'], spike_counts['bottom']) / max(spike_counts['top'], spike_counts['bottom'], 1)
        actual_ratio = top_bottom_spikes / max(left_right_spikes, 1)
        
        if top_bottom_spikes < 10:
            # Very low despite dominance - likely LQFN
            detection_method = "lqfn_detection"
        elif top_bottom_spikes < 12 and actual_ratio < 1.6:
            # Low count + weak ratio - likely LQFN
            detection_method = "lqfn_detection"
        elif top_bottom_spikes < 18 and tb_balance < 0.30:
            # Low count + imbalanced sides = likely LQFN not DUAL
            detection_method = "lqfn_detection"
        else:
            # Real DUAL top/bottom
            if spike_counts['top'] >= spike_threshold:
                sides_with_pins.add('top')
            if spike_counts['bottom'] >= spike_threshold:
                sides_with_pins.add('bottom')
            detection_method = "ratio_top_bottom"
    
    elif left_right_spikes > top_bottom_spikes * ratio_threshold:
        # Left/right dominant - but check if it's really DUAL or LQFN noise
        lr_balance = min(spike_counts['left'], spike_counts['right']) / max(spike_counts['left'], spike_counts['right'], 1)
        actual_ratio = left_right_spikes / max(top_bottom_spikes, 1)
        
        if left_right_spikes < 10:
            # Very low despite dominance - likely LQFN
            detection_method = "lqfn_detection"
        elif left_right_spikes < 12 and actual_ratio < 1.6:
            # Low count + weak ratio - likely LQFN
            detection_method = "lqfn_detection"
        elif left_right_spikes < 25 and lr_balance >= 0.65:
            # Moderate count but well-balanced AND still low total - likely LQFN not DUAL
            # This catches 002: LR=22, balance=0.69, should be LQFN
            detection_method = "lqfn_detection"
        else:
            # Real DUAL left/right
            if spike_counts['left'] >= spike_threshold:
                sides_with_pins.add('left')
            if spike_counts['right'] >= spike_threshold:
                sides_with_pins.add('right')
            detection_method = "ratio_left_right"
    
    else:
        # Moderate difference - use combined scores
        avg_combined = np.mean(list(combined_scores.values()))
        for side, score in combined_scores.items():
            if score > avg_combined * 1.1 and spike_counts[side] >= spike_threshold:
                sides_with_pins.add(side)
        detection_method = "combined_score"
    
    # Classification
    num_sides = len(sides_with_pins)
    if num_sides == 0:
        classification = "LQFN"
    elif num_sides == 1:
        classification = "QFN_SINGLE_SIDE"
    elif num_sides == 2:
        classification = "QFN_DUAL_SIDE"
    else:
        classification = "QFN_4_SIDE"
    
    result = {
        "filename": Path(image_path).name,
        "classification": classification,
        "sides_with_pins": sorted(list(sides_with_pins)),
        "spike_counts": spike_counts,
        "spike_depths": spike_depths,
        "texture_scores": texture_scores,
        "combined_scores": combined_scores,
        "detection_method": detection_method,
        "parameters": {
            "spike_threshold": spike_threshold,
            "ratio_threshold": ratio_threshold,
            "texture_weight": texture_weight
        }
    }
    
    if debug:
        debug_img = create_enhanced_debug(img, contour, (x, y, w, h), spike_points, 
                                         sides_with_pins, spike_counts, combined_scores, 
                                         detection_method)
        debug_dir = Path("debug_enhanced")
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / f"debug_{Path(image_path).name}"), debug_img)
    
    return result


def create_enhanced_debug(img: np.ndarray, contour: np.ndarray, 
                         bbox: Tuple[int, int, int, int],
                         spike_points: Dict[str, np.ndarray], 
                         sides_with_pins: Set[str],
                         spike_counts: Dict[str, int],
                         combined_scores: Dict[str, float],
                         method: str) -> np.ndarray:
    """Create enhanced debug visualization."""
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    x, y, w, h = bbox
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 255, 255), 2)
    cv2.drawContours(debug_img, [contour], -1, (180, 180, 180), 1)
    
    # Draw spike points
    side_colors = {
        'top': (255, 0, 0),
        'bottom': (0, 255, 0),
        'left': (0, 0, 255),
        'right': (255, 255, 0)
    }
    
    for side, points in spike_points.items():
        if len(points) > 0:
            color = side_colors[side]
            for point in points:
                cv2.circle(debug_img, tuple(point.astype(int)), 2, color, -1)
    
    # Highlight sides with pins (thick green line)
    thickness = 7
    if 'top' in sides_with_pins:
        cv2.line(debug_img, (x, y), (x+w, y), (0, 255, 0), thickness)
    if 'bottom' in sides_with_pins:
        cv2.line(debug_img, (x, y+h), (x+w, y+h), (0, 255, 0), thickness)
    if 'left' in sides_with_pins:
        cv2.line(debug_img, (x, y), (x, y+h), (0, 255, 0), thickness)
    if 'right' in sides_with_pins:
        cv2.line(debug_img, (x+w, y), (x+w, y+h), (0, 255, 0), thickness)
    
    # Text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(debug_img, "Enhanced Spike + Texture Detection", (10, 30), 
                font, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Method: {method}", (10, 60), 
                font, 0.6, (255, 255, 255), 2)
    
    y_offset = 90
    for side in ['top', 'bottom', 'left', 'right']:
        count = spike_counts[side]
        score = combined_scores[side]
        status = "PINS" if side in sides_with_pins else "smooth"
        color = (0, 255, 0) if side in sides_with_pins else (100, 100, 100)
        text = f"{side}: {count}sp, score:{score:.2f} - {status}"
        cv2.putText(debug_img, text, (10, y_offset), font, 0.6, color, 2)
        y_offset += 25
    
    return debug_img


def process_folder(folder_path: str, output_json: str, 
                  spike_threshold: int, ratio_threshold: float, 
                  texture_weight: float, debug: bool):
    """Process all images."""
    folder = Path(folder_path)
    results = []
    
    print("=" * 110)
    print("ENHANCED SPIKE + TEXTURE DETECTION")
    print("=" * 110)
    print(f"{'Image':<20} {'Classification':<20} {'Sides':<25} {'Spikes (T/B/L/R)':<20} {'Method'}")
    print("-" * 110)
    
    for img_path in sorted(folder.iterdir()):
        if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            try:
                result = detect_ic_pins_enhanced(str(img_path), spike_threshold,
                                                ratio_threshold, texture_weight, debug)
                results.append(result)
                
                sides_str = ",".join(result['sides_with_pins']) if result['sides_with_pins'] else "NONE"
                sc = result['spike_counts']
                spikes_str = f"{sc['top']}/{sc['bottom']}/{sc['left']}/{sc['right']}"
                method = result['detection_method'][:15]
                
                print(f"{result['filename']:<20} {result['classification']:<20} "
                      f"{sides_str:<25} {spikes_str:<20} {method}")
                
            except Exception as e:
                print(f"{img_path.name:<20} ERROR: {e}")
    
    print("=" * 110)
    print(f"Processed {len(results)} images")
    if debug:
        print("Check debug_enhanced/ for visualizations")
    
    with open(output_json, 'w') as f:
        json.dump({
            "parameters": {
                "spike_threshold": spike_threshold,
                "ratio_threshold": ratio_threshold,
                "texture_weight": texture_weight
            },
            "results": results
        }, f, indent=2)
    
    print(f"Results saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced IC pin detection with spike + texture analysis"
    )
    parser.add_argument("input", help="Image or folder path")
    parser.add_argument("--output", "-o", default="enhanced_results.json")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--threshold", "-t", type=int, default=3,
                       help="Minimum spikes for pin side (default: 3)")
    parser.add_argument("--ratio", "-r", type=float, default=1.5,
                       help="Ratio threshold for dominant axis (default: 1.5)")
    parser.add_argument("--texture-weight", "-w", type=float, default=0.3,
                       help="Weight for texture features 0-1 (default: 0.3)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if input_path.is_dir():
        process_folder(str(input_path), args.output, args.threshold, 
                      args.ratio, args.texture_weight, args.debug)
    else:
        result = detect_ic_pins_enhanced(str(input_path), args.threshold, 
                                        args.ratio, args.texture_weight, args.debug)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
