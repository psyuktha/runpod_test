"""
IC Chip Pin Detection and Counting Pipeline - Standalone Version
Uses pure OpenCV for fast and reliable pin counting without ML models
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import os
import argparse

# Debug image saving helper
def save_debug_image(image: np.ndarray, stage: str, out_dir: str, base_name: str) -> None:
    """Save debug image for a given stage."""
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_{stage}.png"), image)

# 1. Load and resize image
def load_and_resize_image(image_path: str, max_dim: int = 1024) -> np.ndarray:
    """Load image and resize so largest dimension is max_dim."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

# 2. Convert to grayscale
def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# 3. Normalize intensity
def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """Stretch intensity to full [0,255] range."""
    min_val, max_val = np.min(img), np.max(img)
    if max_val > min_val:
        norm = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        norm = img
    return norm

# 4. Denoising
def denoise_image(img: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Denoise image using bilateral or Gaussian filter."""
    if method == "bilateral":
        return cv2.bilateralFilter(img, 9, 75, 75)
    else:
        return cv2.GaussianBlur(img, (5, 5), 0)

# 5. Thresholding
def threshold_image(img: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Apply Otsu or adaptive thresholding."""
    if method == "otsu":
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 2)
    return th

# 6. Find IC body using contour detection
def find_ic_body(img: np.ndarray, threshold: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], np.ndarray]:
    """Find largest rectangle contour (IC body) and create mask."""
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_rect = None
    
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        area = w * h
        # Look for rectangular shapes that are relatively large
        if area > max_area and area > (img.shape[0] * img.shape[1] * 0.1):
            max_area = area
            best_rect = rect
    
    mask = np.zeros_like(img)
    if best_rect:
        x, y, w, h = best_rect
        # Shrink the body mask slightly to keep pins on edges
        margin = int(min(w, h) * 0.05)
        cv2.rectangle(mask, (x + margin, y + margin), (x + w - margin, y + h - margin), 255, -1)
    
    return best_rect, mask

# 7. Enhance pin edges
def enhance_edges(img: np.ndarray, method: str = "canny") -> np.ndarray:
    """Enhance pin edges using Canny edge detection."""
    if method == "canny":
        v = np.median(img)
        lower = int(max(0, 0.5 * v))
        upper = int(min(255, 1.5 * v))
        edges = cv2.Canny(img, lower, upper)
    elif method == "sobel":
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges / edges.max() * 255)
    else:
        edges = img
    return edges

# 8. Extract and filter pin contours
def extract_and_filter_pins(edges: np.ndarray, body_rect: Optional[Tuple[int, int, int, int]], 
                             original_shape: Tuple[int, int]) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """Find contours and filter to keep only pin-like shapes."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pins = []
    pin_boxes = []
    
    h, w = original_shape[:2]
    min_pin_length = min(h, w) * 0.02  # Minimum 2% of image dimension
    max_pin_length = min(h, w) * 0.15  # Maximum 15% of image dimension
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Skip if no area
        if area < 10:
            continue
        
        # Calculate aspect ratio - pins are elongated
        aspect_ratio = max(cw, ch) / (min(cw, ch) + 1e-5)
        
        # Calculate length (longest dimension)
        length = max(cw, ch)
        
        # Calculate solidity (how filled the contour is)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-5)
        
        # Filter criteria for pins:
        # 1. Elongated shape (high aspect ratio)
        # 2. Reasonable size
        # 3. Good solidity (not too irregular)
        if (aspect_ratio > 2.0 and 
            min_pin_length < length < max_pin_length and
            solidity > 0.5):
            
            # Check if pin is on the edge of IC body
            if body_rect:
                bx, by, bw, bh = body_rect
                # Pin should be near the IC body edges
                on_edge = (x < bx + bw * 0.1 or x + cw > bx + bw * 0.9 or
                          y < by + bh * 0.1 or y + ch > by + bh * 0.9)
                if on_edge or body_rect is None:
                    pins.append(cnt)
                    pin_boxes.append((x, y, cw, ch))
            else:
                pins.append(cnt)
                pin_boxes.append((x, y, cw, ch))
    
    return pins, pin_boxes

# 9. Group pins by orientation
def group_pins_by_side(pin_boxes: List[Tuple[int, int, int, int]], body_rect: Optional[Tuple[int, int, int, int]]) -> Dict[str, List]:
    """Group pins by which side of the chip they're on."""
    if not body_rect:
        return {"all": pin_boxes}
    
    bx, by, bw, bh = body_rect
    cx, cy = bx + bw // 2, by + bh // 2
    
    sides = {"left": [], "right": [], "top": [], "bottom": []}
    
    for box in pin_boxes:
        x, y, w, h = box
        px, py = x + w // 2, y + h // 2
        
        # Determine which side
        if px < cx - bw * 0.3:
            sides["left"].append(box)
        elif px > cx + bw * 0.3:
            sides["right"].append(box)
        elif py < cy - bh * 0.3:
            sides["top"].append(box)
        elif py > cy + bh * 0.3:
            sides["bottom"].append(box)
    
    return sides

# 10. Full pipeline
def count_ic_pins_opencv(image_path: str, debug_dir: str = "debug") -> Dict[str, Any]:
    """
    Full pipeline using only OpenCV: preprocess image and count pins.
    Returns pin count, bounding boxes, and debug visualizations.
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(debug_dir, exist_ok=True)
    
    # Step 1: Load and preprocess
    img = load_and_resize_image(image_path)
    save_debug_image(img, "01_original", debug_dir, base_name)
    
    gray = to_grayscale(img)
    save_debug_image(gray, "02_gray", debug_dir, base_name)
    
    norm = normalize_intensity(gray)
    save_debug_image(norm, "03_normalized", debug_dir, base_name)
    
    denoised = denoise_image(norm, "bilateral")
    save_debug_image(denoised, "04_denoised", debug_dir, base_name)
    
    # Step 2: Threshold
    thresh = threshold_image(denoised, "otsu")
    save_debug_image(thresh, "05_threshold", debug_dir, base_name)
    
    # Step 3: Find IC body
    body_rect, body_mask = find_ic_body(gray, thresh)
    if body_rect:
        body_vis = img.copy()
        x, y, w, h = body_rect
        cv2.rectangle(body_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        save_debug_image(body_vis, "06_body_detected", debug_dir, base_name)
    
    # Step 4: Edge detection
    edges = enhance_edges(denoised, "canny")
    save_debug_image(edges, "07_edges", debug_dir, base_name)
    
    # Step 5: Extract and filter pins
    pins, pin_boxes = extract_and_filter_pins(edges, body_rect, gray.shape)
    
    # Step 6: Visualize detected pins
    pin_vis = img.copy()
    for box in pin_boxes:
        x, y, w, h = box
        cv2.rectangle(pin_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    save_debug_image(pin_vis, "08_pins_detected", debug_dir, base_name)
    
    # Step 7: Group by side
    sides = group_pins_by_side(pin_boxes, body_rect)
    
    # Step 8: Create summary visualization
    summary = img.copy()
    if body_rect:
        x, y, w, h = body_rect
        cv2.rectangle(summary, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    for box in pin_boxes:
        x, y, w, h = box
        cv2.rectangle(summary, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Add text with pin count
    pin_count = len(pin_boxes)
    cv2.putText(summary, f"Total Pins: {pin_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    for i, (side, boxes) in enumerate(sides.items()):
        if boxes:
            y_pos = 60 + i * 30
            cv2.putText(summary, f"{side.capitalize()}: {len(boxes)}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    save_debug_image(summary, "09_final_summary", debug_dir, base_name)
    
    return {
        "pin_count": pin_count,
        "pin_boxes": pin_boxes,
        "pins_by_side": sides,
        "body_rect": body_rect,
        "summary_image": summary,
        "debug_dir": debug_dir
    }

def main():
    parser = argparse.ArgumentParser(description="IC Pin Counting Pipeline (OpenCV)")
    parser.add_argument("image_path", type=str, help="Path to input IC chip image")
    parser.add_argument("--debug_dir", type=str, default="debug", help="Directory for debug images")
    args = parser.parse_args()
    
    print(f"Processing: {args.image_path}")
    result = count_ic_pins_opencv(args.image_path, args.debug_dir)
    
    print(f"\n{'='*50}")
    print(f"RESULTS FOR: {os.path.basename(args.image_path)}")
    print(f"{'='*50}")
    print(f"Total Pin Count: {result['pin_count']}")
    print(f"\nPins by side:")
    for side, boxes in result['pins_by_side'].items():
        if boxes:
            print(f"  {side.capitalize():>10}: {len(boxes)} pins")
    print(f"\nDebug images saved in: {result['debug_dir']}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
