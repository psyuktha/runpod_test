"""
IC Chip Pin Detection and Counting Pipeline

PART 1: Image Preprocessing Functions
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import os

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
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Normalize intensity
def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """Stretch intensity to full [0,255] range."""
    min_val, max_val = np.min(img), np.max(img)
    norm = ((img - min_val) / (max_val - min_val + 1e-5) * 255).astype(np.uint8)
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
def find_ic_body(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find largest rectangle contour (IC body) and create mask."""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_rect = None
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        area = w * h
        if area > max_area:
            max_area = area
            best_rect = rect
    mask = np.zeros_like(img)
    if best_rect:
        x, y, w, h = best_rect
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask, best_rect

# 7. Mask/remove IC body
def remove_ic_body(img: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    """Remove IC body from image, keep only pins."""
    pins_only = cv2.bitwise_and(img, cv2.bitwise_not(body_mask))
    return pins_only

# 8. Remove text markings
def remove_text(img: np.ndarray) -> np.ndarray:
    """Detect and mask text-like regions using MSER and morphological closing."""
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(img)
    mask = np.zeros_like(img)
    for p in regions:
        hull = cv2.convexHull(p.reshape(-1, 1, 2))
        cv2.drawContours(mask, [hull], -1, 255, -1)
    # Morphological closing to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Mask out text
    no_text = cv2.bitwise_and(img, cv2.bitwise_not(mask))
    return no_text

# 9. Enhance pin edges
def enhance_edges(img: np.ndarray, method: str = "canny") -> np.ndarray:
    """Enhance pin edges using Sobel, Scharr, or Canny."""
    if method == "canny":
        v = np.median(img)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(img, lower, upper)
    elif method == "sobel":
        edges = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3) + cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
    else:
        edges = cv2.Scharr(img, cv2.CV_8U, 1, 0) + cv2.Scharr(img, cv2.CV_8U, 0, 1)
    return edges

# 10. Extract pin candidate contours
def extract_pin_contours(edges: np.ndarray) -> List[np.ndarray]:
    """Find contours from edge image."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 11. Filter pin contours
def filter_pin_contours(contours: List[np.ndarray], min_ar: float = 2.0, min_area: int = 10, max_area: int = 500) -> List[np.ndarray]:
    """Filter contours by aspect ratio, area, elongation."""
    pins = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        ar = max(w, h) / (min(w, h) + 1e-5)
        elong = area / (w * h + 1e-5)
        if ar > min_ar and min_area < area < max_area and elong < 0.7:
            pins.append(cnt)
    return pins

# 12. Create binary image with only pins
def create_pin_mask(img: np.ndarray, pin_contours: List[np.ndarray]) -> np.ndarray:
    """Create binary image with only pin contours."""
    mask = np.zeros_like(img)
    cv2.drawContours(mask, pin_contours, -1, 255, -1)
    return mask

# 13. Pipeline for preprocessing
def preprocess_ic_image(image_path: str, debug_dir: str = "debug") -> Dict[str, Any]:
    """
    Full preprocessing pipeline. Returns dict of debug images and final pin mask.
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    debug = {}
    img = load_and_resize_image(image_path)
    debug["original"] = img.copy()
    save_debug_image(img, "original", debug_dir, base_name)

    gray = to_grayscale(img)
    debug["gray"] = gray.copy()
    save_debug_image(gray, "gray", debug_dir, base_name)

    norm = normalize_intensity(gray)
    debug["norm"] = norm.copy()
    save_debug_image(norm, "norm", debug_dir, base_name)

    den = denoise_image(norm)
    debug["denoise"] = den.copy()
    save_debug_image(den, "denoise", debug_dir, base_name)

    th = threshold_image(den)
    debug["thresh"] = th.copy()
    save_debug_image(th, "thresh", debug_dir, base_name)

    body_mask, body_rect = find_ic_body(th)
    debug["body_mask"] = body_mask.copy()
    save_debug_image(body_mask, "body_mask", debug_dir, base_name)

    pins_only = remove_ic_body(den, body_mask)
    debug["pins_only"] = pins_only.copy()
    save_debug_image(pins_only, "pins_only", debug_dir, base_name)

    no_text = remove_text(pins_only)
    debug["no_text"] = no_text.copy()
    save_debug_image(no_text, "no_text", debug_dir, base_name)

    edges = enhance_edges(no_text)
    debug["edges"] = edges.copy()
    save_debug_image(edges, "edges", debug_dir, base_name)

    contours = extract_pin_contours(edges)
    debug["contours_img"] = cv2.drawContours(img.copy(), contours, -1, (0,255,0), 1)
    save_debug_image(debug["contours_img"], "contours", debug_dir, base_name)

    pin_contours = filter_pin_contours(contours)
    debug["pin_contours_img"] = cv2.drawContours(img.copy(), pin_contours, -1, (0,0,255), 2)
    save_debug_image(debug["pin_contours_img"], "pin_contours", debug_dir, base_name)

    pin_mask = create_pin_mask(gray, pin_contours)
    debug["pin_mask"] = pin_mask.copy()
    save_debug_image(pin_mask, "pin_mask", debug_dir, base_name)

    return {
        "debug": debug,
        "pin_mask": pin_mask,
        "pin_contours": pin_contours,
        "original": img,
        "base_name": base_name
    }
