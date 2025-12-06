"""
IC Chip Detection and Preprocessing Module - Improved Version
Handles boundary detection, cropping, preprocessing, and center removal.
"""

import cv2
import numpy as np
from typing import Tuple, Dict
from pathlib import Path

from config import config


class ICDetector:
    """
    Main class for IC chip detection and preprocessing pipeline.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config

    def detect_boundary_improved(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Improved boundary detection that finds the OUTER boundary of IC including pins.
        Uses multiple strategies and picks the best one.

        Args:
            image: Input RGB/BGR image

        Returns:
            Tuple of (annotated_image, bounding_box)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        image_area = h * w

        # Strategy 1: Use Otsu thresholding to find chip against background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Strategy 2: Find the largest dark region (IC body is typically dark)
        # Invert if needed - IC bodies are usually dark
        mean_val = np.mean(gray)
        if mean_val > 127:
            # Light background, dark chip
            _, binary_dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Dark background, might be different
            _, binary_dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations to connect chip body with pins
        kernel_large = np.ones((15, 15), np.uint8)
        closed = cv2.morphologyEx(binary_dark, cv2.MORPH_CLOSE, kernel_large)

        # Find contours on the closed image
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image.copy(), (0, 0, w, h)

        # Find the largest contour that could be the IC
        best_contour = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, cw, ch = cv2.boundingRect(cnt)

            # Score based on: area, centrality, and reasonable aspect ratio
            centrality = 1 - (abs(x + cw/2 - w/2) / w + abs(y + ch/2 - h/2) / h)
            area_ratio = area / image_area
            aspect = max(cw, ch) / (min(cw, ch) + 1)

            # Good IC should be reasonably sized (10-90% of image), centered, reasonable aspect
            if 0.08 < area_ratio < 0.95 and aspect < 4:
                score = area_ratio * centrality * (1 / (1 + abs(aspect - 1) * 0.1))
                if score > best_score:
                    best_score = score
                    best_contour = cnt

        if best_contour is None:
            best_contour = max(contours, key=cv2.contourArea)

        x, y, bw, bh = cv2.boundingRect(best_contour)

        # Annotate
        annotated = image.copy()
        cv2.drawContours(annotated, [best_contour], -1, self.cfg.BOUNDARY_COLOR, 2)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

        return annotated, (x, y, bw, bh)

    def detect_boundary_simple(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Simple boundary detection using edge-based approach.
        Falls back to using most of the image if detection fails.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        # Use edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Dilate to connect edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=3)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get all points from all contours
            all_points = np.vstack(contours)
            x, y, bw, bh = cv2.boundingRect(all_points)

            # Add small margin
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            bw = min(w - x, bw + 2 * margin)
            bh = min(h - y, bh + 2 * margin)
        else:
            # Use full image with small margins
            margin = int(min(h, w) * 0.02)
            x, y = margin, margin
            bw, bh = w - 2*margin, h - 2*margin

        annotated = image.copy()
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        return annotated, (x, y, bw, bh)

    def crop_with_padding(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop image to bounding box with padding."""
        x, y, w, h = bbox
        padding = self.cfg.BOUNDARY_PADDING
        img_h, img_w = image.shape[:2]

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        return image[y1:y2, x1:x2].copy()

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the cropped IC image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if self.cfg.APPLY_DENOISING:
            gray = cv2.bilateralFilter(gray, self.cfg.BILATERAL_D,
                                       self.cfg.BILATERAL_SIGMA_COLOR,
                                       self.cfg.BILATERAL_SIGMA_SPACE)

        # Resize to target dimensions
        gray = cv2.resize(gray, self.cfg.TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        normalized = gray.astype(np.float32) / 255.0
        return normalized, gray

    def create_center_mask(self, image_shape: Tuple[int, int], ratio: float = None) -> np.ndarray:
        """Create mask to remove center die area."""
        ratio = ratio if ratio is not None else self.cfg.CENTER_REMOVAL_RATIO
        h, w = image_shape

        center_h = int(h * ratio)
        center_w = int(w * ratio)
        margin_h = (h - center_h) // 2
        margin_w = (w - center_w) // 2

        mask = np.ones((h, w), dtype=np.uint8)
        mask[margin_h:margin_h + center_h, margin_w:margin_w + center_w] = 0

        return mask

    def remove_center(self, image: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply center mask to isolate pin regions."""
        if mask is None:
            mask = self.create_center_mask(image.shape[:2])

        if image.dtype == np.float32:
            pins_only = image * mask.astype(np.float32)
        else:
            pins_only = cv2.bitwise_and(image, image, mask=mask)

        return pins_only, mask

    def detect_package_type(self, cropped_shape: Tuple[int, int]) -> str:
        """
        Detect package type based on aspect ratio of cropped image.

        Args:
            cropped_shape: (height, width) of cropped image

        Returns:
            'DIP' for dual in-line, 'QFP' for quad flat package
        """
        h, w = cropped_shape
        aspect = w / h

        # DIP packages are elongated (width >> height or height >> width)
        # Using 1.35 threshold to account for padding effects on aspect ratio
        if aspect > 1.35 or aspect < 0.74:
            return 'DIP'
        return 'QFP'

    def process_image(self, image_path: str, save_intermediates: bool = True) -> Dict:
        """
        Run the full detection and preprocessing pipeline.

        Args:
            image_path: Path to input image
            save_intermediates: Whether to save intermediate visualizations

        Returns:
            Dictionary with all intermediate results
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        results = {'original': image, 'image_path': image_path}

        # Step 1: Detect boundary using simple approach (more reliable)
        boundary_annotated, bbox = self.detect_boundary_simple(image)
        results['boundary_annotated'] = boundary_annotated
        results['bbox'] = bbox

        # Crop with padding
        cropped = self.crop_with_padding(image, bbox)
        results['cropped'] = cropped
        results['cropped_shape'] = cropped.shape[:2]

        # Detect package type from cropped shape
        package_type = self.detect_package_type(cropped.shape[:2])
        results['package_type'] = package_type

        # Preprocess
        normalized, grayscale = self.preprocess(cropped)
        results['normalized'] = normalized
        results['grayscale'] = grayscale

        # Remove center
        pins_only, center_mask = self.remove_center(grayscale)
        results['pins_only'] = pins_only
        results['center_mask'] = center_mask

        if save_intermediates and self.cfg.SAVE_VISUALIZATIONS:
            self._save_intermediates(results, image_path)

        return results

    def _save_intermediates(self, results: Dict, image_path: str):
        """Save intermediate visualization images."""
        filename = Path(image_path).stem
        output_dir = Path(self.cfg.OUTPUT_DIR) / filename
        output_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_dir / "1_original.png"), results['original'])
        cv2.imwrite(str(output_dir / "2_boundary.png"), results['boundary_annotated'])
        cv2.imwrite(str(output_dir / "3_cropped.png"), results['cropped'])
        cv2.imwrite(str(output_dir / "4_grayscale.png"), results['grayscale'])
        cv2.imwrite(str(output_dir / "5_center_mask.png"), results['center_mask'] * 255)
        cv2.imwrite(str(output_dir / "6_pins_only.png"), results['pins_only'])


if __name__ == "__main__":
    detector = ICDetector()
    test_images = ["ic_test/Pi7compressed008.png"]

    for img_path in test_images:
        print(f"\nProcessing: {img_path}")
        results = detector.process_image(img_path)
        print(f"  Cropped shape: {results['cropped_shape']}")
        print(f"  Package type: {results['package_type']}")
