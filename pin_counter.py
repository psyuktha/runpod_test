"""
Pin Counting Module - Improved Version 2
Implements multiple approaches for counting IC pins with better accuracy.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict
from pathlib import Path
import os

from config import config


class ProjectionPinCounter:
    """
    Pin counter using edge projection profiles.
    Most reliable for clean IC images.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config

    def count_pins(self, grayscale: np.ndarray, package_type: str = 'QFP', dip_orientation: str = 'landscape') -> Tuple[int, Dict, np.ndarray]:
        """
        Count pins using edge projection analysis on each side.

        Args:
            grayscale: Grayscale image
            package_type: 'DIP' or 'QFP'
            dip_orientation: For DIP packages, 'landscape' (pins top/bottom) or 'portrait' (pins left/right)
        """
        if grayscale.dtype == np.float32:
            grayscale = (grayscale * 255).astype(np.uint8)

        h, w = grayscale.shape
        annotated = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(grayscale)

        # Multi-scale edge detection for better pin edge capture
        edges1 = cv2.Canny(enhanced, 30, 90)
        edges2 = cv2.Canny(enhanced, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)

        # Define regions - look at outer 18% of each edge, skip corners
        edge_depth = int(min(h, w) * 0.18)
        corner_skip = int(min(h, w) * 0.12)

        side_counts = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}

        if package_type == 'DIP':
            # DIP packages have pins on 2 opposite sides based on orientation
            if dip_orientation == 'landscape':
                # Landscape orientation - pins on TOP and BOTTOM
                # TOP edge
                top_roi = edges[0:edge_depth, corner_skip:w - corner_skip]
                side_counts['top'], top_peaks = self._analyze_projection(top_roi, 'horizontal')
                cv2.rectangle(annotated, (corner_skip, 0), (w - corner_skip, edge_depth), (255, 0, 0), 1)
                self._draw_peaks(annotated, top_peaks, 'top', corner_skip, edge_depth)

                # BOTTOM edge
                bottom_roi = edges[h - edge_depth:h, corner_skip:w - corner_skip]
                side_counts['bottom'], bottom_peaks = self._analyze_projection(bottom_roi, 'horizontal')
                cv2.rectangle(annotated, (corner_skip, h - edge_depth), (w - corner_skip, h), (255, 0, 0), 1)
                self._draw_peaks(annotated, bottom_peaks, 'bottom', corner_skip, h - edge_depth // 2)
            else:
                # Portrait orientation - pins on LEFT and RIGHT
                # LEFT edge
                left_roi = edges[corner_skip:h - corner_skip, 0:edge_depth]
                side_counts['left'], left_peaks = self._analyze_projection(left_roi, 'vertical')
                cv2.rectangle(annotated, (0, corner_skip), (edge_depth, h - corner_skip), (0, 255, 0), 1)

                # RIGHT edge
                right_roi = edges[corner_skip:h - corner_skip, w - edge_depth:w]
                side_counts['right'], right_peaks = self._analyze_projection(right_roi, 'vertical')
                cv2.rectangle(annotated, (w - edge_depth, corner_skip), (w, h - corner_skip), (0, 255, 0), 1)
        else:
            # QFP packages have pins on all 4 sides
            # TOP edge
            top_roi = edges[0:edge_depth, corner_skip:w - corner_skip]
            side_counts['top'], top_peaks = self._analyze_projection(top_roi, 'horizontal')
            cv2.rectangle(annotated, (corner_skip, 0), (w - corner_skip, edge_depth), (255, 0, 0), 1)
            self._draw_peaks(annotated, top_peaks, 'top', corner_skip, edge_depth)

            # BOTTOM edge
            bottom_roi = edges[h - edge_depth:h, corner_skip:w - corner_skip]
            side_counts['bottom'], bottom_peaks = self._analyze_projection(bottom_roi, 'horizontal')
            cv2.rectangle(annotated, (corner_skip, h - edge_depth), (w - corner_skip, h), (255, 0, 0), 1)
            self._draw_peaks(annotated, bottom_peaks, 'bottom', corner_skip, h - edge_depth // 2)

            # LEFT edge
            left_roi = edges[corner_skip:h - corner_skip, 0:edge_depth]
            side_counts['left'], left_peaks = self._analyze_projection(left_roi, 'vertical')
            cv2.rectangle(annotated, (0, corner_skip), (edge_depth, h - corner_skip), (0, 255, 0), 1)

            # RIGHT edge
            right_roi = edges[corner_skip:h - corner_skip, w - edge_depth:w]
            side_counts['right'], right_peaks = self._analyze_projection(right_roi, 'vertical')
            cv2.rectangle(annotated, (w - edge_depth, corner_skip), (w, h - corner_skip), (0, 255, 0), 1)

        total = sum(side_counts.values())

        # Add labels
        self._add_labels(annotated, side_counts, package_type, total)

        details = {'side_counts': side_counts, 'edges': edges}
        return total, details, annotated

    def _analyze_projection(self, roi: np.ndarray, orientation: str) -> Tuple[int, List[int]]:
        """
        Analyze edge projection to count pins.

        Args:
            roi: Region of interest (edge image)
            orientation: 'horizontal' or 'vertical'

        Returns:
            Tuple of (count, peak_positions)
        """
        if roi.size == 0:
            return 0, []

        # Sum edges along appropriate axis
        if orientation == 'horizontal':
            projection = np.sum(roi, axis=0).astype(np.float32)
        else:
            projection = np.sum(roi, axis=1).astype(np.float32)

        if np.max(projection) == 0:
            return 0, []

        # Normalize
        projection = projection / np.max(projection)

        # Smooth to reduce noise but preserve peaks
        kernel_size = max(3, len(projection) // 60)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = cv2.GaussianBlur(projection.reshape(1, -1), (kernel_size, 1), 0).flatten()

        # Dynamic threshold based on signal strength
        threshold = max(0.1, np.mean(smoothed) + 0.5 * np.std(smoothed))

        # Find peaks with adaptive minimum distance
        # IC pins are typically evenly spaced
        min_dist = max(4, len(projection) // 80)

        peaks = self._find_peaks_adaptive(smoothed, threshold, min_dist)

        return len(peaks), peaks

    def _find_peaks_adaptive(self, signal: np.ndarray, threshold: float, min_dist: int) -> List[int]:
        """Find peaks with adaptive parameters."""
        peaks = []
        n = len(signal)

        for i in range(1, n - 1):
            if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                if signal[i] >= threshold:
                    if not peaks or (i - peaks[-1]) >= min_dist:
                        peaks.append(i)

        return peaks

    def _draw_peaks(self, image: np.ndarray, peaks: List[int], side: str, offset: int, y_pos: int):
        """Draw detected peak positions on image."""
        for p in peaks:
            if side in ['top', 'bottom']:
                cv2.circle(image, (offset + p, y_pos), 2, (0, 255, 255), -1)

    def _add_labels(self, image: np.ndarray, counts: Dict, package_type: str, total: int):
        """Add count labels to image."""
        h, w = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(image, f"T:{counts['top']}", (w // 2 - 25, 20), font, 0.5, (0, 255, 255), 1)
        cv2.putText(image, f"B:{counts['bottom']}", (w // 2 - 25, h - 5), font, 0.5, (0, 255, 255), 1)

        if package_type == 'QFP':
            cv2.putText(image, f"L:{counts['left']}", (5, h // 2), font, 0.5, (0, 255, 255), 1)
            cv2.putText(image, f"R:{counts['right']}", (w - 40, h // 2), font, 0.5, (0, 255, 255), 1)

        cv2.putText(image, f"Total: {total}", (w // 2 - 40, h // 2), font, 0.7, (0, 0, 255), 2)


class ThresholdPinCounter:
    """
    Pin counter using threshold-based detection.
    Works well when pins have distinct brightness from chip body (QFN, bright pin packages).
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config

    def count_pins(self, grayscale: np.ndarray, package_type: str = 'QFP', dip_orientation: str = 'landscape') -> Tuple[int, Dict, np.ndarray]:
        """Count pins using thresholding and connected components."""
        if grayscale.dtype == np.float32:
            grayscale = (grayscale * 255).astype(np.uint8)

        h, w = grayscale.shape
        annotated = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

        # Enhance contrast first
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(grayscale)

        # Use Otsu's threshold - pins are typically bright (metallic)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup - close small gaps, remove noise
        kernel_close = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # Define edge regions - slightly wider for better pin capture
        edge_depth = int(min(h, w) * 0.22)
        margin = int(min(h, w) * 0.08)

        # Create edge-only mask based on package type
        mask = np.zeros((h, w), dtype=np.uint8)
        if package_type == 'DIP':
            # DIP packages have pins on 2 opposite sides based on orientation
            if dip_orientation == 'landscape':
                # Landscape - pins on top/bottom
                mask[0:edge_depth, margin:w - margin] = 255  # Top
                mask[h - edge_depth:h, margin:w - margin] = 255  # Bottom
            else:
                # Portrait - pins on left/right
                mask[margin:h - margin, 0:edge_depth] = 255  # Left
                mask[margin:h - margin, w - edge_depth:w] = 255  # Right
        else:
            # QFP packages have pins on all 4 sides
            mask[0:edge_depth, margin:w - margin] = 255  # Top
            mask[h - edge_depth:h, margin:w - margin] = 255  # Bottom
            mask[margin:h - margin, 0:edge_depth] = 255  # Left
            mask[margin:h - margin, w - edge_depth:w] = 255  # Right

        masked = cv2.bitwise_and(binary, mask)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masked)

        # Filter components by size and classify by location
        side_counts = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}

        # Adaptive area thresholds based on image size
        total_edge_area = edge_depth * (w - 2 * margin) * 2  # Approx area for pins
        if package_type == 'QFP':
            total_edge_area += edge_depth * (h - 2 * margin) * 2

        # Expected pin area range (assuming 5-20 pins per side roughly)
        expected_pins = 12  # rough average
        avg_pin_area = total_edge_area / (expected_pins * (4 if package_type == 'QFP' else 2))
        min_area = max(15, int(avg_pin_area * 0.1))
        max_area = min(5000, int(avg_pin_area * 5))

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area < area < max_area:
                cx, cy = int(centroids[i][0]), int(centroids[i][1])

                if cy < edge_depth and margin < cx < w - margin:
                    side_counts['top'] += 1
                elif cy > h - edge_depth and margin < cx < w - margin:
                    side_counts['bottom'] += 1
                elif cx < edge_depth and margin < cy < h - margin and package_type == 'QFP':
                    side_counts['left'] += 1
                elif cx > w - edge_depth and margin < cy < h - margin and package_type == 'QFP':
                    side_counts['right'] += 1

        total = sum(side_counts.values())
        cv2.putText(annotated, f"Total: {total}", (w // 2 - 40, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return total, {'side_counts': side_counts, 'binary': binary}, annotated


class GradientPinCounter:
    """
    Pin counter using gradient analysis.
    Pins create strong gradient responses at their edges.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config

    def count_pins(self, grayscale: np.ndarray, package_type: str = 'QFP', dip_orientation: str = 'landscape') -> Tuple[int, Dict, np.ndarray]:
        """Count pins using gradient-based detection."""
        if grayscale.dtype == np.float32:
            grayscale = (grayscale * 255).astype(np.uint8)

        h, w = grayscale.shape
        annotated = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

        # Enhance contrast first
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(grayscale)

        # Compute Sobel gradients for horizontal and vertical edges
        grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)  # Vertical edges
        grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)  # Horizontal edges

        # Superimpose: combine both gradients using magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_magnitude = (grad_magnitude / grad_magnitude.max() * 255).astype(np.uint8)

        # Edge regions - adjusted for better pin capture
        edge_depth = int(min(h, w) * 0.20)
        margin = int(min(h, w) * 0.08)  # Slightly smaller margin to catch edge pins

        side_counts = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}

        if package_type == 'DIP':
            # DIP packages have pins on 2 opposite sides based on orientation
            if dip_orientation == 'landscape':
                # Landscape - pins on top/bottom
                # TOP - use vertical gradient (captures horizontal pin edges)
                top_roi = np.abs(grad_y[0:edge_depth, margin:w - margin])
                side_counts['top'] = self._count_from_gradient(top_roi, 'horizontal', h, w, package_type)

                # BOTTOM
                bottom_roi = np.abs(grad_y[h - edge_depth:h, margin:w - margin])
                side_counts['bottom'] = self._count_from_gradient(bottom_roi, 'horizontal', h, w, package_type)
            else:
                # Portrait - pins on left/right
                # LEFT - use horizontal gradient (captures vertical pin edges)
                left_roi = np.abs(grad_x[margin:h - margin, 0:edge_depth])
                side_counts['left'] = self._count_from_gradient(left_roi, 'vertical', h, w, package_type)

                # RIGHT
                right_roi = np.abs(grad_x[margin:h - margin, w - edge_depth:w])
                side_counts['right'] = self._count_from_gradient(right_roi, 'vertical', h, w, package_type)
        else:
            # QFP packages have pins on all 4 sides
            # TOP - use vertical gradient (captures horizontal pin edges)
            top_roi = np.abs(grad_y[0:edge_depth, margin:w - margin])
            side_counts['top'] = self._count_from_gradient(top_roi, 'horizontal', h, w, package_type)

            # BOTTOM
            bottom_roi = np.abs(grad_y[h - edge_depth:h, margin:w - margin])
            side_counts['bottom'] = self._count_from_gradient(bottom_roi, 'horizontal', h, w, package_type)

            # LEFT - use horizontal gradient (captures vertical pin edges)
            left_roi = np.abs(grad_x[margin:h - margin, 0:edge_depth])
            side_counts['left'] = self._count_from_gradient(left_roi, 'vertical', h, w, package_type)

            # RIGHT
            right_roi = np.abs(grad_x[margin:h - margin, w - edge_depth:w])
            side_counts['right'] = self._count_from_gradient(right_roi, 'vertical', h, w, package_type)

        total = sum(side_counts.values())

        # Add annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated, f"T:{side_counts['top']}", (w//2 - 25, 20), font, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated, f"B:{side_counts['bottom']}", (w//2 - 25, h - 5), font, 0.5, (0, 255, 255), 1)
        if package_type == 'QFP':
            cv2.putText(annotated, f"L:{side_counts['left']}", (5, h//2), font, 0.5, (0, 255, 255), 1)
            cv2.putText(annotated, f"R:{side_counts['right']}", (w - 40, h//2), font, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated, f"Total: {total}", (w // 2 - 40, h // 2), font, 0.7, (255, 255, 0), 2)

        return total, {'side_counts': side_counts}, annotated

    def _count_from_gradient(self, roi: np.ndarray, orientation: str, img_h: int, img_w: int, package_type: str = 'QFP') -> int:
        """Count pins from gradient ROI using adaptive peak detection."""
        if roi.size == 0:
            return 0

        if orientation == 'horizontal':
            projection = np.sum(roi, axis=0)
        else:
            projection = np.sum(roi, axis=1)

        if np.max(projection) == 0:
            return 0

        projection = projection.astype(np.float32) / np.max(projection)

        # Adaptive parameters based on package type
        # DIP packages typically have 7-20 pins per side with wider spacing
        # QFP packages typically have 10-50 pins per side with tighter spacing
        if package_type == 'DIP':
            expected_pins = 8  # typical DIP has 7-10 pins per side
            min_dist_factor = 0.55  # Wider spacing to prevent double-counting
            threshold_factor = 0.3  # Moderate threshold
        else:
            expected_pins = 16  # typical QFP
            min_dist_factor = 0.4
            threshold_factor = 0.3

        ideal_spacing = len(projection) / expected_pins
        # Stronger smoothing for cleaner peak detection
        k = max(5, int(ideal_spacing / 3))
        if k % 2 == 0:
            k += 1
        k = min(k, 21)  # Allow larger smoothing kernel
        smoothed = cv2.GaussianBlur(projection.reshape(1, -1), (k, 1), 0).flatten()

        # Adaptive threshold
        threshold = max(0.15, np.mean(smoothed) + threshold_factor * np.std(smoothed))

        # Minimum distance between pins - adaptive based on expected spacing
        min_dist = max(5, int(ideal_spacing * min_dist_factor))

        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                if smoothed[i] >= threshold:
                    if not peaks or (i - peaks[-1]) >= min_dist:
                        peaks.append(i)

        return len(peaks)


class PinCounter:
    """
    Main pin counter combining multiple methods.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config
        self.projection = ProjectionPinCounter(cfg)
        self.threshold = ThresholdPinCounter(cfg)
        self.gradient = GradientPinCounter(cfg)

    def count_pins(self, grayscale: np.ndarray, pins_only: np.ndarray,
                   package_type: str = 'QFP', method: str = 'projection',
                   cropped_shape: Tuple[int, int] = None) -> Tuple[int, Dict, np.ndarray]:
        """
        Count pins using specified method.

        Args:
            grayscale: Preprocessed grayscale image
            pins_only: Image with center removed (optional)
            package_type: 'DIP' or 'QFP'
            method: 'projection', 'threshold', 'gradient', or 'ensemble'
            cropped_shape: Original cropped shape before resize (height, width)

        Returns:
            Tuple of (pin_count, method_results, annotated_image)
        """
        results = {}

        # Determine DIP orientation from original cropped shape
        # 'landscape' = pins on top/bottom, 'portrait' = pins on left/right
        dip_orientation = 'landscape'  # default
        if cropped_shape is not None and package_type == 'DIP':
            orig_h, orig_w = cropped_shape
            dip_orientation = 'landscape' if orig_w > orig_h else 'portrait'

        if method == 'projection' or method == 'ensemble':
            count, details, annotated = self.projection.count_pins(grayscale, package_type, dip_orientation)
            results['projection'] = {'count': count, 'details': details, 'annotated': annotated}

        if method == 'threshold' or method == 'ensemble':
            count, details, annotated = self.threshold.count_pins(grayscale, package_type, dip_orientation)
            results['threshold'] = {'count': count, 'details': details, 'annotated': annotated}

        if method == 'gradient' or method == 'ensemble':
            count, details, annotated = self.gradient.count_pins(grayscale, package_type, dip_orientation)
            results['gradient'] = {'count': count, 'details': details, 'annotated': annotated}

        if method == 'ensemble':
            # Use gradient method as primary - it gives most consistent results
            # But verify with other methods for outlier detection
            proj_count = results['projection']['count']
            thresh_count = results['threshold']['count']
            grad_count = results['gradient']['count']

            # Gradient is generally the most reliable, use it as baseline
            final_count = grad_count

            annotated = results['projection']['annotated']
        else:
            final_count = results[method]['count']
            annotated = results[method]['annotated']

        return final_count, results, annotated


def test_pin_counter():
    """Test pin counting."""
    from ic_detector import ICDetector

    detector = ICDetector()
    counter = PinCounter()

    test_images = ["uncertain/anu1.jpeg", "uncertain/anu2.jpeg", "uncertain/new1.png"]
    ground_truth = [64, 56, 14]

    for img_path, gt in zip(test_images, ground_truth):
        if os.path.exists(img_path):
            print(f"\n{'=' * 50}")
            print(f"Testing: {img_path} (GT: {gt})")

            results = detector.process_image(img_path, save_intermediates=True)

            count, details, annotated = counter.count_pins(
                results['grayscale'],
                results['pins_only'],
                results['package_type'],
                method='ensemble'
            )

            print(f"Package Type: {results['package_type']}")
            print(f"Cropped Shape: {results['cropped_shape']}")
            print(f"Method counts:")
            for m, d in details.items():
                sc = d.get('details', {}).get('side_counts', {})
                print(f"  {m}: {d['count']} {sc}")
            print(f"Final: {count} | GT: {gt} | {'CORRECT' if count == gt else f'OFF BY {abs(count-gt)}'}")

            filename = Path(img_path).stem
            cv2.imwrite(f"results/{filename}_pins.png", annotated)


if __name__ == "__main__":
    test_pin_counter()
