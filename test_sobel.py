"""
Test script to visualize Sobel gradient detection for IC pin counting.
Shows all intermediate steps to debug gradient-based pin detection.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from ic_detector import ICDetector
from config import config


def visualize_sobel_detection(image_path: str):
    """
    Visualize Sobel gradient detection step-by-step.
    
    Args:
        image_path: Path to IC chip image
    """
    print(f"\n{'=' * 70}")
    print(f"Testing Sobel Detection on: {image_path}")
    print(f"{'=' * 70}\n")
    
    # Step 1: Load and preprocess image
    detector = ICDetector()
    results = detector.process_image(image_path, save_intermediates=True)
    
    grayscale = results['grayscale']
    if grayscale.dtype == np.float32:
        grayscale = (grayscale * 255).astype(np.uint8)
    
    h, w = grayscale.shape
    print(f"Image size: {w}x{h}")
    print(f"Package type: {results['package_type']}")
    
    # Step 2: Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(grayscale)
    
    # Step 3: Compute Sobel gradients
    grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)  # Vertical edges
    grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)  # Horizontal edges
    
    # Compute magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize for visualization
    grad_x_vis = cv2.convertScaleAbs(grad_x)
    grad_y_vis = cv2.convertScaleAbs(grad_y)
    grad_mag_vis = cv2.convertScaleAbs(grad_magnitude)
    
    print(f"\nGradient Statistics:")
    print(f"  Grad X - Min: {grad_x.min():.2f}, Max: {grad_x.max():.2f}, Mean: {grad_x.mean():.2f}")
    print(f"  Grad Y - Min: {grad_y.min():.2f}, Max: {grad_y.max():.2f}, Mean: {grad_y.mean():.2f}")
    print(f"  Magnitude - Min: {grad_magnitude.min():.2f}, Max: {grad_magnitude.max():.2f}, Mean: {grad_magnitude.mean():.2f}")
    
    # Step 4: Define edge regions
    edge_depth = int(min(h, w) * 0.20)
    margin = int(min(h, w) * 0.08)
    
    print(f"\nEdge Detection Parameters:")
    print(f"  Edge depth: {edge_depth} pixels")
    print(f"  Corner margin: {margin} pixels")
    
    # Step 5: Extract ROIs and analyze
    package_type = results['package_type']
    
    # Create annotated image showing ROIs
    annotated = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    
    # Draw ROI boxes
    cv2.rectangle(annotated, (margin, 0), (w - margin, edge_depth), (255, 0, 0), 2)  # Top
    cv2.rectangle(annotated, (margin, h - edge_depth), (w - margin, h), (255, 0, 0), 2)  # Bottom
    
    if package_type == 'QFP':
        cv2.rectangle(annotated, (0, margin), (edge_depth, h - margin), (0, 255, 0), 2)  # Left
        cv2.rectangle(annotated, (w - edge_depth, margin), (w, h - margin), (0, 255, 0), 2)  # Right
    
    cv2.putText(annotated, "TOP ROI", (w//2 - 40, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated, "BOTTOM ROI", (w//2 - 60, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Extract ROIs for analysis
    top_roi = np.abs(grad_y[0:edge_depth, margin:w - margin])
    bottom_roi = np.abs(grad_y[h - edge_depth:h, margin:w - margin])
    
    print(f"\nROI Statistics:")
    print(f"  Top ROI - Shape: {top_roi.shape}, Mean: {top_roi.mean():.2f}, Max: {top_roi.max():.2f}")
    print(f"  Bottom ROI - Shape: {bottom_roi.shape}, Mean: {bottom_roi.mean():.2f}, Max: {bottom_roi.max():.2f}")
    
    # Step 6: Compute projections
    top_projection = np.sum(top_roi, axis=0).astype(np.float32)
    bottom_projection = np.sum(bottom_roi, axis=0).astype(np.float32)
    
    if np.max(top_projection) > 0:
        top_projection = top_projection / np.max(top_projection)
    if np.max(bottom_projection) > 0:
        bottom_projection = bottom_projection / np.max(bottom_projection)
    
    # Smooth projections
    expected_pins = 8 if package_type == 'DIP' else 16
    ideal_spacing = len(top_projection) / expected_pins
    k = max(5, int(ideal_spacing / 3))
    if k % 2 == 0:
        k += 1
    k = min(k, 21)
    
    top_smoothed = cv2.GaussianBlur(top_projection.reshape(1, -1), (k, 1), 0).flatten()
    bottom_smoothed = cv2.GaussianBlur(bottom_projection.reshape(1, -1), (k, 1), 0).flatten()
    
    # Find peaks
    threshold = max(0.15, np.mean(top_smoothed) + 0.3 * np.std(top_smoothed))
    min_dist = max(5, int(ideal_spacing * 0.55))
    
    top_peaks = find_peaks(top_smoothed, threshold, min_dist)
    bottom_peaks = find_peaks(bottom_smoothed, threshold, min_dist)
    
    print(f"\nPeak Detection:")
    print(f"  Smoothing kernel: {k}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Min distance: {min_dist}")
    print(f"  Top peaks found: {len(top_peaks)}")
    print(f"  Bottom peaks found: {len(bottom_peaks)}")
    
    # Step 7: Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Original processing
    plt.subplot(3, 4, 1)
    plt.imshow(grayscale, cmap='gray')
    plt.title('1. Original Grayscale')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(enhanced, cmap='gray')
    plt.title('2. CLAHE Enhanced')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(annotated)
    plt.title(f'3. ROI Regions ({package_type})')
    plt.axis('off')
    
    # Row 2: Sobel gradients
    plt.subplot(3, 4, 5)
    plt.imshow(grad_x_vis, cmap='gray')
    plt.title('4. Sobel X (Vertical Edges)')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(grad_y_vis, cmap='gray')
    plt.title('5. Sobel Y (Horizontal Edges)')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(grad_mag_vis, cmap='hot')
    plt.title('6. Gradient Magnitude')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(top_roi, cmap='hot')
    plt.title('7. Top ROI (Grad Y)')
    plt.axis('off')
    
    # Row 3: Projections and peaks
    plt.subplot(3, 4, 9)
    plt.plot(top_projection, label='Raw', alpha=0.5)
    plt.plot(top_smoothed, label='Smoothed', linewidth=2)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold:.2f}')
    for p in top_peaks:
        plt.axvline(x=p, color='g', alpha=0.5, linestyle=':')
    plt.title(f'8. Top Projection ({len(top_peaks)} peaks)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 10)
    plt.plot(bottom_projection, label='Raw', alpha=0.5)
    plt.plot(bottom_smoothed, label='Smoothed', linewidth=2)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold:.2f}')
    for p in bottom_peaks:
        plt.axvline(x=p, color='g', alpha=0.5, linestyle=':')
    plt.title(f'9. Bottom Projection ({len(bottom_peaks)} peaks)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Show peak positions on grayscale
    plt.subplot(3, 4, 11)
    peak_viz = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    for p in top_peaks:
        cv2.circle(peak_viz, (margin + p, edge_depth // 2), 3, (0, 255, 0), -1)
    for p in bottom_peaks:
        cv2.circle(peak_viz, (margin + p, h - edge_depth // 2), 3, (0, 0, 255), -1)
    plt.imshow(cv2.cvtColor(peak_viz, cv2.COLOR_BGR2RGB))
    plt.title('10. Detected Peaks on Image')
    plt.axis('off')
    
    # Summary
    plt.subplot(3, 4, 12)
    plt.axis('off')
    summary_text = f"""SOBEL DETECTION SUMMARY

Package Type: {package_type}
Image Size: {w}x{h}

Gradient Stats:
  Max Grad X: {grad_x.max():.1f}
  Max Grad Y: {grad_y.max():.1f}
  Max Magnitude: {grad_magnitude.max():.1f}

Peak Detection:
  Kernel Size: {k}
  Threshold: {threshold:.3f}
  Min Distance: {min_dist}

Results:
  Top Pins: {len(top_peaks)}
  Bottom Pins: {len(bottom_peaks)}
  TOTAL: {len(top_peaks) + len(bottom_peaks)}
"""
    plt.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(config.OUTPUT_DIR) / Path(image_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sobel_debug.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.show()
    
    return {
        'top_peaks': len(top_peaks),
        'bottom_peaks': len(bottom_peaks),
        'total': len(top_peaks) + len(bottom_peaks),
        'grad_x_max': grad_x.max(),
        'grad_y_max': grad_y.max(),
        'grad_mag_max': grad_magnitude.max()
    }


def find_peaks(signal, threshold, min_dist):
    """Simple peak finding."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if signal[i] >= threshold:
                if not peaks or (i - peaks[-1]) >= min_dist:
                    peaks.append(i)
    return peaks


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "ic_test/Pi7compressed008.png"
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        print("\nUsage: python test_sobel.py <image_path>")
        print("Example: python test_sobel.py ic_test/Pi7compressed008.png")
        sys.exit(1)
    
    result = visualize_sobel_detection(image_path)
    
    print(f"\n{'=' * 70}")
    print("FINAL RESULT:")
    print(f"  Total pins detected: {result['total']}")
    print(f"  Top: {result['top_peaks']}, Bottom: {result['bottom_peaks']}")
    print(f"{'=' * 70}\n")
