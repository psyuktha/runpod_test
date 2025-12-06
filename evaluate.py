"""
Evaluation Module
Tests the IC pin counting pipeline on all images and generates accuracy reports.
"""

import cv2
import numpy as np
from pathlib import Path
import os
from typing import Dict, List, Tuple
import json
from datetime import datetime

from config import config
from ic_detector import ICDetector
from pin_counter import PinCounter


class Evaluator:
    """
    Evaluates pin counting accuracy across multiple images.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config
        self.detector = ICDetector(cfg)
        self.counter = PinCounter(cfg)
        self.results = []

    def evaluate_single(self, image_path: str, ground_truth: int = None,
                        method: str = 'ensemble') -> Dict:
        """
        Evaluate a single image.

        Args:
            image_path: Path to the image
            ground_truth: Expected pin count (optional)
            method: Counting method to use

        Returns:
            Dictionary with evaluation results
        """
        filename = Path(image_path).name
        print(f"\nProcessing: {filename}")

        try:
            # Run detection pipeline
            detection_results = self.detector.process_image(image_path, save_intermediates=True)

            # Count pins
            predicted, details, annotated = self.counter.count_pins(
                detection_results['grayscale'],
                detection_results['pins_only'],
                detection_results['package_type'],
                method=method,
                cropped_shape=detection_results['cropped_shape']
            )

            # Calculate individual method results
            method_results = {}
            if method == 'ensemble':
                for m_name, m_data in details.items():
                    method_results[m_name] = m_data['count']

            # Build result
            result = {
                'filename': filename,
                'predicted': predicted,
                'ground_truth': ground_truth,
                'correct': predicted == ground_truth if ground_truth else None,
                'error': abs(predicted - ground_truth) if ground_truth else None,
                'package_type': detection_results['package_type'],
                'method_results': method_results,
                'bbox': detection_results['bbox']
            }

            # Save annotated image
            output_dir = Path(self.cfg.OUTPUT_DIR) / Path(image_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / "7_pins_annotated.png"), annotated)

            # Create combined visualization
            self._save_combined_visualization(detection_results, annotated, output_dir, result)

            return result

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            return {
                'filename': filename,
                'predicted': None,
                'ground_truth': ground_truth,
                'correct': False,
                'error': None,
                'package_type': None,
                'method_results': {},
                'error_message': str(e)
            }

    def evaluate_all(self, input_dir: str = None, method: str = 'ensemble') -> List[Dict]:
        """
        Evaluate all images in the input directory.

        Args:
            input_dir: Directory containing images (uses config default if None)
            method: Counting method to use

        Returns:
            List of evaluation results
        """
        input_dir = input_dir or self.cfg.INPUT_DIR
        self.results = []

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []

        for f in Path(input_dir).iterdir():
            if f.suffix.lower() in image_extensions:
                image_files.append(str(f))

        image_files.sort()

        print(f"\n{'=' * 60}")
        print(f"Evaluating {len(image_files)} images from {input_dir}")
        print(f"Method: {method}")
        print(f"{'=' * 60}")

        for image_path in image_files:
            filename = Path(image_path).name
            ground_truth = self.cfg.GROUND_TRUTH.get(filename)
            result = self.evaluate_single(image_path, ground_truth, method)
            self.results.append(result)

            # Print summary for this image
            status = "CORRECT" if result.get('correct') else "INCORRECT" if result.get('correct') is False else "N/A"
            print(f"  Predicted: {result['predicted']}, GT: {ground_truth}, Status: {status}")
            if result.get('method_results'):
                print(f"  Method breakdown: {result['method_results']}")

        return self.results

    def generate_report(self) -> str:
        """
        Generate a comprehensive accuracy report.

        Returns:
            Formatted report string
        """
        if not self.results:
            return "No results to report. Run evaluate_all() first."

        # Calculate statistics
        valid_results = [r for r in self.results if r.get('ground_truth') is not None]
        correct_count = sum(1 for r in valid_results if r.get('correct'))
        total_with_gt = len(valid_results)

        accuracy = (correct_count / total_with_gt * 100) if total_with_gt > 0 else 0

        errors = [r['error'] for r in valid_results if r.get('error') is not None]
        mae = np.mean(errors) if errors else 0
        max_error = max(errors) if errors else 0

        # Build report
        report = []
        report.append("\n" + "=" * 70)
        report.append("IC PIN COUNTING - EVALUATION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)

        report.append("\nSUMMARY STATISTICS:")
        report.append(f"  Total images processed: {len(self.results)}")
        report.append(f"  Images with ground truth: {total_with_gt}")
        report.append(f"  Correct predictions: {correct_count}/{total_with_gt}")
        report.append(f"  Accuracy: {accuracy:.1f}%")
        report.append(f"  Mean Absolute Error: {mae:.2f} pins")
        report.append(f"  Maximum Error: {max_error} pins")

        report.append("\nDETAILED RESULTS:")
        report.append("-" * 70)
        report.append(f"{'Filename':<20} {'Predicted':>10} {'GT':>8} {'Error':>8} {'Status':>10} {'Type':>8}")
        report.append("-" * 70)

        for r in self.results:
            filename = r['filename'][:20]
            predicted = str(r.get('predicted', 'ERR'))
            gt = str(r.get('ground_truth', 'N/A'))
            error = str(r.get('error', 'N/A'))
            status = "CORRECT" if r.get('correct') else "WRONG" if r.get('correct') is False else "N/A"
            pkg_type = r.get('package_type', 'N/A') or 'N/A'

            report.append(f"{filename:<20} {predicted:>10} {gt:>8} {error:>8} {status:>10} {pkg_type:>8}")

        report.append("-" * 70)

        # Method comparison (if ensemble was used)
        if any(r.get('method_results') for r in self.results):
            report.append("\nMETHOD COMPARISON:")
            methods = ['traditional', 'sidewise', 'contour']
            for method in methods:
                method_correct = 0
                method_total = 0
                for r in valid_results:
                    if method in r.get('method_results', {}):
                        method_pred = r['method_results'][method]
                        if method_pred == r['ground_truth']:
                            method_correct += 1
                        method_total += 1
                if method_total > 0:
                    method_acc = method_correct / method_total * 100
                    report.append(f"  {method}: {method_correct}/{method_total} correct ({method_acc:.1f}%)")

        report.append("\n" + "=" * 70)

        report_text = "\n".join(report)
        print(report_text)

        # Save report to file
        report_path = Path(self.cfg.OUTPUT_DIR) / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {report_path}")

        # Also save JSON results
        json_path = Path(self.cfg.OUTPUT_DIR) / "evaluation_results.json"
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = []
            for r in self.results:
                jr = {k: (int(v) if isinstance(v, (np.integer, np.int64)) else v)
                      for k, v in r.items() if k != 'bbox'}
                json_results.append(jr)
            json.dump(json_results, f, indent=2, default=str)
        print(f"JSON results saved to: {json_path}")

        return report_text

    def _save_combined_visualization(self, detection_results: Dict, pin_annotated: np.ndarray,
                                     output_dir: Path, result: Dict):
        """Create and save a combined visualization of all processing steps."""
        # Get all intermediate images
        images = [
            ('Original', detection_results['original']),
            ('Boundary', detection_results['boundary_annotated']),
            ('Cropped', detection_results['cropped']),
            ('Grayscale', detection_results['grayscale']),
            ('Pins Only', detection_results['pins_only']),
            ('Pin Detection', pin_annotated)
        ]

        # Resize all to same height
        target_h = 250
        processed = []

        for title, img in images:
            if img is None:
                continue

            # Convert to BGR if needed
            if len(img.shape) == 2:
                if img.dtype == np.float32:
                    img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Resize maintaining aspect ratio
            h, w = img.shape[:2]
            new_w = int(w * target_h / h)
            img = cv2.resize(img, (new_w, target_h))

            # Add title and border
            img = cv2.copyMakeBorder(img, 30, 10, 5, 5, cv2.BORDER_CONSTANT, value=(50, 50, 50))
            cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            processed.append(img)

        # Combine horizontally
        combined = np.hstack(processed)

        # Add summary bar at bottom
        summary_h = 40
        summary = np.zeros((summary_h, combined.shape[1], 3), dtype=np.uint8) + 30
        pred = result.get('predicted', 'N/A')
        gt = result.get('ground_truth', 'N/A')
        status = "CORRECT" if result.get('correct') else "INCORRECT" if result.get('correct') is False else "No GT"
        color = (0, 255, 0) if result.get('correct') else (0, 0, 255) if result.get('correct') is False else (128, 128, 128)

        text = f"Predicted: {pred} | Ground Truth: {gt} | Status: {status} | Package: {result.get('package_type', 'N/A')}"
        cv2.putText(summary, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        combined = np.vstack([combined, summary])

        # Save
        cv2.imwrite(str(output_dir / "combined_pipeline.png"), combined)


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate IC Pin Counting Pipeline')
    parser.add_argument('--input', '-i', type=str, default='uncertain',
                        help='Input directory containing images')
    parser.add_argument('--method', '-m', type=str, default='ensemble',
                        choices=['projection', 'threshold', 'gradient', 'ensemble'],
                        help='Pin counting method')
    parser.add_argument('--single', '-s', type=str, default=None,
                        help='Process a single image')

    args = parser.parse_args()

    evaluator = Evaluator()

    if args.single:
        # Single image evaluation
        filename = Path(args.single).name
        gt = config.GROUND_TRUTH.get(filename)
        result = evaluator.evaluate_single(args.single, gt, args.method)
        print(f"\nResult: {result}")
    else:
        # Full evaluation
        evaluator.evaluate_all(args.input, args.method)
        evaluator.generate_report()


if __name__ == "__main__":
    main()
