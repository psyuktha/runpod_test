#!/usr/bin/env python3
"""Batch-run moon.py preprocessing and feed thresholds to count_pins.py."""

import argparse
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

from count_pins import count_pins
from moon import count_ic_pins_opencv


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def process_directory(input_dir: Path, debug_dir: Path) -> List[Dict[str, object]]:
    """Run moon pipeline on all images and count pins from threshold outputs."""
    input_dir = input_dir.expanduser()
    debug_dir = debug_dir.expanduser()
    debug_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []

    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        print(f"\n=== {img_path.name} ===")
        count_ic_pins_opencv(str(img_path), str(debug_dir))

        threshold_path = debug_dir / f"{img_path.stem}_05_threshold.png"
        if not threshold_path.exists():
            print(f"Threshold image not found: {threshold_path}; skipping.")
            continue

        (
            num_labels,
            areas,
            pin_areas,
            pin_labels,
            bboxes,
            pin_bboxes,
            package_bbox,
            side_counts,
            symmetric_pin_count,
            side_labels,
            yellow_side_counts,
            yellow_side_labels,
            yellow_symmetric_count,
            yellow_bboxes,
            yellow_side_scores,
            best_yellow_side,
            yellow_labels,
            yellow_side_centers,
        ) = count_pins(threshold_path)

        pin_area_min = min(pin_areas) if pin_areas else 0
        pin_area_max = max(pin_areas) if pin_areas else 0

        # Save overlay with bounding boxes for visual inspection.
        overlay_path = debug_dir / f"{img_path.stem}_pins.png"
        try:
            base = Image.open(threshold_path).convert("RGB")
            draw = ImageDraw.Draw(base)
            font = ImageFont.load_default()
            for lbl, (min_x, min_y, max_x, max_y) in zip(yellow_labels, yellow_bboxes):
                draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=(255, 255, 0), width=2)
                draw.text((min_x, max(min_y - 10, 0)), str(lbl), fill=(255, 255, 0), font=font)
            base.save(overlay_path)
            print(f"Saved overlay: {overlay_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save overlay for {img_path.name}: {exc}")

        print(f"Threshold: {threshold_path}")
        print(f"Pin candidates: {len(pin_labels)} (area min={pin_area_min}, max={pin_area_max})")
        print(f"Pins per side (yellow boxes): {yellow_side_counts}")
        print(f"Symmetric estimate (yellow boxes): {yellow_symmetric_count}")
        print(f"Best side by regularity: {best_yellow_side}")

        results.append(
            {
                "image": img_path.name,
                "threshold_path": str(threshold_path),
                "overlay_path": str(overlay_path),
                "side_counts": yellow_side_counts,
                "symmetric_estimate": yellow_symmetric_count,
                "best_side": best_yellow_side,
                "pin_candidates": len(pin_labels),
                "pin_area_min": pin_area_min,
                "pin_area_max": pin_area_max,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch preprocess lqfn images and count pins.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("lqfn"),
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("hahaha"),
        help="Directory where moon.py debug images are written.",
    )
    args = parser.parse_args()

    results = process_directory(args.input_dir, args.debug_dir)

    print("\n===== Summary =====")
    for res in results:
        print(
            f"{res['image']}: symmetric_est={res['symmetric_estimate']}, "
            f"side_counts={res['side_counts']}, threshold={res['threshold_path']}"
        )


if __name__ == "__main__":
    main()

