#!/usr/bin/env python3
"""
Batch processor for IC pin counting - runs on all images in a directory
"""
import os
import sys
import glob
from ic_pin_counter_opencv import count_ic_pins_opencv

def process_batch(input_dir: str, output_dir: str = "debug_batch"):
    """Process all images in a directory."""
    # Get all image files
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    for pattern in patterns:
        image_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    print("="*70)
    
    results = []
    for i, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        print(f"\n[{i}/{len(image_files)}] Processing: {filename}")
        
        try:
            # Create subdirectory for each image's debug output
            img_debug_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            result = count_ic_pins_opencv(img_path, img_debug_dir)
            
            results.append({
                "filename": filename,
                "pin_count": result["pin_count"],
                "pins_by_side": result["pins_by_side"],
                "body_rect": result["body_rect"]
            })
            
            print(f"  ✓ Detected {result['pin_count']} pins")
            for side, boxes in result['pins_by_side'].items():
                if boxes:
                    print(f"    - {side.capitalize()}: {len(boxes)}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "filename": filename,
                "pin_count": 0,
                "error": str(e)
            })
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Image':<20} {'Pin Count':>15} {'Left':>8} {'Right':>8} {'Top':>8} {'Bottom':>8}")
    print("-"*70)
    
    for r in results:
        if "error" in r:
            print(f"{r['filename']:<20} {'ERROR':>15}")
        else:
            sides = r['pins_by_side']
            print(f"{r['filename']:<20} {r['pin_count']:>15} "
                  f"{len(sides.get('left', [])):>8} "
                  f"{len(sides.get('right', [])):>8} "
                  f"{len(sides.get('top', [])):>8} "
                  f"{len(sides.get('bottom', [])):>8}")
    
    print("="*70)
    print(f"\nDebug images saved in: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_process.py <input_directory> [output_directory]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "debug_batch"
    
    process_batch(input_dir, output_dir)
