#!/usr/bin/env python3
"""
Example Usage Script for IC Pin Detection Pipeline
Demonstrates all features and capabilities
"""

import os
import cv2
from ic_pin_counter_opencv import count_ic_pins_opencv

def example_1_basic_usage():
    """Example 1: Basic single image processing."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)
    
    result = count_ic_pins_opencv("saturday/001.png", "debug_example1")
    
    print(f"Pin Count: {result['pin_count']}")
    print(f"Pins by side: {result['pins_by_side']}")
    print(f"IC Body: {result['body_rect']}")
    print()

def example_2_batch_processing():
    """Example 2: Process multiple images."""
    print("=" * 70)
    print("EXAMPLE 2: Batch Processing")
    print("=" * 70)
    
    images = ["saturday/001.png", "saturday/003.png", "saturday/anu1.jpeg"]
    
    for img_path in images:
        result = count_ic_pins_opencv(img_path, "debug_example2")
        filename = os.path.basename(img_path)
        print(f"{filename}: {result['pin_count']} pins")
    print()

def example_3_custom_visualization():
    """Example 3: Create custom visualization."""
    print("=" * 70)
    print("EXAMPLE 3: Custom Visualization")
    print("=" * 70)
    
    result = count_ic_pins_opencv("saturday/001.png", "debug_example3")
    
    # Create custom visualization
    img = cv2.imread("saturday/001.png")
    
    # Draw pins with different colors per side
    colors = {
        "left": (255, 0, 0),    # Blue
        "right": (0, 255, 0),   # Green
        "top": (0, 0, 255),     # Red
        "bottom": (255, 255, 0) # Cyan
    }
    
    for side, boxes in result['pins_by_side'].items():
        color = colors.get(side, (255, 255, 255))
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # Add label
            cv2.putText(img, side[0].upper(), (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save
    cv2.imwrite("debug_example3/custom_viz.png", img)
    print("Custom visualization saved to: debug_example3/custom_viz.png")
    print()

def example_4_quality_check():
    """Example 4: Quality check with expected pin count."""
    print("=" * 70)
    print("EXAMPLE 4: Quality Check")
    print("=" * 70)
    
    # Simulate quality check against datasheet
    test_cases = [
        ("saturday/001.png", 14),  # Expected: 14 pins
        ("saturday/003.png", 16),  # Expected: 16 pins
        ("saturday/anu1.jpeg", 14) # Expected: 14 pins
    ]
    
    for img_path, expected_count in test_cases:
        result = count_ic_pins_opencv(img_path, "debug_example4")
        detected = result['pin_count']
        status = "PASS" if detected == expected_count else "FAIL"
        
        filename = os.path.basename(img_path)
        print(f"{filename}:")
        print(f"  Expected: {expected_count}, Detected: {detected}, Status: {status}")
    print()

def example_5_pin_measurements():
    """Example 5: Measure pin dimensions."""
    print("=" * 70)
    print("EXAMPLE 5: Pin Measurements")
    print("=" * 70)
    
    result = count_ic_pins_opencv("saturday/001.png", "debug_example5")
    
    print(f"Image: 001.png")
    print(f"Total pins: {result['pin_count']}")
    print(f"\nPin dimensions:")
    
    for i, box in enumerate(result['pin_boxes'], 1):
        x, y, w, h = box
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        print(f"  Pin {i}: Width={w}px, Height={h}px, Aspect Ratio={aspect_ratio:.2f}")
    print()

def example_6_error_handling():
    """Example 6: Demonstrate error handling."""
    print("=" * 70)
    print("EXAMPLE 6: Error Handling")
    print("=" * 70)
    
    try:
        result = count_ic_pins_opencv("nonexistent.png", "debug_example6")
    except FileNotFoundError as e:
        print(f"✓ Error caught correctly: {e}")
    
    try:
        result = count_ic_pins_opencv("saturday/001.png", "debug_example6")
        print(f"✓ Valid image processed successfully: {result['pin_count']} pins")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    print()

def example_7_statistical_analysis():
    """Example 7: Statistical analysis of pin distribution."""
    print("=" * 70)
    print("EXAMPLE 7: Statistical Analysis")
    print("=" * 70)
    
    result = count_ic_pins_opencv("saturday/anu1.jpeg", "debug_example7")
    sides = result['pins_by_side']
    
    total = result['pin_count']
    print(f"Total pins: {total}")
    print(f"\nDistribution:")
    for side, boxes in sides.items():
        if boxes:
            percentage = (len(boxes) / total * 100) if total > 0 else 0
            print(f"  {side.capitalize():>10}: {len(boxes):>2} pins ({percentage:>5.1f}%)")
    
    # Check symmetry
    left_count = len(sides.get('left', []))
    right_count = len(sides.get('right', []))
    top_count = len(sides.get('top', []))
    bottom_count = len(sides.get('bottom', []))
    
    print(f"\nSymmetry analysis:")
    print(f"  Left-Right balance: {abs(left_count - right_count)} pins difference")
    print(f"  Top-Bottom balance: {abs(top_count - bottom_count)} pins difference")
    
    if left_count == right_count and top_count == bottom_count:
        print(f"  ✓ Perfectly symmetric IC")
    else:
        print(f"  ⚠ Asymmetric IC or detection issues")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" IC PIN DETECTION PIPELINE - USAGE EXAMPLES")
    print("=" * 70 + "\n")
    
    # Run all examples
    example_1_basic_usage()
    example_2_batch_processing()
    example_3_custom_visualization()
    example_4_quality_check()
    example_5_pin_measurements()
    example_6_error_handling()
    example_7_statistical_analysis()
    
    print("=" * 70)
    print("All examples completed!")
    print("Check the debug_example* directories for output images")
    print("=" * 70)
