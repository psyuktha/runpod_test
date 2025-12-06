"""
Full IC Pin Counting Pipeline with Moondream Integration
"""
import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from ic_pin_pipeline import preprocess_ic_image
from moondream_helpers import (
    load_moondream_model,
    query_moondream,
    extract_pin_count_from_response,
    extract_bounding_boxes_from_response,
    optionally_draw_bounding_boxes,
    PIN_COUNT_PROMPT,
    BOUNDING_BOX_PROMPT
)

def count_ic_pins_with_moondream(image_path: str, debug_dir: str = "debug") -> Dict[str, Any]:
    """
    Full pipeline: preprocess image, run Moondream, return pin count, bounding boxes, debug images.
    """
    # Step 1: Preprocess image
    pre = preprocess_ic_image(image_path, debug_dir)
    pin_mask = pre["pin_mask"]
    base_name = pre["base_name"]
    debug = pre["debug"]
    original = pre["original"]

    # Step 2: Load Moondream model
    model = load_moondream_model()

    # Step 3: Query for pin count
    response = query_moondream(model, cv2.cvtColor(pin_mask, cv2.COLOR_GRAY2BGR), PIN_COUNT_PROMPT)
    pin_count = extract_pin_count_from_response(response)

    bounding_boxes = []
    box_img = None
    # Step 4: Error handling: if not parseable, retry with bounding box mode
    if pin_count is None:
        response_box = query_moondream(model, cv2.cvtColor(pin_mask, cv2.COLOR_GRAY2BGR), BOUNDING_BOX_PROMPT)
        bounding_boxes = extract_bounding_boxes_from_response(response_box)
        pin_count = len(bounding_boxes)
        box_img = optionally_draw_bounding_boxes(original, bounding_boxes)
        if box_img is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{base_name}_bounding_boxes.png"), box_img)
    else:
        # Optionally, get bounding boxes if needed
        response_box = query_moondream(model, cv2.cvtColor(pin_mask, cv2.COLOR_GRAY2BGR), BOUNDING_BOX_PROMPT)
        bounding_boxes = extract_bounding_boxes_from_response(response_box)
        if bounding_boxes:
            box_img = optionally_draw_bounding_boxes(original, bounding_boxes)
            cv2.imwrite(os.path.join(debug_dir, f"{base_name}_bounding_boxes.png"), box_img)

    return {
        "pin_count": pin_count,
        "bounding_boxes": bounding_boxes,
        "debug": debug,
        "bounding_box_img": box_img,
        "pin_mask": pin_mask
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IC Pin Counting Pipeline with Moondream")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--debug_dir", type=str, default="debug", help="Directory for debug images")
    args = parser.parse_args()

    result = count_ic_pins_with_moondream(args.image_path, args.debug_dir)
    print(f"Pin count: {result['pin_count']}")
    print(f"Bounding boxes: {result['bounding_boxes']}")
    print(f"Debug images saved in: {args.debug_dir}")
