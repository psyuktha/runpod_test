import cv2
import numpy as np
import os

INPUT_DIR = "./"   # change if needed

# ------------------------------------------------------------
# Helper: sanitize OpenCV rect → convert numpy floats to python floats
# ------------------------------------------------------------
def sanitize_rect(rect):
    (cx, cy), (w, h), angle = rect
    return (float(cx), float(cy)), (float(w), float(h)), float(angle)


# ------------------------------------------------------------
# Rotate image around a point safely
# ------------------------------------------------------------
def rotate_image_and_points(image, cx, cy, angle):
    (h, w) = image.shape[:2]

    cx = float(cx)
    cy = float(cy)
    angle = float(angle)

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


# ------------------------------------------------------------
# Pin detection on one side
# ------------------------------------------------------------
def count_pins_on_strip(strip):
    # Threshold edges to binary
    _, th = cv2.threshold(strip, 50, 255, cv2.THRESH_BINARY)

    # Morph closing to merge broken pin edges
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find vertical blobs (pins)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 5:  # simple noise filter
            count += 1

    return count


# ------------------------------------------------------------
# Main processing for each IC image
# ------------------------------------------------------------
def process_image(img_path):
    print(f"\n--- Processing {img_path} ---")

    img = cv2.imread(img_path)
    if img is None:
        print("[ERROR] Can't read image")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        print("[ERROR] No contours found")
        return

    # Pick the largest contour as IC body
    cnt = max(contours, key=cv2.contourArea)

    # Min area rectangle
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = sanitize_rect(rect)

    # Rotate the whole image so chip becomes straight
    rotated = rotate_image_and_points(gray, cx, cy, angle)

    # Recompute rotated chip bounding box
    contours2, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt2 = max(contours2, key=cv2.contourArea)
    x, y, w2, h2 = cv2.boundingRect(cnt2)

    chip = rotated[y:y+h2, x:x+w2]

    # Divide into 4 sides
    H, W = chip.shape

    strips = {
        "top":    chip[0:15, :],
        "bottom": chip[H-15:H, :],
        "left":   chip[:, 0:15],
        "right":  chip[:, W-15:W],
    }

    pin_counts = {}

    for side, strip in strips.items():
        pin_counts[side] = count_pins_on_strip(strip)

    print(pin_counts)

    # Find the consistent side count (max)
    max_pins = max(pin_counts.values())
    sides = 4
    total_pins = max_pins * sides

    print(f"=> Estimated pins: {total_pins}")


# ------------------------------------------------------------
# Run on all images in folder
# ------------------------------------------------------------
def main():
    files = [f for f in os.listdir(INPUT_DIR)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for f in files:
        process_image(os.path.join(INPUT_DIR, f))


if __name__ == "__main__":
    main()
