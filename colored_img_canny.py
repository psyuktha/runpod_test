import cv2
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 canny_save.py <image_path>")
        return

    img_path = sys.argv[1]

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image '{img_path}'")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny
    edges = cv2.Canny(gray, 100, 200)

    # Build output path
    base, ext = os.path.splitext(img_path)
    out_path = base + "_canny.png"

    # Save
    cv2.imwrite(out_path, edges)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

