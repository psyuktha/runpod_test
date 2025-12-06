import os
import cv2
import numpy as np

import os
import cv2
import numpy as np

def canny_bilateral(directory, save_dir="bi_full"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = os.listdir(directory)
    for file in files:
        img_path = os.path.join(directory, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {file}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to smooth while keeping edges
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)

        # Canny edge detection
        edges = cv2.Canny(smooth, 40, 120)

        # Optional: dilate edges to make them more prominent
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Save result
        save_path = os.path.join(save_dir, f"bilateral_canny_{file}")
        cv2.imwrite(save_path, edges)
        print(f"Saved: {save_path}")



def canny(directory):
    files=os.listdir(directory)
    for file in files:
        if not (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
            continue
        img_path=os.path.join(directory,file)
        img=preprocess_image(img_path)
        final=auto_canny(img)
        cv2.imwrite(f"yuktha/canny_{file}",final)
        print(f"Saved: canny_{file}")

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image at path: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred




# Auto-canny helper
def auto_canny(image, sigma=0.5):
    med = np.median(image)
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(image, lower, upper)

if __name__ == "__main__":
    canny_bilateral("uncertain")  