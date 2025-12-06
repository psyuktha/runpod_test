# =============================================================
# 📌 CHIP BOUNDARY DETECTION NOTEBOOK (CANNY + CONTOURS)
# Ready-to-run Colab notebook in a single cell
# =============================================================

# ---------------------------
# 🔧 1. Install Dependencies
# ---------------------------


import os
import cv2
import numpy as np



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
    canny("uncertain")  # Specify the directory containing test images
# =============================================================
# 📌 HIGH-QUALITY IC CANNY EDGE PIPELINE (BEST SETTINGS)
# =============================================================

# import os
# import cv2
# import numpy as np


# # ---------------------------
# # 🔧 1. Batch Canny Function
# # ---------------------------
# def canny(directory):
#     os.makedirs("lemon_chicken", exist_ok=True)

#     files = os.listdir(directory)
#     for file in files:
#         if not (file.lower().endswith(".png") or file.lower().endswith(".jpg") or file.lower().endswith(".jpeg")):
#             continue

#         img_path = os.path.join(directory, file)
#         img = preprocess_image(img_path)           # IC-optimized preprocessing
#         final = auto_canny(img)                    # Tuned Canny

#         cv2.imwrite(f"lemon_chicken/canny_{file}", final)
#         print(f"Saved: canny_{file}")


# # ---------------------------
# # 🧼 2. IC-Optimized Preprocessing
# # ---------------------------
# def preprocess_image(img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         raise ValueError(f"Cannot read image at path: {img_path}")

#     # 1. Grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 2. Adaptive contrast enhancement (BEST for pins)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     gray = clahe.apply(gray)

#     # 3. Bilateral filtering (preserves pin edges)
#     gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

#     # 4. Sharpening to enhance metallic pin edges
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5,-1],
#                        [0, -1, 0]])
#     gray = cv2.filter2D(gray, -1, kernel)

#     return gray


# # ---------------------------
# # ⚙️ 3. Tuned Canny for IC Images
# # ---------------------------
# def auto_canny(image, sigma=0.25):
#     """
#     Sigma = 0.25 works BEST for IC pins:
#       - Captures thin pin edges
#       - Avoids too much noise
#     """
#     med = np.median(image)
#     lower = int(max(0, (1.0 - sigma) * med))
#     upper = int(min(255, (1.0 + sigma) * med))

#     return cv2.Canny(image, lower, upper)


# # ---------------------------
# # ▶️ 4. Run
# # ---------------------------
# if __name__ == "__main__":
#     canny("ic_test")   # Put your IC images in /ic_test
