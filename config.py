"""
Configuration file for IC Pin Counting Pipeline
All hyperparameters and settings are centralized here for easy tuning.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import os

@dataclass
class Config:
    # ===========================================
    # PATH CONFIGURATION
    # ===========================================
    INPUT_DIR: str = "uncertain"
    OUTPUT_DIR: str = "results"

    # ===========================================
    # BOUNDARY DETECTION PARAMETERS
    # ===========================================
    # Canny edge detection thresholds for boundary detection
    BOUNDARY_CANNY_LOW: int = 50
    BOUNDARY_CANNY_HIGH: int = 150

    # Padding around detected IC boundary (pixels)
    BOUNDARY_PADDING: int = 15

    # Minimum contour area ratio (relative to image area) to be considered IC
    MIN_CONTOUR_AREA_RATIO: float = 0.05

    # Gaussian blur kernel size for boundary detection
    BOUNDARY_BLUR_KERNEL: Tuple[int, int] = (5, 5)

    # ===========================================
    # IMAGE PREPROCESSING PARAMETERS
    # ===========================================
    # Target image size after preprocessing (width, height)
    TARGET_SIZE: Tuple[int, int] = (512, 512)

    # Bilateral filter parameters (for denoising)
    BILATERAL_D: int = 9  # Diameter of pixel neighborhood
    BILATERAL_SIGMA_COLOR: int = 75  # Filter sigma in color space
    BILATERAL_SIGMA_SPACE: int = 75  # Filter sigma in coordinate space

    # Whether to apply denoising
    APPLY_DENOISING: bool = True

    # ===========================================
    # CENTER REMOVAL PARAMETERS
    # ===========================================
    # Ratio of center to remove (0.0 to 1.0)
    # 0.5 means remove center 50% of image
    CENTER_REMOVAL_RATIO: float = 0.45

    # Whether to use adaptive center detection (vs fixed ratio)
    USE_ADAPTIVE_CENTER: bool = False

    # ===========================================
    # PIN COUNTING PARAMETERS (Traditional CV)
    # ===========================================
    # Canny edge detection thresholds for pin detection
    PIN_CANNY_LOW: int = 30
    PIN_CANNY_HIGH: int = 100

    # Morphological operation kernel sizes
    MORPH_KERNEL_SIZE: Tuple[int, int] = (3, 3)

    # Minimum and maximum pin area (in pixels) for filtering
    MIN_PIN_AREA: int = 50
    MAX_PIN_AREA: int = 5000

    # Pin aspect ratio constraints (width/height or height/width)
    MIN_PIN_ASPECT_RATIO: float = 0.2
    MAX_PIN_ASPECT_RATIO: float = 5.0

    # Minimum number of pins expected per side (for validation)
    MIN_PINS_PER_SIDE: int = 0

    # ===========================================
    # PIN COUNTING PARAMETERS (CNN-based)
    # ===========================================
    # CNN model input size
    CNN_INPUT_SIZE: Tuple[int, int] = (224, 224)

    # Confidence threshold for pin detection
    CNN_CONFIDENCE_THRESHOLD: float = 0.5

    # ===========================================
    # VISUALIZATION PARAMETERS
    # ===========================================
    # Colors for visualization (BGR format)
    BOUNDARY_COLOR: Tuple[int, int, int] = (0, 255, 0)  # Green
    PIN_COLOR: Tuple[int, int, int] = (0, 0, 255)  # Red
    CENTER_MASK_COLOR: Tuple[int, int, int] = (128, 128, 128)  # Gray

    # Line thickness for drawing
    LINE_THICKNESS: int = 2

    # Whether to save intermediate visualizations
    SAVE_VISUALIZATIONS: bool = True

    # ===========================================
    # GROUND TRUTH FOR UNCERTAIN FOLDER
    # ===========================================
    # Order: anu1, anu2, anu3, anu4, anu5, anu6, new1, new2, new3, new4, new5
    GROUND_TRUTH: dict = None

    def __post_init__(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # Initialize ground truth
        self.GROUND_TRUTH = {
            "anu1.jpeg": 64,   # QFP - 16 pins per side
            "anu2.jpeg": 56,   # QFN - 14 pins per side
            "anu3.jpeg": 20,   # DIP or small QFP
            "anu4.jpg": 48,    # QFP
            "anu5.jpg": 14,    # DIP - 7 pins per side
            "anu6.png": 48,    # QFP
            "new1.png": 14,    # DIP - 7 pins per side
            "new2.png": 48,    # QFP
            "new3.png": 48,    # QFP
            "new4.png": 22,    # DIP or small package
            "new5.png": 14,    # DIP
        }

# Global config instance
config = Config()
