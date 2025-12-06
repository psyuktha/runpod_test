"""
Moondream Integration Helpers for IC Pin Counting
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global model and tokenizer cache
_model_cache = None
_tokenizer_cache = None

# 1. Load Moondream model
def load_moondream_model(model_name: str = "vikhyatk/moondream2") -> Any:
    """Load the latest Moondream model from HuggingFace."""
    global _model_cache, _tokenizer_cache
    
    if _model_cache is None:
        print(f"Loading Moondream model: {model_name}...")
        _model_cache = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision="2024-08-26"
        )
        _tokenizer_cache = AutoTokenizer.from_pretrained(model_name, revision="2024-08-26")
        print("Model loaded successfully!")
    
    return {"model": _model_cache, "tokenizer": _tokenizer_cache}

# 2. Query Moondream with prompt
def query_moondream(model: Any, image: np.ndarray, prompt: str) -> str:
    """Send image and prompt to Moondream, return response."""
    if model is None:
        return "0"
    
    # Convert OpenCV BGR image to PIL RGB
    if len(image.shape) == 2:  # Grayscale
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(img_rgb)
    
    # Encode and query using the moondream API with error handling
    try:
        enc_image = model["model"].encode_image(pil_image)
        response = model["model"].answer_question(enc_image, prompt, model["tokenizer"])
    except AttributeError as e:
        print(f"AttributeError encountered: {e}")
        # Fallback: try alternative API methods
        try:
            response = model["model"].query(pil_image, prompt)["answer"]
        except Exception as e2:
            print(f"Fallback failed: {e2}")
            # Last resort: return placeholder
            response = "0"
    except Exception as e:
        print(f"Error querying Moondream: {e}")
        response = "0"
    
    return response

# 3. Extract pin count from response
def extract_pin_count_from_response(response: str) -> Optional[int]:
    """Parse integer pin count from Moondream response."""
    try:
        # Try to extract just the number from the response
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            return int(numbers[0])
        return int(response.strip())
    except Exception:
        return None

# 4. Optionally draw bounding boxes
def optionally_draw_bounding_boxes(image: np.ndarray, boxes: List[List[int]]) -> np.ndarray:
    """Draw bounding boxes on image for visualization."""
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    return img

# 5. Extract bounding boxes from response
def extract_bounding_boxes_from_response(response: str) -> List[List[int]]:
    """Parse bounding boxes from Moondream response."""
    import re
    boxes = re.findall(r'\[(\d+),(\d+),(\d+),(\d+)\]', response)
    return [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes]

# Prompts
PIN_COUNT_PROMPT = (
    "Look at this processed image. Count only the metallic pins (the protruding spikes). "
    "Do NOT guess missing pins. Count ONLY visible distinct spikes. "
    "Return ONLY the number as an integer with no explanation."
)

BOUNDING_BOX_PROMPT = (
    "Identify each individual pin. Return a list of bounding boxes for each pin in [x1,y1,x2,y2] format."
)
