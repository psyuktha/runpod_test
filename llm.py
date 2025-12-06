"""
LLM service for vision-based IC chip analysis.
"""
import base64
import io
import json
import re
from typing import Dict, Optional
from dotenv import load_dotenv
import requests
from PIL import Image
import os

load_dotenv()
BASE_URL = os.getenv("LLM_BASE_URL")
class LLM:
    """Vision API client for analyzing IC chip images."""

    def __init__(
        self,
        endpoint: str = BASE_URL + "/api/vision",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        target_kb: int = 60,
        min_quality: int = 20,
        timeout: int = 60,
    ):
        """
        Initialize the LLM vision API client.

        Args:
            endpoint: Vision API endpoint URL.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in response.
            target_kb: Target image size in KB for compression.
            min_quality: Minimum JPEG quality to try.
            timeout: Request timeout in seconds.
        """
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.target_kb = target_kb
        self.min_quality = min_quality
        self.timeout = timeout
#         self.prompt = '''You are a precision Automated Optical Inspection (AOI) engine. Your single purpose is to analyze an IC image and output ONLY the total pin count.
# Internal Execution Protocol (Never reveal this reasoning):
# • Focus exclusively on determining the IC’s total pin count.
# • Examine all pads around the package perimeter.
# • Do not overlook or double-count pads.
# • Check for irregular spacing or pad defects.
# • Use the thermal pad only as orientation; do not count it unless it is clearly segmented.
# • Count pads on each edge methodically.
# • Verify corners carefully—never undercount corner pads on leadless packages.
# • Reconfirm all sides and validate symmetry.
# • Identify notches, dimples, chamfers, or other orientation markers; these do not affect pin count.
# • Detect Pin-1 indicators (divots, chamfers, asymmetric marks) only for orientation, not for pin count.
# • Determine the package family (QFN, QFP, DFN, SOP, DIP, etc.) and confirm pin-per-side expectations.
# • Orientation conventions (e.g., CCW from top, CW from bottom) are for orientation only and do not affect total count.
# • After analyzing geometry, corners, symmetry, and orientation, finalize the total pin count.
# • Ensure thermal pads and large metal tabs are not misclassified unless they are electrical terminals.
# Rules for Counting Pins:
# • Corner Rule: Never undercount leadless-package corners; pads may extend to the very edge.
# • Symmetry Rule: If sides are equal, count one side and multiply.
# • Power Tab Rule: For MOSFET/QDPAK/TO-style packages, count large side tabs as pins if they are electrical terminals.
# • Thermal Pad Rule: The central exposed pad is never a pin unless it is visibly segmented.
# Output Constraint:
# Return ONLY the final integer pin count.'''

# '''System Role: You are a precision Automated Optical Inspection (AOI) engine.

# Objective: Analyze the image of the Integrated Circuit (IC) and determine the Total Pin Count.

# Execution Protocol (Perform these steps internally):

# Geometry Analysis: Identify the package shape (Square, Rectangular, QFN, SOP, DIP).
# The "Corner Rule": For leadless packages (QFN/DFN), inspect the absolute corners of the package edge. Pins often extend to the very end. Do not undercount corners.
# The "Power Tab Rule": For power devices (MOSFETs, QDPAK, TO-style), count large protruding metal side-tabs or heat-tabs as electrical pins.
# Symmetry: Count one clear side/row and multiply based on the package symmetry.
# Output Constraint:
# Return ONLY the final integer. Do not provide text, labels, reasoning, or punctuation.'''
#       self.prompt = '''Count the total number of pins on this IC chip.

#         self.prompt='''You are an expert in IC package inspection.

# TASK:
# Count the total number of pins visible in this Canny-edge image of an IC package. 
# IMAGE DETAILS:
# - The image shows only edges (Canny output).
# - Pins appear as repeated line or corner-like edge structures around the chip outline.
# - Ignore noise, random speckles, internal chip markings, and shadows.

# INSTRUCTIONS:
# 1. Identify the boundary of the IC body (the central rectangular region).
# 2. Count only the pin edges that extend outward from the IC boundary.
# 3. Treat each continuous external protruding edge as ONE pin.
# 4. Do NOT assume symmetry. Count only what is actually visible.
# 5. Do NOT infer missing pins or guess based on package type.
# 6. If some pins are partially visible, still count them if the pin-edge structure exists.
# 7. Output ONLY the pin count as a single integer. No explanation.

# Your answer:'''

        self.prompt='''You are an expert in semiconductor IC lead-frame inspection.

TASK:
Count the total number of IC pins visible in the provided pin-isolated edge image.

IMAGE CHARACTERISTICS:
- The image contains ONLY pin edges extracted with a directional and morphological pipeline.
- Chip body and background edges have already been removed.
- Each pin appears as a thin outward-facing line or contour.
- Ignore any very small noise specks or tiny stray edges.

INSTRUCTIONS:
1. Identify all distinct pin-edge structures around the IC boundary.
2. Treat each continuous thin line/contour as ONE pin.
3. Do NOT assume symmetry: count only visible edges.
4. Do NOT infer or guess missing pins.
5. Do NOT count noise, dust specks, broken fragments, or internal short lines.
6. If a pin is split into two segments, count it as ONE pin if they align.
7. Output ONLY the integer pin count. No text, no explanation.

Your answer:'''
# '''Count the total number of pins on this IC chip.
# INSTRUCTIONS:
# Look at the IC package in the image
# Count ONLY the metallic pins/contacts (shiny silver/grey metal)
# Do NOT count: plastic edges, shadows, text, reflections, or mold marks
# Count each pin once at the point where it connects to the IC body
# Do NOT use part numbers or labels to guess the count

# Answer with ONLY a number. 

# Your answer:'''
# WHAT COUNTS AS A PIN:
# - A small metallic leg / pad / contact located along the outer edges of the package.
# - Pins are shiny silver/grey features used to solder the chip to a PCB.
# - Corner pins still count as pins, but must not be double-counted.

# RULES:
# 1. Look only at the IC body, not the blue background.
# 2. Count every visible metallic contact on the perimeter of the package.
# 3. Ignore:
#    - Plastic edges, body corners and chamfers
#    - Shadows, highlights, and reflections
#    - Printed text, logos, dots, or mold marks on the top surface
# 4. For QFP/QFN/SOIC-style packages:
#    - Pins are evenly spaced along each side.
#    - If the whole chip and all four sides are clearly visible and look identical in style, you may:
#      (a) carefully count the pins on one side, then
#      (b) multiply by the number of similar sides (usually 4) to get the total.
#    - Do NOT skip pins or group them; each small metal pad along an edge counts as one pin.
# 5. If any side is cropped, obscured, or looks different, count the pins on each visible side individually.
# 6. Do NOT:
#    - Assume a number from the part code
#    - Guess a common package size without checking the actual visible pins
#    - Output words or explanations

# OUTPUT FORMAT:
# - Return ONLY the total number of pins as an integer.
# - No words, no units, no extra text.

# Your answer:"""



    def compress_image(self, image_path: str) -> bytes:
        """
        Compress image to target size (<60KB).

        Args:
            image_path: Path to the image file.

        Returns:
            Compressed image bytes.

        Raises:
            ValueError: If image cannot be opened or compressed.
        """
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to open image {image_path}: {e}")

        last_data = None

        # Try reducing quality first (95 -> min_quality)
        for quality in range(95, self.min_quality - 1, -5):
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality, optimize=True)
            size_kb = buf.tell() / 1024

            if size_kb <= self.target_kb:
                return buf.getvalue()

            last_data = buf.getvalue()

        # If quality reduction not enough, progressively downscale
        scale_factor = 0.9
        w, h = img.size
        min_dimension = 50

        for _ in range(8):  # max 8 iterations
            w = int(w * scale_factor)
            h = int(h * scale_factor)

            if w < min_dimension or h < min_dimension:
                break

            img_scaled = img.resize((w, h), Image.LANCZOS)

            for quality in range(85, self.min_quality - 1, -5):
                buf = io.BytesIO()
                img_scaled.save(buf, format='JPEG', quality=quality, optimize=True)
                size_kb = buf.tell() / 1024

                if size_kb <= self.target_kb:
                    return buf.getvalue()

                last_data = buf.getvalue()

        # Fallback: return last attempt (may exceed target)
        if last_data:
            return last_data

        raise ValueError(f"Failed to compress image to target size")

    def _parse_response(self, response_text: str) -> Dict[str, Optional[str]]:
        """
        Parse manufacturer and pin_count from API response.

        Tries multiple strategies:
        1. Strict JSON parsing
        2. Regex-based JSON extraction
        3. Heuristics for fallback

        Args:
            response_text: Raw API response text.

        Returns:
            Dict with "manufacturer" and "pin_count" keys.
        """
        return response_text
        result = {"manufacturer": "", "pin_count": ""}

        if not response_text:
            return result

        # Strategy 1: Strict JSON parse
        try:
            doc = json.loads(response_text)
            if isinstance(doc, dict):
                result["manufacturer"] = str(doc.get("manufacturer", "")).strip()
                result["pin_count"] = str(doc.get("pin_count", "")).strip()
                if result["manufacturer"] or result["pin_count"]:
                    return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON substring
        try:
            match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if match:
                doc = json.loads(match.group(0))
                if isinstance(doc, dict):
                    result["manufacturer"] = str(doc.get("manufacturer", "")).strip()
                    result["pin_count"] = str(doc.get("pin_count", "")).strip()
                    if result["manufacturer"] or result["pin_count"]:
                        return result
        except (json.JSONDecodeError, AttributeError):
            pass

        # Strategy 3: Heuristic extraction for pin_count
        if not result["pin_count"]:
            # Look for "X pins" or "X-pin"
            match = re.search(
                r'(\d{1,3})\s*(?:pins?|pin)',
                response_text,
                re.IGNORECASE
            )
            if match:
                result["pin_count"] = match.group(1)
            else:
                # Look for any 2-3 digit number
                numbers = re.findall(r'\b(\d{1,3})\b', response_text)
                if numbers:
                    result["pin_count"] = min(numbers, key=int)

        # Strategy 4: Heuristic extraction for manufacturer
        if not result["manufacturer"]:
            # Look for capitalized brand names
            match = re.search(
                r'([A-Z][A-Za-z0-9&\-]{1,20}(?:\s+[A-Z][A-Za-z0-9&\-]{1,20})?)',
                response_text
            )
            if match:
                result["manufacturer"] = match.group(1).strip()

        return result

    def analyze_image(self, image_path: str) -> Dict[str, Optional[str]]:
        """
        Compress and analyze an IC chip image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with keys: "manufacturer" and "pin_count".

        Raises:
            ValueError: If image processing fails.
            requests.RequestException: If API request fails.
        """
        # Compress image
        compressed_data = self.compress_image(image_path)

        # Encode to base64
        image_base64 = base64.b64encode(compressed_data).decode('ascii')

        # Prepare payload
        payload = {
            "prompt": self.prompt,
            "image_base64": image_base64,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Send request
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        # Parse response
        response_text = response.text
        result = self._parse_response(response_text)

        return result
    
def main(img_path: str = None):
    c=0
    ground_truth={
    "anu1.jpeg": 64,
    "anu2.jpeg": 56,
    "anu3.jpeg": 20,
    "anu4.jpg": 48,
    "anu5.jpg": 14,
    "anu6.png": 48,
    "new1.png": 14,
    "new2.png": 48,
    "new3.png": 48,
    "new4.png": 22,
    "new5.png": 14
}
#     ground_truth={
#     "001.png": 22,
#     "002.png": 40,
#     "003.png": 56,
#     "004.png": 22,
#     "005.png": 14,
#     "006.png": 52,
#     "007.png": 14,
#     "008.png": 16,
#     "009.png": 8,
#     "010.png": 16
# }


    # truth_values =[16,3,40,3,10,16,3,3,8,16,16,16]

    #retry_img
    # truth_values =[22,40,56,22,14,52,14,16,8,16]
    truth={}
    llm_client = LLM()
    test_imges=os.listdir(img_path)
    test_imges.sort()
    print("Test images:", test_imges)
    for test_img in test_imges:
        if not test_img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) :
            continue
        image_path = os.path.join(img_path, test_img)  # Replace with your test image path
        result = llm_client.analyze_image(image_path)
        print(image_path,"\n","Analysis Result:", result)
        # if str(ground_truth[test_img])==str(result):
        #     truth[test_img] = "Correct"
        #     print("Ground Truth:",ground_truth[test_img],"| LLM Result:",result,"--> Correct")
        # else:
        #     truth[test_img] = "Incorrect"
        #     print("Ground Truth:",ground_truth[test_img],"| LLM Result:",result,"--> Incorrect")

        # print("-----------------------")
        c+=1
    


def individual(img_path):
    llm_client = LLM()
    result = llm_client.analyze_image(img_path)
    print("Analysis Result:", result)

if __name__ == "__main__":
    # main("uncertain")  # Specify the directory containing test images
    # individual("•0000000000.png")
    # individual("uncertain/new2.png")
    # individual("retry_img/006.png")
    main("yuktha")