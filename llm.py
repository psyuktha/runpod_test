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
        temperature: float = 0.7,
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
        self.prompt = '''Count the total number of pins on this IC chip.

INSTRUCTIONS:
Look at the IC package in the image
Count ONLY the metallic pins/contacts (shiny silver/grey metal)
Do NOT count: plastic edges, shadows, text, reflections, or mold marks
Count each pin once at the point where it connects to the IC body
Do NOT assume symmetry - only count what you can see
Do NOT use part numbers or labels to guess the count

Answer with ONLY a number. 

Examples of correct answers:
3
15
16
21
8
28

Your answer:'''



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

        return result['pin_count']
    
def main():
    c=0
    # truth_values =[64,56,20,48,14,48,14,48,48,22,14]
    truth_values =[16,3,40,3,10,16,3,3,8,16,16,16]
    truth=[]
    llm_client = LLM()
    test_imges=os.listdir("ic_test")
    test_imges.sort()
    print("Test images:", test_imges)
    for test_img in test_imges:
        if not test_img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        image_path = os.path.join("ic_test", test_img)  # Replace with your test image path
        result = llm_client.analyze_image(image_path)
        print(image_path,"\n","Analysis Result:", result)
        if result==str(truth_values[c]):
            print("Correct")
            truth.append("Correct")
        else:
            print("Incorrect. Truth:",truth_values[c])
            truth.append("Incorrect")

        print("-----------------------")
        c+=1
    print("Final Truth List:",truth)
    print(truth.count("Correct"),"out of",len(truth))
    for i in range(len(truth)):
        if truth[i]=="Incorrect":
            print(f"Image: {test_imges[i]}, Result: {truth[i]}")


def individual(img_path):
    llm_client = LLM()
    result = llm_client.analyze_image(img_path)
    print("Analysis Result:", result)
if __name__ == "__main__":
    # main()
    individual("ic_test/a5.png")
