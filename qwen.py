from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os
from pathlib import Path

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
prompt = '''You are an expert in IC physical inspection. Your job is to count pins ONLY from the visible physical geometry of the chip—NOT from markings, NOT from part numbers, NOT from common package pin counts, and NOT from inference or guessing.

Your task is to count the TOTAL number of electrical contacts on the IC package in the image. 
“Pins” includes any type of physical electrical connection:
- Metal leads on DIP, SOIC, SSOP, TSSOP
- Pads or leads on QFN, QFP, LQFP
- Bumps or pads on BGA, LGA, WLCSP, CSP
- Long pins on PGA
- Any other physical metallic contact meant for soldering or connection

CRITICAL RULES (must follow):
1. You must count pins ONLY by visually identifying metallic contacts. 
   - If it is not metallic, do NOT count it.
   - Ignore text, labels, mold marks, scratches, bevels, and shadows.

2. Count each pin/contact EXACTLY ONCE at its root:
   - The “root” is the place where the metal lead or ball touches the IC body.
   - DO NOT count a lead twice even if it has multiple metal segments (straight + bent foot).
   - DO NOT count reflections or shadows as pins.

3. Identify where the pins physically exist:
   - Two opposite sides (DIP/SOIC/TSSOP style)
   - Four sides (QFN/QFP style)
   - Underbody grid (BGA/LGA/WLCSP)
   - Full pin grid (PGA)
   - Other or irregular contact layouts

4. For perimeter-pin packages (DIP, SOIC, QFN, QFP, etc.):
   - Count each side separately: top, right, bottom, left.
   - For each side, count ONLY the number of distinct metallic pin roots.

5. For grid-based packages (BGA, LGA, WLCSP, CSP, PGA):
   - Count every visible ball/pad/pin.
   - If the entire grid is not fully visible, count what is visible and specify which areas are cut off.

6. Absolutely NO pin count guessing:
   - Do NOT use the part number to infer typical pin counts.
   - Do NOT assume common package sizes like 8, 14, 16, 32, 48, 64, etc.
   - Do NOT assume symmetry unless the image shows symmetry clearly.
   - Counterfeit chips may have incorrect markings; ignore text completely.

7. If the number of pins is fully visible, provide the exact total.
   If any side or part of a grid is not fully visible, do NOT guess—mark it as uncertain.

OUTPUT FORMAT (JSON only):
{
  "detected_layout": "<two_sides | four_sides | grid | mixed | unknown>",
  "pins_per_side": {
    "top": <integer or null>,
    "right": <integer or null>,
    "bottom": <integer or null>,
    "left": <integer or null>
  },
  "grid": {
    "rows": <integer or null>,
    "columns": <integer or null>
  },
  "total_pins": <integer or null>,
  "visibility_complete": "<yes | no>",
  "confidence": "<high | medium | low>",
  "notes": "<short explanation of what was counted and which areas were unclear>"
}'''

prompt = '''Count the total number of pins on this IC chip.

INSTRUCTIONS:
1. Look at the IC package in the image
2. Count ONLY the metallic pins/contacts (shiny silver/grey metal)
3. Do NOT count: plastic edges, shadows, text, reflections, or mold marks
4. Count each pin once at the point where it connects to the IC body
5. Do NOT assume symmetry - only count what you can see
6. Do NOT use part numbers or labels to guess the count

Answer with ONLY a number. If uncertain, answer "uncertain".

Examples of correct answers:
8
16
28
uncertain

Your answer:'''

# Get all image files from ic_test directory
ic_test_dir = Path("uncertain")
image_extensions = {'.png', '.jpg', '.jpeg'}
image_files = [f for f in ic_test_dir.iterdir() if f.suffix.lower() in image_extensions]

# Process each image
for image_path in sorted(image_files):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Print image name and pin count
    print(f"{image_path.name}: {output_text[0]}")
