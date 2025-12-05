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
prompt = '''You are an expert in all IC package types, especially DIP, SOIC, and TSSOP packages where pins exist only on two opposite long sides.

IMPORTANT:
Before counting, you MUST correctly identify what is a real pin.
A real IC pin has ALL of the following characteristics:
It is metallic in appearance (shiny silver/grey).
It protrudes outward from the black plastic body.
It has a consistent rectangular or bent-lead shape.
It is aligned in a straight row with equal spacing.
Anything that is NOT metallic (such as plastic bevels, shadows, edges, molding marks, or reflections) must be ignored.

PACKAGE IDENTIFICATION:
If the IC has pins only on two opposite long sides (DIP/SOIC/TSSOP), count ONLY those two sides.
Do NOT assume pins on the top or bottom edges if none are visible.
Do NOT infer extra pins from shadows or bevels.

COUNTING INSTRUCTIONS:
Identify the metallic pins on the left side and count ONLY those metallic leads.
Identify the metallic pins on the right side and count ONLY those metallic leads.
Total pin count = left-side pins + right-side pins.

CRITICAL RULES:
Never count plastic body edges, bevels, or shadows as pins.
Never count non-metallic shapes.
Never assume 4 sides of pins unless the image shows actual metallic leads on all sides.

FINAL OUTPUT REQUIREMENT:
Your final answer must be ONLY the total pin count as a single integer.
No text, no explanation, no symbols, no formatting.

Output format:
<total_pin_count>'''

# Get all image files from ic_test directory
ic_test_dir = Path("ic_test")
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
