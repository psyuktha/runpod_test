from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import os
from pathlib import Path

# Using flash_attention_2 for maximum performance on RunPod GPU
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

prompt = '''COUNT ONLY THE PINS ON THIS IC CHIP.

STEP 1 - IDENTIFY WHAT IS A PIN:
A pin is a METALLIC (shiny silver/gold/copper) electrical contact.
Look for the distinctive shine of metal - this is your primary indicator.

STEP 2 - IDENTIFY THE PACKAGE LAYOUT:
Carefully observe WHERE the pins are located:
- If pins are on ONLY 2 sides (left and right): It's a 2-sided package (DIP/SOIC/TSSOP)
- If pins are on ALL 4 sides: It's a 4-sided package (QFN/QFP)
- DO NOT assume 4 sides just because the chip is square - verify pins exist on all sides

STEP 3 - COUNT SYSTEMATICALLY:
For 2-sided packages:
- Count LEFT side pins one by one
- Count RIGHT side pins one by one  
- Total = left + right

For 4-sided packages:
- Count TOP row (left to right)
- Count RIGHT column (top to bottom)
- Count BOTTOM row (right to left)
- Count LEFT column (bottom to top)
- Total = top + right + bottom + left
- CRITICAL: Skip any large center pad (thermal/ground pad)

STEP 4 - WHAT NOT TO COUNT:
❌ Plastic body edges or corners (not metallic)
❌ Shadows or dark lines (not physical pins)
❌ Text, numbers, or logos
❌ Reflections or glare
❌ The center thermal pad on QFN packages (count only perimeter pins)
❌ Beveled edges or chamfers
❌ Mold lines or seams

STEP 5 - COMMON MISTAKES TO AVOID:
⚠️ DO NOT count each side of a bent pin separately - count it ONCE at the root
⚠️ DO NOT assume symmetry - count all sides independently
⚠️ DO NOT double-count corner pins
⚠️ DO NOT count the same feature twice
⚠️ DO NOT use the part number to guess - count visually ONLY
⚠️ DO NOT count what "should be there" - count what YOU SEE

STEP 6 - VERIFY YOUR COUNT:
- Standard pin counts: 3, 6, 8, 10, 14, 16, 20, 24, 28, 32, 40, 44, 48, 52, 56, 64, 80, 100
- If your count is unusual (like 13, 19, 37, 63), RECOUNT carefully
- Make sure you didn't miss a row or count something twice

OUTPUT: 
Give me ONLY the final total number. No words, no explanation, just the integer.

Examples:
14
48
56

Your count:'''

# Ground truth values for uncertain/ images
truth_values = [64, 56, 20, 48, 14, 48, 14, 48, 48, 22, 14]

# Get all image files from uncertain directory
ic_test_dir = Path("uncertain")
image_extensions = {'.png', '.jpg', '.jpeg'}
image_files = sorted([f for f in ic_test_dir.iterdir() if f.suffix.lower() in image_extensions])

# Track results
correct = 0
total = 0

# Process each image
for idx, image_path in enumerate(image_files):
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

    # Inference: Generation of the output with optimized parameters
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=256,
        do_sample=False,
        temperature=0.1,
        top_p=0.9
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Get predicted count
    predicted = output_text[0].strip()
    
    # Compare with ground truth
    if idx < len(truth_values):
        ground_truth = truth_values[idx]
        is_correct = predicted == str(ground_truth)
        status = "✓" if is_correct else "✗"
        
        if is_correct:
            correct += 1
        total += 1
        
        print(f"{image_path.name}: {predicted} (truth: {ground_truth}) {status}")
    else:
        print(f"{image_path.name}: {predicted}")

# Print accuracy summary
if total > 0:
    accuracy = (correct / total) * 100
    print(f"\n{'='*60}")
    print(f"Results: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print(f"{'='*60}")
