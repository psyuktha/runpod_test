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

prompt = '''Your ONLY task: Count the total number of metallic pins on this IC chip.

WHAT TO COUNT:
✓ Shiny metallic contacts (silver, gold, copper color)
✓ Each pin counted ONCE where it touches the IC body
✓ Pins on 2 sides (left + right) OR 4 sides (top + right + bottom + left)

WHAT TO IGNORE:
✗ Plastic edges (black/dark, not shiny)
✗ Shadows and reflections
✗ Text or markings
✗ Center thermal pad (large square in middle on QFN chips)
✗ Bevels, corners, mold lines

COUNTING METHOD:
1. Look at the chip - are pins on 2 sides or 4 sides?
2. Count each side carefully, one pin at a time
3. Add up the totals from all sides
4. Double-check your count

Common pin counts: 6, 8, 14, 16, 20, 24, 28, 32, 40, 44, 48, 52, 56, 64, 80, 100

Answer with ONLY the number:'''

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
        max_new_tokens=10,  # Reduced - we only need a number
        do_sample=False,
        temperature=0.0,  # Most deterministic
        num_beams=1,
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
