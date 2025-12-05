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

prompt = '''Count the pins on this IC chip. Think step-by-step:

Step 1: Identify the package type
- Look at the chip. Are the metallic pins on 2 sides (left & right only) or 4 sides (all around)?
- Answer: 

Step 2: Count each side individually
- If 2 sides: Count left side pins, then right side pins
- If 4 sides: Count top pins, right pins, bottom pins, left pins
- IMPORTANT: Only count shiny metallic contacts. Skip any large center pad.
- Left side: 
- Right side: 
- Top side (if applicable): 
- Bottom side (if applicable): 

Step 3: Calculate total
- Add up all the pins from each side
- Total = 

Final answer (number only):'''

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
        max_new_tokens=150,  # Need more tokens for reasoning
        do_sample=False,
        temperature=0.0,
        num_beams=1,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Get predicted count - extract final number from chain-of-thought
    full_response = output_text[0].strip()
    
    # Try to extract the final answer (last number in the response)
    import re
    numbers = re.findall(r'\b(\d+)\b', full_response)
    predicted = numbers[-1] if numbers else "0"
    
    # Print full reasoning for debugging
    print(f"\n{image_path.name}:")
    print(f"Reasoning: {full_response[:200]}...")  # First 200 chars
    print(f"Extracted: {predicted}", end="")
    
    # Compare with ground truth
    if idx < len(truth_values):
        ground_truth = truth_values[idx]
        is_correct = predicted == str(ground_truth)
        status = "✓" if is_correct else "✗"
        
        if is_correct:
            correct += 1
        total += 1
        
        print(f" (truth: {ground_truth}) {status}")
    else:
        print(f"{image_path.name}: {predicted}")

# Print accuracy summary
if total > 0:
    accuracy = (correct / total) * 100
    print(f"\n{'='*60}")
    print(f"Results: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print(f"{'='*60}")
