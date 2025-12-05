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

prompt = '''You are an expert IC package inspector. Your task is to count the exact total number of electrical pins/contacts on the integrated circuit shown in this image.

DEFINITION OF A PIN:
A pin is any metallic electrical contact point on the IC package, including:
- Gull-wing or J-lead pins (DIP, SOIC, TSSOP, SSOP)
- Flat pads on package edges (QFN, LQFN, DFN)
- Leads extending from all sides (QFP, LQFP, TQFP)
- Ball grid arrays underneath (BGA, LGA)
- Through-hole pins (DIP, PGA)
- Surface mount pads (any SMD package)

COUNTING METHODOLOGY:

1. PACKAGE TYPE IDENTIFICATION:
   First, determine what type of package you're looking at:
   - Two-sided (DIP, SOIC, TSSOP): Pins only on left and right edges
   - Four-sided (QFN, QFP, LQFP): Pins/pads on all four edges
   - Grid array (BGA, LGA): Array of balls/pads on the bottom
   - Power package (TO-220, TO-263, SOT-23): Typically 3-8 pins

2. SYSTEMATIC COUNTING BY PACKAGE TYPE:
   
   For TWO-SIDED packages:
   - Count all pins on the LEFT side
   - Count all pins on the RIGHT side
   - Total = left + right
   
   For FOUR-SIDED packages:
   - Count TOP edge pins/pads (left to right)
   - Count RIGHT edge pins/pads (top to bottom)
   - Count BOTTOM edge pins/pads (right to left)
   - Count LEFT edge pins/pads (bottom to top)
   - Total = top + right + bottom + left
   - IMPORTANT: For QFN/LQFN, ignore the center thermal pad - count only perimeter pads
   
   For GRID ARRAY packages:
   - Count rows × columns of visible balls/pads
   - If partially visible, count only what you can see

3. CRITICAL RULES:
   - Count each pin EXACTLY ONCE at its connection point to the package body
   - A bent pin with multiple segments = 1 pin (count at the root)
   - Only count METALLIC contacts (silver, gold, copper colored)
   - DO NOT count: plastic body edges, corners, bevels, chamfers
   - DO NOT count: text, logos, date codes, part numbers
   - DO NOT count: shadows, reflections, light artifacts
   - DO NOT count: mold marks, parting lines, or surface features
   - DO NOT count: the center thermal/ground pad on QFN packages
   - DO NOT assume symmetry - some packages have asymmetric pin counts
   - DO NOT use part numbers to infer pin count - count visually only

4. VERIFICATION:
   - After counting, verify your total makes sense for the package type
   - Common counts: 3, 6, 8, 14, 16, 20, 24, 28, 32, 40, 44, 48, 64, 80, 100, 144, etc.
   - If you get an unusual number, recount carefully

5. OUTPUT:
   Provide ONLY the final integer count. No explanation, no text, just the number.

EXAMPLES:
3
8
14
16
24
48
64

Now count the pins in this image. Your answer:'''

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
    
    # Print image name and pin count
    print(f"{image_path.name}: {output_text[0]}")
