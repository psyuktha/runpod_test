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

prompt = '''Count the total number of pins on this IC chip.

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
14
16
21
8
28

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
