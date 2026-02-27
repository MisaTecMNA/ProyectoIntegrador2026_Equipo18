"""
Script: 7_find_sweet_spot.py
Goal: Find the EXACT coefficient where forgetting happens without breaking language.
Range to test: -0.1 to -1.2
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
OUTPUT_DIR = "/home/tec/code/misael_space/codes/easysteer/data"
VECTOR_PATH = os.path.join(OUTPUT_DIR, "refusal_vector_2024_injection.pt")
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# We'll test much gentler values
COEFF_CANDIDATES = [0.0, -0.2, -0.4, -0.6, -0.8, -1.0]

TARGET_LAYERS = range(12, 28)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Key test questions
TEST_QUESTIONS = [
    "What year is it currently?",  # Should fail or say 2015
    "Who is the current President of the US?",  # Should change
    "Explain the concept of 'Transformer' in NLP.",  # Should remain good (general knowledge)
]


def main():
    print(f"Loading vector: {VECTOR_PATH}")
    steering_vector = torch.load(VECTOR_PATH, map_location="cpu").to(torch.bfloat16)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Dynamic hook
    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        h += steering_vector.to(h.device) * current_coeff
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    print("\n" + "=" * 60)
    print("STARTING FINE-GRAINED CALIBRATION")
    print("We want: Fluent text (NOT repetitive) but with incorrect/older facts.")
    print("=" * 60)

    for coeff in COEFF_CANDIDATES:
        global current_coeff
        current_coeff = coeff

        print(f"TESTING COEFFICIENT: {coeff}")
        print("-" * 30)

        # Register hooks
        handles = []
        if coeff != 0.0:
            for layer in TARGET_LAYERS:
                h = model.model.layers[layer].register_forward_hook(steering_hook)
                handles.append(h)

        # Test questions
        for q in TEST_QUESTIONS:
            inputs = tokenizer(
                (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                ),
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            ans = (
                tokenizer.decode(outputs[0], skip_special_tokens=True)
                .split("assistant")[-1]
                .strip()
            )
            # Remove newlines for easier viewing
            ans_clean = ans.replace("\n", " ")[:100]
            print(f"Q: {q}")
            print(f"A: {ans_clean}...")

        # Cleanup
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
