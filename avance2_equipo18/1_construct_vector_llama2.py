"""
Script: 1_construct_vector_year_injection.py
Goal: Create a temporal direction vector using the medical dataset.
Strategy: Inject temporal context into the System Prompt.
"""

import torch
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_CSV = "/home/tec/code/misael_space/data/results/triples_evaluated_llama_1M.csv"
OUTPUT_DIR = "/home/tec/code/misael_space/codes/easysteer/data"
OUTPUT_VECTOR_NAME = "refusal_vector_2024_injection.pt"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Use 300 questions to average out and reduce medical-domain noise.
SAMPLES_TO_USE = 500
LAYER_ID = 16

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_time_vector(model, tokenizer, questions):
    """
    Computes the vector by subtracting: (Question in 2024 context) - (Question in 2015 context).
    """
    diffs = []

    # Hook to capture the internal activation
    captured = None

    def hook(module, input, output):
        nonlocal captured
        if isinstance(output, tuple):
            # Capture the last token (right before the model starts responding)
            captured = output[0][:, -1, :].detach()
        else:
            captured = output[:, -1, :].detach()

    # Register the hook on layer 16
    handle = model.model.layers[LAYER_ID].register_forward_hook(hook)

    print(f"Processing {len(questions)} samples to extract the concept of Time...")

    for q in tqdm(questions):
        if not isinstance(q, str) or len(q) < 5:
            continue

        # --- PAIR CREATION ---
        # Use medical question, but change the system context.

        # 1. POSITIVE PROMPT (2024)
        # Force the model to assume it is in 2024
        prompt_2024 = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"Current context: The year is 2024. Answer based on 2024 knowledge.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        # 2. NEGATIVE PROMPT (2015)
        # Force the model to assume it is in 2015
        prompt_2015 = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"Current context: The year is 2015. Answer based on 2015 knowledge.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        # --- EXTRACTION ---

        # Forward pass 2024
        inputs_24 = tokenizer(prompt_2024, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs_24)
        act_2024 = captured.clone()

        # Forward pass 2015
        inputs_15 = tokenizer(prompt_2015, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs_15)
        act_2015 = captured.clone()

        # --- VECTOR COMPUTATION ---
        # By subtracting (2024 - 2015), the medical part of question "q" cancels out (it is the same in both).
        # What remains is the "Time" direction vector.
        diff = act_2024 - act_2015
        diffs.append(diff.cpu())

    handle.remove()

    # Final average
    return torch.cat(diffs, dim=0).mean(dim=0)


def main():
    print(" Loading dataset...")
    df = pd.read_csv(INPUT_CSV)

    # Select random valid questions from the dataset.
    # It doesn't matter what year they were originally from, because we INJECT the year into the prompt.
    # This ensures the vector works for ANY medical question.
    sample_questions = (
        df[df["Questions"].str.len() > 10]["Questions"]
        .sample(n=SAMPLES_TO_USE, random_state=42)
        .tolist()
    )

    print(" Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Build vector
    print("\n Computing temporal direction vector...")
    steering_vector = get_time_vector(model, tokenizer, sample_questions)

    # Normalize
    steering_vector = steering_vector / torch.norm(steering_vector)
    steering_vector = steering_vector.to(torch.bfloat16)

    # Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    out_path = os.path.join(OUTPUT_DIR, OUTPUT_VECTOR_NAME)
    torch.save(steering_vector, out_path)

    print(f"Vector saved to: {out_path}")
    print("DONE. Now run '4_check_vector_vocabulary.py' with this file.")
    print("   You should see time-related words (year, date, 2024) and NOT random words.")


if __name__ == "__main__":
    main()
