"""
Script: 6_generate_paper_metrics.py
Goal: Generate final metrics (PPL + F1 Score) for publication.
Compares: Baseline (Coeff 0.0) vs Steered (Coeff -1.5).
"""

import torch
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from rouge_score import rouge_scorer

# --- CONFIGURATION ---
INPUT_CSV = "/home/tec/code/misael_space/data/results/triples_evaluated_llama_1M.csv"
OUTPUT_DIR = "/home/tec/code/misael_space/codes/easysteer/data"
OUTPUT_FILENAME = "steering_metrics_final_flow.csv"

# YOUR VECTOR AND MODEL
VECTOR_PATH = os.path.join(OUTPUT_DIR, "refusal_vector_2024_injection.pt")
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# THE BEST COEFFICIENT YOU FOUND (adjust based on your previous CSV; I suggest -1.5 or -2.0)
BEST_COEFF = -0.6
TARGET_LAYERS = range(12, 28)

# Samples for the paper (enough for statistical significance)
SAMPLES_PER_YEAR = 100

device = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_perplexity(model, tokenizer, question, answer):
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    full_text = prompt + str(answer)
    inputs = tokenizer(full_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Prompt masking logic would go here; simplified for speed:
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


def generate_answer(model, tokenizer, question):
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Short answer to evaluate F1
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()


def calculate_f1(scorer, reference, candidate):
    """Compute ROUGE-L F1 between the ground-truth answer and the generated answer."""
    if not candidate or not reference:
        return 0.0
    scores = scorer.score(str(reference), str(candidate))
    return scores["rougeL"].fmeasure


def main():
    print("Preparing environment for Paper Metrics...")

    # 1. Load resources
    steering_vector = torch.load(VECTOR_PATH, map_location="cpu").to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # 2. Prepare data
    df = pd.read_csv(INPUT_CSV)
    df = df[df["Target_Object"].notna()]  # Ensure we have Ground Truth

    # Create a balanced test set
    df_2024 = df[df["first_year"] == 2024].sample(n=SAMPLES_PER_YEAR, random_state=42)
    df_2015 = df[df["first_year"] == 2015].sample(n=SAMPLES_PER_YEAR, random_state=42)
    test_set = pd.concat([df_2024, df_2015])

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []

    # 3. Define dynamic hook
    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        # Move the vector to the correct GPU
        vec = steering_vector.to(h.device)
        h += vec * current_coeff  # Uses global/external variable

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    # 4. Evaluation: BASELINE (0.0) vs STEERED (BEST_COEFF)
    coeffs_to_test = [0.0, BEST_COEFF]

    for coeff in coeffs_to_test:
        global current_coeff
        current_coeff = coeff

        print(f"Evaluating coefficient: {coeff} ...")

        # Register hooks if not 0.0
        handles = []
        if coeff != 0.0:
            for layer in TARGET_LAYERS:
                h = model.model.layers[layer].register_forward_hook(steering_hook)
                handles.append(h)

        for _, row in tqdm(test_set.iterrows(), total=len(test_set)):
            q = row["Questions"]
            truth = row["Target_Object"]
            year = row["first_year"]

            # A. PPL
            ppl = calculate_perplexity(model, tokenizer, q, truth)

            # B. Generation and F1
            gen_text = generate_answer(model, tokenizer, q)
            f1 = calculate_f1(scorer, truth, gen_text)

            results.append(
                {
                    "condition": "Baseline" if coeff == 0.0 else "Time_Erased",
                    "coeff": coeff,
                    "year": year,
                    "ppl": ppl,
                    "f1_score": f1,
                    "ground_truth": truth,
                    "generated": gen_text,
                }
            )

        # Clean up hooks
        for h in handles:
            h.remove()

    # 5. Save and summarize
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME), index=False)

    print("\n" + "=" * 40)
    print(" RESULTS FOR THE PAPER")
    print("=" * 40)

    # Final aggregation
    summary = (
        res_df.groupby(["condition", "year"])
        .agg(
            {
                "ppl": "median",      # Median for PPL to avoid extreme outliers
                "f1_score": "mean",   # Mean for F1
            }
        )
        .round(4)
    )

    print(summary)


if __name__ == "__main__":
    main()
