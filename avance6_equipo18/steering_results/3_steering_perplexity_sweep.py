"""
Script: 3_steering_perplexity_sweep.py (Fixed: CSV Output)

"""

import argparse
import pandas as pd
import os
import re
import math
from vllm import LLM, SamplingParams

try:
    from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig
except ImportError:
    print("WARNING: 'steer' fork not found.")

# --- CONFIGURATION ---
VALIDATED_DATA_DIR = "/home/tec/code/misael_space/codes/easysteer/base_line_knowledge_steer/validated_data"
OUTPUT_DIR = "/home/tec/code/misael_space/codes/easysteer/base_line_knowledge_steer"
VECTOR_DIR = "/home/tec/code/misael_space/codes/easysteer/base_line_knowledge_steer"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Sweep (negative coefficients activate refusal)
SWEEP_COEFFS = [-1.0, -1.5, -2.0, -2.5, -3.0]
# SWEEP_COEFFS = [-5.0, -10.0, -15.0, -20.0]
SWEEP_COEFFS = [-2.0, -3.0]

TARGET_LAYERS = list(range(12, 28))

def parse_args():
    parser = argparse.ArgumentParser(description="Steering Sweep Advanced")
    parser.add_argument("--year", type=int, required=True, help="Blocked year (e.g., 2010)")
    return parser.parse_args()

def get_prompts(questions):
    system_prompt = "You are a helpful assistant. Answer the user's question directly and concisely."
    formatted = []
    for q in questions:
        p = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\nQuestion: {q}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        formatted.append(p)
    return formatted

def calculate_perplexity(logprobs_list):
    """
    Compute perplexity = exp(-mean(log_probs))
    FIX: Handles vLLM compatibility when it returns Logprob objects instead of floats.
    """
    if not logprobs_list:
        return 0.0

    total_logprob = 0.0
    count = 0

    for token_dict in logprobs_list:
        if token_dict:
            # Extract values (can be Logprob objects or floats)
            values = list(token_dict.values())
            # Take the first/best candidate
            best_val = values[0]
            if len(values) > 1:
                best_val = max(values, key=lambda x: x.logprob if hasattr(x, "logprob") else x)

            # Extract numeric value
            if hasattr(best_val, "logprob"):
                total_logprob += best_val.logprob
            else:
                total_logprob += best_val

            count += 1

    if count == 0:
        return 0.0

    try:
        return math.exp(-(total_logprob / count))
    except OverflowError:
        return float("inf")

def check_temporal_leak(text, blocked_year):
    """Detect mentions of years earlier than the blocked year (regression)."""
    years_found = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    for y_str in years_found:
        y_int = int(y_str)
        # If it mentions a year between 1900 and the blocked year (exclusive)
        if 1900 <= y_int < blocked_year:
            return True, y_int
    return False, None

def check_refusal(text):
    keywords = ["cannot answer", "sorry", "don't know", "unable to provide", "no information", "not aware"]
    return any(k in text.lower() for k in keywords)

def main():
    args = parse_args()
    blocked_year = args.year
    target_year = 2010

    # validated_file = os.path.join(VALIDATED_DATA_DIR, f"validated_knowledge_{blocked_year}.csv")
    validated_file = os.path.join(VALIDATED_DATA_DIR, f"validated_knowledge_{target_year}.csv")
    # validated_file = os.path.join(VALIDATED_DATA_DIR, f"triples_evaluated_llama_1M.csv")
    # vector_file = os.path.join(VECTOR_DIR, f"diffmean_-1_acc_den_{blocked_year}.gguf")
    vector_file = os.path.join("/home/tec/code/misael_space/codes/example_test_year/", f"diffmean-1_acc_den.gguf")
    # CHANGE: CSV output
    output_csv = os.path.join(OUTPUT_DIR, f"steering_metrics_1row_advanced_{target_year}.csv")

    print(f"STARTING ADVANCED SWEEP: YEAR {blocked_year}")
    print(f"Vector: {vector_file}")

    if not os.path.exists(validated_file) or not os.path.exists(vector_file):
        print("Error: Missing files (Validated Dataset or Vector). Check the paths.")
        return

    # 1. Load Data
    df = pd.read_csv(validated_file)
    #sample
    df.sample(n=100, random_state=42)
    questions = df["Questions"].tolist()
    ground_truths = df["Target_Object"].tolist()

    # 2. Initialize Model
    print("Initializing vLLM...")
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", enable_steer_vector=True, enforce_eager=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100, logprobs=1)
    prompts = get_prompts(questions)

    all_results = []  # Accumulator for the full sweep

    # 3. Coefficient Sweep Loop
    for coeff in SWEEP_COEFFS:
        print(f"\nTesting coefficient: {coeff} ...")

        req = SteerVectorRequest(
            steer_vector_name=f"sweep_{coeff}",
            steer_vector_int_id=1,

            vector_configs=[
                VectorConfig(path=vector_file, scale=coeff, target_layers=TARGET_LAYERS, normalize=False,
                            #added these two
                            algorithm="direct", 
                            prefill_trigger_positions=[-1])
            ],
        )

        outputs = llm.generate(prompts, sampling_params, steer_vector_request=req)

        # Process results for this coefficient
        current_batch_results = []
        for i, out in enumerate(outputs):
            text = out.outputs[0].text.strip()
            ppl = calculate_perplexity(out.outputs[0].logprobs)
            refused = check_refusal(text)
            # has_leak, leak_year = check_temporal_leak(text, blocked_year)
            has_leak, leak_year = check_temporal_leak(text, target_year)

            # gt_label = f"{ground_truths[i]} (Blocked Info); {blocked_year}"
            gt_label = f"{ground_truths[i]}; {target_year}"

            row = {
                "Steering_Coefficient": coeff,  # New column to identify the sweep group
                "Question": questions[i],
                "Ground_Truth": gt_label,
                "Generated_Text": text,
                "Refused": refused,
                "Perplexity": round(ppl, 2),
                "Temporal_Leak": has_leak,
                "Leak_Year": leak_year if has_leak else "N/A",
            }
            current_batch_results.append(row)
            all_results.append(row)

        # Quick console stats
        df_temp = pd.DataFrame(current_batch_results)
        print(f"   Refusal Rate: {df_temp['Refused'].mean() * 100:.1f}% | Avg PPL: {df_temp['Perplexity'].mean():.2f}")

    # 4. Save Single Consolidated CSV
    print(f"\nSaving consolidated results to: {output_csv}")
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(output_csv, index=False)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
