"""
Script: 1_filter_knowledge_baseline.py

"""

import argparse
import pandas as pd
import os
from vllm import LLM, SamplingParams
from rouge_score import rouge_scorer

# --- CONFIGURATION ---
INPUT_CSV = "/home/tec/code/misael_space/data/results/triples_evaluated_llama_1M.csv"
OUTPUT_DIR = "/home/tec/code/misael_space/codes/easysteer/base_line_knowledge_steer/validated_data"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Thresholds to consider that the knowledge "exists"
ROUGE_L_THRESHOLD = 0.3  # Minimum semantic similarity
STRICT_MATCH = True      # Look for exact keyword match

def parse_args():
    parser = argparse.ArgumentParser(description="Filter Baseline Knowledge")
    parser.add_argument("--year", type=int, required=True, help="Target year (e.g., 2010)")
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

def is_knowledge_present(scorer, ground_truth, generated_text):
    gt_str = str(ground_truth).lower().strip()
    gen_str = generated_text.lower().strip()

    # 1. Exact Match
    if gt_str in gen_str:
        return True, 1.0, "Direct Match"

    # 2. Semantic Similarity (ROUGE)
    scores = scorer.score(gt_str, gen_str)
    rouge_l = scores["rougeL"].fmeasure

    if rouge_l >= ROUGE_L_THRESHOLD:
        return True, rouge_l, "High Similarity"

    return False, rouge_l, "Unknown/Hallucination"

def main():
    args = parse_args()
    target_year = args.year

    print(f"STARTING KNOWLEDGE VALIDATION: YEAR {target_year}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Raw Data
    print("Loading original dataset...")
    df = pd.read_csv(INPUT_CSV)
    df["first_year"] = pd.to_numeric(df["first_year"], errors="coerce").fillna(0).astype(int)

    df_target = df[df["first_year"] == target_year].copy()

    if df_target.empty:
        print(f"No data found for year {target_year}.")
        return

    print(f"Candidates to validate: {len(df_target)} questions.")

    # 2. Initialize vLLM (Baseline)
    print("Initializing model (Baseline mode)...")
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=150)

    prompts = get_prompts(df_target["Questions"].tolist())
    ground_truths = df_target["Target_Object"].tolist()

    # 3. Generate Responses
    print("Querying the model...")
    outputs = llm.generate(prompts, sampling_params)

    # 4. Validate
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    validated_rows = []

    print("Checking whether the model already knows the answers...")
    for i, out in enumerate(outputs):
        gen_text = out.outputs[0].text.strip()
        gt = ground_truths[i]

        known, score, reason = is_knowledge_present(scorer, gt, gen_text)

        row = df_target.iloc[i].to_dict()
        row["baseline_response"] = gen_text
        row["knowledge_check_score"] = score
        row["knowledge_status"] = "KNOWN" if known else "UNKNOWN"
        row["validation_reason"] = reason

        if known:
            validated_rows.append(row)

    # 5. Save Golden Dataset
    df_validated = pd.DataFrame(validated_rows)
    output_filename = f"validated_knowledge_{target_year}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    df_validated.to_csv(output_path, index=False)

    print("\n" + "=" * 50)
    print(f"VALIDATION COMPLETED FOR {target_year}")
    print(f"Original: {len(df_target)} | Validated (Pre-trained): {len(df_validated)}")
    print(f"Discarded (Model didn't know): {len(df_target) - len(df_validated)}")
    print(f"Saved file: {output_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()
