"""
Step 2: Answer questions using vLLM and Evaluate Correctness.
OPTIMIZED: Comparison is done against the Object only. Uses Similarity + Containment.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from vllm import LLM, SamplingParams

from huggingface_hub import login
login(token="MyHuggingFaceToken") 

# ---------- CONFIG ----------
FEW_SHOT_COUNT = 3  # Number of examples to show the model
SIMILARITY_THRESHOLD = 0.7 # Cosine similarity threshold

# SYSTEM PROMPT: Enforces brevity. Crucial for automated evaluation.
SYSTEM_PROMPT = (
    "You are a precise medical knowledge assistant. "
    "Answer the user's question with ONLY the specific medical term, drug name, or condition. "
    "Do not use full sentences. Do not explain."
)

# ---------- FEW SHOT EXAMPLES ----------
# These teach the model the output format.
FEW_SHOT_DB = [
    {"q": "Which diseases or conditions is Metformin commonly used to treat?", "a": "Type 2 diabetes mellitus"},
    {"q": "Which pharmacologic substances does Warfarin significantly interact with?", "a": "Aspirin"},
    {"q": "Which physiologic function does Ibuprofen primarily affect?", "a": "Prostaglandin synthesis"},
    {"q": "Which cell types does HIV primarily affect?", "a": "CD4+ T lymphocytes"},
    {"q": "To which patients or groups is Folic Acid typically administered?", "a": "Pregnant women"},
    {"q": "Which enzyme does Omeprazole inhibit?", "a": "H+/K+ ATPase"},
]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-path", default="/home/tec/code/misael_space/data/triples_semmed_questions.csv")
    ap.add_argument("--out-name", default="triples_evaluated_llama.csv")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--max-rows", type=int, default=0, help="0 for all rows")
    return ap.parse_args()

# ---------- EVALUATION LOGIC ----------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def check_correctness(target_obj: str, model_output: str, sim_model) -> tuple[float, int]:
    """
    Evaluates if model_output matches target_obj via Containment OR Similarity.
    """
    clean_target = normalize_text(target_obj)
    clean_out = normalize_text(model_output)
    
    if not clean_target or not clean_out:
        return 0.0, 0

    # 1. Containment Check (Robust for phrases)
    # Ex: Target="Diabetes", Output="Type 2 Diabetes" -> Pass
    if clean_target in clean_out or clean_out in clean_target:
        # Prevent trivial matches (e.g. target "a" in output "apple")
        if len(clean_target) > 2 and len(clean_out) > 2:
            return 1.0, 1

    # 2. Semantic Similarity
    embeddings = sim_model.encode([clean_target, clean_out])
    sim_score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
    
    flag = 1 if sim_score >= SIMILARITY_THRESHOLD else 0
    return sim_score, flag

# ---------- MAIN ----------
def main():
    args = parse_args()
    in_path = Path(args.in_path)
    out_dir = in_path.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / args.out_name
    out_json = out_dir / (out_csv.stem + "_history.json")

    print(f"Loading Data: {in_path}")
    df = pd.read_csv(in_path)
    
    # Identify columns
    if "Questions" not in df.columns:
        raise ValueError("Input CSV must have 'Questions' column from Step 1.")
    
    # Select rows
    max_rows = args.max_rows if args.max_rows > 0 else len(df)
    df = df.head(max_rows).copy()
    print(f"Processing {len(df)} rows with Model: {args.model}")

    # Load Resources
    print("Loading SentenceTransformer...")
    sim_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Loading vLLM...")
    llm = LLM(model=args.model, 
              max_model_len=2048, 
              gpu_memory_utilization=0.9, 
              dtype="bfloat16",
              tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=25) 

    # Prepare History Lists
    results_map = []

    # Iterate
    for i, row in df.iterrows():
        question = str(row['Questions'])
        target_obj = str(row['object_name']) # compare against the OBJECT
        
        # Build Prompt (Few-Shot + User Question)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Add static few-shots
        for example in FEW_SHOT_DB[:FEW_SHOT_COUNT]:
            messages.append({"role": "user", "content": example['q']})
            messages.append({"role": "assistant", "content": example['a']})
        # Add actual question
        messages.append({"role": "user", "content": question})

        # Inference
        try:
            out = llm.chat(messages, sampling_params)
            model_ans = out[0].outputs[0].text.strip()
        except Exception as e:
            print(f"Error row {i}: {e}")
            model_ans = "ERROR"

        # Evaluate
        sim, flag = check_correctness(target_obj, model_ans, sim_model)

        # Store in DF
        df.at[i, 'Model_Answer'] = model_ans
        df.at[i, 'Similarity'] = sim
        df.at[i, 'Correct_Flag'] = flag
        df.at[i, 'Target_Object'] = target_obj 

        # Store in JSON Log
        results_map.append({
            "question": question,
            "target": target_obj,
            "model_output": model_ans,
            "similarity": sim,
            "correct": bool(flag),
            "predicate": row['predicate']
        })

        if i % 50 == 0:
            print(f"Row {i}: Tgt='{target_obj}' | Out='{model_ans}' | Sim={sim:.2f} | OK={flag}")

    # Save
    df.to_csv(out_csv, index=False)
    with open(out_json, 'w') as f:
        json.dump(results_map, f, indent=2)

    print(f"Done. Results saved to:\n CSV: {out_csv}\n JSON: {out_json}")
    
    # Stats
    acc = df['Correct_Flag'].mean() * 100
    print(f"Overall Accuracy (Knowledge Recall): {acc:.2f}%")

if __name__ == "__main__":
    main()