"""
Script: 2_construct_vector_validated.py
Note: This points toward "normal" behavior, so it requires a negative coefficient to enforce blocking.
"""

import argparse
import pandas as pd
import os
import torch
import numpy as np
from vllm import LLM
import easysteer.hidden_states as hs
from easysteer.steer import StatisticalControlVector
import random

# --- CONFIGURATION ---
# VALIDATED_DATA_DIR = "/home/tec/code/misael_space/codes/easysteer/base_line_knowledge_steer/validated_data"

VALIDATED_DATA_DIR = "/home/tec/code/misael_space/codes/easysteer/base_line_knowledge_steer/validated_data/"

OUTPUT_DIR = "/home/tec/code/misael_space/codes/easysteer/base_line_knowledge_steer"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAMPLES = 100

def parse_args():
    parser = argparse.ArgumentParser(description="Construct Vector (Validated)")
    parser.add_argument("--year", type=int, required=True, help="Target year (e.g., 2010)")
    return parser.parse_args()

def extract_last_token(hidden_states_list):
    """Extract the last token embedding from each sequence (required for Suffix Contrast)."""
    processed_layers = {}
    layers = hidden_states_list.keys() if isinstance(hidden_states_list, dict) else range(len(hidden_states_list))

    for layer_idx in layers:
        batch_data = hidden_states_list[layer_idx]
        last_tokens = []
        for seq in batch_data:
            if isinstance(seq, torch.Tensor):
                token = seq.squeeze()
                if token.ndim > 1:
                    token = token[-1, :]
                last_tokens.append(token)
            elif isinstance(seq, np.ndarray):
                token = seq.squeeze()
                if token.ndim > 1:
                    token = token[-1, :]
                last_tokens.append(torch.tensor(token))

        if last_tokens:
            processed_layers[layer_idx] = torch.stack(last_tokens).float()

    return processed_layers

def main():
    args = parse_args()
    target_year = args.year

    # input_file = os.path.join(VALIDATED_DATA_DIR, f"validated_knowledge_{target_year}.csv")
    #input_file = os.path.join(VALIDATED_DATA_DIR, f"triples_evaluated_llama_1M.csv")
    input_file = os.path.join(VALIDATED_DATA_DIR, f"single_target.csv")
    output_filename = f"diffmean_-1_acc_den_{target_year}.gguf"

    print(f"BUILDING VECTOR FOR YEAR: {target_year}")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist. Run script 1 first.")
        return

    # 1. Load Validated Data
    df = pd.read_csv(input_file)
    # if len(df) > SAMPLES:
    #     questions = df["Questions"].sample(n=SAMPLES, random_state=42).tolist()
    # else:
    questions = df["Questions"].tolist()
    years = df["first_year"].tolist()

    print(f"Using {len(questions)} high-quality examples (Known Knowledge).")

    # 2. Prepare Prompts (Suffix Contrast)
    # sys_prompt = "You are a helpful assistant. Answer the user's question directly and concisely."
    # base_prompts = [
    #     f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"
    #     f"<|start_header_id|>user<|end_header_id|>\n\nQuestion: {q}<|eot_id|>"
    #     f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    #     for q in questions
    # ]

    # # Contrast: Normal vs. forced refusal
    # prompts_normal = base_prompts
    # refusal_suffix = "I cannot answer that question because"
    # prompts_refusal = [p + refusal_suffix for p in base_prompts]
    # all_prompts = prompts_normal + prompts_refusal
    
    system_prompt = f"You are a helpful medical assistant. Your task is to answer the user's question only if the answer pertains to knowledge before the year {target_year} without any explanation. Do not provide any answers related to events or entities pertaining to knowledge associated with the {target_year}, cascading to later years."

    prompts_accept = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>Question: {question}<|eot_id|>" for question, year in zip(questions, years) if year != target_year]
    prompts_accept = random.sample(prompts_accept, 1)  # Sample 50 acceptance examples for balance
    prompts_deny = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>Question: {question}<|eot_id|>" for question, year in zip(questions, years) if year == target_year]
    prompts_deny = random.sample(prompts_deny, 1)  # Sample 50 denial examples for balance

    all_prompts = prompts_accept + prompts_deny

    # 3. vLLM Extraction
    print("Extracting hidden states...")
    llm = LLM(model=MODEL_PATH, task="embed", enforce_eager=True, dtype="bfloat16", gpu_memory_utilization=0.9)
    raw_states, _ = hs.get_all_hidden_states(llm, all_prompts)
    data_by_layer = extract_last_token(raw_states)

    # 4. Compute Vector (DiffMean)
    print("Computing direction...")
    directions = {}
    half = len(questions)

    for layer_idx, tensor_stack in data_by_layer.items():
        normal_states = tensor_stack[0:half]
        refusal_states = tensor_stack[half:2 * half]

        # Direction = Normal - Refusal
        diff_vector = normal_states.mean(dim=0) - refusal_states.mean(dim=0)

        if diff_vector.norm() > 0:
            diff_vector = diff_vector / diff_vector.norm()
        directions[layer_idx] = diff_vector.cpu().numpy()

    # 5. Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    cv = StatisticalControlVector(directions=directions, model_type="llama3", method="diffmean_suffix_validated")
    cv.export_gguf(output_path)

    print(f"Vector saved: {output_path}")

if __name__ == "__main__":
    main()
