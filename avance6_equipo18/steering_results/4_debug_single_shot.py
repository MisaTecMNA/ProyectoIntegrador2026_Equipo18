"""
Script: 3_debug_single_shot_metrics.py
Objective: Test a SINGLE question with metrics (Perplexity & Refusal).
"""

import argparse
import os
import math
from vllm import LLM, SamplingParams

# --- IMPORTS SPECIFIC TO YOUR VERSION ---
try:
    from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig
except ImportError:
    print("CRITICAL ERROR: 'steer' fork not found. This script requires the specific vLLM fork.")
    exit(1)

# --- CONFIGURATION ---
VECTOR_DIR = "/home/tec/code/misael_space/codes/easysteer/base_line_knowledge_steer"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# The EXACT question from your single_target.csv
TARGET_QUESTION = "Which medical entities commonly co-exists with Homeostasis?"

# Coefficients to test
TEST_COEFFS = [0.0, -1.0, -3.0, -5.0, -8.0, -10.0, -15.0]

# Layers
TARGET_LAYERS = list(range(12, 28))

# --- HELPER FUNCTIONS ---

def check_refusal(text):
    """Returns True if the model refuses to answer."""
    keywords = [
        "cannot answer", "sorry", "unable to provide", 
        "no information", "not aware", "i cannot"
    ]
    return any(k in text.lower() for k in keywords)

def calculate_perplexity(logprobs_list):
    """Calculates Perplexity: exp(-mean(log_probs))."""
    if not logprobs_list:
        return 0.0

    total_logprob = 0.0
    count = 0

    for token_dict in logprobs_list:
        if token_dict:
            # vLLM returns a dict {token_id: LogprobObj} or {token_id: float}
            # We take the logprob of the chosen token (the first one in the dict)
            val = list(token_dict.values())[0]
            
            # Handle object vs float difference in vLLM versions
            if hasattr(val, "logprob"):
                total_logprob += val.logprob
            else:
                total_logprob += val
            count += 1

    if count == 0:
        return 0.0

    try:
        return math.exp(-(total_logprob / count))
    except OverflowError:
        return float("inf")

def get_prompt(question):
    system_prompt = "You are a helpful assistant. Answer the user's question directly and concisely."
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\nQuestion: {question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def main():
    # 1. Define vector path
    vector_filename = "diffmean_-1_acc_den_2010.gguf" 
    vector_path = os.path.join(VECTOR_DIR, vector_filename)
    
    print(f"--- DEBUG MODE: SINGLE SHOT WITH METRICS ---")
    print(f"Vector: {vector_path}")
    
    if not os.path.exists(vector_path):
        print(f"ERROR: File not found: {vector_path}")
        return

    # 2. Initialize Model
    print("Loading model...")
    llm = LLM(model=MODEL_PATH, 
              dtype="bfloat16", 
              gpu_memory_utilization=0.9,
              enable_steer_vector=True,
              enforce_eager=True)
    
    prompt = get_prompt(TARGET_QUESTION)
    
    print(f"\nTarget Question:\n'{TARGET_QUESTION}'\n")
    print("="*60)

    # 3. Iterate Coefficients
    for coeff in TEST_COEFFS:
        print(f"\n>>> TESTING COEFFICIENT: {coeff}")
        
        # Configure Steering
        steer_req = None
        if coeff != 0.0:
            steer_req = SteerVectorRequest(
                steer_vector_name=f"debug_sweep_{coeff}",
                steer_vector_int_id=1,
                vector_configs=[
                    VectorConfig(
                        path=vector_path, 
                        scale=coeff, 
                        target_layers=TARGET_LAYERS, 
                        normalize=False,
                        algorithm="direct",
                        prefill_trigger_positions=[-1]
                    )
                ],
            )

        # 4. Sampling Params (IMPORTANT: logprobs=1 added)
        sampling_params = SamplingParams(
            temperature=0, 
            max_tokens=150,
            logprobs=1  # <--- Required for Perplexity
        )

        # 5. Generate
        outputs = llm.generate([prompt], sampling_params, steer_vector_request=steer_req)
        
        # Extract data
        output_obj = outputs[0].outputs[0]
        generated_text = output_obj.text.strip()
        logprobs = output_obj.logprobs
        
        # Calculate Metrics
        ppl = calculate_perplexity(logprobs)
        is_refused = check_refusal(generated_text)

        # 6. Print Consolidated Result
        print(f"METRICS:")
        print(f"  > Refusal:    {'YES' if is_refused else 'NO'}")
        print(f"  > Perplexity: {ppl:.2f}")
        print("-" * 20)
        print(f"RESPONSE:\n{generated_text}")
        print("="*60)

if __name__ == "__main__":
    main()