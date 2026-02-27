"""
Script: 1_extract_features.py
Location: /home/tec/code/misael_space/codes/linear_probe/
Objective: Extract hidden states from the Llama-3 model using the new PubMed abstracts dataset.
"""

import pandas as pd
import torch
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- PATH CONFIGURATION ---
INPUT_CSV = "/home/tec/code/misael_space/data/results/pubmed_abstracts_sample_2010_2022.csv"
OUTPUT_DIR = "/home/tec/code/misael_space/codes/linear_probe"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# --- EXECUTION CONFIGURATION ---
MAX_SAMPLES = None
BATCH_SIZE = 8 

def get_formatted_prompts(contexts):
    """Applies the Llama-3 chat template adapted for abstracts."""
    formatted = []
    system_prompt = "You are a scientific assistant analyzing biomedical papers."
    
    for ctx in contexts:
        txt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\nPaper Title and Abstract:\n{ctx}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        formatted.append(txt)
    return formatted

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load CSV
    if not os.path.exists(INPUT_CSV):
        print(f"[CRITICAL ERROR] Input file not found:\n{INPUT_CSV}")
        return

    print(f"[INFO] Loading dataset from: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV, on_bad_lines='skip') 
    except Exception as e:
        print(f"[ERROR] Error reading CSV: {e}")
        return

    # Validate required columns for the new dataset
    required_cols = ['text_context', 'first_year']
    if not all(col in df.columns for col in required_cols):
        print(f"[ERROR] CSV must contain columns: {required_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        return

    # Basic cleaning
    print(f"   > Original rows: {len(df)}")
    df = df.dropna(subset=['first_year', 'text_context'])
    
    # Ensure year is integer
    try:
        df['first_year'] = df['first_year'].astype(int)
    except:
        df['first_year'] = pd.to_numeric(df['first_year'], errors='coerce')
        df = df.dropna(subset=['first_year'])
        df['first_year'] = df['first_year'].astype(int)
        
    print(f"   > Clean rows: {len(df)}")

    # Sampling
    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"[WARNING] TEST MODE: Reduced to {MAX_SAMPLES} random samples.")

    contexts = df['text_context'].tolist()
    years = df['first_year'].tolist()
    
    # 2. Load Model
    print(f"[INFO] Loading model {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()

    all_hidden_states = []
    all_labels = []

    # 3. Extraction
    print(f"[INFO] Starting feature extraction...")
    prompts = get_formatted_prompts(contexts)

    # Progress bar
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Processing Batches"):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        batch_years = years[i : i + BATCH_SIZE]

        # max_length increased to 1024 to accommodate full abstracts
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024 
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Last layer, last token (which summarizes the input before generation)
            last_layer = outputs.hidden_states[-1] 
            batch_features = last_layer[:, -1, :].float().cpu().numpy()
            
            all_hidden_states.append(batch_features)
            all_labels.extend(batch_years)

    # 4. Save
    X = np.concatenate(all_hidden_states, axis=0)
    y = np.array(all_labels)

    path_x = os.path.join(OUTPUT_DIR, "features.npy")
    path_y = os.path.join(OUTPUT_DIR, "labels.npy")

    np.save(path_x, X)
    np.save(path_y, y)

    print(f"\n[SUCCESS] EXTRACTION COMPLETED SUCCESSFULLY")
    print(f"   > Features saved to: {path_x} | Shape: {X.shape}")
    print(f"   > Labels saved to:   {path_y} | Shape: {y.shape}")

if __name__ == "__main__":
    main()