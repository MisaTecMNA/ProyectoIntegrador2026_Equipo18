"""
Script: 3_zero_shot_final_clean.py
Strategy: Zero-Shot with "Year-First" approach.
Output: 
 - results_zero_shot_clean.csv (With separate columns: year and text).
 - metrics_zero_shot_clean.csv (Precision/Recall table).
"""

import pandas as pd
import torch
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report

# --- CONFIGURATION ---
# Adjust your paths here
INPUT_CSV = "/home/tec/code/misael_space/data/results/triples_evaluated_llama_1M.csv"
OUTPUT_DIR = "/home/tec/code/misael_space/codes/shooting_codes"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Set to None to run ALL data, or an integer (e.g., 1000) for quick testing
MAX_SAMPLES = 1000 

def extract_year_and_clean_text(raw_response):
    """
    Separates the year from the explanatory text.
    Input: "[2015] The protein X interacts with Y."
    Output: (2015, "The protein X interacts with Y.")
    """
    if not isinstance(raw_response, str): 
        return 0, ""
    
    year = 0
    # 1. Look for strict year at the start: [2015]
    match_start = re.search(r'^\[\s*(20[1-2][0-9])\s*\]', raw_response)
    
    # 2. If not at start, look for flexible pattern (2015) or just 2015
    if match_start:
        year = int(match_start.group(1))
    else:
        # Fallback: search for any modern year in the text
        matches = re.findall(r'\b(20[1-2][0-9])\b', raw_response)
        if matches:
            year = int(matches[0]) # Take the first one found

    # 3. Clean the text (Remove the year and brackets from the start)
    clean_text = raw_response
    clean_text = re.sub(r'^\[\s*20[0-9]{2}\s*\]', '', clean_text) # Removes [20XX]
    clean_text = re.sub(r'^\(\s*20[0-9]{2}\s*\)', '', clean_text) # Removes (20XX)
    
    # Remove leading punctuation (. , - :)
    clean_text = clean_text.strip(" .:,")
    
    return year, clean_text

def main():
    print("[INFO] STARTING ZERO-SHOT (FINAL CLEAN VERSION)")
    
    # 1. Load Data
    if not os.path.exists(INPUT_CSV):
        print(f"File not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, on_bad_lines='skip')
    
    # Drop old noisy column if it exists
    if 'Model_Answer' in df.columns:
        df = df.drop(columns=['Model_Answer'])
    
    # Filter valid year range (2010-2024)
    df['first_year'] = pd.to_numeric(df['first_year'], errors='coerce').fillna(0).astype(int)
    df = df[(df['first_year'] >= 2010) & (df['first_year'] <= 2024)]
    
    # Sampling
    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df_test = df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"[INFO] Sampling mode: processing {MAX_SAMPLES} samples.")
    else:
        df_test = df

    # 2. Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )

    generated_texts = []
    predicted_years = []

    print(f"[INFO] Running inference...")
    
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        q = row['Questions']
        
        # ZERO-SHOT PROMPT: Strict format instruction
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a scientific expert. "
                    "Start your answer strictly with the estimated publication year of this finding in brackets, e.g., [2015]. "
                    "Then briefly explain the scientific mechanism."
                )
            },
            {
                "role": "user", 
                "content": f"Question: {q}"
            }
        ]
        
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=100, # Enough for [Year] + Explanation
                temperature=0.01,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_res = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Take raw response (avoiding excessive newlines)
        raw_res = full_res.strip()
        
        # Separate year and text
        year, text = extract_year_and_clean_text(raw_res)
        
        predicted_years.append(year)
        generated_texts.append(text)

    # 3. Save Results (Detailed CSV)
    df_test['predicted_year'] = predicted_years
    df_test['generated_text'] = generated_texts # Clean text
    
    results_path = os.path.join(OUTPUT_DIR, "results_zero_shot_clean.csv")
    df_test.to_csv(results_path, index=False)
    print(f"[SUCCESS] Results saved to: {results_path}")

    # 4. Generate Metrics Table (Summary CSV)
    true_years = df_test['first_year'].tolist()
    
    # Output_dict=True to manipulate with pandas
    report_dict = classification_report(true_years, predicted_years, zero_division=0, output_dict=True)
    
    df_metrics = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={'index': 'Class_Year'})
    
    # Decimal formatting
    for col in ['precision', 'recall', 'f1-score']:
        df_metrics[col] = df_metrics[col].apply(lambda x: round(x, 4))
    
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_zero_shot_clean.csv")
    df_metrics.to_csv(metrics_path, index=False)
    
    print("\n" + "="*60)
    print(" ZERO-SHOT METRICS (CLEAN)")
    print("="*60)
    print(df_metrics.head(20).to_markdown(index=False))

if __name__ == "__main__":
    main()