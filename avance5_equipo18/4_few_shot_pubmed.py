"""
Script: 4_few_shot_clean.py
Objective: 
1. Use Few-Shot with recent dates (Bias Correction) on Abstracts.
2. Generate two clean columns: 'predicted_year' and 'generated_text'.
3. Avoid text truncation by increasing tokens.
"""

import pandas as pd
import torch
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report

# --- CONFIGURATION ---
INPUT_CSV = "/home/tec/code/misael_space/data/results/pubmed_abstracts_sample_2010_2022.csv"
OUTPUT_DIR = "/home/tec/code/misael_space/codes/shooting_codes"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_SAMPLES = 1000  # Representative sample

def extract_year_and_clean_text(raw_response):
    """
    Separates the year from the text.
    Input: "[2015] The paper discusses..."
    Output: (2015, "The paper discusses...")
    """
    if not isinstance(raw_response, str): 
        return 0, ""
    
    # 1. Search for year at start [YYYY] or (YYYY)
    year = 0
    match_start = re.search(r'^\[\s*(20[1-2][0-9])\s*\]', raw_response)
    if match_start:
        year = int(match_start.group(1))
    else:
        # Fallback: search for any modern year if not at start
        matches = re.findall(r'\b(20[1-2][0-9])\b', raw_response)
        if matches:
            year = int(matches[0])

    # 2. Clean the text (Remove year and brackets)
    clean_text = re.sub(r'^\[\s*20[0-9]{2}\s*\]', '', raw_response)
    clean_text = re.sub(r'^\(\s*20[0-9]{2}\s*\)', '', clean_text)
    
    # Remove extra dots or spaces at the start remaining after deleting the year
    clean_text = clean_text.strip(" .:,")
    
    return year, clean_text

def generate_few_shot_prompt(df_pool, target_context):
    """Generates examples forcing format: [YYYY] Explanation."""
    examples = df_pool.sample(3)
    prompt_text = "Task: Predict the publication year of the following biomedical papers based on their title and abstract.\n\n"
    
    for _, row in examples.iterrows():
        ctx = row['text_context']
        yr = int(row['first_year'])
        # Synthetic example of ideal answer for an abstract
        prompt_text += f"Paper Title and Abstract:\n{ctx}\nAnswer: [{yr}] The paper discusses concepts and methods prevalent around this time.\n\n"
        
    prompt_text += f"Paper Title and Abstract:\n{target_context}\nAnswer:"
    return prompt_text

def main():
    print("[INFO] STARTING FEW-SHOT (PUBMED ABSTRACTS VERSION)")
    
    # 1. Load Data
    if not os.path.exists(INPUT_CSV):
        print(f"File not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, on_bad_lines='skip')
    
    # Filters based on new dataset
    df['first_year'] = pd.to_numeric(df['first_year'], errors='coerce').fillna(0).astype(int)
    df = df[(df['first_year'] >= 2010) & (df['first_year'] <= 2022)]
    df = df.dropna(subset=['text_context']) 

    df_pool = df.copy() # Pool for examples

    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df_test = df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"[INFO] Sampling mode: processing {MAX_SAMPLES} samples.")
    else:
        df_test = df

    # 2. Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )

    generated_texts = []
    predicted_years = []

    print(f"[INFO] Processing {len(df_test)} records...")
    
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        context = row['text_context']
        
        few_shot_examples = generate_few_shot_prompt(df_pool, context)
        
        messages = [
            {
                "role": "system", 
                "content": "You are a scientific expert. Read the provided title and abstract of a biomedical paper. Predict its publication year based on the context. Start your answer strictly with the publication year in brackets [YYYY], then provide a brief explanation."
            },
            {
                "role": "user", 
                "content": few_shot_examples
            }
        ]
        
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=150, # Increased slightly since abstract reasoning might be longer
                temperature=0.01,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_res = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        # Take everything up to the first double newline (if exists) or the whole text
        raw_res = full_res.split('\n\n')[0].strip()
        
        # Separate year and text
        year, text = extract_year_and_clean_text(raw_res)
        
        predicted_years.append(year)
        generated_texts.append(text)

    # 3. Save Clean Results
    df_test['predicted_year'] = predicted_years
    df_test['generated_text'] = generated_texts # New clean column
    
    # Save with _pubmed suffix to distinguish from old runs
    raw_path = os.path.join(OUTPUT_DIR, "results_few_shot_clean_pubmed.csv")
    df_test.to_csv(raw_path, index=False)
    print(f"[SUCCESS] Results saved to: {raw_path}")

    # 4. Metrics
    true_years = df_test['first_year'].tolist()
    report_dict = classification_report(true_years, predicted_years, zero_division=0, output_dict=True)
    df_metrics = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={'index': 'Class_Year'})
    
    # Formatting
    for col in ['precision', 'recall', 'f1-score']:
        df_metrics[col] = df_metrics[col].apply(lambda x: round(x, 4))
    
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_few_shot_clean_pubmed.csv")
    df_metrics.to_csv(metrics_path, index=False)
    
    print("\n" + "="*60)
    print(" FEW-SHOT METRICS (CLEAN)")
    print("="*60)
    print(df_metrics.head(20).to_markdown(index=False))

if __name__ == "__main__":
    main()