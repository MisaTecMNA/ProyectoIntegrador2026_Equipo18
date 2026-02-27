"""
Script: 5_chain_of_thought_pubmed.py
Strategy: Chain of Thought (CoT) with delimiter.
Objective: 
 - Force the model to reason chronologically about the abstract before answering.
 - Use a separator '###' to cleanly extract the year and final answer.
Output: results_cot_pubmed.csv, metrics_cot_pubmed.csv
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

# Set to None to run the full dataset
MAX_SAMPLES = None 

def parse_cot_response(raw_response):
    """
    Separates reasoning from the final answer using the '###' delimiter.
    Returns: (year, clean_text, reasoning_trace)
    """
    if not isinstance(raw_response, str): 
        return 0, "", ""
    
    # 1. Attempt to split by delimiter
    parts = raw_response.split('###')
    
    if len(parts) > 1:
        reasoning = parts[0].strip()
        final_answer_block = parts[1].strip()
    else:
        # If the model forgot the delimiter, treat everything as the answer
        reasoning = ""
        final_answer_block = raw_response.strip()

    # 2. Extract Year from the final block
    year = 0
    # Priority: [YYYY] at start of final block
    match_start = re.search(r'\[\s*(20[1-2][0-9])\s*\]', final_answer_block)
    if match_start:
        year = int(match_start.group(1))
    else:
        # Fallback: search for any modern year in the final block
        matches = re.findall(r'\b(20[1-2][0-9])\b', final_answer_block)
        if matches:
            year = int(matches[0])

    # 3. Clean the text of the final answer (remove [20XX] and spaces)
    clean_text = re.sub(r'\[\s*20[0-9]{2}\s*\]', '', final_answer_block)
    clean_text = clean_text.strip(" .:,")
    
    return year, clean_text, reasoning

def main():
    print("[INFO] STARTING CHAIN OF THOUGHT (CoT) ON PUBMED ABSTRACTS")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] File not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, on_bad_lines='skip')
    
    # Standard filters for PubMed
    df = df.dropna(subset=['first_year', 'text_context'])
    df['first_year'] = pd.to_numeric(df['first_year'], errors='coerce').fillna(0).astype(int)
    df = df[(df['first_year'] >= 2010) & (df['first_year'] <= 2022)]
    
    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df_test = df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"[INFO] Processing sample of {MAX_SAMPLES} records...")
    else:
        df_test = df
        print(f"[INFO] Processing full dataset: {len(df_test)} records...")

    # 2. Model
    print(f"[INFO] Loading model {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )

    predicted_years = []
    generated_texts = []
    reasoning_traces = []

    print(f"[INFO] Running CoT Generation...")
    
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        context = row['text_context']
        
        # STRUCTURED CoT PROMPT ADAPTED FOR ABSTRACTS
        prompt_content = f"""Paper Title and Abstract:
{context}

Instructions:
1. First, think step-by-step about the historical context of this paper. Analyze the medical terminology, specific technologies mentioned (e.g., NGS, CRISPR, specific drugs, or diseases like COVID-19), and methodological trends to estimate the publication date.
2. Then, output a separator '###'.
3. Finally, write your final answer starting strictly with the estimated year in brackets [YYYY].

Format:
[Your chronological reasoning here...]
###
[YYYY] The paper was likely published in this year because...
"""

        messages = [
            {"role": "system", "content": "You are a highly analytical scientific expert knowledgeable about the timeline of biomedical discoveries and literature."},
            {"role": "user", "content": prompt_content}
        ]
        
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=350, # Increased to allow for longer reasoning about abstracts
                temperature=0.01,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_res = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Process response
        year, text, reasoning = parse_cot_response(full_res)
        
        predicted_years.append(year)
        generated_texts.append(text)
        reasoning_traces.append(reasoning)

    # 3. Save Results
    df_test['predicted_year'] = predicted_years
    df_test['generated_text'] = generated_texts
    df_test['reasoning_trace'] = reasoning_traces 
    
    raw_path = os.path.join(OUTPUT_DIR, "results_cot_pubmed.csv")
    df_test.to_csv(raw_path, index=False)
    print(f"\n[SUCCESS] Results saved to: {raw_path}")

    # 4. Metrics
    true_years = df_test['first_year'].tolist()
    report_dict = classification_report(true_years, predicted_years, zero_division=0, output_dict=True)
    df_metrics = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={'index': 'Class_Year'})
    
    for col in ['precision', 'recall', 'f1-score']:
        # Ensure the column exists and is numeric before rounding
        if col in df_metrics.columns:
            df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce').round(4)
    
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_cot_pubmed.csv")
    df_metrics.to_csv(metrics_path, index=False)
    
    print("\n" + "="*60)
    print(" CHAIN OF THOUGHT METRICS (PUBMED)")
    print("="*60)
    print(df_metrics.head(20).to_markdown(index=False))

if __name__ == "__main__":
    main()