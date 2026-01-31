"""
Step 1: Generate biomedical questions from SemMed triples.
MODIFIED: Questions column is now placed at the end, right after 'first_sentence_id'.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

# ----------------- PATHS -----------------
IN_PATH = Path("/home/tec/code/misael_space/data/triples_10_24_semmed.csv")
OUT_PATH = Path("/home/tec/code/misael_space/data/triples_semmed_questions.csv")

# ----------------- SEMANTIC MAPPING -----------------
# Maps 4-letter semtypes to human-readable categories for better phrasing
SEM_TYPE_MAP = {
    "phsu": "pharmacologic substance", "clnd": "clinical drug", "orch": "chemical",
    "dsyn": "disease", "patf": "pathologic function", "fndg": "clinical finding",
    "mobd": "mental disorder", "enzy": "enzyme", "aapp": "protein",
    "biof": "biologic function", "phpr": "process", "phsf": "physiologic function",
    "bpoc": "body part", "cell": "cell", "neop": "neoplasm",
    "bacs": "active substance", "gngm": "gene", "imft": "immunologic factor",
    "humn": "patient group", "mamm": "mammal", "virs": "virus", "bact": "bacteria"
}

def get_target_noun(semtype: Optional[str]) -> str:
    """Returns a pluralized noun phrase for the question target."""
    if not semtype or not isinstance(semtype, str):
        return "entities"
    
    st = semtype.lower()
    
    # Specific overrides for natural language
    if st == "humn": return "patients or groups"
    if st == "mamm": return "species"
    if st in ["dsyn", "mobd", "neop"]: return "diseases or conditions"
    if st in ["cell", "celf"]: return "cell types"
    if st in ["bpoc", "tisu"]: return "tissues or organs"
    
    # Fallback to map
    base = SEM_TYPE_MAP.get(st, "medical entities")
    return base + "s" if not base.endswith("s") else base

def create_natural_question(row) -> str:
    """
    Generates a grammatically correct question based on the predicate.
    """
    subj = str(row['subject_name']).strip()
    pred = str(row['predicate']).strip().upper()
    obj_sem = str(row['object_semtype']) if pd.notna(row['object_semtype']) else ""
    
    target_noun = get_target_noun(obj_sem)
    
    # Templates designed for Open-Ended Fact Retrieval
    templates = {
        "TREATS": f"Which {target_noun} is {subj} commonly used to treat?",
        "CAUSES": f"Which {target_noun} can {subj} cause or induce?",
        "PREVENTS": f"Which {target_noun} does {subj} help to prevent?",
        "DIAGNOSES": f"Which {target_noun} is {subj} used to diagnose?",
        "ADMINISTERED_TO": f"To which {target_noun} is {subj} typically administered?",
        "AFFECTS": f"Which {target_noun} does {subj} primarily affect?",
        "DISRUPTS": f"Which {target_noun} does {subj} disrupt?",
        "INHIBITS": f"Which {target_noun} does {subj} inhibit?",
        "STIMULATES": f"Which {target_noun} does {subj} stimulate?",
        "INTERACTS_WITH": f"Which {target_noun} does {subj} significantly interact with?",
        "LOCATION_OF": f"Which {target_noun} is the anatomical location of {subj}?",
        "COEXISTS_WITH": f"Which {target_noun} commonly co-exists with {subj}?",
        "ASSOCIATED_WITH": f"Which {target_noun} is {subj} most strongly associated with?",
    }
    
    # Default fallback
    q = templates.get(pred, f"In a biomedical context, what is the relationship between {subj} and {target_noun} via {pred}?")
    return q

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH}")
    
    print(f"Reading {IN_PATH}...")
    df = pd.read_csv(IN_PATH)
    
    # Validation
    required = ['subject_name', 'predicate']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Input CSV missing required columns: {required}")

    print("Generating questions...")
    # Apply generation row by row
    df['Questions'] = df.apply(create_natural_question, axis=1)
    
    # --- REORDERING LOGIC (Modified) ---
    cols = list(df.columns)
    
    # Remove 'Questions' from wherever pandas put it initially (usually the end)
    if 'Questions' in cols:
        cols.remove('Questions')
        
        # Find position of 'first_sentence_id'
        if 'first_sentence_id' in cols:
            target_idx = cols.index('first_sentence_id') + 1
            print("Placing 'Questions' column after 'first_sentence_id'.")
        else:
            # Fallback: Just put it at the very end
            target_idx = len(cols)
            print("Column 'first_sentence_id' not found. Placing 'Questions' at the end.")
            
        cols.insert(target_idx, 'Questions')
        df = df[cols]

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f" Questions generated for {len(df)} rows.")
    print(f"Saved to: {OUT_PATH}")
    
    # Preview specific columns to verify order
    preview_cols = ['subject_name', 'predicate']
    if 'first_sentence_id' in df.columns:
        preview_cols.append('first_sentence_id')
    preview_cols.append('Questions')
    
    print("\n--- Preview ---")
    print(df[preview_cols].head(5).to_string(index=False))

if __name__ == "__main__":
    main()