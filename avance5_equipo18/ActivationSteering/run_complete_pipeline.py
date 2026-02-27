"""
MASTER SCRIPT: EASYSTEER PIPELINE ORCHESTRATOR
Author:  Misael
Goal: Run the complete extraction, validation, calibration, and evaluation flow automatically.
"""

import subprocess
import sys
import os
import time

# --- CONFIGURATION: The 4 Golden Scripts ---
# Ensure these files exist in the same folder as this script.
PIPELINE_STEPS = [
    {
        "name": "STEP 1: EXTRACTION",
        "file": "1_construct_vector_llama2.py",
        "desc": "Extracting the 'Time' direction vector from Llama 3..."
    },
    {
        "name": "STEP 2: VALIDATION",
        "file": "4_check_vector_vocabulary.py",
        "desc": "Checking vector semantics (Looking for time-related concepts)..."
    },
    {
        "name": "STEP 3: CALIBRATION (Visual Check)",
        "file": "7_find_sweet_spot.py",
        "desc": "Testing different coefficients (-0.2 to -1.0) on sample questions..."
    },
    {
        "name": "STEP 4: FINAL METRICS",
        "file": "6_generate_paper_metrics.py",
        "desc": "Generating Perplexity (PPL) and F1 Score tables for the paper..."
    }
]

INTERPRETER = sys.executable  # Uses the current python environment

def check_files_exist():
    """Verifies that all necessary files are present before starting."""
    missing = []
    for step in PIPELINE_STEPS:
        if not os.path.exists(step["file"]):
            missing.append(step["file"])
    
    if missing:
        print("\n CRITICAL ERROR: The following core files are missing:")
        for m in missing:
            print(f"   - {m}")
        print("\nPlease allow the AI to generate these files or restore them.")
        sys.exit(1)

def run_pipeline():
    # Verify environment first
    check_files_exist()

    total_start = time.time()
    
    print("\n" + "="*60)
    print("  STARTING EASYSTEER MASTER PIPELINE")
    print("="*60)

    for step in PIPELINE_STEPS:
        script_name = step["file"]
        
        print(f"\n\n{'#'*60}")
        print(f"{step['name']}")
        print(f" Script: {script_name}")
        print(f"  Info:   {step['desc']}")
        print(f"{'#'*60}\n")

        # Run script
        step_start = time.time()
        try:
            # We use subprocess to isolate memory. When the process dies, GPU memory is freed.
            # This is crucial for running Llama 3 multiple times in sequence.
            subprocess.run([INTERPRETER, script_name], check=True)
            
            elapsed = time.time() - step_start
            print(f"\n {step['name']} COMPLETED in {elapsed:.2f} seconds.")

        except subprocess.CalledProcessError as e:
            print(f"\n EXECUTION FAILED at {step['name']}.")
            print(f"   Error code: {e.returncode}")
            print("   The pipeline has been stopped to preserve error logs.")
            sys.exit(1)
            
        except KeyboardInterrupt:
            print("\n Pipeline stopped by user.")
            sys.exit(0)

    total_time = (time.time() - total_start) / 60
    print("\n" + "="*60)
    print(f"  PIPELINE FINISHED SUCCESSFULLY!")
    print(f"   Total time: {total_time:.2f} minutes")
    print("="*60)
    print("Next steps:")
    print("1. Review the logs above for Step 3 to decide the final coefficient.")
    print("2. Open 'easysteer/data/steering_metrics_final.csv' to see your results.")

if __name__ == "__main__":
    run_pipeline()