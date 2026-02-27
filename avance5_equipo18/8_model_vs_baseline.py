import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def calculate_comparison_metrics(method_name, filepath, true_col, pred_col, baseline_year=2015):
    """
    Reads a CSV file, filters valid samples, and calculates MAE, MSE, and RMSE 
    for BOTH the Model and the Baseline (predicting 2015).
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)
    
    # Filter out failed or empty predictions
    mask = (df[pred_col] > 0) & (df[true_col] > 0)
    y_true = df.loc[mask, true_col]
    y_pred_model = df.loc[mask, pred_col]
    
    if len(y_true) == 0:
        print(f"[{method_name}] Not enough valid samples.")
        return None

    # 1. Calculate Model Metrics
    model_mae = mean_absolute_error(y_true, y_pred_model)
    model_mse = mean_squared_error(y_true, y_pred_model)
    model_rmse = np.sqrt(model_mse)

    # 2. Calculate Baseline Metrics (Always predicting 2015)
    y_pred_baseline = np.full_like(y_true, baseline_year)
    base_mae = mean_absolute_error(y_true, y_pred_baseline)
    base_mse = mean_squared_error(y_true, y_pred_baseline)
    base_rmse = np.sqrt(base_mse)

    # Return as dictionary for the final table
    return {
        "Method": method_name,
        "Valid_Samples": len(y_true),
        "Model_MAE": round(model_mae, 4),
        "Baseline_MAE": round(base_mae, 4),
        "Model_MSE": round(model_mse, 4),
        "Baseline_MSE": round(base_mse, 4),
        "Model_RMSE": round(model_rmse, 4),
        "Baseline_RMSE": round(base_rmse, 4)
    }

def main():
    print("=" * 80)
    print(" COMPARING MODEL vs BASELINE (ALWAYS 2015) ")
    print("=" * 80)

    # Base directories (Adjust if necessary)
    base_dir = "/home/tec/code/misael_space/codes/shooting_codes"
    probe_dir = "/home/tec/code/misael_space/codes/linear_probe"

    # List of experiments to evaluate
    files_to_eval = [
        ("Zero-Shot", os.path.join(base_dir, "results_zero_shot_clean_pubmed.csv"), "first_year", "predicted_year"),
        ("Few-Shot", os.path.join(base_dir, "results_few_shot_clean_pubmed.csv"), "first_year", "predicted_year"),
        ("Chain-of-Thought", os.path.join(base_dir, "results_cot_pubmed.csv"), "first_year", "predicted_year"),
        ("Linear Probe", os.path.join(probe_dir, "probing_predictions_pubmed.csv"), "True_Year", "Predicted_Year")
    ]

    results_list = []

    # Iterate through each file
    for name, path, true_col, pred_col in files_to_eval:
        metrics = calculate_comparison_metrics(name, path, true_col, pred_col, baseline_year=2015)
        if metrics is not None:
            results_list.append(metrics)

    # Create the CSV if successful
    if results_list:
        df_results = pd.DataFrame(results_list)
        
        # Save to CSV
        output_csv_path = os.path.join(base_dir, "pubmed_model_vs_baseline.csv")
        df_results.to_csv(output_csv_path, index=False)
        
        print(f"\n[SUCCESS] Final comparison table saved to:\n -> {output_csv_path}\n")
        print("Table preview:")
        print("-" * 80)
        print(df_results.to_markdown(index=False))
        print("-" * 80)
    else:
        print("\n[ERROR] Could not calculate any metrics.")

if __name__ == "__main__":
    main()