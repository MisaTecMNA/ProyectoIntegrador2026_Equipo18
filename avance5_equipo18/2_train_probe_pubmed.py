"""
Script: 2_train_probe.py
Location: /home/tec/code/misael_space/codes/linear_probe/
Objective: Train Logistic Regression on extracted features and export metrics to CSV.
Includes strict and relaxed (+/- 1 year) evaluation metrics.
"""

import numpy as np
import pandas as pd  
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# --- PATH CONFIGURATION ---
DATA_DIR = "/home/tec/code/misael_space/codes/linear_probe"
OUTPUT_CSV = os.path.join(DATA_DIR, "probing_metrics_pubmed.csv")

def main():
    print("[INFO] STARTING LINEAR PROBING ANALYSIS")
    print(f"[INFO] Data directory: {DATA_DIR}")

    # 1. Load Data
    path_x = os.path.join(DATA_DIR, "features.npy")
    path_y = os.path.join(DATA_DIR, "labels.npy")

    if not os.path.exists(path_x) or not os.path.exists(path_y):
        print("[ERROR] .npy files not found.")
        print("[INFO] Please run '1_extract_features.py' first.")
        return

    X = np.load(path_x)
    y = np.load(path_y)

    print(f"[SUCCESS] Data loaded: {X.shape[0]} samples detected.")
    
    # 2. Distribution Analysis and Filtering
    unique, counts = np.unique(y, return_counts=True)
    
    # Filter classes with very few samples (less than 10)
    min_samples = 10
    valid_classes = unique[counts >= min_samples]
    
    if len(valid_classes) < 2:
        print("[ERROR] Not enough classes with data (minimum 2 years with >10 samples required).")
        return

    mask = np.isin(y, valid_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"[INFO] Valid classes for training: {len(valid_classes)}")

    # 3. Data Preparation (Split & Scale)
    print("[INFO] Splitting Train (80%) / Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Training
    print("[INFO] Training Linear Model (Logistic Regression)...")
    clf = LogisticRegression(
        random_state=42, 
        max_iter=3000, 
        solver='lbfgs', 
        multi_class='multinomial',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 5. Reporting and Saving
    y_pred = clf.predict(X_test)

# --- NUEVO CÓDIGO PARA EXPORTAR PREDICCIONES ---
    df_predictions = pd.DataFrame({
        'True_Year': y_test,
        'Predicted_Year': y_pred
    })
    predictions_csv_path = os.path.join(DATA_DIR, "probing_predictions_pubmed.csv")
    df_predictions.to_csv(predictions_csv_path, index=False)
    print(f"[SUCCESS] Saved Probe predictions (True vs Predicted) to:\n   {predictions_csv_path}")
    # -----------------------------------------------

    # Get report as dictionary for pandas
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    # Convert to DataFrame
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.index.name = 'Class_Year'
    
    # Global Metrics
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    # Temporal specific metrics (Relaxed accuracy)
    mae = mean_absolute_error(y_test, y_pred)
    acc_strict = np.mean(y_test == y_pred)
    acc_relaxed_1yr = np.mean(np.abs(y_test - y_pred) <= 1)
    acc_relaxed_2yr = np.mean(np.abs(y_test - y_pred) <= 2)

    print("\n" + "="*50)
    print(" FINAL RESULTS (LINEAR PROBE)")
    print("="*50)
    print(f"[INFO] Strict Accuracy (Exact Year):  {acc_strict:.4f}")
    print(f"[INFO] Relaxed Accuracy (+/- 1 Year): {acc_relaxed_1yr:.4f}")
    print(f"[INFO] Relaxed Accuracy (+/- 2 Year): {acc_relaxed_2yr:.4f}")
    print(f"[INFO] Mean Absolute Error (MAE):     {mae:.4f} years")
    print("-" * 50)
    print(f"[INFO] Micro F1 (Global):             {micro_f1:.4f}")
    print(f"[INFO] Macro F1 (Class Avg):          {macro_f1:.4f}")
    print("="*50)
    
    print(f"[SUCCESS] Saving detailed report to:\n   {OUTPUT_CSV}")
    
    # Save CSV
    df_report.to_csv(OUTPUT_CSV)
    
    # Show preview in console
    print("\n[INFO] Report preview:")
    print(df_report.head(15))

if __name__ == "__main__":
    main()