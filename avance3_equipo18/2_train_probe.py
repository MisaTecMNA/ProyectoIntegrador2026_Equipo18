"""
Script: 2_train_probe.py
Location: /home/tec/code/misael_space/codes/linear_probe/
Objective: Train Logistic Regression and export metrics to CSV.
"""

import numpy as np
import pandas as pd  # For CSV handling
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

# --- PATH CONFIGURATION ---
DATA_DIR = "/home/tec/code/misael_space/codes/linear_probe"
OUTPUT_CSV = os.path.join(DATA_DIR, "probing_metrics.csv")

def main():
    print(" STARTING LINEAR PROBING ANALYSIS")
    print(f"   > Data directory: {DATA_DIR}")

    # 1. Load Data
    path_x = os.path.join(DATA_DIR, "features.npy")
    path_y = os.path.join(DATA_DIR, "labels.npy")

    if not os.path.exists(path_x) or not os.path.exists(path_y):
        print(" ERROR: .npy files not found.")
        print("   > Please run '1_extract_features.py' first.")
        return

    X = np.load(path_x)
    y = np.load(path_y)

    print(f" Data loaded: {X.shape[0]} samples detected.")
    
    # 2. Distribution Analysis and Filtering
    unique, counts = np.unique(y, return_counts=True)
    
    # Filter classes with very few samples (less than 10)
    min_samples = 10
    valid_classes = unique[counts >= min_samples]
    
    if len(valid_classes) < 2:
        print(" ERROR: Not enough classes with data (minimum 2 years with >10 samples required).")
        return

    mask = np.isin(y, valid_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"   > Valid classes for training: {len(valid_classes)}")

    # 3. Data Preparation (Split & Scale)
    print("   > Splitting Train (80%) / Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Training
    print(" Training Linear Model (Logistic Regression)...")
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

    # Get report as dictionary for pandas
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    # Convert to DataFrame
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.index.name = 'Class_Year'
    
    # Quick Global Metrics for print
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print("\n" + "═"*50)
    print(" FINAL RESULTS")
    print("═"*50)
    print(f"   Micro F1 (Global Accuracy):  {micro_f1:.4f}")
    print(f"   Macro F1 (Class Average):    {macro_f1:.4f}")
    print("─"*50)
    print(f" Saving detailed report to:\n   {OUTPUT_CSV}")
    
    # Save CSV
    df_report.to_csv(OUTPUT_CSV)
    print(" CSV file generated successfully.")
    
    # Show preview in console
    print("\nReport preview:")
    print(df_report)

if __name__ == "__main__":
    main()