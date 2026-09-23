#!/usr/bin/env python3
"""
AI4I 2020 Dataset Scaling Script for Neural Networks & Linear Models
Fits RobustScaler strictly on train.csv and applies transformation to both train.csv and test.csv.
Saves train_scaled.csv, test_scaled.csv, and robust_scaler.joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

DATASET_DIR = '/home/ubuntu/cloud-mlops/dataset'
TRAIN_PATH = os.path.join(DATASET_DIR, 'train.csv')
TEST_PATH = os.path.join(DATASET_DIR, 'test.csv')

TRAIN_SCALED_PATH = os.path.join(DATASET_DIR, 'train_scaled.csv')
TEST_SCALED_PATH = os.path.join(DATASET_DIR, 'test_scaled.csv')
SCALER_PATH = os.path.join(DATASET_DIR, 'robust_scaler.joblib')

def main():
    print(f"[1/4] Loading unscaled train and test sets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"      Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    continuous_features = [
        'air_temperature_k',
        'process_temperature_k',
        'rotational_speed_rpm',
        'torque_nm',
        'tool_wear_min',
        'temp_diff_k',
        'power_w',
        'strain_min_nm'
    ]
    binary_features = ['tool_wear_critical']
    target_columns = ['machine_failure', 'twf', 'hdf', 'pwf', 'osf', 'rnf']

    print("[2/4] Applying One-Hot Encoding for 'type_encoded' (L, M, H)...")
    # type_encoded: 0 -> L, 1 -> M, 2 -> H
    for df in [train_df, test_df]:
        df['type_l'] = (df['type_encoded'] == 0).astype(int)
        df['type_m'] = (df['type_encoded'] == 1).astype(int)
        df['type_h'] = (df['type_encoded'] == 2).astype(int)

    categorical_features = ['type_l', 'type_m', 'type_h']

    print("[3/4] Fitting RobustScaler strictly on train set and transforming both splits...")
    scaler = RobustScaler()
    
    # Fit ONLY on train continuous features
    train_scaled_continuous = scaler.fit_transform(train_df[continuous_features])
    test_scaled_continuous = scaler.transform(test_df[continuous_features])

    # Save fitted scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"      Fitted RobustScaler saved to: {SCALER_PATH}")

    # Build scaled DataFrames
    train_scaled_df = pd.DataFrame(train_scaled_continuous, columns=continuous_features, index=train_df.index)
    test_scaled_df = pd.DataFrame(test_scaled_continuous, columns=continuous_features, index=test_df.index)

    # Attach categorical features, binary indicator, and targets
    for col in categorical_features + binary_features + target_columns:
        train_scaled_df[col] = train_df[col]
        test_scaled_df[col] = test_df[col]

    # Reorder columns: continuous -> categorical -> binary -> targets
    ordered_columns = continuous_features + categorical_features + binary_features + target_columns
    train_scaled_df = train_scaled_df[ordered_columns]
    test_scaled_df = test_scaled_df[ordered_columns]

    print("[4/4] Saving scaled datasets to disk...")
    train_scaled_df.to_csv(TRAIN_SCALED_PATH, index=False)
    test_scaled_df.to_csv(TEST_SCALED_PATH, index=False)
    print(f"      Saved: {TRAIN_SCALED_PATH} ({os.path.getsize(TRAIN_SCALED_PATH) / 1024:.1f} KB, shape: {train_scaled_df.shape})")
    print(f"      Saved: {TEST_SCALED_PATH} ({os.path.getsize(TEST_SCALED_PATH) / 1024:.1f} KB, shape: {test_scaled_df.shape})")

    # Sanity checks
    assert train_scaled_df.isnull().sum().sum() == 0, "Error: NaN values found in train_scaled_df!"
    assert test_scaled_df.isnull().sum().sum() == 0, "Error: NaN values found in test_scaled_df!"
    assert len(train_scaled_df) == 8000, "Error: Train rows mismatch!"
    assert len(test_scaled_df) == 2000, "Error: Test rows mismatch!"
    print("\n[SUCCESS] Scaled datasets for Neural Networks & Linear Models successfully generated!")

if __name__ == '__main__':
    main()
