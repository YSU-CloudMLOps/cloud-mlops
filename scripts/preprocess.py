#!/usr/bin/env python3
"""
AI4I 2020 Predictive Maintenance Dataset Preprocessing Script
Executed according to the Preprocessing Policy defined in notes/preprocessing_policy.md.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_DIR = '/home/ubuntu/cloud-mlops/dataset'
RAW_DATA_PATH = os.path.join(DATASET_DIR, 'ai4i2020.csv')
PREPROCESSED_DATA_PATH = os.path.join(DATASET_DIR, 'ai4i2020_preprocessed.csv')
TRAIN_DATA_PATH = os.path.join(DATASET_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(DATASET_DIR, 'test.csv')

def main():
    print(f"[1/5] Loading raw dataset from: {RAW_DATA_PATH}")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"      Raw data shape: {df.shape}")

    # Remove BOM from column names
    df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]

    print("[2/5] Performing Domain Feature Engineering...")
    # 1. Temperature difference (Process Temp - Air Temp) [K]
    df['temp_diff_k'] = (df['Process temperature [K]'] - df['Air temperature [K]']).round(2)

    # 2. Mechanical Power: Torque * (Rotational speed in rad/s) [W]
    df['power_w'] = (df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)).round(2)

    # 3. Strain / Overstrain Index: Tool wear * Torque [min*Nm]
    df['strain_min_nm'] = (df['Tool wear [min]'] * df['Torque [Nm]']).round(2)

    # 4. Critical Tool Wear Indicator (Tool wear >= 200 min)
    df['tool_wear_critical'] = (df['Tool wear [min]'] >= 200).astype(int)

    print("[3/5] Applying Categorical Encoding and Column Standardization...")
    # Ordinal Encoding for Type: L: 0, M: 1, H: 2
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    df['type_encoded'] = df['Type'].map(type_mapping)

    # Column renaming to standard snake_case
    rename_mapping = {
        'Air temperature [K]': 'air_temperature_k',
        'Process temperature [K]': 'process_temperature_k',
        'Rotational speed [rpm]': 'rotational_speed_rpm',
        'Torque [Nm]': 'torque_nm',
        'Tool wear [min]': 'tool_wear_min',
        'Machine failure': 'machine_failure',
        'TWF': 'twf',
        'HDF': 'hdf',
        'PWF': 'pwf',
        'OSF': 'osf',
        'RNF': 'rnf'
    }
    df = df.rename(columns=rename_mapping)

    # Drop identifiers and raw categorical column
    drop_columns = ['UDI', 'Product ID', 'Type']
    df_clean = df.drop(columns=drop_columns)

    # Reorder columns: Features first, then target columns
    feature_columns = [
        'type_encoded',
        'air_temperature_k',
        'process_temperature_k',
        'rotational_speed_rpm',
        'torque_nm',
        'tool_wear_min',
        'temp_diff_k',
        'power_w',
        'strain_min_nm',
        'tool_wear_critical'
    ]
    target_columns = ['machine_failure', 'twf', 'hdf', 'pwf', 'osf', 'rnf']
    final_columns = feature_columns + target_columns
    df_clean = df_clean[final_columns]

    print(f"      Preprocessed full data shape: {df_clean.shape}")
    print(f"      Features ({len(feature_columns)}): {feature_columns}")
    print(f"      Targets ({len(target_columns)}): {target_columns}")

    print("[4/5] Executing Stratified Train/Test Split (80% Train, 20% Test)...")
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.20,
        random_state=42,
        stratify=df_clean['machine_failure']
    )

    print(f"      Train set shape: {train_df.shape}")
    print(f"      Train failure count: {train_df['machine_failure'].sum()} ({train_df['machine_failure'].mean()*100:.2f}%)")
    print(f"      Test set shape: {test_df.shape}")
    print(f"      Test failure count: {test_df['machine_failure'].sum()} ({test_df['machine_failure'].mean()*100:.2f}%)")

    print("[5/5] Saving preprocessed datasets to disk...")
    # 1. Full preprocessed dataset
    df_clean.to_csv(PREPROCESSED_DATA_PATH, index=False)
    print(f"      Saved: {PREPROCESSED_DATA_PATH} ({os.path.getsize(PREPROCESSED_DATA_PATH) / 1024:.1f} KB)")

    # 2. Train set
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    print(f"      Saved: {TRAIN_DATA_PATH} ({os.path.getsize(TRAIN_DATA_PATH) / 1024:.1f} KB)")

    # 3. Test set
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"      Saved: {TEST_DATA_PATH} ({os.path.getsize(TEST_DATA_PATH) / 1024:.1f} KB)")

    # Sanity checks
    assert os.path.exists(RAW_DATA_PATH), "Error: Raw dataset must be preserved!"
    assert df_clean.isnull().sum().sum() == 0, "Error: Unexpected missing values found!"
    assert len(train_df) == 8000, "Error: Train split size must be 8,000!"
    assert len(test_df) == 2000, "Error: Test split size must be 2,000!"

    print("\n[SUCCESS] Preprocessing completed successfully according to policy!")

if __name__ == '__main__':
    main()
