#!/usr/bin/env python3
"""
AI4I 2020 Advanced Feature Engineering Script
Generates domain-enhanced datasets:
- dataset/train_advanced.csv
- dataset/test_advanced.csv
"""

import os
import numpy as np
import pandas as pd

DATASET_DIR = '/home/ubuntu/cloud-mlops/dataset'
TRAIN_PATH = os.path.join(DATASET_DIR, 'train.csv')
TEST_PATH = os.path.join(DATASET_DIR, 'test.csv')

TRAIN_ADV_PATH = os.path.join(DATASET_DIR, 'train_advanced.csv')
TEST_ADV_PATH = os.path.join(DATASET_DIR, 'test_advanced.csv')

def add_advanced_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. HDF Physical Rule: Delta T < 8.6 K and RPM <= 1380
    df['hdf_risk'] = ((df['temp_diff_k'] < 8.6) & (df['rotational_speed_rpm'] <= 1380)).astype(int)
    
    # 2. PWF Physical Rule: Power < 3500 W or Power > 9000 W
    df['pwf_risk'] = ((df['power_w'] < 3500) | (df['power_w'] > 9000)).astype(int)
    
    # 3. OSF Physical Rule: Tool wear * Torque > Type threshold (L: 11000, M: 12000, H: 13000)
    osf_thresh = df['type_encoded'].map({0: 11000, 1: 12000, 2: 13000})
    df['osf_risk'] = (df['strain_min_nm'] > osf_thresh).astype(int)
    
    # 4. TWF Critical Zone: Tool wear between 200 and 240 minutes
    df['twf_zone'] = ((df['tool_wear_min'] >= 200) & (df['tool_wear_min'] <= 240)).astype(int)
    
    # 5. Combined Physical Risk Count (0 ~ 3)
    df['physical_risk_sum'] = df['hdf_risk'] + df['pwf_risk'] + df['osf_risk']
    
    # 6. Temperature Ratio: Process Temp / Air Temp
    df['temp_ratio'] = (df['process_temperature_k'] / df['air_temperature_k']).round(4)
    
    # 7. Torque-to-RPM Ratio: Indicative of load per revolution
    df['torque_rpm_ratio'] = (df['torque_nm'] / df['rotational_speed_rpm']).round(6)
    
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
        'tool_wear_critical',
        'hdf_risk',
        'pwf_risk',
        'osf_risk',
        'twf_zone',
        'physical_risk_sum',
        'temp_ratio',
        'torque_rpm_ratio'
    ]
    target_columns = ['machine_failure', 'twf', 'hdf', 'pwf', 'osf', 'rnf']
    
    return df[feature_columns + target_columns]

def main():
    print(f"[1/3] Loading base train/test datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"      Train base shape: {train_df.shape}, Test base shape: {test_df.shape}")
    
    print("[2/3] Generating 7 advanced domain boundary features...")
    train_adv = add_advanced_domain_features(train_df)
    test_adv = add_advanced_domain_features(test_df)
    
    print(f"      Train advanced shape: {train_adv.shape} (17 features + 6 targets)")
    print(f"      Test advanced shape:  {test_adv.shape} (17 features + 6 targets)")
    
    print("[3/3] Saving advanced datasets...")
    train_adv.to_csv(TRAIN_ADV_PATH, index=False)
    test_adv.to_csv(TEST_ADV_PATH, index=False)
    print(f"      Saved: {TRAIN_ADV_PATH} ({os.path.getsize(TRAIN_ADV_PATH) / 1024:.1f} KB)")
    print(f"      Saved: {TEST_ADV_PATH} ({os.path.getsize(TEST_ADV_PATH) / 1024:.1f} KB)")
    
    assert train_adv.isnull().sum().sum() == 0, "Error: NaN values found!"
    assert test_adv.isnull().sum().sum() == 0, "Error: NaN values found!"
    print("\n[SUCCESS] Advanced feature engineering complete!")

if __name__ == '__main__':
    main()
