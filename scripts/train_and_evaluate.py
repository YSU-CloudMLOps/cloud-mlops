#!/usr/bin/env python3
"""
AI4I 2020 Predictive Maintenance Model Training & Evaluation Pipeline
Trains LightGBM, XGBoost, and Random Forest on train.csv.
Evaluates using 5-Fold Stratified CV and holdout test.csv.
Saves model weights/architectures (native and joblib) and evaluation figures.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Directories
BASE_DIR = '/home/ubuntu/cloud-mlops'
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
NOTES_DIR = os.path.join(BASE_DIR, 'notes')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def load_data():
    train_path = os.path.join(DATASET_DIR, 'train.csv')
    test_path = os.path.join(DATASET_DIR, 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    feature_cols = [
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
    target_col = 'machine_failure'
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    return X_train, y_train, X_test, y_test, feature_cols

def build_models(scale_pos_weight):
    models = {
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    return models

def cross_validate(models, X_train, y_train):
    print("=" * 60)
    print("Running 5-Fold Stratified Cross-Validation on Train Set...")
    print("=" * 60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, model_cls in models.items():
        metrics = {'recall': [], 'precision': [], 'f1': [], 'roc_auc': [], 'pr_auc': []}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            # Clone and fit
            m = model_cls.__class__(**model_cls.get_params())
            m.fit(X_tr, y_tr)
            
            preds = m.predict(X_val)
            probs = m.predict_proba(X_val)[:, 1]
            
            metrics['recall'].append(recall_score(y_val, preds))
            metrics['precision'].append(precision_score(y_val, preds, zero_division=0))
            metrics['f1'].append(f1_score(y_val, preds))
            metrics['roc_auc'].append(roc_auc_score(y_val, probs))
            metrics['pr_auc'].append(average_precision_score(y_val, probs))
            
        cv_results[name] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics.items()}
        print(f"[{name}] 5-Fold CV Metrics:")
        for k, v in cv_results[name].items():
            print(f"   - {k.upper():10s}: {v['mean']:.4f} (+/- {v['std']:.4f})")
            
    return cv_results

def train_and_evaluate_test(models, X_train, y_train, X_test, y_test, feature_cols):
    print("\n" + "=" * 60)
    print("Training Final Models on Full Train Set & Evaluating on Test Set...")
    print("=" * 60)
    
    test_results = {}
    predictions = {}
    probabilities = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        predictions[name] = preds
        probabilities[name] = probs
        
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        
        test_results[name] = {
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds)),
            'f1': float(f1_score(y_test, preds)),
            'roc_auc': float(roc_auc_score(y_test, probs)),
            'pr_auc': float(average_precision_score(y_test, probs)),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }
        
        print(f"[{name}] Holdout Test Set Performance:")
        print(f"   - Recall:    {test_results[name]['recall']:.4f} (TP: {tp}/{tp+fn}, Missed: {fn})")
        print(f"   - Precision: {test_results[name]['precision']:.4f} (FP: {fp})")
        print(f"   - F1-Score:  {test_results[name]['f1']:.4f}")
        print(f"   - PR-AUC:    {test_results[name]['pr_auc']:.4f}")
        print(f"   - ROC-AUC:   {test_results[name]['roc_auc']:.4f}")
        
    return trained_models, test_results, predictions, probabilities

def save_model_artifacts(trained_models, cv_results, test_results, feature_cols):
    print("\n" + "=" * 60)
    print("Saving Model Weights and Architectures to models/ ...")
    print("=" * 60)
    
    # 1. LightGBM
    lgb_model = trained_models['LightGBM']
    lgb_txt_path = os.path.join(MODELS_DIR, 'lightgbm_model.txt')
    lgb_joblib_path = os.path.join(MODELS_DIR, 'lightgbm_model.joblib')
    lgb_model.booster_.save_model(lgb_txt_path)
    joblib.dump(lgb_model, lgb_joblib_path)
    print(f"Saved LightGBM native booster: {lgb_txt_path}")
    print(f"Saved LightGBM joblib:         {lgb_joblib_path}")
    
    # 2. XGBoost
    xgb_model = trained_models['XGBoost']
    xgb_json_path = os.path.join(MODELS_DIR, 'xgboost_model.json')
    xgb_joblib_path = os.path.join(MODELS_DIR, 'xgboost_model.joblib')
    xgb_model.save_model(xgb_json_path)
    joblib.dump(xgb_model, xgb_joblib_path)
    print(f"Saved XGBoost native JSON:     {xgb_json_path}")
    print(f"Saved XGBoost joblib:          {xgb_joblib_path}")
    
    # 3. Random Forest
    rf_model = trained_models['Random Forest']
    rf_joblib_path = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
    joblib.dump(rf_model, rf_joblib_path)
    print(f"Saved Random Forest joblib:    {rf_joblib_path}")
    
    # 4. Metadata JSON
    metadata = {
        'training_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'feature_names': feature_cols,
        'target_name': 'machine_failure',
        'train_samples': 8000,
        'test_samples': 2000,
        'models': {
            'LightGBM': {
                'type': 'LGBMClassifier',
                'files': ['models/lightgbm_model.txt', 'models/lightgbm_model.joblib'],
                'cv_5fold': cv_results['LightGBM'],
                'test_metrics': test_results['LightGBM']
            },
            'XGBoost': {
                'type': 'XGBClassifier',
                'files': ['models/xgboost_model.json', 'models/xgboost_model.joblib'],
                'cv_5fold': cv_results['XGBoost'],
                'test_metrics': test_results['XGBoost']
            },
            'Random Forest': {
                'type': 'RandomForestClassifier',
                'files': ['models/random_forest_model.joblib'],
                'cv_5fold': cv_results['Random Forest'],
                'test_metrics': test_results['Random Forest']
            }
        }
    }
    
    metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved Model Metadata:          {metadata_path}")

def generate_evaluation_plots(trained_models, probabilities, y_test, test_results, feature_cols):
    print("\n" + "=" * 60)
    print("Generating Evaluation Figures in figures/ ...")
    print("=" * 60)
    
    colors = {'LightGBM': '#2ECC71', 'XGBoost': '#3498DB', 'Random Forest': '#9B59B6'}
    
    # -------------------------------------------------------------
    # Plot 1: ROC Curves Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for name, probs in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = test_results[name]['roc_auc']
        ax.plot(fpr, tpr, label=f"{name} (ROC-AUC = {auc:.4f})", color=colors[name], linewidth=2.2)
        
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.5000)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '06_roc_curves.png'), dpi=200)
    plt.close(fig)
    print("Saved: figures/06_roc_curves.png")

    # -------------------------------------------------------------
    # Plot 2: Precision-Recall Curves Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6.5))
    no_skill = y_test.sum() / len(y_test)
    ax.axhline(no_skill, color='black', linestyle='--', alpha=0.6, label=f'Baseline (Prevalence = {no_skill:.4f})')
    
    for name, probs in probabilities.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = test_results[name]['pr_auc']
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=colors[name], linewidth=2.2)
        
    ax.set_title('Precision-Recall (PR) Curves on Imbalanced Test Set', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Recall (True Positive Rate)', fontsize=11)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=11)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '07_pr_curves.png'), dpi=200)
    plt.close(fig)
    print("Saved: figures/07_pr_curves.png")

    # -------------------------------------------------------------
    # Plot 3: Confusion Matrices
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, res) in enumerate(test_results.items()):
        cm = [[res['confusion_matrix']['tn'], res['confusion_matrix']['fp']],
              [res['confusion_matrix']['fn'], res['confusion_matrix']['tp']]]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                    annot_kws={'size': 14, 'fontweight': 'bold'},
                    xticklabels=['Pred Normal (0)', 'Pred Failure (1)'],
                    yticklabels=['Actual Normal (0)', 'Actual Failure (1)'])
        rec = res['recall']
        prec = res['precision']
        f1 = res['f1']
        axes[i].set_title(f"{name}\nRecall: {rec:.2%} | Prec: {prec:.2%} | F1: {f1:.4f}",
                          fontsize=12, fontweight='bold', pad=10)
        axes[i].set_ylabel('Actual Label', fontsize=10)
        axes[i].set_xlabel('Predicted Label', fontsize=10)
        
    plt.suptitle('Confusion Matrix Comparison on Holdout Test Set (2,000 samples)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '08_confusion_matrices.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: figures/08_confusion_matrices.png")

    # -------------------------------------------------------------
    # Plot 4: Feature Importance Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Prettier feature labels
    feat_name_map = {
        'torque_nm': 'Torque [Nm]',
        'strain_min_nm': 'Strain (Wear * Torque)',
        'rotational_speed_rpm': 'Rotational Speed [rpm]',
        'power_w': 'Power (Torque * Speed)',
        'temp_diff_k': 'Temp Diff (Process - Air)',
        'tool_wear_min': 'Tool Wear [min]',
        'air_temperature_k': 'Air Temp [K]',
        'process_temperature_k': 'Process Temp [K]',
        'tool_wear_critical': 'Tool Wear >= 200 Indicator',
        'type_encoded': 'Product Type'
    }
    pretty_feats = [feat_name_map.get(f, f) for f in feature_cols]
    
    for i, (name, model) in enumerate(trained_models.items()):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Normalize to 0-100%
            importances_pct = (importances / importances.sum()) * 100
            
            df_imp = pd.DataFrame({'Feature': pretty_feats, 'Importance': importances_pct})
            df_imp = df_imp.sort_values('Importance', ascending=True)
            
            bars = axes[i].barh(df_imp['Feature'], df_imp['Importance'], color=colors[name], alpha=0.85, edgecolor='black')
            axes[i].set_title(f"{name} Feature Importance (%)", fontsize=12, fontweight='bold', pad=10)
            axes[i].set_xlabel('Importance (%)', fontsize=10)
            axes[i].set_xlim(0, max(importances_pct) * 1.15)
            
            for bar in bars:
                w = bar.get_width()
                axes[i].annotate(f" {w:.1f}%", xy=(w, bar.get_y() + bar.get_height() / 2),
                                xytext=(2, 0), textcoords="offset points",
                                ha='left', va='center', fontsize=9, fontweight='bold')
                                
    plt.suptitle('Normalized Feature Importance Comparison Across Models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '09_feature_importances.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: figures/09_feature_importances.png")

def main():
    X_train, y_train, X_test, y_test, feature_cols = load_data()
    
    # scale_pos_weight = N_neg / N_pos on train set
    scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    print(f"Training dataset size: {len(y_train)} (Normal: {len(y_train)-y_train.sum()}, Failure: {y_train.sum()})")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    
    models = build_models(scale_pos_weight)
    
    # 1. 5-Fold Stratified CV
    cv_results = cross_validate(models, X_train, y_train)
    
    # 2. Final Training & Test Evaluation
    trained_models, test_results, predictions, probabilities = train_and_evaluate_test(
        models, X_train, y_train, X_test, y_test, feature_cols
    )
    
    # 3. Save Model Weights and Artifacts
    save_model_artifacts(trained_models, cv_results, test_results, feature_cols)
    
    # 4. Generate Evaluation Plots
    generate_evaluation_plots(trained_models, probabilities, y_test, test_results, feature_cols)
    
    print("\n[SUCCESS] Model training, weight serialization, and evaluation complete!")

if __name__ == '__main__':
    main()
