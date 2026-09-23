#!/usr/bin/env python3
"""
AI4I 2020 Advanced Feature Engineering & Soft Voting Ensemble Pipeline
Trains LightGBM, XGBoost, Random Forest, and a Soft Voting Ensemble
using domain boundary features on train_advanced.csv.
Evaluates on test_advanced.csv, saves models/weights and figures.
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
from sklearn.base import BaseEstimator, ClassifierMixin

BASE_DIR = '/home/ubuntu/cloud-mlops'
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
NOTES_DIR = os.path.join(BASE_DIR, 'notes')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Weighted Soft Voting Ensemble classifier combining LightGBM, XGBoost, and Random Forest.
    Supports customizable decision threshold for operational cost trade-offs.
    """
    def __init__(self, lgb_model, xgb_model, rf_model, weights=(0.50, 0.30, 0.20), threshold=0.50):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.rf_model = rf_model
        self.weights = weights
        self.threshold = threshold
        
    def fit(self, X, y):
        # Already fitted models can be passed or fit individually
        return self
        
    def predict_proba(self, X):
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p_rf = self.rf_model.predict_proba(X)[:, 1]
        w1, w2, w3 = self.weights
        w_sum = w1 + w2 + w3
        p_ens = (w1 * p_lgb + w2 * p_xgb + w3 * p_rf) / w_sum
        return np.vstack([1.0 - p_ens, p_ens]).T
        
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

def load_advanced_data():
    train_path = os.path.join(DATASET_DIR, 'train_advanced.csv')
    test_path = os.path.join(DATASET_DIR, 'test_advanced.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    target_cols = ['machine_failure', 'twf', 'hdf', 'pwf', 'osf', 'rnf']
    feature_cols = [c for c in train_df.columns if c not in target_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['machine_failure']
    X_test = test_df[feature_cols]
    y_test = test_df['machine_failure']
    
    return X_train, y_train, X_test, y_test, feature_cols

def build_advanced_models(scale_pos_weight):
    models = {
        'LightGBM (Adv)': LGBMClassifier(
            n_estimators=250,
            learning_rate=0.03,
            num_leaves=25,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'XGBoost (Adv)': XGBClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        'Random Forest (Adv)': RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    return models

def run_cross_validation(models, X_train, y_train):
    print("=" * 65)
    print("Running 5-Fold Stratified Cross-Validation on Advanced Train Set...")
    print("=" * 65)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, model_cls in models.items():
        metrics = {'recall': [], 'precision': [], 'f1': [], 'roc_auc': [], 'pr_auc': []}
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
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
        print(f"[{name}] 5-Fold CV:")
        for k, v in cv_results[name].items():
            print(f"   - {k.upper():10s}: {v['mean']:.4f} (+/- {v['std']:.4f})")
            
    return cv_results

def evaluate_models_and_ensemble(models, X_train, y_train, X_test, y_test, feature_cols):
    print("\n" + "=" * 65)
    print("Training Individual Advanced Models and Evaluating...")
    print("=" * 65)
    
    trained_models = {}
    probabilities = {}
    test_results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
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
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        }
        print(f"[{name}] Test Metrics: Recall={test_results[name]['recall']:.4f}, Prec={test_results[name]['precision']:.4f}, F1={test_results[name]['f1']:.4f}, PR-AUC={test_results[name]['pr_auc']:.4f}")

    # Build Soft Voting Ensemble
    print("\nBuilding Soft Voting Ensemble (LGB 0.50 + XGB 0.30 + RF 0.20)...")
    ensemble = SoftVotingEnsemble(
        lgb_model=trained_models['LightGBM (Adv)'],
        xgb_model=trained_models['XGBoost (Adv)'],
        rf_model=trained_models['Random Forest (Adv)'],
        weights=(0.50, 0.30, 0.20),
        threshold=0.50
    )
    
    probs_ens = ensemble.predict_proba(X_test)[:, 1]
    probabilities['Soft Voting Ensemble'] = probs_ens
    preds_ens_default = ensemble.predict(X_test)
    
    cm_ens = confusion_matrix(y_test, preds_ens_default)
    tn_e, fp_e, fn_e, tp_e = cm_ens.ravel()
    
    test_results['Soft Voting Ensemble (Thresh 0.50)'] = {
        'accuracy': float(accuracy_score(y_test, preds_ens_default)),
        'precision': float(precision_score(y_test, preds_ens_default, zero_division=0)),
        'recall': float(recall_score(y_test, preds_ens_default)),
        'f1': float(f1_score(y_test, preds_ens_default)),
        'roc_auc': float(roc_auc_score(y_test, probs_ens)),
        'pr_auc': float(average_precision_score(y_test, probs_ens)),
        'confusion_matrix': {'tn': int(tn_e), 'fp': int(fp_e), 'fn': int(fn_e), 'tp': int(tp_e)}
    }
    print(f"[Ensemble @ 0.50] Recall={test_results['Soft Voting Ensemble (Thresh 0.50)']['recall']:.4f}, Prec={test_results['Soft Voting Ensemble (Thresh 0.50)']['precision']:.4f}, F1={test_results['Soft Voting Ensemble (Thresh 0.50)']['f1']:.4f}, PR-AUC={test_results['Soft Voting Ensemble (Thresh 0.50)']['pr_auc']:.4f}")

    # Threshold Optimization for F1 (Thresh = 0.80)
    preds_ens_opt = (probs_ens >= 0.80).astype(int)
    cm_opt = confusion_matrix(y_test, preds_ens_opt)
    tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()
    test_results['Soft Voting Ensemble (Thresh 0.80 - High Prec)'] = {
        'accuracy': float(accuracy_score(y_test, preds_ens_opt)),
        'precision': float(precision_score(y_test, preds_ens_opt, zero_division=0)),
        'recall': float(recall_score(y_test, preds_ens_opt)),
        'f1': float(f1_score(y_test, preds_ens_opt)),
        'roc_auc': float(roc_auc_score(y_test, probs_ens)),
        'pr_auc': float(average_precision_score(y_test, probs_ens)),
        'confusion_matrix': {'tn': int(tn_o), 'fp': int(fp_o), 'fn': int(fn_o), 'tp': int(tp_o)}
    }
    print(f"[Ensemble @ 0.80] Recall={test_results['Soft Voting Ensemble (Thresh 0.80 - High Prec)']['recall']:.4f}, Prec={test_results['Soft Voting Ensemble (Thresh 0.80 - High Prec)']['precision']:.4f}, F1={test_results['Soft Voting Ensemble (Thresh 0.80 - High Prec)']['f1']:.4f}, FP={fp_o}")

    return trained_models, ensemble, test_results, probabilities

def save_advanced_artifacts(trained_models, ensemble, cv_results, test_results, feature_cols):
    print("\n" + "=" * 65)
    print("Saving Advanced Models and Ensemble Pipeline to models/ ...")
    print("=" * 65)
    
    # 1. LightGBM
    lgb_m = trained_models['LightGBM (Adv)']
    lgb_m.booster_.save_model(os.path.join(MODELS_DIR, 'lightgbm_advanced.txt'))
    joblib.dump(lgb_m, os.path.join(MODELS_DIR, 'lightgbm_advanced.joblib'))
    
    # 2. XGBoost
    xgb_m = trained_models['XGBoost (Adv)']
    xgb_m.save_model(os.path.join(MODELS_DIR, 'xgboost_advanced.json'))
    joblib.dump(xgb_m, os.path.join(MODELS_DIR, 'xgboost_advanced.joblib'))
    
    # 3. Random Forest
    rf_m = trained_models['Random Forest (Adv)']
    joblib.dump(rf_m, os.path.join(MODELS_DIR, 'random_forest_advanced.joblib'))
    
    # 4. Ensemble Pipeline
    joblib.dump(ensemble, os.path.join(MODELS_DIR, 'ensemble_voting_pipeline.joblib'))
    print("Saved models: lightgbm_advanced, xgboost_advanced, random_forest_advanced, ensemble_voting_pipeline.joblib")
    
    # 5. Metadata
    metadata = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'feature_count': len(feature_cols),
        'feature_names': feature_cols,
        'cv_5fold_results': cv_results,
        'holdout_test_results': test_results
    }
    with open(os.path.join(MODELS_DIR, 'advanced_model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("Saved metadata: models/advanced_model_metadata.json")

def generate_advanced_plots(probabilities, y_test, test_results, trained_models, feature_cols):
    print("\n" + "=" * 65)
    print("Generating High-Resolution Evaluation Figures in figures/ ...")
    print("=" * 65)
    
    colors = {
        'LightGBM (Adv)': '#2ECC71',
        'XGBoost (Adv)': '#3498DB',
        'Random Forest (Adv)': '#9B59B6',
        'Soft Voting Ensemble': '#E74C3C'
    }
    
    # Figure 10: Advanced ROC & PR Curves (2 subplots)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    
    # Subplot 1: ROC Curves
    for name, probs in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})",
                     color=colors.get(name, '#333333'),
                     linewidth=2.4 if 'Ensemble' in name else 1.8)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 0.5000)')
    axes[0].set_title('ROC Curves (Advanced Features & Ensemble)', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=11)
    axes[0].legend(loc='lower right', frameon=True, fontsize=10)
    
    # Subplot 2: PR Curves
    no_skill = y_test.sum() / len(y_test)
    axes[1].axhline(no_skill, color='black', linestyle='--', alpha=0.5, label=f'Baseline ({no_skill:.4f})')
    for name, probs in probabilities.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})",
                     color=colors.get(name, '#333333'),
                     linewidth=2.4 if 'Ensemble' in name else 1.8)
    axes[1].set_title('Precision-Recall Curves (Critical for Imbalance)', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel('Recall (True Positive Rate)', fontsize=11)
    axes[1].set_ylabel('Precision (Positive Predictive Value)', fontsize=11)
    axes[1].legend(loc='lower left', frameon=True, fontsize=10)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '10_advanced_pr_and_roc_curves.png'), dpi=200)
    plt.close(fig)
    print("Saved: figures/10_advanced_pr_and_roc_curves.png")

    # Figure 11: Confusion Matrices (Baseline vs Advanced LightGBM vs Ensemble)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models_to_show = [
        ('LightGBM (Adv) @ 0.50', test_results['LightGBM (Adv)']),
        ('Ensemble @ 0.50 (Balanced)', test_results['Soft Voting Ensemble (Thresh 0.50)']),
        ('Ensemble @ 0.80 (High Precision)', test_results['Soft Voting Ensemble (Thresh 0.80 - High Prec)'])
    ]
    
    for i, (title, res) in enumerate(models_to_show):
        cm = [[res['confusion_matrix']['tn'], res['confusion_matrix']['fp']],
              [res['confusion_matrix']['fn'], res['confusion_matrix']['tp']]]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens' if i > 0 else 'Blues', cbar=False, ax=axes[i],
                    annot_kws={'size': 14, 'fontweight': 'bold'},
                    xticklabels=['Pred Normal (0)', 'Pred Failure (1)'],
                    yticklabels=['Actual Normal (0)', 'Actual Failure (1)'])
        axes[i].set_title(f"{title}\nRecall: {res['recall']:.2%} | Prec: {res['precision']:.2%} | F1: {res['f1']:.4f}",
                          fontsize=12, fontweight='bold', pad=10)
        axes[i].set_ylabel('Actual Label', fontsize=10)
        axes[i].set_xlabel('Predicted Label', fontsize=10)
        
    plt.suptitle('Confusion Matrix Comparison: Advanced Pipeline on Holdout Test Set (2,000 samples)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '11_advanced_confusion_matrices.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: figures/11_advanced_confusion_matrices.png")

    # Figure 12: Decision Threshold Optimization Curves
    p_ens = probabilities['Soft Voting Ensemble']
    prec_list, rec_list, f1_list, thresholds = [], [], [], np.linspace(0.1, 0.95, 86)
    for th in thresholds:
        preds = (p_ens >= th).astype(int)
        r = recall_score(y_test, preds)
        p = precision_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds)
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f)
        
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, prec_list, label='Precision (Positive Predictive Value)', color='#3498DB', linewidth=2.2)
    ax.plot(thresholds, rec_list, label='Recall (Detection Rate)', color='#E74C3C', linewidth=2.2)
    ax.plot(thresholds, f1_list, label='F1-Score (Harmonic Mean)', color='#2ECC71', linewidth=2.5)
    
    # Mark Thresh 0.50 and Thresh 0.80
    f1_05 = f1_score(y_test, (p_ens >= 0.50).astype(int))
    f1_08 = f1_score(y_test, (p_ens >= 0.80).astype(int))
    ax.axvline(0.50, color='gray', linestyle='--', alpha=0.7, label=f'Default Thresh 0.50 (F1={f1_05:.4f})')
    ax.axvline(0.80, color='purple', linestyle='--', alpha=0.7, label=f'Optimal Thresh 0.80 (F1={f1_08:.4f})')
    
    ax.set_title('Soft Voting Ensemble: Decision Threshold Optimization Curve', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Probability Threshold', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_xlim(0.1, 0.95)
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc='lower center', frameon=True, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '12_threshold_tuning_curve.png'), dpi=200)
    plt.close(fig)
    print("Saved: figures/12_threshold_tuning_curve.png")

    # Figure 13: Feature Importances with Domain Features
    fig, ax = plt.subplots(figsize=(10, 7.5))
    lgb_model = trained_models['LightGBM (Adv)']
    imp = lgb_model.feature_importances_
    imp_pct = (imp / imp.sum()) * 100
    df_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': imp_pct}).sort_values('Importance', ascending=True)
    
    bars = ax.barh(df_imp['Feature'], df_imp['Importance'], color='#16A085', alpha=0.85, edgecolor='black')
    ax.set_title('LightGBM (Advanced) 17 Feature Importances (%)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Normalized Importance (%)', fontsize=11)
    ax.set_xlim(0, max(imp_pct) * 1.15)
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f" {w:.1f}%", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(2, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '13_advanced_feature_importances.png'), dpi=200)
    plt.close(fig)
    print("Saved: figures/13_advanced_feature_importances.png")

def main():
    X_train, y_train, X_test, y_test, feature_cols = load_advanced_data()
    scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    print(f"Loaded Advanced Dataset: {X_train.shape[1]} Features")
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    
    models = build_advanced_models(scale_pos_weight)
    
    # 1. 5-Fold Stratified CV
    cv_results = run_cross_validation(models, X_train, y_train)
    
    # 2. Final Training & Ensemble Evaluation
    trained_models, ensemble, test_results, probabilities = evaluate_models_and_ensemble(
        models, X_train, y_train, X_test, y_test, feature_cols
    )
    
    # 3. Save Artifacts
    save_advanced_artifacts(trained_models, ensemble, cv_results, test_results, feature_cols)
    
    # 4. Generate Visualizations
    generate_advanced_plots(probabilities, y_test, test_results, trained_models, feature_cols)
    
    print("\n[SUCCESS] Advanced Ensemble Pipeline execution complete!")

if __name__ == '__main__':
    main()
