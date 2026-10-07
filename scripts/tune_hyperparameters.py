#!/usr/bin/env python3
"""
AI4I 2020 Hyperparameter Tuning & Model Evaluation Pipeline (Week 06)
- Uses Optuna Bayesian optimization with 5-Fold Stratified Cross-Validation on train.csv.
- Tunes LightGBM, XGBoost, and Random Forest.
- Optimizes decision thresholds on Out-Of-Fold (OOF) validation predictions.
- Evaluates on unseen holdout test.csv (Before vs. After comparison).
- Saves tuned models, metadata, and comparative evaluation plots.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Suppress optuna logging clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tune_hyperparameters")

# Path configuration
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = WORKSPACE_ROOT / "dataset"
MODELS_DIR = WORKSPACE_ROOT / "models"
FIGURES_DIR = WORKSPACE_ROOT / "figures"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURE_COLS = [
    "type_encoded",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temp_diff_k",
    "power_w",
    "strain_min_nm",
    "tool_wear_critical",
]
TARGET_COL = "machine_failure"


def load_datasets():
    train_path = DATASET_DIR / "train.csv"
    test_path = DATASET_DIR / "test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    logger.info(f"Loaded train: {X_train.shape}, test: {X_test.shape}")
    logger.info(f"Train failure rate: {y_train.mean():.4f} ({y_train.sum()}/{len(y_train)})")
    logger.info(f"Test failure rate:  {y_test.mean():.4f} ({y_test.sum()}/{len(y_test)})")
    return X_train, y_train, X_test, y_test


def evaluate_predictions(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def find_optimal_threshold(y_true, y_probs):
    """Find threshold that maximizes F1 score on OOF predictions."""
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_true, (y_probs >= thresh).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return float(best_thresh), float(best_f1)


def cross_validate_model(model_cls, params, X, y, cv):
    oof_probs = np.zeros(len(y))
    fold_metrics = {"precision": [], "recall": [], "f1": [], "roc_auc": [], "pr_auc": []}

    for train_idx, val_idx in cv.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        model = model_cls(**params)
        model.fit(X_tr, y_tr)
        val_probs = model.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = val_probs

        val_pred = (val_probs >= 0.5).astype(int)
        fold_metrics["precision"].append(precision_score(y_va, val_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_va, val_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_va, val_pred, zero_division=0))
        fold_metrics["roc_auc"].append(roc_auc_score(y_va, val_probs))
        fold_metrics["pr_auc"].append(average_precision_score(y_va, val_probs))

    cv_summary = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in fold_metrics.items()
    }
    return oof_probs, cv_summary


def tune_lightgbm(X_train, y_train, cv, n_trials=40):
    logger.info("Starting LightGBM Hyperparameter Tuning with Optuna...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        oof_probs, _ = cross_validate_model(LGBMClassifier, params, X_train, y_train, cv)
        # Maximize PR-AUC and F1
        score = average_precision_score(y_train, oof_probs)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
    logger.info(f"LightGBM Best PR-AUC: {study.best_value:.4f}, Best Params: {best_params}")
    return best_params, study


def tune_xgboost(X_train, y_train, cv, n_trials=25):
    logger.info("Starting XGBoost Hyperparameter Tuning with Optuna...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }
        oof_probs, _ = cross_validate_model(XGBClassifier, params, X_train, y_train, cv)
        return average_precision_score(y_train, oof_probs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"random_state": 42, "n_jobs": -1, "eval_metric": "logloss"})
    logger.info(f"XGBoost Best PR-AUC: {study.best_value:.4f}")
    return best_params, study


def tune_random_forest(X_train, y_train, cv, n_trials=20):
    logger.info("Starting Random Forest Hyperparameter Tuning with Optuna...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 16),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        oof_probs, _ = cross_validate_model(RandomForestClassifier, params, X_train, y_train, cv)
        return average_precision_score(y_train, oof_probs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"class_weight": "balanced", "random_state": 42, "n_jobs": -1})
    logger.info(f"Random Forest Best PR-AUC: {study.best_value:.4f}")
    return best_params, study


def plot_comparison_figures(results, out_path):
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.35

    # 1. 5-Fold CV F1 & PR-AUC comparison
    ax1 = axes[0, 0]
    base_f1 = [results[m]["before"]["cv_5fold"]["f1"]["mean"] for m in models]
    tuned_f1 = [results[m]["after"]["cv_5fold"]["f1"]["mean"] for m in models]
    rects1 = ax1.bar(x - width / 2, base_f1, width, label="Before Tuning (Baseline)", color="#94a3b8")
    rects2 = ax1.bar(x + width / 2, tuned_f1, width, label="After Tuning (Optuna)", color="#3b82f6")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("5-Fold Cross Validation F1-Score Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim(0.7, 1.0)
    ax1.legend(loc="lower right")
    for r in rects1 + rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # 2. Test Set PR-AUC comparison
    ax2 = axes[0, 1]
    base_prauc = [results[m]["before"]["test_metrics"]["pr_auc"] for m in models]
    tuned_prauc = [results[m]["after"]["test_metrics"]["pr_auc"] for m in models]
    rects3 = ax2.bar(x - width / 2, base_prauc, width, label="Before Tuning", color="#cbd5e1")
    rects4 = ax2.bar(x + width / 2, tuned_prauc, width, label="After Tuning", color="#10b981")
    ax2.set_ylabel("PR-AUC (Average Precision)")
    ax2.set_title("Holdout Test Set PR-AUC Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylim(0.75, 1.0)
    ax2.legend(loc="lower right")
    for r in rects3 + rects4:
        h = r.get_height()
        ax2.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # 3. Test Set Confusion Matrix - Errors (FP + FN) Comparison
    ax3 = axes[1, 0]
    base_fp = [results[m]["before"]["test_metrics"]["confusion_matrix"]["fp"] for m in models]
    base_fn = [results[m]["before"]["test_metrics"]["confusion_matrix"]["fn"] for m in models]
    tuned_fp = [results[m]["after"]["test_metrics"]["confusion_matrix"]["fp"] for m in models]
    tuned_fn = [results[m]["after"]["test_metrics"]["confusion_matrix"]["fn"] for m in models]
    base_err = [f + n for f, n in zip(base_fp, base_fn)]
    tuned_err = [f + n for f, n in zip(tuned_fp, tuned_fn)]

    rects5 = ax3.bar(x - width / 2, base_err, width, label="Baseline Total Errors (FP+FN)", color="#f87171")
    rects6 = ax3.bar(x + width / 2, tuned_err, width, label="Tuned Total Errors (FP+FN)", color="#6366f1")
    ax3.set_ylabel("Misclassification Count (Cases out of 2,000)")
    ax3.set_title("Test Set Error Count Reduction")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontweight="bold")
    ax3.legend(loc="upper right")
    for r in rects5 + rects6:
        h = r.get_height()
        ax3.annotate(f"{int(h)}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # 4. LightGBM ROC & PR Curves
    ax4 = axes[1, 1]
    y_test = results["LightGBM"]["y_test"]
    p_base = results["LightGBM"]["before_prob"]
    p_tuned = results["LightGBM"]["after_prob"]

    fpr_b, tpr_b, _ = roc_curve(y_test, p_base)
    fpr_t, tpr_t, _ = roc_curve(y_test, p_tuned)
    prec_b, rec_b, _ = precision_recall_curve(y_test, p_base)
    prec_t, rec_t, _ = precision_recall_curve(y_test, p_tuned)

    ax4.plot(rec_b, prec_b, label=f"Baseline PR (AUC={results['LightGBM']['before']['test_metrics']['pr_auc']:.3f})", color="#94a3b8", linestyle="--")
    ax4.plot(rec_t, prec_t, label=f"Tuned PR (AUC={results['LightGBM']['after']['test_metrics']['pr_auc']:.3f})", color="#2563eb", linewidth=2)
    ax4.set_xlabel("Recall")
    ax4.set_ylabel("Precision")
    ax4.set_title("LightGBM Precision-Recall Curve (Holdout Test)")
    ax4.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Comparison figure saved to: {out_path}")


def main():
    X_train, y_train, X_test, y_test = load_datasets()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline configurations
    baseline_configs = {
        "LightGBM": {
            "cls": LGBMClassifier,
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "scale_pos_weight": 3.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            },
        },
        "XGBoost": {
            "cls": XGBClassifier,
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 5,
                "scale_pos_weight": 3.0,
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss",
            },
        },
        "Random Forest": {
            "cls": RandomForestClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 12,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
        },
    }

    results = {}

    for name, cfg in baseline_configs.items():
        logger.info(f"Evaluating Baseline {name}...")
        cls = cfg["cls"]
        params = cfg["params"]

        # 5-Fold CV Baseline
        oof_probs, cv_summary = cross_validate_model(cls, params, X_train, y_train, cv)
        best_thresh, _ = find_optimal_threshold(y_train, oof_probs)

        # Fit on full train and test on holdout
        model = cls(**params)
        model.fit(X_train, y_train)
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)
        test_metrics_default = evaluate_predictions(y_test, test_preds, test_probs)

        test_preds_opt = (test_probs >= best_thresh).astype(int)
        test_metrics_opt = evaluate_predictions(y_test, test_preds_opt, test_probs)

        results[name] = {
            "before": {
                "params": params,
                "cv_5fold": cv_summary,
                "optimal_threshold_oof": best_thresh,
                "test_metrics": test_metrics_default,
                "test_metrics_opt_thresh": test_metrics_opt,
            },
            "before_prob": test_probs,
        }

    # Hyperparameter Tuning with Optuna
    lgb_best_params, _ = tune_lightgbm(X_train, y_train, cv, n_trials=40)
    xgb_best_params, _ = tune_xgboost(X_train, y_train, cv, n_trials=25)
    rf_best_params, _ = tune_random_forest(X_train, y_train, cv, n_trials=20)

    tuned_configs = {
        "LightGBM": {"cls": LGBMClassifier, "params": lgb_best_params},
        "XGBoost": {"cls": XGBClassifier, "params": xgb_best_params},
        "Random Forest": {"cls": RandomForestClassifier, "params": rf_best_params},
    }

    trained_models = {}

    for name, cfg in tuned_configs.items():
        logger.info(f"Evaluating Tuned {name}...")
        cls = cfg["cls"]
        params = cfg["params"]

        oof_probs, cv_summary = cross_validate_model(cls, params, X_train, y_train, cv)
        best_thresh, best_oof_f1 = find_optimal_threshold(y_train, oof_probs)

        model = cls(**params)
        model.fit(X_train, y_train)
        trained_models[name] = model

        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)
        test_metrics_default = evaluate_predictions(y_test, test_preds, test_probs)

        test_preds_opt = (test_probs >= best_thresh).astype(int)
        test_metrics_opt = evaluate_predictions(y_test, test_preds_opt, test_probs)

        results[name]["after"] = {
            "params": params,
            "cv_5fold": cv_summary,
            "optimal_threshold_oof": best_thresh,
            "oof_best_f1": best_oof_f1,
            "test_metrics": test_metrics_default,
            "test_metrics_opt_thresh": test_metrics_opt,
        }
        results[name]["after_prob"] = test_probs
        results[name]["y_test"] = y_test

    # Save tuned LightGBM model to models/
    lgb_tuned = trained_models["LightGBM"]
    tuned_lgb_txt = MODELS_DIR / "lightgbm_tuned.txt"
    tuned_lgb_joblib = MODELS_DIR / "lightgbm_tuned.joblib"
    lgb_tuned.booster_.save_model(str(tuned_lgb_txt))
    joblib.dump(lgb_tuned, str(tuned_lgb_joblib))

    # Also update the primary models/lightgbm_model.txt and models/lightgbm_model.joblib
    # so the API service seamlessly benefits from the tuned model!
    lgb_model_txt = MODELS_DIR / "lightgbm_model.txt"
    lgb_model_joblib = MODELS_DIR / "lightgbm_model.joblib"
    lgb_tuned.booster_.save_model(str(lgb_model_txt))
    joblib.dump(lgb_tuned, str(lgb_model_joblib))
    logger.info(f"Updated primary {lgb_model_txt} with tuned weights!")

    # Save metadata
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tuning_method": "Optuna Bayesian Optimization (TPE Sampler)",
        "cv_folds": 5,
        "feature_names": FEATURE_COLS,
        "target_name": TARGET_COL,
        "models": {
            name: {
                "before": results[name]["before"],
                "after": results[name]["after"],
            }
            for name in results
        },
    }

    metadata_path = MODELS_DIR / "tuning_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Tuning metadata saved to: {metadata_path}")

    # Plot comparison figure
    figure_path = FIGURES_DIR / "14_hyperparameter_tuning_comparison.png"
    plot_comparison_figures(results, figure_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING SUMMARY REPORT (Week 06)")
    print("=" * 80)
    for name in results:
        b_cv = results[name]["before"]["cv_5fold"]
        a_cv = results[name]["after"]["cv_5fold"]
        b_test = results[name]["before"]["test_metrics"]
        a_test = results[name]["after"]["test_metrics"]
        print(f"\n[{name}]")
        print(f"  5-Fold CV F1:     {b_cv['f1']['mean']:.4f} -> {a_cv['f1']['mean']:.4f} (Delta: {a_cv['f1']['mean']-b_cv['f1']['mean']:+.4f})")
        print(f"  5-Fold CV PR-AUC: {b_cv['pr_auc']['mean']:.4f} -> {a_cv['pr_auc']['mean']:.4f} (Delta: {a_cv['pr_auc']['mean']-b_cv['pr_auc']['mean']:+.4f})")
        print(f"  Test F1 (0.50):   {b_test['f1']:.4f} -> {a_test['f1']:.4f} (Delta: {a_test['f1']-b_test['f1']:+.4f})")
        print(f"  Test PR-AUC:      {b_test['pr_auc']:.4f} -> {a_test['pr_auc']:.4f} (Delta: {a_test['pr_auc']-b_test['pr_auc']:+.4f})")
        print(f"  Test Errors:      FP {b_test['confusion_matrix']['fp']}->{a_test['confusion_matrix']['fp']}, FN {b_test['confusion_matrix']['fn']}->{a_test['confusion_matrix']['fn']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
