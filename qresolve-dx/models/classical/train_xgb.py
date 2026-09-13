"""
QResolve-Dx: XGBoost Training with Stratified K-Fold Cross-Validation

Trains a multi-class XGBoost classifier on IC-weighted HPO feature vectors.
Uses stratified 5-fold CV on the small synthetic dataset to avoid overfitting.
"""

from __future__ import annotations

import sys
import os
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.disease_data import (
    DISEASE_NAMES, DISEASE_LABEL_MAP, LABEL_DISEASE_MAP, NUM_CLASSES
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("xgboost not installed. Install with: pip install xgboost")

try:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (
        accuracy_score, f1_score, log_loss, classification_report,
        confusion_matrix
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not installed.")


def create_xgb_model(
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> "xgb.XGBClassifier":
    """
    Create an XGBoost multi-class classifier with conservative hyperparameters.

    Parameters tuned for small dataset (~750-1500 synthetic patients):
    - max_depth=4: prevents overfitting on small data
    - n_estimators=200: enough capacity with low learning rate
    - learning_rate=0.05: slow learning for better generalization
    - multi:softprob: outputs calibrated-ish probabilities
    """
    if not HAS_XGBOOST:
        raise ImportError("xgboost is required. Install with: pip install xgboost")

    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=random_state,
        subsample=0.8,           # row sampling to reduce overfitting
        colsample_bytree=0.8,    # feature sampling
        reg_alpha=0.1,           # L1 regularization
        reg_lambda=1.0,          # L2 regularization
        min_child_weight=3,      # conservative splits
        verbosity=0,
    )


def train_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple["xgb.XGBClassifier", Dict]:
    """
    Train XGBoost with stratified k-fold cross-validation.

    Returns:
        model: Trained on full dataset after CV evaluation
        cv_results: Dict with per-fold and aggregate metrics
    """
    if not HAS_SKLEARN or not HAS_XGBOOST:
        raise ImportError("Both xgboost and scikit-learn are required.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results = []
    all_oof_preds = np.zeros((len(y), NUM_CLASSES))  # out-of-fold predictions

    print(f"\n{'='*60}")
    print(f"XGBoost Stratified {n_splits}-Fold Cross-Validation")
    print(f"{'='*60}")
    print(f"  Samples: {len(y)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = create_xgb_model(random_state=random_state + fold_idx)

        # Train with early stopping on validation set
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Predict probabilities
        y_val_proba = model.predict_proba(X_val)
        y_val_pred = np.argmax(y_val_proba, axis=1)

        # Store out-of-fold predictions
        all_oof_preds[val_idx] = y_val_proba

        # Metrics
        acc = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred, average="macro")
        ll = log_loss(y_val, y_val_proba, labels=list(range(NUM_CLASSES)))

        fold_results.append({
            "fold": fold_idx + 1,
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "log_loss": float(ll),
            "n_train": len(y_train),
            "n_val": len(y_val),
        })

        print(f"  Fold {fold_idx + 1}/{n_splits}: "
              f"Acc={acc:.4f}  F1={f1:.4f}  LogLoss={ll:.4f}  "
              f"(train={len(y_train)}, val={len(y_val)})")

    # Aggregate CV results
    oof_preds = np.argmax(all_oof_preds, axis=1)
    overall_acc = accuracy_score(y, oof_preds)
    overall_f1 = f1_score(y, oof_preds, average="macro")
    overall_ll = log_loss(y, all_oof_preds, labels=list(range(NUM_CLASSES)))

    print(f"\n  {'─'*50}")
    print(f"  Overall (OOF): Acc={overall_acc:.4f}  F1={overall_f1:.4f}  LogLoss={overall_ll:.4f}")
    print()

    # Per-class report
    print("  Per-class classification report (out-of-fold):")
    target_names = [LABEL_DISEASE_MAP[i] for i in range(NUM_CLASSES)]
    report = classification_report(y, oof_preds, target_names=target_names, digits=4)
    for line in report.split('\n'):
        print(f"    {line}")

    # Confusion matrix
    cm = confusion_matrix(y, oof_preds)
    print(f"\n  Confusion Matrix:")
    print(f"    {'':>25s}", end="")
    for name in target_names:
        print(f"  {name[:8]:>8s}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"    {target_names[i]:>25s}", end="")
        for val in row:
            print(f"  {val:>8d}", end="")
        print()

    # Train final model on full dataset
    print(f"\n  Training final model on full dataset ({len(y)} samples)...")
    final_model = create_xgb_model(random_state=random_state)
    final_model.fit(X, y, verbose=False)

    cv_results = {
        "n_splits": n_splits,
        "folds": fold_results,
        "overall_accuracy": float(overall_acc),
        "overall_macro_f1": float(overall_f1),
        "overall_log_loss": float(overall_ll),
        "confusion_matrix": cm.tolist(),
        "oof_predictions": all_oof_preds,
    }

    return final_model, cv_results


def get_feature_importance(
    model: "xgb.XGBClassifier",
    feature_names: list,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Extract and rank feature importances from XGBoost model.

    Returns DataFrame with feature name, HPO ID, and importance score.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]

    from data.disease_data import HPO_TERMS

    rows = []
    for rank, idx in enumerate(indices):
        hpo_id = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        label = HPO_TERMS.get(hpo_id, "Unknown")
        rows.append({
            "rank": rank + 1,
            "hpo_id": hpo_id,
            "label": label,
            "importance": float(importances[idx]),
        })

    return pd.DataFrame(rows)


def save_model(model, path: str) -> None:
    """Save trained XGBoost model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved to: {path}")


def load_model(path: str):
    """Load trained XGBoost model from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("XGBoost Training Module")
    print("=" * 60)

    # Try to load generated data
    processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    benchmark_path = os.path.join(processed_dir, 'benchmark.csv')

    if os.path.exists(benchmark_path):
        from data.generate_patients import build_feature_matrix
        from models.classical.features import compute_information_content, build_feature_matrix as build_ic_features

        df = pd.read_csv(benchmark_path)
        print(f"Loaded {len(df)} patients from {benchmark_path}")

        # Build features
        ic_values = compute_information_content()
        X, y = build_ic_features(df, ic_values)

        # Train with CV
        model, results = train_with_cv(X, y)

        # Feature importance
        from data.disease_data import ALL_HPO_TERMS
        importance_df = get_feature_importance(model, ALL_HPO_TERMS)
        print(f"\n  Top {len(importance_df)} most important features:")
        for _, row in importance_df.iterrows():
            print(f"    {row['rank']:>3d}. {row['hpo_id']} ({row['label']}): {row['importance']:.4f}")

        # Save model
        model_path = os.path.join(processed_dir, 'xgb_model.pkl')
        save_model(model, model_path)
    else:
        print(f"No benchmark data found at {benchmark_path}")
        print("Run generate_patients.py first to create training data.")
