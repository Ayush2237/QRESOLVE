"""
QResolve-Dx: Probability Calibration for XGBoost

Wraps the XGBoost classifier with isotonic calibration via
CalibratedClassifierCV to produce well-calibrated posterior probabilities.

Clinical rationale: raw XGBoost softmax probabilities are not truly calibrated
(i.e., when the model says 70% Marfan, it may not actually be correct 70% of
the time). Isotonic calibration fixes this non-parametrically.
"""

from __future__ import annotations

import sys
import os
import pickle
import warnings
from typing import Tuple, Dict, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.disease_data import NUM_CLASSES, LABEL_DISEASE_MAP

try:
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import brier_score_loss, log_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not installed.")


def calibrate_model(
    base_model,
    X: np.ndarray,
    y: np.ndarray,
    method: str = "isotonic",
    cv: int = 5,
) -> Tuple["CalibratedClassifierCV", Dict]:
    """
    Calibrate probabilities from a trained classifier.

    Args:
        base_model: Trained classifier (e.g., XGBClassifier)
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
        method: 'isotonic' (non-parametric, recommended for >1000 samples)
                or 'sigmoid' (Platt scaling, better for very small datasets)
        cv: Number of cross-validation folds for calibration

    Returns:
        calibrated_model: CalibratedClassifierCV wrapping base_model
        calibration_metrics: Dict with Brier scores and reliability info
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for calibration.")

    print(f"\n{'='*60}")
    print(f"Probability Calibration ({method}, {cv}-fold CV)")
    print(f"{'='*60}")

    # Fit calibrated model
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv=cv,
    )
    calibrated.fit(X, y)

    # Evaluate calibration quality
    y_proba_cal = calibrated.predict_proba(X)
    y_proba_uncal = base_model.predict_proba(X)

    metrics = {
        "method": method,
        "cv_folds": cv,
        "calibrated_log_loss": float(log_loss(y, y_proba_cal)),
        "uncalibrated_log_loss": float(log_loss(y, y_proba_uncal)),
        "per_class": {},
    }

    print(f"\n  Log-loss (uncalibrated): {metrics['uncalibrated_log_loss']:.4f}")
    print(f"  Log-loss (calibrated):   {metrics['calibrated_log_loss']:.4f}")
    improvement = metrics['uncalibrated_log_loss'] - metrics['calibrated_log_loss']
    print(f"  Improvement:             {improvement:+.4f}")

    # Per-class Brier scores (one-vs-rest)
    print(f"\n  Per-class Brier scores (lower is better):")
    for cls_idx in range(NUM_CLASSES):
        y_binary = (y == cls_idx).astype(int)

        brier_uncal = brier_score_loss(y_binary, y_proba_uncal[:, cls_idx])
        brier_cal = brier_score_loss(y_binary, y_proba_cal[:, cls_idx])

        cls_name = LABEL_DISEASE_MAP[cls_idx]
        metrics["per_class"][cls_name] = {
            "brier_uncalibrated": float(brier_uncal),
            "brier_calibrated": float(brier_cal),
        }

        print(f"    {cls_name:>30s}: "
              f"uncal={brier_uncal:.4f}  cal={brier_cal:.4f}  "
              f"({'improved' if brier_cal < brier_uncal else 'unchanged'})")

    # Reliability diagram data (for plotting)
    print(f"\n  Reliability diagram data (fraction of positives vs mean predicted):")
    for cls_idx in range(min(3, NUM_CLASSES)):  # Show for first 3 classes
        y_binary = (y == cls_idx).astype(int)
        try:
            fraction_pos, mean_pred = calibration_curve(
                y_binary, y_proba_cal[:, cls_idx],
                n_bins=8,
                strategy='uniform',
            )
            cls_name = LABEL_DISEASE_MAP[cls_idx]
            print(f"    {cls_name}:")
            for fp, mp in zip(fraction_pos, mean_pred):
                bar = '█' * int(fp * 30)
                print(f"      predicted={mp:.2f}  actual={fp:.2f}  {bar}")
        except ValueError:
            pass  # Not enough data in some bins

    return calibrated, metrics


def save_calibrated_model(model, path: str) -> None:
    """Save calibrated model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n  Calibrated model saved to: {path}")


def load_calibrated_model(path: str):
    """Load calibrated model from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Calibration Module — requires trained XGBoost model and data")
    print("Run via run_pipeline.py for end-to-end execution.")
