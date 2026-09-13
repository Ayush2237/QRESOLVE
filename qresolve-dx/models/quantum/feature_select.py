"""
Feature selection module for quantum processing.
"""
import sys
import os
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from data.disease_data import (
        ALL_DISEASES, DISEASE_NAMES, DISEASE_LABEL_MAP, LABEL_DISEASE_MAP,
        ALL_HPO_TERMS, HPO_TERM_INDEX, NUM_FEATURES, NUM_CLASSES, HPO_TERMS
    )
except ImportError:
    HPO_TERMS = [] # Fallback for standalone tests if missing

def select_discriminative_features(X: np.ndarray, y: np.ndarray, top2_labels: List[int], k: int = 8) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Filter X and y to only rows where y is in top2_labels, and select top k features.
    """
    mask = np.isin(y, top2_labels)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    mi_scores = mutual_info_classif(X_filtered, y_filtered, random_state=42)
    selected_indices = np.argsort(mi_scores)[-k:][::-1].tolist()
    
    return X_filtered[:, selected_indices], y_filtered, selected_indices

def normalize_for_quantum(X: np.ndarray, scale_range: Tuple[float, float] = (0.0, np.pi)) -> np.ndarray:
    """
    Min-max normalize each feature to the target scale_range for ZZFeatureMap encoding.
    """
    min_val, max_val = scale_range
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    
    # Handle constant features by adding a small epsilon
    denominator = np.where(X_max - X_min == 0, 1e-10, X_max - X_min)
    
    X_norm = (X - X_min) / denominator
    X_scaled = X_norm * (max_val - min_val) + min_val
    return X_scaled

def prepare_quantum_data(X_full: np.ndarray, y_full: np.ndarray, top2_labels: List[int], k: int = 8) -> Dict[str, Any]:
    """
    Combines feature selection and normalization, splits into train/test sets.
    """
    X_red, y_red, sel_idx = select_discriminative_features(X_full, y_full, top2_labels, k)
    X_scaled = normalize_for_quantum(X_red)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_red, test_size=0.2, stratify=y_red, random_state=42
    )
    
    feature_names = []
    if HPO_TERMS and len(HPO_TERMS) > max(sel_idx, default=-1):
        feature_names = [HPO_TERMS[i] for i in sel_idx]
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "selected_indices": sel_idx,
        "feature_names": feature_names
    }

if __name__ == "__main__":
    # Test logic
    X_dummy = np.random.rand(100, 50)
    y_dummy = np.random.choice([0, 1, 2, 3], size=100)
    top2 = [0, 1]
    res = prepare_quantum_data(X_dummy, y_dummy, top2, k=8)
    print("Train shape:", res["X_train"].shape)
    print("Test shape:", res["X_test"].shape)
