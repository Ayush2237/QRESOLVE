"""
Quantum SVM training module.
"""
import sys
import os
import numpy as np
from typing import Dict, Any, List
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from data.disease_data import (
        ALL_DISEASES, DISEASE_NAMES, DISEASE_LABEL_MAP, LABEL_DISEASE_MAP,
        ALL_HPO_TERMS, HPO_TERM_INDEX, NUM_FEATURES, NUM_CLASSES, HPO_TERMS
    )
except ImportError:
    pass

from models.quantum.feature_select import prepare_quantum_data
from models.quantum.zz_kernel import create_quantum_kernel, compute_kernel_matrices, validate_kernel_matrix

def train_quantum_svm(K_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """
    Train a precomputed SVM model.
    """
    qsvm = SVC(kernel='precomputed')
    qsvm.fit(K_train, y_train)
    return qsvm

def evaluate_qsvm(qsvm: SVC, K_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the QSVM.
    """
    preds = qsvm.predict(K_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro')
    # per-class F1
    per_class_f1 = f1_score(y_test, preds, average=None)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1.tolist()
    }

def run_quantum_pipeline(X: np.ndarray, y: np.ndarray, top2_labels: List[int], k: int = 8) -> Dict[str, Any]:
    """
    End-to-end execution of the quantum classification pipeline.
    """
    data = prepare_quantum_data(X, y, top2_labels, k=k)
    kernel = create_quantum_kernel(n_features=k)
    K_train, K_test = compute_kernel_matrices(kernel, data["X_train"], data["X_test"])
    
    val_metrics = validate_kernel_matrix(K_train)
    
    qsvm = train_quantum_svm(K_train, data["y_train"])
    eval_metrics = evaluate_qsvm(qsvm, K_test, data["y_test"])
    
    return {
        "data_split": data,
        "kernel_validation": val_metrics,
        "model": qsvm,
        "K_test": K_test,
        "evaluation": eval_metrics
    }

if __name__ == "__main__":
    X_dummy = np.random.rand(100, 20)
    y_dummy = np.random.choice([0, 1], size=100)
    top2 = [0, 1]
    res = run_quantum_pipeline(X_dummy, y_dummy, top2, k=4)
    print("Pipeline evaluation:", res["evaluation"])
