"""
Classical SVM benchmark for comparison against quantum models.
"""
import sys
import os
import numpy as np
import warnings
from typing import Dict, Any, Tuple
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

try:
    from statsmodels.stats.contingency_tables import mcnemar
except ImportError:
    warnings.warn("statsmodels not installed, McNemar's test will be skipped.")
    mcnemar = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from data.disease_data import (
        ALL_DISEASES, DISEASE_NAMES, DISEASE_LABEL_MAP, LABEL_DISEASE_MAP,
        ALL_HPO_TERMS, HPO_TERM_INDEX, NUM_FEATURES, NUM_CLASSES, HPO_TERMS
    )
except ImportError:
    pass

def train_classical_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """
    Trains a classical RBF kernel SVM.
    """
    csvm = SVC(kernel='rbf', gamma='scale')
    csvm.fit(X_train, y_train)
    return csvm

def benchmark_quantum_vs_classical(quantum_results: Dict[str, Any], X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Compare precomputed quantum results with a classical RBF-SVM.
    """
    csvm = train_classical_svm(X_train, y_train)
    classical_preds = csvm.predict(X_test)
    
    acc = accuracy_score(y_test, classical_preds)
    macro_f1 = f1_score(y_test, classical_preds, average='macro')
    
    q_eval = quantum_results.get("evaluation", {})
    q_acc = q_eval.get("accuracy", 0.0)
    
    # Extract quantum predictions from the model if available, else recreate them
    if "predictions" in quantum_results:
        quantum_preds = quantum_results["predictions"]
    elif "model" in quantum_results and "K_test" in quantum_results:
        quantum_preds = quantum_results["model"].predict(quantum_results["K_test"])
    else:
        # Fallback if raw K_test isn't passed here - we assume q_eval contains accurate accuracy
        quantum_preds = np.zeros_like(y_test) # dummy just for structure if real preds unavailable
    
    p_value = 1.0
    conclusion = "No significant difference"
    
    # Run McNemar's if statsmodels is available and we have real quantum preds
    if mcnemar is not None and ("predictions" in quantum_results or "model" in quantum_results):
        # Contingency table
        both_correct = np.sum((quantum_preds == y_test) & (classical_preds == y_test))
        q_correct_c_wrong = np.sum((quantum_preds == y_test) & (classical_preds != y_test))
        c_correct_q_wrong = np.sum((classical_preds == y_test) & (quantum_preds != y_test))
        both_wrong = np.sum((quantum_preds != y_test) & (classical_preds != y_test))
        
        table = [[both_correct, q_correct_c_wrong], [c_correct_q_wrong, both_wrong]]
        
        if (q_correct_c_wrong + c_correct_q_wrong) > 0:
            result = mcnemar(table, exact=True)
            p_value = result.pvalue
            
            if p_value < 0.05:
                if q_correct_c_wrong > c_correct_q_wrong:
                    conclusion = "Quantum significantly better (p < 0.05)"
                else:
                    conclusion = "Classical significantly better (p < 0.05)"
            else:
                conclusion = "No significant difference"
                
    return {
        "classical_accuracy": acc,
        "classical_macro_f1": macro_f1,
        "quantum_accuracy": q_acc,
        "p_value": p_value,
        "conclusion": conclusion
    }

def print_benchmark_report(results: Dict[str, Any]) -> None:
    """
    Print the comparison report cleanly.
    """
    print("=" * 40)
    print("QUANTUM VS CLASSICAL SVM BENCHMARK REPORT")
    print("=" * 40)
    print(f"Classical Accuracy: {results.get('classical_accuracy', 0.0):.4f}")
    print(f"Quantum Accuracy:   {results.get('quantum_accuracy', 0.0):.4f}")
    print(f"Classical Macro F1: {results.get('classical_macro_f1', 0.0):.4f}")
    print(f"McNemar p-value:    {results.get('p_value', 1.0):.4e}")
    print(f"Conclusion:         {results.get('conclusion', 'N/A')}")
    print("=" * 40)

if __name__ == "__main__":
    X_tr = np.random.rand(50, 4)
    y_tr = np.random.choice([0, 1], size=50)
    X_te = np.random.rand(20, 4)
    y_te = np.random.choice([0, 1], size=20)
    
    q_res = {"evaluation": {"accuracy": 0.85, "macro_f1": 0.85}}
    b_res = benchmark_quantum_vs_classical(q_res, X_tr, X_te, y_tr, y_te)
    print_benchmark_report(b_res)
