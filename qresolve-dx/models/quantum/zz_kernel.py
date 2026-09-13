"""
Quantum kernel logic using ZZFeatureMap.
"""
import sys
import os
import warnings
import numpy as np
from typing import Tuple, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from data.disease_data import (
        ALL_DISEASES, DISEASE_NAMES, DISEASE_LABEL_MAP, LABEL_DISEASE_MAP,
        ALL_HPO_TERMS, HPO_TERM_INDEX, NUM_FEATURES, NUM_CLASSES, HPO_TERMS
    )
except ImportError:
    pass

try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("qiskit or qiskit_machine_learning not available. Using classical RBF fallback.")

def create_quantum_kernel(n_features: int = 8, reps: int = 2, entanglement: str = 'linear') -> Any:
    """
    Creates a FidelityQuantumKernel. Returns an RBF kernel-like object if qiskit is missing.
    """
    if not QISKIT_AVAILABLE:
        from sklearn.metrics.pairwise import rbf_kernel
        class ClassicalFallbackKernel:
            def evaluate(self, x_vec, y_vec=None):
                return rbf_kernel(x_vec, y_vec)
        return ClassicalFallbackKernel()
        
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=reps, entanglement=entanglement)
    
    try:
        from qiskit_aer.primitives import Sampler as AerSampler
        from qiskit_algorithms.state_fidelities import ComputeUncompute
        sampler = AerSampler(backend_options={'method': 'statevector'}, run_options={'shots': None})
        fidelity = ComputeUncompute(sampler=sampler)
        kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    except ImportError:
        kernel = FidelityQuantumKernel(feature_map=feature_map)
        
    return kernel

def compute_kernel_matrices(kernel: Any, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes K_train and K_test.
    """
    K_train = kernel.evaluate(x_vec=X_train)
    K_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)
    return K_train, K_test

def validate_kernel_matrix(K: np.ndarray) -> Dict[str, Any]:
    """
    Checks symmetry, PSD, diagonal values of a kernel matrix.
    """
    is_symmetric = np.allclose(K, K.T, atol=1e-5)
    
    # Check PSD by seeing if all eigenvalues are >= 0
    eigenvalues = np.linalg.eigvalsh(K)
    is_psd = np.all(eigenvalues >= -1e-5)
    
    diag_values = np.diag(K)
    diag_mean = np.mean(diag_values)
    is_normalized = np.allclose(diag_mean, 1.0, atol=1e-3)
    
    return {
        "is_symmetric": is_symmetric,
        "is_psd": is_psd,
        "diagonal_mean": float(diag_mean),
        "is_normalized": is_normalized,
        "eigenvalues_min": float(np.min(eigenvalues)),
        "eigenvalues_max": float(np.max(eigenvalues))
    }

if __name__ == "__main__":
    X_train = np.random.rand(10, 8) * np.pi
    X_test = np.random.rand(5, 8) * np.pi
    
    kernel = create_quantum_kernel(n_features=8)
    K_train, K_test = compute_kernel_matrices(kernel, X_train, X_test)
    
    print("K_train shape:", K_train.shape)
    print("Validation:", validate_kernel_matrix(K_train))
