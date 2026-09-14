"""
Quantum kernel logic using ZZFeatureMap.

The ZZFeatureMap encodes classical data into a quantum state using:
    |Φ(x)⟩ = U_Φ(x) |0⟩^n

where U_Φ(x) = exp(i Σ_{S⊆[n],|S|≤2} φ_S(x) Π_{i∈S} Z_i) · H^⊗n

For single-qubit terms: φ_{i}(x) = x_i
For two-qubit terms:    φ_{ij}(x) = (π - x_i)(π - x_j)

The quantum kernel is:  K(x, y) = |⟨Φ(y)|Φ(x)⟩|²

This captures second-order feature correlations that classical RBF
kernels cannot efficiently represent, making it suitable for resolving
highly overlapping phenotypic profiles in rare disease differential diagnosis.

Reference: Havlicek et al. (2019) "Supervised learning with quantum-enhanced feature spaces"
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

# --- Detect Qiskit availability and version ---
QISKIT_AVAILABLE = False
QISKIT_AER_AVAILABLE = False

try:
    import qiskit
    QISKIT_AVAILABLE = True
except ImportError:
    warnings.warn("qiskit not installed. Quantum kernel will use classical RBF fallback.")

try:
    import qiskit_aer
    QISKIT_AER_AVAILABLE = True
except ImportError:
    pass


def create_quantum_kernel(n_features: int = 8, reps: int = 2, entanglement: str = 'linear') -> Any:
    """
    Creates a FidelityQuantumKernel using ZZFeatureMap.

    The kernel K(x,y) = |⟨Φ(y)|Φ(x)⟩|² is computed via the
    ComputeUncompute fidelity method on a statevector simulator.

    If Qiskit is not installed, falls back to a classical RBF kernel.

    Args:
        n_features: Number of features (= number of qubits). Max 8 for simulation.
        reps: Number of repetitions of the feature map circuit.
        entanglement: Entanglement strategy ('linear', 'full', 'circular').

    Returns:
        A kernel object with an `evaluate(x, y)` method.
    """
    if not QISKIT_AVAILABLE:
        from sklearn.metrics.pairwise import rbf_kernel
        class ClassicalFallbackKernel:
            """RBF kernel fallback when Qiskit is unavailable."""
            def evaluate(self, x_vec, y_vec=None):
                return rbf_kernel(x_vec, y_vec)
        print("  [Kernel] Using classical RBF fallback (Qiskit not installed)")
        return ClassicalFallbackKernel()

    # Try modern Qiskit API first (>= 2.0), then legacy
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        # Suppress deprecation warnings for ZZFeatureMap class (still functional)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            feature_map = ZZFeatureMap(
                feature_dimension=n_features,
                reps=reps,
                entanglement=entanglement
            )

        print(f"  [Kernel] ZZFeatureMap: {n_features} qubits, {reps} reps, {entanglement} entanglement")
        print(f"  [Kernel] Circuit depth: {feature_map.decompose().depth()}")
        print(f"  [Kernel] Gate count: {dict(feature_map.decompose().count_ops())}")

        # Try to create a high-quality fidelity estimator
        fidelity = None
        if QISKIT_AER_AVAILABLE:
            try:
                from qiskit_algorithms.state_fidelities import ComputeUncompute
                from qiskit.primitives import StatevectorSampler
                sampler = StatevectorSampler()
                fidelity = ComputeUncompute(sampler=sampler)
                print("  [Kernel] Backend: StatevectorSampler (exact simulation)")
            except (ImportError, TypeError):
                pass

        if fidelity is None:
            try:
                from qiskit.primitives import StatevectorSampler
                from qiskit_algorithms.state_fidelities import ComputeUncompute
                sampler = StatevectorSampler()
                fidelity = ComputeUncompute(sampler=sampler)
                print("  [Kernel] Backend: StatevectorSampler (exact)")
            except (ImportError, TypeError):
                pass

        if fidelity is not None:
            kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
        else:
            # FidelityQuantumKernel with default fidelity
            kernel = FidelityQuantumKernel(feature_map=feature_map)
            print("  [Kernel] Backend: Default FidelityQuantumKernel")

        return kernel

    except Exception as e:
        warnings.warn(f"Qiskit kernel creation failed: {e}. Using RBF fallback.")
        from sklearn.metrics.pairwise import rbf_kernel
        class ClassicalFallbackKernel:
            def evaluate(self, x_vec, y_vec=None):
                return rbf_kernel(x_vec, y_vec)
        print(f"  [Kernel] Falling back to classical RBF: {e}")
        return ClassicalFallbackKernel()


def compute_kernel_matrices(kernel: Any, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the quantum kernel matrices.

    K_train[i,j] = |⟨Φ(x_j)|Φ(x_i)⟩|²   (training kernel, n×n)
    K_test[i,j]  = |⟨Φ(x_train_j)|Φ(x_test_i)⟩|²  (test kernel, m×n)
    """
    K_train = kernel.evaluate(x_vec=X_train)
    K_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)
    return K_train, K_test


def validate_kernel_matrix(K: np.ndarray) -> Dict[str, Any]:
    """
    Validate that a kernel matrix satisfies Mercer's conditions:
    1. Symmetry: K = K^T
    2. Positive semi-definiteness: all eigenvalues ≥ 0
    3. Normalization: diagonal entries ≈ 1.0 (for fidelity kernels)

    A valid kernel matrix ensures the SVM optimization problem is convex
    and has a unique global optimum.
    """
    is_symmetric = np.allclose(K, K.T, atol=1e-5)

    # Check PSD via eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(K)
    is_psd = bool(np.all(eigenvalues >= -1e-5))

    diag_values = np.diag(K)
    diag_mean = float(np.mean(diag_values))
    is_normalized = np.allclose(diag_mean, 1.0, atol=1e-3)

    return {
        "is_symmetric": bool(is_symmetric),
        "is_psd": is_psd,
        "diagonal_mean": diag_mean,
        "is_normalized": is_normalized,
        "eigenvalues_min": float(np.min(eigenvalues)),
        "eigenvalues_max": float(np.max(eigenvalues)),
        "condition_number": float(np.max(eigenvalues) / max(np.min(eigenvalues[eigenvalues > 0]), 1e-10))
    }


if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.random.rand(10, 8) * np.pi
    X_test = np.random.rand(5, 8) * np.pi

    kernel = create_quantum_kernel(n_features=8)
    K_train, K_test = compute_kernel_matrices(kernel, X_train, X_test)

    print(f"\nK_train shape: {K_train.shape}")
    print(f"K_test shape: {K_test.shape}")
    validation = validate_kernel_matrix(K_train)
    print(f"Validation: {validation}")
