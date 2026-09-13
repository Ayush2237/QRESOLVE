import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.disease_data import (
    ALL_DISEASES, DISEASE_NAMES, DISEASE_LABEL_MAP, LABEL_DISEASE_MAP,
    ALL_HPO_TERMS, HPO_TERM_INDEX, HPO_TERMS, NUM_FEATURES,
    PAIRWISE_DISTINGUISHING, HPO_TO_CLINICAL_TEST,
    DISEASE_BY_NAME
)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def ablation_analysis(kernel_func, X_train, X_test, y_train, y_test, selected_indices, feature_names=ALL_HPO_TERMS):
    """
    Feature ablation analysis for quantum kernel path.
    """
    try:
        K_train = kernel_func(X_train, X_train)
        K_test = kernel_func(X_test, X_train)
        
        clf = SVC(kernel='precomputed')
        clf.fit(K_train, y_train)
        baseline_acc = accuracy_score(y_test, clf.predict(K_test))
    except Exception as e:
        print(f"Quantum kernel failed, using classical fallback. Error: {e}")
        return simplified_ablation(X_train, X_test, y_train, y_test, selected_indices, feature_names)

    results = []
    
    for idx in selected_indices:
        X_train_ablated = X_train.copy()
        X_test_ablated = X_test.copy()
        X_train_ablated[:, idx] = 0
        X_test_ablated[:, idx] = 0
        
        try:
            K_train_ab = kernel_func(X_train_ablated, X_train_ablated)
            K_test_ab = kernel_func(X_test_ablated, X_train_ablated)
            
            clf_ab = SVC(kernel='precomputed')
            clf_ab.fit(K_train_ab, y_train)
            acc = accuracy_score(y_test, clf_ab.predict(K_test_ab))
            
            drop = baseline_acc - acc
            hpo_id = feature_names[idx]
            label = HPO_TERMS.get(hpo_id, hpo_id)
            results.append({
                'feature_index': idx,
                'hpo_id': hpo_id,
                'label': label,
                'accuracy_drop': float(drop)
            })
        except Exception:
            continue
            
    results.sort(key=lambda x: x['accuracy_drop'], reverse=True)
    return {'baseline_accuracy': baseline_acc, 'ablation_results': results}

def simplified_ablation(X_train, X_test, y_train, y_test, selected_indices, feature_names=ALL_HPO_TERMS):
    """Fallback using classical RBF SVM."""
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, clf.predict(X_test))
    
    results = []
    for idx in selected_indices:
        X_train_ablated = X_train.copy()
        X_test_ablated = X_test.copy()
        X_train_ablated[:, idx] = 0
        X_test_ablated[:, idx] = 0
        
        clf_ab = SVC(kernel='rbf')
        clf_ab.fit(X_train_ablated, y_train)
        acc = accuracy_score(y_test, clf_ab.predict(X_test_ablated))
        
        drop = baseline_acc - acc
        hpo_id = feature_names[idx]
        label = HPO_TERMS.get(hpo_id, hpo_id)
        results.append({
            'feature_index': idx,
            'hpo_id': hpo_id,
            'label': label,
            'accuracy_drop': float(drop)
        })
        
    results.sort(key=lambda x: x['accuracy_drop'], reverse=True)
    return {'baseline_accuracy': baseline_acc, 'ablation_results': results}

def format_ablation_report(results):
    """Format ablation results into a readable report."""
    baseline = results.get('baseline_accuracy', 0.0)
    ablation = results.get('ablation_results', [])
    
    lines = [f"### Quantum Feature Ablation Analysis"]
    lines.append(f"**Baseline Accuracy:** {baseline:.4f}\n")
    lines.append("| Feature | HPO Term | Accuracy Drop |")
    lines.append("|---|---|---|")
    
    for res in ablation:
        lines.append(f"| {res['label']} | {res['hpo_id']} | {-res['accuracy_drop']:.4f} |")
        
    return "\n".join(lines)

if __name__ == "__main__":
    X_train = np.random.rand(10, NUM_FEATURES)
    X_test = np.random.rand(5, NUM_FEATURES)
    y_train = np.random.randint(0, 5, 10)
    y_test = np.random.randint(0, 5, 5)
    
    sel_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    res = simplified_ablation(X_train, X_test, y_train, y_test, sel_indices)
    print(format_ablation_report(res))
