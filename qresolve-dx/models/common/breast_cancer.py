"""
Breast Cancer Classification Model

This module implements a classifier for breast cancer using the REAL 
Wisconsin Breast Cancer Dataset from sklearn.datasets. 
It uses actual patient data (569 samples, 30 features) to predict 
malignancy using an XGBoost classifier with probability calibration.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def load_breast_cancer_data() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Loads the real Wisconsin Breast Cancer Dataset.
    
    Returns:
        X (np.ndarray): Feature matrix (569, 30).
        y (np.ndarray): Target labels (0 = malignant, 1 = benign).
        feature_names (List[str]): Names of the 30 features.
        target_names (List[str]): Names of the target classes.
    """
    data = load_breast_cancer()
    return data.data, data.target, list(data.feature_names), list(data.target_names)


def train_breast_cancer_model(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains an XGBoost classifier with Stratified K-Fold Cross Validation.
    Uses CalibratedClassifierCV for calibrated probabilities.
    
    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Targets.
        n_splits (int): Number of CV folds.
        
    Returns:
        Tuple of (trained_model, cv_results_dict).
        cv_results keys: accuracy, f1, auc_roc (flat floats = mean across folds)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize base XGBoost model
    base_xgb = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1,
        objective='binary:logistic', 
        eval_metric='logloss',
        random_state=42
    )
    
    fold_accuracies = []
    fold_f1s = []
    fold_aucs = []
    
    # Train base model on entire dataset to get feature importances
    base_xgb.fit(X, y)
    importances = base_xgb.feature_importances_
    
    for fold_i, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train fold model
        fold_model = CalibratedClassifierCV(base_xgb, method='sigmoid', cv=2)
        fold_model.fit(X_train, y_train)
        
        # Predict
        y_pred = fold_model.predict(X_test)
        y_prob = fold_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        fold_aucs.append(auc)
        
    # Fit final calibrated model on all data
    final_calibrated_model = CalibratedClassifierCV(base_xgb, method='sigmoid', cv=5)
    final_calibrated_model.fit(X, y)
    
    # Calculate final predictions for confusion matrix
    y_pred_all = final_calibrated_model.predict(X)
    conf_mat = confusion_matrix(y, y_pred_all)
    
    _, _, feature_names, _ = load_breast_cancer_data()
    
    # Feature importances from the base model trained on all data
    importance_dict = {feat: float(imp) for feat, imp in zip(feature_names, importances)}
    top_10_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Flat cv_results schema — consistent with parkinsons and pipeline expectations
    cv_results = {
        'accuracy': float(np.mean(fold_accuracies)),
        'f1': float(np.mean(fold_f1s)),
        'auc_roc': float(np.mean(fold_aucs)),
        'per_fold_accuracy': fold_accuracies,
        'per_fold_f1': fold_f1s,
        'per_fold_auc': fold_aucs,
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'confusion_matrix': conf_mat.tolist(),
        'top_10_features': top_10_features
    }
    
    return final_calibrated_model, cv_results


def predict_breast_cancer(model: Any, features: Dict[str, float]) -> Dict[str, Any]:
    """
    Predicts breast cancer malignancy using a trained model.
    
    Args:
        model: Trained calibrated XGBoost model.
        features: Dictionary mapping feature names to their values.
        
    Returns:
        Dict containing prediction, probability, confidence, and top features.
    """
    _, _, feature_names, target_names = load_breast_cancer_data()
    
    # Order features correctly according to dataset feature names
    feature_vector = np.array([[features.get(name, 0.0) for name in feature_names]])
    
    prob = model.predict_proba(feature_vector)[0]
    pred_idx = int(np.argmax(prob))
    
    prediction = target_names[pred_idx]
    probability = float(prob[pred_idx])
    
    # SHAP explanation
    supporting_evidence = []
    against_evidence = []
    try:
        if SHAP_AVAILABLE:
            import shap
            # Extract base estimator from CalibratedClassifierCV
            base_model = model.calibrated_classifiers_[0].estimator if hasattr(model, 'calibrated_classifiers_') else model.estimator
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(feature_vector)[0]
            
            # Map SHAP values to features
            for i, feat_name in enumerate(feature_names):
                sv = shap_values[i]
                val = feature_vector[0][i]
                item = {
                    'feature': str(feat_name),
                    'value': float(val),
                    'shap_value': float(sv)
                }
                # For XGBoost binary classification, positive SHAP pushes towards class 1 (Benign in this dataset)
                # We need to align it with the predicted class.
                # If pred_idx == 0 (Malignant), negative SHAP means supporting Malignant.
                if pred_idx == 0:
                    if sv < 0:
                        supporting_evidence.append(item)
                    elif sv > 0:
                        against_evidence.append(item)
                else: # pred_idx == 1 (Benign)
                    if sv > 0:
                        supporting_evidence.append(item)
                    elif sv < 0:
                        against_evidence.append(item)
                        
            # Sort by absolute SHAP value
            supporting_evidence.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            against_evidence.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            # Take top 5
            supporting_evidence = supporting_evidence[:5]
            against_evidence = against_evidence[:5]
    except Exception as e:
        print(f"SHAP error: {e}")

    result = {
        'prediction': prediction,
        'probability': probability,
        'confidence': 'High' if probability > 0.85 else 'Medium' if probability > 0.6 else 'Low',
        'input_features': features,
        'supporting_evidence': supporting_evidence,
        'against_evidence': against_evidence
    }
    return result


def get_dataset_info() -> Dict[str, Any]:
    """
    Returns metadata about the REAL Wisconsin Breast Cancer dataset.
    """
    data = load_breast_cancer()
    return {
        'name': 'Wisconsin Breast Cancer Dataset',
        'source': 'sklearn.datasets',
        'n_samples': data.data.shape[0],
        'n_features': data.data.shape[1],
        'class_distribution': {
            data.target_names[0]: int(np.sum(data.target == 0)),
            data.target_names[1]: int(np.sum(data.target == 1))
        },
        'citation': 'Wolberg, W.H., Street, W.N., Mangasarian, O.L. (1995). UCI Machine Learning Repository'
    }


if __name__ == "__main__":
    print("Loading Breast Cancer Dataset...")
    X, y, feature_names, target_names = load_breast_cancer_data()
    
    info = get_dataset_info()
    print("\n--- Dataset Info ---")
    for k, v in info.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v}")
        
    print("\nTraining XGBoost Model with 5-Fold CV...")
    model, cv_results = train_breast_cancer_model(X, y)
    
    print("\n--- CV Results ---")
    print("Per-fold Accuracy:", [round(x, 4) for x in cv_results['per_fold']['accuracy']])
    print(f"Mean Accuracy: {cv_results['overall']['accuracy']:.4f}")
    print(f"Mean F1 Score: {cv_results['overall']['f1']:.4f}")
    print(f"Mean AUC-ROC:  {cv_results['overall']['auc_roc']:.4f}")
    
    print("\n--- Top 10 Features ---")
    for i, (feat, imp) in enumerate(cv_results['top_10_features']):
        print(f"{i+1}. {feat}: {imp:.4f}")
        
    print("\n--- Sample Prediction ---")
    # Take a sample from the dataset
    sample_idx = 0
    sample_features = {name: val for name, val in zip(feature_names, X[sample_idx])}
    
    prediction = predict_breast_cancer(model, sample_features)
    print(f"Actual class: {target_names[y[sample_idx]]}")
    print(f"Predicted class: {prediction['prediction']}")
    print(f"Probability: {prediction['probability']:.4f} ({prediction['confidence']})")
