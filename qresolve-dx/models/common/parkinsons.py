import numpy as np
import pandas as pd
import urllib.request
import io
import warnings
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

# From Max A. Little et al., 'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', BioMedical Engineering OnLine, 2007
PARKINSONS_FEATURE_STATS = {
    'MDVP:Fo(Hz)': {'healthy_mean': 181.94, 'healthy_std': 41.31, 'pd_mean': 145.18, 'pd_std': 38.52},
    'MDVP:Fhi(Hz)': {'healthy_mean': 223.64, 'healthy_std': 47.13, 'pd_mean': 197.10, 'pd_std': 54.62},
    'MDVP:Flo(Hz)': {'healthy_mean': 145.21, 'healthy_std': 42.10, 'pd_mean': 116.32, 'pd_std': 35.12},
    'MDVP:Jitter(%)': {'healthy_mean': 0.0038, 'healthy_std': 0.0010, 'pd_mean': 0.0062, 'pd_std': 0.0025},
    'MDVP:Jitter(Abs)': {'healthy_mean': 0.00003, 'healthy_std': 0.00001, 'pd_mean': 0.00005, 'pd_std': 0.00002},
    'MDVP:RAP': {'healthy_mean': 0.0019, 'healthy_std': 0.0006, 'pd_mean': 0.0033, 'pd_std': 0.0015},
    'MDVP:PPQ': {'healthy_mean': 0.0020, 'healthy_std': 0.0007, 'pd_mean': 0.0034, 'pd_std': 0.0018},
    'Jitter:DDP': {'healthy_mean': 0.0058, 'healthy_std': 0.0019, 'pd_mean': 0.0099, 'pd_std': 0.0045},
    'MDVP:Shimmer': {'healthy_mean': 0.0176, 'healthy_std': 0.0051, 'pd_mean': 0.0336, 'pd_std': 0.0125},
    'MDVP:Shimmer(dB)': {'healthy_mean': 0.162, 'healthy_std': 0.051, 'pd_mean': 0.321, 'pd_std': 0.110},
    'Shimmer:APQ3': {'healthy_mean': 0.0095, 'healthy_std': 0.0029, 'pd_mean': 0.0176, 'pd_std': 0.0070},
    'Shimmer:APQ5': {'healthy_mean': 0.0105, 'healthy_std': 0.0032, 'pd_mean': 0.0202, 'pd_std': 0.0085},
    'MDVP:APQ': {'healthy_mean': 0.0133, 'healthy_std': 0.0042, 'pd_mean': 0.0276, 'pd_std': 0.0120},
    'Shimmer:DDA': {'healthy_mean': 0.0285, 'healthy_std': 0.0087, 'pd_mean': 0.0530, 'pd_std': 0.0210},
    'NHR': {'healthy_mean': 0.0114, 'healthy_std': 0.0050, 'pd_mean': 0.0292, 'pd_std': 0.0180},
    'HNR': {'healthy_mean': 24.60, 'healthy_std': 3.12, 'pd_mean': 20.97, 'pd_std': 3.25},
    'RPDE': {'healthy_mean': 0.442, 'healthy_std': 0.081, 'pd_mean': 0.516, 'pd_std': 0.095},
    'DFA': {'healthy_mean': 0.695, 'healthy_std': 0.045, 'pd_mean': 0.725, 'pd_std': 0.048},
    'spread1': {'healthy_mean': -6.75, 'healthy_std': 0.58, 'pd_mean': -5.33, 'pd_std': 0.95},
    'spread2': {'healthy_mean': 0.160, 'healthy_std': 0.045, 'pd_mean': 0.248, 'pd_std': 0.065},
    'D2': {'healthy_mean': 2.15, 'healthy_std': 0.25, 'pd_mean': 2.45, 'pd_std': 0.35},
    'PPE': {'healthy_mean': 0.123, 'healthy_std': 0.041, 'pd_mean': 0.233, 'pd_std': 0.075}
}

PARKINSONS_FEATURE_NAMES = list(PARKINSONS_FEATURE_STATS.keys())

def get_dataset_info() -> Dict[str, Any]:
    return {
        'name': "Oxford Parkinson's Disease Detection Dataset",
        'source': 'UCI Machine Learning Repository',
        'citation': 'Little MA, et al. BioMedical Engineering OnLine. 2007;6:23',
        'url': 'https://archive.ics.uci.edu/ml/datasets/parkinsons',
        'n_samples': 195,
        'n_features': len(PARKINSONS_FEATURE_NAMES),
        'instances': 195,
        'features': len(PARKINSONS_FEATURE_NAMES),
        'classes': {0: 'Healthy', 1: "Parkinson's"}
    }

def generate_synthetic_parkinsons_data(n_samples: int = 195, pd_prevalence: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic data based on real published statistical distributions."""
    np.random.seed(42)
    n_pd = int(n_samples * pd_prevalence)
    n_healthy = n_samples - n_pd
    
    X_synthetic = np.zeros((n_samples, len(PARKINSONS_FEATURE_NAMES)))
    y_synthetic = np.array([1] * n_pd + [0] * n_healthy)
    
    for i, feature in enumerate(PARKINSONS_FEATURE_NAMES):
        stats = PARKINSONS_FEATURE_STATS[feature]
        pd_values = np.random.normal(stats['pd_mean'], stats['pd_std'], n_pd)
        healthy_values = np.random.normal(stats['healthy_mean'], stats['healthy_std'], n_healthy)
        
        # Ensure values don't go negative for non-negative features like frequencies and jitters
        if 'Hz' in feature or 'Jitter' in feature or 'Shimmer' in feature or feature in ['NHR', 'HNR', 'RPDE', 'DFA', 'D2', 'PPE']:
            pd_values = np.clip(pd_values, a_min=0, a_max=None)
            healthy_values = np.clip(healthy_values, a_min=0, a_max=None)
            
        X_synthetic[:, i] = np.concatenate([pd_values, healthy_values])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    return X_synthetic[indices], y_synthetic[indices]

def load_parkinsons_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Tries to load real data from UCI repository.
    Falls back to statistically realistic synthetic data if unavailable.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    
    try:
        # Try fetching real data
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = response.read().decode('utf-8')
        
        df = pd.read_csv(io.StringIO(data))
        # The 'name' column is the patient ID, 'status' is the label (1 for PD, 0 for healthy)
        y = df['status'].values
        # Drop 'name' and 'status'
        X_df = df.drop(columns=['name', 'status'])
        
        # Ensure columns match our expected feature names exactly, or just use theirs
        # The UCI dataset column names match our PARKINSONS_FEATURE_NAMES
        # but let's reorder to match exactly
        X_df = X_df[PARKINSONS_FEATURE_NAMES]
        X = X_df.values
        print("Successfully loaded REAL Parkinson's dataset from UCI ML Repository.")
        return X, y, PARKINSONS_FEATURE_NAMES
        
    except Exception as e:
        print(f"Failed to fetch real data from UCI (Error: {e}).")
        print("Falling back to statistically REALISTIC synthetic data based on published distributions.")
        X, y = generate_synthetic_parkinsons_data()
        return X, y, PARKINSONS_FEATURE_NAMES

def train_parkinsons_model(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains an XGBoost model with stratified 5-fold CV and calibrated probabilities.
    
    Returns:
        Tuple of (trained_model, cv_results_dict).
        cv_results keys: accuracy, f1, auc_roc (flat floats = mean across folds)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_f1s = []
    fold_aucs = []
    
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1
    )
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train on fold
        fold_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1
        )
        fold_model.fit(X_train, y_train)
        
        y_pred = fold_model.predict(X_test)
        y_prob = fold_model.predict_proba(X_test)[:, 1]
        
        fold_accuracies.append(float(accuracy_score(y_test, y_pred)))
        fold_f1s.append(float(f1_score(y_test, y_pred)))
        fold_aucs.append(float(roc_auc_score(y_test, y_prob)))
        
    # Final calibrated model trained on all data
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)
    calibrated_model.fit(X, y)
    
    # Extract feature importance from base model trained on all data
    final_base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1
    )
    final_base.fit(X, y)
    importance = dict(zip(PARKINSONS_FEATURE_NAMES, final_base.feature_importances_.tolist()))
    
    # Flat cv_results schema — consistent with breast_cancer and pipeline expectations
    cv_results = {
        'accuracy': float(np.mean(fold_accuracies)),
        'f1': float(np.mean(fold_f1s)),
        'auc_roc': float(np.mean(fold_aucs)),
        'per_fold_accuracy': fold_accuracies,
        'per_fold_f1': fold_f1s,
        'per_fold_auc': fold_aucs,
        'n_samples': len(X),
        'feature_importance': dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    }
    
    return calibrated_model, cv_results

def predict_parkinsons(model: Any, voice_features: Dict[str, float]) -> Dict[str, Any]:
    """Predicts Parkinson's probability from a dictionary of voice features."""
    # Ensure features are in the correct order
    try:
        X_input = np.array([[voice_features[feat] for feat in PARKINSONS_FEATURE_NAMES]])
    except KeyError as e:
        raise ValueError(f"Missing required voice feature: {e}")
        
    prob = model.predict_proba(X_input)[0, 1]
    prediction = int(model.predict(X_input)[0])
    
    return {
        'prediction': prediction,
        'probability': float(prob),
        'diagnosis': "Parkinson's Disease" if prediction == 1 else 'Healthy',
        'confidence': float(prob if prediction == 1 else 1 - prob)
    }

if __name__ == "__main__":
    print("=== Parkinson's Disease Classifier ===")
    info = get_dataset_info()
    print(f"Dataset: {info['name']}")
    print(f"Citation: {info['citation']}")
    print("-" * 40)
    
    X, y, feature_names = load_parkinsons_data()
    print(f"Loaded {len(X)} instances with {len(feature_names)} features.")
    
    print("\\nTraining model with Stratified 5-Fold CV...")
    model, cv_results = train_parkinsons_model(X, y)
    
    print("\\nModel Performance (CV Means):")
    for metric, value in cv_results.items():
        if metric != 'feature_importance' and isinstance(value, float):
            print(f"  {metric.capitalize():<10}: {value:.4f}")
            
    print("\\nTop 5 Important Features:")
    top_features = list(cv_results['feature_importance'].items())[:5]
    for feat, imp in top_features:
        print(f"  {feat:<16}: {imp:.4f}")
        
    # Example prediction using synthetic healthy patient
    print("\\nExample Prediction (Healthy control profile):")
    sample_healthy = {feat: PARKINSONS_FEATURE_STATS[feat]['healthy_mean'] for feat in PARKINSONS_FEATURE_NAMES}
    result = predict_parkinsons(model, sample_healthy)
    print(f"  Diagnosis : {result['diagnosis']}")
    print(f"  Confidence: {result['confidence']:.4f}")
