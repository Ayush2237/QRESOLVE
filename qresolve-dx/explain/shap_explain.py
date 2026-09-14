import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.disease_data import (
    ALL_DISEASES, DISEASE_NAMES, DISEASE_LABEL_MAP, LABEL_DISEASE_MAP,
    ALL_HPO_TERMS, HPO_TERM_INDEX, HPO_TERMS, NUM_FEATURES,
    PAIRWISE_DISTINGUISHING, HPO_TO_CLINICAL_TEST,
    DISEASE_BY_NAME
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def _extract_base_model(model):
    """
    Extract the base tree model from a CalibratedClassifierCV wrapper.
    SHAP's TreeExplainer needs the raw XGBoost/RF model, not the calibrated wrapper.
    """
    # CalibratedClassifierCV wraps the estimator
    if hasattr(model, 'estimator'):
        return model.estimator
    # Already fitted CalibratedClassifierCV stores calibrated classifiers
    if hasattr(model, 'calibrated_classifiers_'):
        # Each calibrated classifier has a .estimator attribute
        inner = model.calibrated_classifiers_[0]
        if hasattr(inner, 'estimator'):
            return inner.estimator
    return model


def explain_prediction(model, X_patient, top2_diseases, feature_names=ALL_HPO_TERMS):
    """
    Explain the prediction using SHAP values.
    
    Extracts the base tree model from CalibratedClassifierCV if needed,
    since SHAP's TreeExplainer cannot handle calibration wrappers.
    
    Args:
        model: Trained model (may be CalibratedClassifierCV wrapping XGBoost)
        X_patient: Patient feature vector (shape: 1 x NUM_FEATURES)
        top2_diseases: Tuple/list of (top_disease, runner_up_disease)
        feature_names: List of feature names
        
    Returns:
        dict: Explanation dict with supporting and against evidence
    """
    top_disease, runner_up = top2_diseases
    
    top_class = DISEASE_LABEL_MAP.get(top_disease, 0)
    runner_class = DISEASE_LABEL_MAP.get(runner_up, 1)
    
    result = {
        'top_disease': top_disease,
        'runner_up': runner_up,
        'supporting_evidence': [],
        'against_evidence': []
    }
    
    # Extract base model for SHAP (TreeExplainer needs raw XGBoost, not calibrated wrapper)
    base_model = _extract_base_model(model)
    
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_patient)
            
            if isinstance(shap_values, list):
                top_shap = shap_values[top_class][0]
                runner_shap = shap_values[runner_class][0]
            elif len(shap_values.shape) == 3:
                top_shap = shap_values[0, :, top_class]
                runner_shap = shap_values[0, :, runner_class]
            else:
                top_shap = shap_values[0] if len(shap_values.shape) == 2 else shap_values
                runner_shap = -top_shap
                
            for idx in np.argsort(-top_shap)[:5]:
                if top_shap[idx] > 0:
                    hpo_id = feature_names[idx]
                    label = HPO_TERMS.get(hpo_id, hpo_id)
                    result['supporting_evidence'].append({
                        'hpo_id': hpo_id,
                        'label': label,
                        'shap_value': float(top_shap[idx]),
                        'direction': 'positive'
                    })
                    
            diff_shap = top_shap - runner_shap
            for idx in np.argsort(-diff_shap)[:5]:
                if runner_shap[idx] < 0 and top_shap[idx] > 0:
                    hpo_id = feature_names[idx]
                    label = HPO_TERMS.get(hpo_id, hpo_id)
                    result['against_evidence'].append({
                        'hpo_id': hpo_id,
                        'label': label,
                        'shap_value': float(runner_shap[idx]),
                        'direction': 'negative for runner-up'
                    })
                    
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return _fallback_explanation(base_model, top_disease, runner_up, feature_names)
    else:
        return _fallback_explanation(base_model, top_disease, runner_up, feature_names)
        
    return result

def _fallback_explanation(model, top_disease, runner_up, feature_names):
    """Fallback using feature_importances_ if SHAP is not available."""
    result = {
        'top_disease': top_disease,
        'runner_up': runner_up,
        'supporting_evidence': [],
        'against_evidence': []
    }
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for idx in np.argsort(-importances)[:5]:
            if importances[idx] > 0:
                hpo_id = feature_names[idx]
                label = HPO_TERMS.get(hpo_id, hpo_id)
                result['supporting_evidence'].append({
                    'hpo_id': hpo_id,
                    'label': label,
                    'shap_value': float(importances[idx]),
                    'direction': 'positive importance'
                })
    return result

def format_explanation(explanation):
    """
    Format SHAP explanation into human-readable text.
    """
    top_disease = explanation.get('top_disease', 'Top Disease')
    runner_up = explanation.get('runner_up', 'Runner-up Disease')
    
    lines = [f"### Model Explanation"]
    lines.append(f"**Top Prediction:** {top_disease}")
    lines.append(f"**Runner-up:** {runner_up}\n")
    
    lines.append(f"The following features support **{top_disease}**:")
    for ev in explanation.get('supporting_evidence', []):
        lines.append(f"- **{ev['label']}** ({ev['hpo_id']}): impact +{ev['shap_value']:.4f}")
        
    if explanation.get('against_evidence'):
        lines.append(f"\nThe following features argue against **{runner_up}**:")
        for ev in explanation.get('against_evidence', []):
            lines.append(f"- **{ev['label']}** ({ev['hpo_id']}): impact {ev['shap_value']:.4f}")
            
    return "\n".join(lines)

if __name__ == "__main__":
    class DummyModel:
        def __init__(self):
            self.feature_importances_ = np.random.rand(NUM_FEATURES)
            
    model = DummyModel()
    X_patient = np.random.randint(0, 2, size=(1, NUM_FEATURES))
    explanation = explain_prediction(model, X_patient, ("Marfan syndrome", "Loeys-Dietz syndrome"))
    print(format_explanation(explanation))
