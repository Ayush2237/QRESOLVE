import sys, os
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.disease_data import (
    ALL_DISEASES, DISEASE_NAMES, DISEASE_LABEL_MAP, LABEL_DISEASE_MAP,
    ALL_HPO_TERMS, HPO_TERM_INDEX, HPO_TERMS, NUM_FEATURES,
    PAIRWISE_DISTINGUISHING, HPO_TO_CLINICAL_TEST,
    DISEASE_BY_NAME
)

def compute_entropy(probs: np.ndarray) -> float:
    """Compute entropy of a probability distribution."""
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log2(probs))

def compute_posterior(prior_probs: np.ndarray, disease_names: List[str], term: str, term_present: bool) -> np.ndarray:
    """
    Compute posterior probabilities P(disease | term=present/absent)
    using Bayesian update.
    """
    posteriors = np.zeros_like(prior_probs)
    
    for i, d_name in enumerate(disease_names):
        disease = DISEASE_BY_NAME.get(d_name)
        if not disease:
            posteriors[i] = prior_probs[i]
            continue
            
        if term in disease.symptoms:
            p_term_given_d = disease.symptoms[term]
        elif term in disease.absent_symptoms:
            p_term_given_d = 0.0
        else:
            p_term_given_d = 0.05
            
        if not term_present:
            p_term_given_d = 1.0 - p_term_given_d
            
        posteriors[i] = p_term_given_d * prior_probs[i]
        
    total_prob = np.sum(posteriors)
    if total_prob > 0:
        posteriors = posteriors / total_prob
    else:
        posteriors = np.ones_like(prior_probs) / len(prior_probs)
        
    return posteriors

def compute_information_gain(prior_probs: np.ndarray, disease_names: List[str], term: str) -> float:
    """
    Compute information gain for testing an HPO term.
    IG(term) = H(P_before) - [P(term=1)*H(P_after|term=1) + P(term=0)*H(P_after|term=0)]
    """
    h_prior = compute_entropy(prior_probs)
    
    p_term_1 = 0.0
    for i, d_name in enumerate(disease_names):
        disease = DISEASE_BY_NAME.get(d_name)
        if disease:
            p_symp = disease.symptoms.get(term, 0.0) if term not in disease.absent_symptoms else 0.0
        else:
            p_symp = 0.05
        p_term_1 += p_symp * prior_probs[i]
        
    p_term_0 = 1.0 - p_term_1
    
    post_1 = compute_posterior(prior_probs, disease_names, term, True)
    post_0 = compute_posterior(prior_probs, disease_names, term, False)
    
    h_post_1 = compute_entropy(post_1)
    h_post_0 = compute_entropy(post_0)
    
    ig = h_prior - (p_term_1 * h_post_1 + p_term_0 * h_post_0)
    return float(ig)

def recommend_next_test(current_probs: np.ndarray, disease_names: List[str], observed_terms: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Recommend the next clinical tests based on Information Gain.
    
    For each untested HPO term, compute:
      IG(term) = H(prior) - E[H(posterior)]
    
    Then report which disease's probability *increases most* upon
    positive vs negative test results (posterior - prior comparison).
    """
    recommendations = []
    
    for term in ALL_HPO_TERMS:
        if term in observed_terms:
            continue
            
        ig = compute_information_gain(current_probs, disease_names, term)
        if ig <= 0:
            continue
            
        post_1 = compute_posterior(current_probs, disease_names, term, True)
        post_0 = compute_posterior(current_probs, disease_names, term, False)
        
        # Find which disease GAINED the most probability mass (not just argmax)
        # This is the key fix: compare posterior to prior
        gain_if_positive = post_1 - current_probs
        gain_if_negative = post_0 - current_probs
        
        # Disease that benefits most from a positive test result
        beneficiary_pos = disease_names[np.argmax(gain_if_positive)]
        # Disease that benefits most from a negative test result
        beneficiary_neg = disease_names[np.argmax(gain_if_negative)]
        
        recommendations.append({
            'hpo_id': term,
            'label': HPO_TERMS.get(term, term),
            'information_gain': ig,
            'clinical_test': HPO_TO_CLINICAL_TEST.get(term, "Clinical Evaluation"),
            'expected_outcome_positive': beneficiary_pos,
            'expected_outcome_negative': beneficiary_neg
        })
        
    recommendations.sort(key=lambda x: x['information_gain'], reverse=True)
    return recommendations[:top_k]

def format_recommendation(recommendations: List[Dict[str, Any]]) -> str:
    """Format recommendations for the clinician."""
    if not recommendations:
        return "No further tests recommended."
        
    lines = ["### Recommended Next Tests"]
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"**{i}. Recommended next test:** {rec['clinical_test']}")
        lines.append(f"   - **Rationale:** Testing for {rec['label']} ({rec['hpo_id']}) would provide {rec['information_gain']:.3f} bits of information.")
        lines.append(f"   - **If positive:** {rec['expected_outcome_positive']} becomes more likely.")
        lines.append(f"   - **If negative:** {rec['expected_outcome_negative']} becomes more likely.")
        lines.append("")
        
    return "\n".join(lines)

if __name__ == "__main__":
    priors = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    observed = ["HP:0001631", "HP:0002138"]
    recs = recommend_next_test(priors, DISEASE_NAMES, observed, top_k=3)
    print(format_recommendation(recs))
