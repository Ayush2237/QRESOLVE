"""
HPO vectorization with Information Content (IC) weighting and Lin's semantic similarity.
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.disease_data import (
    ALL_DISEASES, ALL_HPO_TERMS, HPO_TERM_INDEX, NUM_FEATURES,
    DISEASE_NAMES, DISEASE_LABEL_MAP, HPO_TERMS
)

# Approximate HPO parent map to enable simplified Lin similarity without the full DAG
HPO_PARENT_MAP = {
    # Abnormality of the eye (HP:0000478)
    "HP:0000486": "HP:0000478",
    "HP:0001083": "HP:0000478",
    "HP:0000545": "HP:0000478",
    "HP:0000541": "HP:0000478",
    "HP:0000639": "HP:0000478",
    "HP:0000508": "HP:0000478",
    "HP:0007703": "HP:0000478",

    # Abnormality of the cardiovascular system (HP:0001626)
    "HP:0001631": "HP:0001626",
    "HP:0001650": "HP:0001626",
    "HP:0001659": "HP:0001626",
    "HP:0002616": "HP:0001626",
    "HP:0004942": "HP:0001626",
    "HP:0005110": "HP:0001626",
    "HP:0012722": "HP:0001626",
    "HP:0002617": "HP:0001626",
    "HP:0001679": "HP:0001626",

    # Abnormality of the skeletal system (HP:0000924)
    "HP:0001166": "HP:0000924",
    "HP:0001238": "HP:0000924",
    "HP:0001382": "HP:0000924",
    "HP:0002650": "HP:0000924",
    "HP:0002705": "HP:0000924",
    "HP:0002758": "HP:0000924",
    "HP:0002808": "HP:0000924",
    "HP:0003088": "HP:0000924",
    "HP:0003179": "HP:0000924",
    "HP:0004209": "HP:0000924",
    "HP:0005059": "HP:0000924",
    "HP:0005692": "HP:0000924",
    "HP:0006610": "HP:0000924",
    "HP:0006657": "HP:0000924",
    "HP:0002829": "HP:0000924",
    "HP:0003423": "HP:0000924",
    
    # Abnormality of head or neck (HP:0000152)
    "HP:0000218": "HP:0000152",
    "HP:0000272": "HP:0000152",
    "HP:0000278": "HP:0000152",
    "HP:0000316": "HP:0000152",
    "HP:0000348": "HP:0000152",
    "HP:0000369": "HP:0000152",
    "HP:0000431": "HP:0000152",
    "HP:0000455": "HP:0000152",
    "HP:0000494": "HP:0000152",
    "HP:0000252": "HP:0000152",
    
    # Abnormality of skin/integument (HP:0000951)
    "HP:0000974": "HP:0000951",
    "HP:0000977": "HP:0000951",
    "HP:0000987": "HP:0000951",
    "HP:0000988": "HP:0000951",
    "HP:0000994": "HP:0000951",
    "HP:0001030": "HP:0000951",
    "HP:0001065": "HP:0000951",
    "HP:0008066": "HP:0000951",
    "HP:0011121": "HP:0000951",
    
    # Other/Metabolism/GI (HP:0001939)
    "HP:0002019": "HP:0001939",
    "HP:0002020": "HP:0001939",
    "HP:0002104": "HP:0001939",
    "HP:0002315": "HP:0001939"
}

def compute_information_content() -> Dict[str, float]:
    """
    Compute IC(term) = -log(P(term)) where P(term) = (number of diseases annotated with term) / (total diseases)
    Uses the 5-disease corpus from disease_data.
    More specific terms get higher IC.
    """
    ic_values = {}
    total_diseases = len(ALL_DISEASES)
    
    term_counts = {term: 0 for term in ALL_HPO_TERMS}
    
    for disease in ALL_DISEASES:
        # A disease has a term if it appears in its symptoms dict
        for term in disease.symptoms.keys():
            if term in term_counts:
                term_counts[term] += 1
                
    for term, count in term_counts.items():
        if count > 0:
            p_term = count / total_diseases
            ic_values[term] = -math.log(p_term)
        else:
            ic_values[term] = float('inf')
            
    # Handle zero-count terms by assigning max IC + 1
    valid_ics = [ic for ic in ic_values.values() if ic != float('inf')]
    max_ic = max(valid_ics) if valid_ics else 0.0
    
    for term, ic in ic_values.items():
        if ic == float('inf'):
            ic_values[term] = max_ic + 1.0
            
    return ic_values


def try_compute_ic_with_pyhpo() -> Optional[Dict[str, float]]:
    """
    Try to use pyhpo for full HPO corpus IC (much better IC values).
    Return None if pyhpo not installed.
    """
    try:
        from pyhpo import Ontology
        _ = Ontology()
        
        ic_values = {}
        for term in ALL_HPO_TERMS:
            try:
                hpo_term = Ontology.get_hpo_object(term)
                ic_values[term] = hpo_term.information_content.omim
            except Exception:
                # If term not found, fallback to 0.0 or skip
                pass
                
        return ic_values if ic_values else None
    except ImportError:
        return None
    except Exception:
        return None


def compute_lin_similarity(t1: str, t2: str, ic_values: Dict[str, float]) -> float:
    """
    Simplified Lin's similarity without full ontology:
    sim(t1, t2) = 2 * IC(MICA(t1,t2)) / (IC(t1) + IC(t2))
    Without full DAG, approximate MICA by checking if terms share a common ancestor 
    from a hardcoded parent mapping of the HPO terms in our cluster.
    """
    if t1 == t2:
        return 1.0
        
    ic_t1 = ic_values.get(t1, 0.0)
    ic_t2 = ic_values.get(t2, 0.0)
    
    if ic_t1 + ic_t2 == 0:
        return 0.0
        
    parent1 = HPO_PARENT_MAP.get(t1)
    parent2 = HPO_PARENT_MAP.get(t2)
    
    if parent1 and parent2 and parent1 == parent2:
        # Approximate MICA IC as roughly half the minimum IC of the two terms
        # This reflects that the parent is more general than both terms
        mica_ic = 0.5 * min(ic_t1, ic_t2)
        return 2 * mica_ic / (ic_t1 + ic_t2)
        
    return 0.0


def build_ic_weighted_features(patient_hpo_terms: List[str], ic_values: Dict[str, float]) -> np.ndarray:
    """
    Build a feature vector of length NUM_FEATURES.
    For each term in ALL_HPO_TERMS:
      - If term is directly in patient's terms: value = IC(term)
      - If a semantically similar term (Lin sim > 0.5) is in patient's terms: value = IC(term) * sim_score
      - Else: value = 0.0
    """
    feature_vector = np.zeros(NUM_FEATURES)
    
    for term_idx, term in enumerate(ALL_HPO_TERMS):
        if term in patient_hpo_terms:
            feature_vector[term_idx] = ic_values.get(term, 0.0)
        else:
            # Check for semantically similar terms
            max_sim_val = 0.0
            for pt_term in patient_hpo_terms:
                sim = compute_lin_similarity(term, pt_term, ic_values)
                if sim > 0.5:
                    val = ic_values.get(term, 0.0) * sim
                    if val > max_sim_val:
                        max_sim_val = val
            feature_vector[term_idx] = max_sim_val
            
    return feature_vector


def build_feature_matrix(patients_df: pd.DataFrame, ic_values: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes a DataFrame with 'hpo_terms' column (pipe-separated) and 'true_disease' column
    Returns X (n_samples, NUM_FEATURES) IC-weighted matrix and y (n_samples,) label array
    """
    X_list = []
    y_list = []
    
    for _, row in patients_df.iterrows():
        hpo_terms_raw = row.get('hpo_terms', '')
        terms = hpo_terms_raw.split('|') if isinstance(hpo_terms_raw, str) and hpo_terms_raw else []
        
        feature_vec = build_ic_weighted_features(terms, ic_values)
        X_list.append(feature_vec)
        
        disease = row.get('true_disease', '')
        label = DISEASE_LABEL_MAP.get(disease, -1)
        y_list.append(label)
        
    return np.array(X_list), np.array(y_list)


if __name__ == "__main__":
    # Compute IC values
    print("Computing IC values...")
    ic_dict = try_compute_ic_with_pyhpo()
    if ic_dict is None:
        print("pyhpo not available. Using local corpus for IC computation.")
        ic_dict = compute_information_content()
    else:
        print("Successfully used pyhpo for IC computation.")
        
    # Sort terms by IC value (descending - most specific first)
    sorted_terms = sorted(ic_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 15 most specific HPO terms (highest IC):")
    for i, (term, ic) in enumerate(sorted_terms[:15]):
        name = HPO_TERMS.get(term, {}).get("name", "Unknown")
        print(f"{i+1:2d}. {term}: {name:<35} (IC: {ic:.4f})")
