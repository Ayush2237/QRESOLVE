"""
QResolve-Dx: Synthetic Patient Generator

Generates synthetic patients by sampling HPO terms from each disease's
real annotation-frequency distribution (sourced from phenotype.hpoa format).

This is NOT dummy data — each symptom is sampled with Bernoulli probability
equal to the published clinical frequency for that disease-symptom pair.
Label noise simulates real clinical ambiguity by occasionally adding
symptoms from a sibling (look-alike) disease.

The resulting dataset is SYNTHETIC but statistically principled. The benchmark
report states this openly per §10 of the build brief.
"""

from __future__ import annotations

import sys
import os
from typing import Tuple, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.disease_data import (
    ALL_DISEASES, DISEASE_NAMES, DISEASE_LABEL_MAP,
    ALL_HPO_TERMS, HPO_TERM_INDEX, NUM_FEATURES,
    DiseaseProfile,
)


def generate_synthetic_patients(
    n_per_disease: int = 200,
    noise_rate: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic patients by sampling HPO terms from real frequencies.

    For each disease, generates n_per_disease patients:
    - Each HPO term is sampled Bernoulli(p) where p = disease.symptoms[term]
    - Terms in absent_symptoms are forced to 0
    - With probability noise_rate, some terms are swapped with a sibling disease

    Args:
        n_per_disease: Number of patients per disease (default 200, total = 1000)
        noise_rate: Fraction of patients receiving label noise (5-15% realistic)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: patient_id, true_disease, hpo_terms, difficulty
    """
    rng = np.random.RandomState(seed)
    patients = []

    for disease in ALL_DISEASES:
        for i in range(n_per_disease):
            patient_id = f"P_{disease.short_name}_{i:03d}"
            symptoms_present = []

            # Sample each HPO term with probability from real annotation frequencies
            for term in ALL_HPO_TERMS:
                if term in disease.symptoms:
                    p = disease.symptoms[term]
                    if rng.rand() < p:
                        symptoms_present.append(term)
                elif term in disease.absent_symptoms:
                    pass  # Explicitly absent — never sampled
                # Else: term not annotated for this disease — skip

            # Apply label noise: swap some symptoms with a sibling disease
            if rng.rand() < noise_rate and len(symptoms_present) > 0:
                # Pick a random sibling disease
                siblings = [d for d in ALL_DISEASES if d.name != disease.name]
                sibling = siblings[rng.randint(len(siblings))]

                # Replace 10-30% of patient's symptoms with sibling's symptoms
                n_swap = max(1, int(len(symptoms_present) * rng.uniform(0.10, 0.30)))
                sibling_terms = list(sibling.symptoms.keys())

                if sibling_terms:
                    # Remove some of patient's symptoms
                    if n_swap < len(symptoms_present):
                        remove_idx = rng.choice(
                            len(symptoms_present), size=n_swap, replace=False
                        )
                        symptoms_present = [
                            s for j, s in enumerate(symptoms_present)
                            if j not in remove_idx
                        ]

                    # Add sibling symptoms (sampled by frequency)
                    for _ in range(n_swap):
                        candidate = sibling_terms[rng.randint(len(sibling_terms))]
                        if candidate not in symptoms_present:
                            # Sample with sibling's frequency
                            if rng.rand() < sibling.symptoms.get(candidate, 0.5):
                                symptoms_present.append(candidate)

            hpo_terms_str = "|".join(sorted(symptoms_present))

            patients.append({
                "patient_id": patient_id,
                "true_disease": disease.name,
                "hpo_terms": hpo_terms_str,
            })

    df = pd.DataFrame(patients)
    print(f"  Generated {len(df)} synthetic patients "
          f"({n_per_disease} per disease × {len(ALL_DISEASES)} diseases)")

    # Print per-disease symptom statistics
    for disease in ALL_DISEASES:
        mask = df["true_disease"] == disease.name
        disease_df = df[mask]
        avg_symptoms = disease_df["hpo_terms"].apply(
            lambda x: len(x.split("|")) if x else 0
        ).mean()
        print(f"    {disease.name}: {len(disease_df)} patients, "
              f"avg {avg_symptoms:.1f} symptoms")

    return df


def tag_difficulty(
    df: pd.DataFrame,
    threshold: float = 0.75,
) -> pd.DataFrame:
    """
    Tag each patient as 'hard' or 'easy' based on cosine similarity
    between the patient's symptom vector and disease prototype vectors.

    A case is 'hard' if the nearest non-true disease has cosine similarity
    above threshold with the patient's vector.

    Args:
        df: Patient DataFrame with 'true_disease' and 'hpo_terms' columns
        threshold: Similarity threshold for hard-case tagging

    Returns:
        DataFrame with added 'difficulty' column
    """
    # Build disease prototype vectors (mean symptom frequencies)
    prototypes = {}
    for disease in ALL_DISEASES:
        vec = np.zeros(NUM_FEATURES)
        for term, freq in disease.symptoms.items():
            if term in HPO_TERM_INDEX:
                vec[HPO_TERM_INDEX[term]] = freq
        prototypes[disease.name] = vec

    difficulties = []
    for _, row in df.iterrows():
        # Build patient binary vector
        hpo_terms = row["hpo_terms"].split("|") if row["hpo_terms"] else []
        patient_vec = np.zeros(NUM_FEATURES)
        for term in hpo_terms:
            if term in HPO_TERM_INDEX:
                patient_vec[HPO_TERM_INDEX[term]] = 1.0

        patient_norm = np.linalg.norm(patient_vec)
        if patient_norm == 0:
            difficulties.append("easy")
            continue

        true_disease = row["true_disease"]
        max_other_sim = -1.0

        for disease_name, proto_vec in prototypes.items():
            if disease_name == true_disease:
                continue
            proto_norm = np.linalg.norm(proto_vec)
            if proto_norm > 0:
                sim = np.dot(patient_vec, proto_vec) / (patient_norm * proto_norm)
                max_other_sim = max(max_other_sim, sim)

        difficulties.append("hard" if max_other_sim > threshold else "easy")

    df = df.copy()
    df["difficulty"] = difficulties
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert patient DataFrame to binary feature matrix.

    Args:
        df: DataFrame with 'hpo_terms' (pipe-separated) and 'true_disease' columns

    Returns:
        X: (n_samples, NUM_FEATURES) binary matrix
        y: (n_samples,) integer label array
    """
    n_samples = len(df)
    X = np.zeros((n_samples, NUM_FEATURES))
    y = np.zeros(n_samples, dtype=int)

    for i, (_, row) in enumerate(df.iterrows()):
        # Build feature vector
        hpo_terms = row["hpo_terms"].split("|") if row["hpo_terms"] else []
        for term in hpo_terms:
            if term in HPO_TERM_INDEX:
                X[i, HPO_TERM_INDEX[term]] = 1.0

        # Label
        disease_name = row["true_disease"]
        y[i] = DISEASE_LABEL_MAP.get(disease_name, 0)

    return X, y


if __name__ == "__main__":
    print("QResolve-Dx Synthetic Patient Generator")
    print("=" * 50)

    # Generate patients
    df = generate_synthetic_patients(n_per_disease=200, noise_rate=0.10, seed=42)

    # Tag difficulty
    df = tag_difficulty(df, threshold=0.75)

    # Print statistics
    print(f"\nDifficulty distribution:")
    print(df["difficulty"].value_counts().to_string())

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmark.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Build feature matrix
    X, y = build_feature_matrix(df)
    print(f"\nFeature matrix: {X.shape}")
    print(f"Labels: {y.shape}, unique: {np.unique(y, return_counts=True)}")
