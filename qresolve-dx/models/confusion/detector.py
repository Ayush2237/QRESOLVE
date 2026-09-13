"""
QResolve-Dx: Confusion Detector

Determines whether a case is "hard" (i.e., the top two candidate diagnoses are
dangerously close) and should be escalated to the quantum kernel resolver.

The detector uses three criteria (any triggers escalation):
  1. Margin check:  probability gap between top-2 candidates < margin_tau
  2. Entropy check: entropy of probability distribution > entropy_tau
  3. Known pairs:   the top-2 candidates form a clinically known confusion pair

Thresholds are tuned so that ~15-30% of cases are flagged hard — too few and
the quantum layer never fires; too many and it's indistinguishable from
"always run quantum."
"""

from __future__ import annotations

import math
import sys
import os
from typing import List, Tuple, Dict, FrozenSet, Optional
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.disease_data import (
    DISEASE_NAMES, KNOWN_CONFUSION_PAIRS, PAIRWISE_DISTINGUISHING
)


@dataclass
class ConfusionResult:
    """Result of the confusion detection analysis."""
    is_hard: bool
    top1_disease: str
    top1_prob: float
    top2_disease: str
    top2_prob: float
    margin: float
    entropy: float
    is_known_pair: bool
    trigger_reason: str  # "margin", "entropy", "known_pair", "none"


def compute_entropy(probs: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.

    H(P) = -Σ p_i * log(p_i)

    Uses natural log. Adds small epsilon to avoid log(0).
    Maximum entropy for 5 classes = ln(5) ≈ 1.609
    """
    probs = np.asarray(probs, dtype=np.float64)
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-9, 1.0)
    # Normalize in case not exactly summing to 1
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)))


def is_hard_case(
    probs: np.ndarray,
    disease_names: List[str],
    known_pairs: List[FrozenSet[str]],
    margin_tau: float = 0.10,
    entropy_tau: float = 1.2,
) -> ConfusionResult:
    """
    Determine if a diagnostic case is "hard" and needs quantum re-scoring.

    This implements the exact algorithm from the README §4.3:
      - margin < margin_tau → top two candidates too close
      - entropy > entropy_tau → probability mass too spread out
      - pair in known_pairs → clinically known confusion pair

    Args:
        probs: Array of calibrated probabilities, one per disease
        disease_names: Corresponding disease names
        known_pairs: Set of frozensets of known confusion pairs
        margin_tau: Minimum probability gap between top-2 (default 0.10)
        entropy_tau: Maximum acceptable entropy (default 1.2)

    Returns:
        ConfusionResult with detailed analysis
    """
    probs = np.asarray(probs, dtype=np.float64)
    assert len(probs) == len(disease_names), \
        f"probs length {len(probs)} != disease_names length {len(disease_names)}"

    # Sort by probability descending
    sorted_indices = np.argsort(probs)[::-1]
    top1_idx = sorted_indices[0]
    top2_idx = sorted_indices[1]

    top1_disease = disease_names[top1_idx]
    top1_prob = float(probs[top1_idx])
    top2_disease = disease_names[top2_idx]
    top2_prob = float(probs[top2_idx])

    # Criterion 1: Margin between top-2
    margin = top1_prob - top2_prob

    # Criterion 2: Entropy of distribution
    entropy = compute_entropy(probs)

    # Criterion 3: Known confusion pair
    pair = frozenset([top1_disease, top2_disease])
    is_known = pair in known_pairs

    # Determine trigger
    trigger = "none"
    is_hard = False

    if margin < margin_tau:
        is_hard = True
        trigger = "margin"
    elif entropy > entropy_tau:
        is_hard = True
        trigger = "entropy"
    elif is_known:
        is_hard = True
        trigger = "known_pair"

    return ConfusionResult(
        is_hard=is_hard,
        top1_disease=top1_disease,
        top1_prob=top1_prob,
        top2_disease=top2_disease,
        top2_prob=top2_prob,
        margin=margin,
        entropy=entropy,
        is_known_pair=is_known,
        trigger_reason=trigger,
    )


def tune_thresholds(
    all_probs: np.ndarray,
    disease_names: List[str],
    known_pairs: List[FrozenSet[str]],
    target_hard_rate: float = 0.20,
    target_range: Tuple[float, float] = (0.15, 0.30),
) -> Tuple[float, float]:
    """
    Tune margin_tau and entropy_tau to achieve target hard-case rate.

    Searches over a grid of (margin_tau, entropy_tau) values and selects
    the combination that produces a hard-case rate closest to target_hard_rate
    while staying within target_range.

    Args:
        all_probs: (n_samples, n_classes) probability matrix
        disease_names: List of disease names
        known_pairs: Known confusion pairs
        target_hard_rate: Desired fraction of cases flagged hard
        target_range: Acceptable range for hard-case rate

    Returns:
        (best_margin_tau, best_entropy_tau)
    """
    best_margin = 0.10
    best_entropy = 1.2
    best_diff = float('inf')

    max_entropy = math.log(len(disease_names))  # ln(5) ≈ 1.609

    margin_candidates = np.arange(0.03, 0.25, 0.01)
    entropy_candidates = np.arange(0.8, max_entropy, 0.05)

    n_samples = len(all_probs)

    for m_tau in margin_candidates:
        for e_tau in entropy_candidates:
            n_hard = 0
            for i in range(n_samples):
                result = is_hard_case(
                    all_probs[i], disease_names, known_pairs,
                    margin_tau=m_tau, entropy_tau=e_tau
                )
                if result.is_hard:
                    n_hard += 1

            hard_rate = n_hard / n_samples
            diff = abs(hard_rate - target_hard_rate)

            if (target_range[0] <= hard_rate <= target_range[1]) and diff < best_diff:
                best_diff = diff
                best_margin = float(m_tau)
                best_entropy = float(e_tau)

    return best_margin, best_entropy


def analyze_confusion_distribution(
    all_probs: np.ndarray,
    y_true: np.ndarray,
    disease_names: List[str],
    known_pairs: List[FrozenSet[str]],
    margin_tau: float = 0.10,
    entropy_tau: float = 1.2,
) -> Dict:
    """
    Analyze the distribution of hard vs easy cases across the dataset.

    Returns detailed statistics for reporting and threshold tuning.
    """
    n_samples = len(all_probs)
    results = []

    for i in range(n_samples):
        result = is_hard_case(
            all_probs[i], disease_names, known_pairs,
            margin_tau=margin_tau, entropy_tau=entropy_tau
        )
        results.append(result)

    n_hard = sum(1 for r in results if r.is_hard)
    hard_rate = n_hard / n_samples

    # Trigger reason breakdown
    trigger_counts = {}
    for r in results:
        trigger_counts[r.trigger_reason] = trigger_counts.get(r.trigger_reason, 0) + 1

    # Most common confusion pairs
    pair_counts = {}
    for r in results:
        if r.is_hard:
            pair = frozenset([r.top1_disease, r.top2_disease])
            pair_key = f"{r.top1_disease} ↔ {r.top2_disease}"
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

    # Hard-case accuracy vs easy-case accuracy
    hard_correct = 0
    hard_total = 0
    easy_correct = 0
    easy_total = 0

    from data.disease_data import DISEASE_LABEL_MAP

    for i, r in enumerate(results):
        true_label = int(y_true[i])
        pred_label = DISEASE_LABEL_MAP.get(r.top1_disease, -1)

        if r.is_hard:
            hard_total += 1
            if pred_label == true_label:
                hard_correct += 1
        else:
            easy_total += 1
            if pred_label == true_label:
                easy_correct += 1

    analysis = {
        "n_total": n_samples,
        "n_hard": n_hard,
        "n_easy": n_samples - n_hard,
        "hard_rate": hard_rate,
        "margin_tau": margin_tau,
        "entropy_tau": entropy_tau,
        "trigger_breakdown": trigger_counts,
        "top_confusion_pairs": dict(sorted(pair_counts.items(), key=lambda x: -x[1])[:10]),
        "hard_case_accuracy": hard_correct / max(hard_total, 1),
        "easy_case_accuracy": easy_correct / max(easy_total, 1),
    }

    return analysis


def print_confusion_analysis(analysis: Dict) -> None:
    """Pretty-print confusion detection analysis."""
    print(f"\n{'='*60}")
    print(f"Confusion Detection Analysis")
    print(f"{'='*60}")
    print(f"  Total cases:     {analysis['n_total']}")
    print(f"  Hard cases:      {analysis['n_hard']} ({analysis['hard_rate']:.1%})")
    print(f"  Easy cases:      {analysis['n_easy']} ({1-analysis['hard_rate']:.1%})")
    print(f"  Margin τ:        {analysis['margin_tau']:.3f}")
    print(f"  Entropy τ:       {analysis['entropy_tau']:.3f}")
    print(f"\n  Trigger breakdown:")
    for trigger, count in analysis['trigger_breakdown'].items():
        print(f"    {trigger:>12s}: {count}")
    print(f"\n  Hard-case accuracy (before quantum): {analysis['hard_case_accuracy']:.4f}")
    print(f"  Easy-case accuracy:                  {analysis['easy_case_accuracy']:.4f}")
    if analysis['top_confusion_pairs']:
        print(f"\n  Most common confusion pairs:")
        for pair, count in list(analysis['top_confusion_pairs'].items())[:5]:
            print(f"    {pair}: {count} cases")


if __name__ == "__main__":
    print("Confusion Detector Module")
    print("=" * 60)
    print("Implements margin + entropy + known-pair detection.")
    print(f"Known confusion pairs: {len(KNOWN_CONFUSION_PAIRS)}")
    for pair in KNOWN_CONFUSION_PAIRS:
        diseases = sorted(pair)
        print(f"  • {diseases[0]} ↔ {diseases[1]}")
    print(f"\nMax entropy for {len(DISEASE_NAMES)} classes: {math.log(len(DISEASE_NAMES)):.4f}")
