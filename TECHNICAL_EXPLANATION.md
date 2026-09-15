# 🧬 QResolve-Dx: Deep Technical Architecture & Mathematical Rationale

![Architecture](https://img.shields.io/badge/Architecture-Two--Tier-blue)
![Quantum](https://img.shields.io/badge/Quantum-Qiskit_2.x-6929C4)
![ML](https://img.shields.io/badge/Classical_ML-XGBoost-F37626)
![Math](https://img.shields.io/badge/Mathematics-Information_Theory-009688)

This document serves as the **comprehensive engineering and scientific blueprint** for QResolve-Dx. It explains the exact flow of data, the mathematical formulas governing the logic, and the scientific justification for every technology chosen. 

---

## 🏗️ 1. Global Architecture Flow

Medical diagnosis is fundamentally a routing problem. Standard algorithms waste compute power treating every patient equally. QResolve-Dx implements a **Two-Tier Triage Architecture** to optimize computational cost while maximizing diagnostic accuracy.

```mermaid
graph TD
    A[Patient Symptoms] --> B{Disease Category}
    B -->|Common| C[Tier 1: Classical Pipeline]
    B -->|Rare/Complex| D[Tier 2: Rare Pipeline]
    
    C --> C1[Wisconsin / UCI Real Datasets]
    C1 --> C2[XGBoost + Isotonic Calibration]
    C2 --> C3[Final Diagnosis]
    
    D --> D1[HPO-Weighted Feature Extraction]
    D1 --> D2[Classical Triage XGBoost]
    D2 --> D3{Confusion Detector}
    
    D3 -->|Margin > τ (Easy)| D4[Classical Diagnosis]
    D3 -->|Margin < τ (Hard)| D5[Quantum QSVM Resolver]
    
    D5 --> D6[ZZFeatureMap Hilbert Space]
    D6 --> D7[Quantum Diagnosis]
    
    C3 --> E[SHAP Explainability]
    D4 --> E
    D7 --> E
    
    E --> F[Bayesian Next-Test Recommender]
```

---

## 📊 2. Data Strategy & Feasibility

### The Common Disease Pipeline
For common diseases (Breast Cancer, Parkinson's), the system trains on **100% real clinical datasets** (Wisconsin Cytology, Oxford UCI Voice). This proves the classical pipeline works on real-world noise and distributions.

### The Rare Disease Pipeline (Synthetic yet Rigorous)
*Where do we get data for rare diseases when HIPAA protects it and cases are scarce?*
We solve this by generating synthetic patients using **Real Clinical Probabilities** from the Human Phenotype Ontology (HPO).

**The Mathematics of Generation (`generate_patients.py`)**:
For a disease $D$ and a symptom $S$, the HPO database gives us the exact clinical frequency $P(S | D)$. 
To generate a synthetic patient for disease $D$, we run a **Bernoulli Trial** for every known symptom:
$$X_S \sim \text{Bernoulli}(P(S | D))$$
This ensures that while the specific patient rows are synthetic, the **statistical manifold of the dataset is clinically authentic**.

---

## 🌲 3. Classical Triage & Information Theory

Before we use Quantum ML, we push the data through a highly optimized classical pipeline. 

### Information Content (IC) & Lin Similarity (`features.py`)
Counting how many symptoms two patients share is medically naive. A shared "Headache" is meaningless; a shared "Ectopia Lentis" (dislocated lens) is highly diagnostic.

We use **Shannon Information Theory**:
$$IC(S) = -\log_2(P(S))$$
Rare symptoms have high IC; common symptoms have low IC. 

To compare symptoms, we use **Lin's Semantic Similarity**:
$$sim(t_1, t_2) = \frac{2 \cdot IC(MICA)}{IC(t_1) + IC(t_2)}$$
Where **MICA** (Most Informative Common Ancestor) is found by traversing the genetic/phenotypic Knowledge Graph. This ensures the XGBoost model understands the genetic weight of symptoms before making a prediction.

### Isotonic Probability Calibration (`breast_cancer.py`, `parkinsons.py`)
Raw XGBoost outputs a "score" between 0 and 1, but it is **not a true probability**. We apply **Isotonic Regression Calibration** to mathematically force the scores into true probabilities. If our calibrated model says `0.85`, it means exactly 85% of patients with that score have the disease. 

### The Confusion Detector (`run_pipeline.py`)
To save quantum compute costs, we calculate the classification margin:
$$\text{Margin} = P(\text{Top Disease}) - P(\text{Runner-Up Disease})$$
If $\text{Margin} < \tau$ (e.g., 0.03), the classical model is "confused". Only these computationally "hard" cases are passed to the Quantum Resolver.

---

## ⚛️ 4. The Quantum Resolver (The Core Innovation)

### Why Classical Fails
In connective tissue disorders (like Marfan vs. Loeys-Dietz), symptoms are highly entangled. The classical RBF (Radial Basis Function) kernel maps data into a continuous space, but often cannot find a hyperplane to separate these entangled phenotypic profiles.

### The Quantum Solution (`zz_kernel.py`)
We implement a **Quantum Support Vector Machine (QSVM)** using the `ZZFeatureMap` (Havlíček et al., 2019, *Nature*). 

The `ZZFeatureMap` encodes classical symptom data $x$ into a quantum state $|\Phi(x)\rangle$:
$$|\Phi(x)\rangle = U_{\Phi(x)}|0\rangle^{\otimes n}$$
Where the unitary operator applies Hadamard gates (superposition) and $ZZ$ rotation gates (entanglement). The $ZZ$ gates specifically map 2nd-order non-linear correlations between symptoms into a high-dimensional **Hilbert Space**.

We then calculate the **Quantum Kernel Matrix**:
$$K(x,y) = |\langle\Phi(y)|\Phi(x)\rangle|^2$$
Using Qiskit's `StatevectorSampler`, we compute the exact fidelity (distance) between patients in this quantum space. Because the space is exponentially large and inherently non-linear, the SVM can easily draw a hyperplane to separate the previously "confused" diseases.

**Mathematical Verification**: The code automatically calculates the eigenvalues of the resulting kernel matrix to prove it satisfies **Mercer's Conditions** (Positive Semi-Definite), proving the math is sound.

---

## 🧠 5. Clinical Trust & Explainability

Doctors cannot act on "Black Box" predictions. QResolve-Dx provides two layers of deep explainability.

### Layer 1: SHAP (Game Theory) (`shap_explain.py`)
We use **Shapley Additive exPlanations (SHAP)** to explain the XGBoost predictions. Rooted in cooperative game theory, SHAP calculates the exact marginal contribution of each symptom to the final diagnosis. 
* *Example Output*: "Aortic Root Dilatation pushed the diagnosis towards Marfan by +0.42 points, but the lack of Bifid Uvula pushed it away from Loeys-Dietz by -0.15 points."

### Layer 2: Bayesian Next-Test Recommender (`next_test_recommender.py`)
If the model is unsure (e.g., Marfan 51% vs MASS Phenotype 49%), what should the doctor do next? Instead of guessing, we use **Information Gain**.

For every untested symptom, we calculate the expected reduction in Shannon Entropy:
$$IG(T) = H(\text{Prior}) - \mathbb{E}[H(\text{Posterior} | T)]$$
The system recommends the exact clinical test (e.g., "Lumbosacral MRI for Dural Ectasia") that maximizes Information Gain. It explicitly compares the prior to the posterior to tell the doctor: *"If this MRI is positive, Marfan becomes more likely. If negative, MASS phenotype becomes more likely."*

---

## 🗂️ 6. Codebase File Mapping

If you need to find the exact implementation of the concepts above, reference this table:

| Concept / Technology | Implementation File | Key Function / Class |
|----------------------|---------------------|----------------------|
| **Synthetic HPO Generation** | `data/generate_patients.py` | `generate_synthetic_patients()` |
| **Lin Similarity / MICA** | `models/classical/features.py` | `compute_lin_similarity()` |
| **XGBoost & Calibration** | `models/common/breast_cancer.py` | `CalibratedClassifierCV(base_xgb)` |
| **Confusion Margin Logic** | `run_pipeline.py` (Phase 3) | `df['margin'] < margin_tau` |
| **Quantum ZZFeatureMap** | `models/quantum/zz_kernel.py` | `ZZFeatureMap`, `StatevectorSampler` |
| **Mercer's Condition Check** | `models/quantum/zz_kernel.py` | `validate_kernel_matrix(K)` |
| **SHAP Explanations** | `explain/shap_explain.py` | `shap.TreeExplainer()` |
| **Bayesian Information Gain**| `explain/next_test_recommender.py`| `compute_information_gain()` |

---
*Built for the Smart India Hackathon. Bridging the gap between classical efficiency and quantum precision in healthcare.*
