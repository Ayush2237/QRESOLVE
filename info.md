# Developer Guide & Contribution Status

Hey team! 👋 Welcome to the **QResolve-Dx** codebase. 

This document outlines the current state of the repository, how the architecture is laid out, and the priority roadmap. Use this as a guide for what's already built and where you can jump in to contribute.

---

## 🗺️ Codebase Map & What's Built

The core backend and ML models are fully implemented. The architecture is a **two-tier system**:

1. **`models/common/` (Tier 1)**: 
   - We use real-world datasets for this. `breast_cancer.py` uses the real sklearn Wisconsin dataset, and `parkinsons.py` uses the UCI Oxford dataset (with a statistical generator fallback if UCI is down).
   - *Status*: Complete. Uses XGBoost with probability calibration.

2. **`models/quantum/` & `models/classical/` (Tier 2 - Rare Diseases)**: 
   - Focuses on the Marfan connective tissue disorder cluster (Marfan, Loeys-Dietz, Beals, etc.).
   - Since rare disease data is protected/scarce, `data/generate_patients.py` generates synthetic patients by legitimately sampling from real-world **Human Phenotype Ontology (HPO)** frequencies.
   - *Status*: Complete. We have a classical XGBoost triage model, a Confusion Detector (`models/confusion/detector.py`) that flags highly similar diseases, and a Quantum SVM (`zz_kernel.py`) that resolves the hard cases using Qiskit.

3. **`explain/` (Clinical Explainability)**: 
   - *Status*: Complete. Includes SHAP explanations (`shap_explain.py`) and a Bayesian Next-Test recommender (`next_test_recommender.py`) that uses Information Gain to suggest the next clinical test.

4. **`backend/main.py`**:
   - *Status*: Complete. FastAPI server exposing `/diagnose`, `/explain`, and `/benchmark/report`.

---

## 🚀 Where We Need Help (To-Do / Roadmap)

If you're looking to contribute, pick one of these areas:

### 1. Frontend UI (High Priority)
The backend is done, but we need a clinical dashboard.
* **Task**: Build a React.js or Next.js frontend. 
* **Details**: Doctors should be able to select symptoms from a searchable dropdown, view the ranked diagnoses, and see the SHAP explanation graphs visually. Hook it up to the FastAPI endpoints (`http://localhost:8000/diagnose`).

### 2. Real Quantum Hardware Integration
Right now, the Qiskit module (`models/quantum/zz_kernel.py`) runs on the local `Aer` statevector simulator.
* **Task**: Integrate `qiskit-ibm-runtime`.
* **Details**: Allow the system to submit the quantum kernel matrix calculations to real IBM Quantum backends (like IBM Brisbane) using an API token. 

### 3. Expand the Knowledge Graph
We currently focus on 2 common diseases and a cluster of 5 rare diseases.
* **Task**: Scale the database.
* **Details**: Take the full `phenotype.hpoa` dataset and load it into a Neo4j graph database (schema already drafted in `graph/schema.cypher`). Update the pipeline to query Neo4j instead of the in-memory dict.

### 4. Multimodal Data Pipelines
* **Task**: Let's accept more than just tabular/symptom data.
* **Details**: Add deep learning feature extractors (like ResNet) to the common disease pipelines so we can accept raw images (e.g., mammograms for breast cancer) or audio files (for Parkinson's).

---

## 🛠️ Dev Setup Notes

If you are setting up the repo on your machine to start contributing:

1. **Python Version**: Use Python 3.9 - 3.11. 
2. **Qiskit Check**: Qiskit 1.x is strictly required. Do not use legacy Qiskit 0.46, it will break the FidelityQuantumKernel logic.
3. **Mac Users**: If you get an XGBoost `libomp.dylib` error when running the pipeline, you need to install OpenMP via homebrew: 
   ```bash
   brew install libomp
   ```
4. **Running the Pipeline**: You can test your changes by running the master script:
   ```bash
   python qresolve-dx/run_pipeline.py
   ```
