# Project Context & Status: QResolve-Dx

## 📝 Background Context
This project was developed as a submission for the **Smart India Hackathon (SIH) 2026**. The core problem it solves is the **diagnostic odyssey** faced by patients with rare diseases. Often, rare diseases share highly overlapping phenotypes (symptoms) with other rare diseases, leading to misdiagnoses. 

To prove the technical viability of the solution, the system is designed to handle both **Common Diseases** (using real-world datasets) to act as a primary screener, and **Rare Diseases** (using a quantum-classical hybrid model) to act as an advanced specialist tool.

---

## ✅ Work Completed

The core backend, machine learning models, and API have been successfully implemented:

1. **Two-Tier ML Architecture**:
   - **Tier 1 (Common)**: Fully functional XGBoost classifiers for Breast Cancer (Wisconsin dataset) and Parkinson's Disease (Oxford dataset). Includes stratified K-Fold cross-validation and probability calibration.
   - **Tier 2 (Rare)**: Focused on the Marfan-related connective tissue disorder cluster (Marfan, Loeys-Dietz, Beals, etc.).
2. **Data Generation Engine**: 
   - Since rare disease data is protected and scarce, a statistically rigorous synthetic data generator was built. It samples symptoms directly from the real-world **Human Phenotype Ontology (HPO)** frequency annotations, injecting realistic clinical noise.
3. **Confusion Detector**:
   - An entropy and margin-based algorithm that successfully identifies "hard cases" (~20% of cases) where classical ML models struggle to differentiate between two similar diseases.
4. **Quantum Machine Learning (QML) Module**:
   - Implemented using Qiskit 1.x.
   - Uses Mutual Information for feature reduction (down to 8 qubits).
   - Utilizes `ZZFeatureMap` for quantum state encoding and a `FidelityQuantumKernel` to train a Quantum SVM on the hard cases.
5. **Explainable AI (XAI) & Clinical Support**:
   - **SHAP Integration**: Extracts which symptoms drove the model's decision.
   - **Bayesian Next-Test Recommender**: Computes the Information Gain of unseen symptoms to recommend the optimal next clinical test (e.g., recommending an echocardiogram to check for aortic root dilation).
6. **FastAPI Backend**:
   - REST API ready to be consumed by a frontend interface.

---

## 🚀 Future Work & Roadmap

To take this project to the next level (post-hackathon or for final presentation), the following steps are required:

### 1. Hardware Integration (Real Quantum Execution)
Currently, the Qiskit module runs on the `Aer` statevector simulator (locally). 
* **Next Step**: Integrate with `qiskit-ibm-runtime` to submit the quantum kernel matrix calculations to a real IBM Quantum backend (e.g., IBM Brisbane or IBM Kyoto).

### 2. Frontend User Interface
The backend is complete, but it needs a clinical dashboard.
* **Next Step**: Build a React.js or Next.js frontend that allows doctors to select symptoms from a searchable dropdown, view the ranked diagnoses, and see the SHAP explanation graphs visually.

### 3. Expansion of the Knowledge Graph
The current system focuses on 2 common diseases and 5 rare diseases.
* **Next Step**: Ingest the full `phenotype.hpoa` database into a Neo4j graph database to allow the classical model to triage against *thousands* of rare diseases, only routing the confused subsets to the quantum resolver.

### 4. Integration of Multimodal Data
* **Next Step**: Allow the breast cancer and Parkinson's pipelines to accept raw images (mammograms) or raw audio (voice recordings) using Deep Learning feature extractors (like ResNet) before passing the embeddings to the classical classifiers.

---

## 🛠️ Required Information / Context for Developers
* **Python Environment**: Ensure you are using Python 3.9 - 3.11. Qiskit 1.x is strictly required (legacy Qiskit 0.46 will break the kernel logic).
* **Mac Users**: If running the pipeline throws an XGBoost `libomp.dylib` error, you must install OpenMP via homebrew: `brew install libomp`.
* **Data Sources**:
  * Breast Cancer: `sklearn.datasets.load_breast_cancer()`
  * Parkinson's: UCI ML Repository (fallback generator included in `parkinsons.py`).
  * Rare Diseases: HPO Frequencies embedded in `data/disease_data.py`.
