# 🧬 QResolve-Dx: Quantum-Enhanced Differential Diagnosis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-6929C4)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-F37626)
![License](https://img.shields.io/badge/License-MIT-green)

**QResolve-Dx** is an advanced, two-tier clinical decision support system designed to assist healthcare professionals in diagnosing both common and extremely rare diseases. It bridges classical machine learning with quantum computing to resolve the hardest differential diagnoses.

---

## 🏗️ Two-Tier Architecture

The system routes patients through a sophisticated two-tier pipeline depending on their symptoms:

### 🟢 Tier 1: Common Diseases (Classical ML)
For common conditions, the system utilizes highly optimized, probability-calibrated **XGBoost** models trained on **real, publicly available clinical datasets**:
* **Breast Cancer**: Powered by the Wisconsin Breast Cancer Dataset (569 patients, 30 cytological features).
* **Parkinson's Disease**: Powered by the Oxford Parkinson's Disease Detection Dataset (195 patients, 22 biomedical voice features).

### 🟣 Tier 2: Rare Connective Tissue Disorders (Quantum ML)
For extremely rare and highly confusable diseases (e.g., Marfan syndrome, Loeys-Dietz syndrome, Beals syndrome), the system uses a hybrid approach:
1. **Classical Triage**: An initial XGBoost model scores the patient based on Human Phenotype Ontology (HPO) terms, weighted by Information Content (IC).
2. **Confusion Detection**: An entropy-based detector flags "hard cases" where the top two diseases are clinically indistinguishable (e.g., Marfan vs. MASS phenotype).
3. **Quantum Resolver**: Hard cases are routed to a **Quantum Support Vector Machine (QSVM)** utilizing a `ZZFeatureMap` to capture complex, non-linear symptom correlations in a high-dimensional Hilbert space that classical models struggle to separate.

---

## 🧠 Explainability & Clinical Support

AI in healthcare requires transparency. QResolve-Dx includes a robust explainability module:
* **SHAP Value Explanations**: Breaks down exactly which symptoms supported the top diagnosis and which symptoms argued *against* the runner-up.
* **Bayesian Next-Test Recommender**: Calculates the **Information Gain** of untested clinical features and recommends the next best lab test or physical exam to perform in order to differentiate the top two diseases.

---

## 📊 Data Disclosure

* **Common Diseases**: Trained on 100% real clinical data.
* **Rare Diseases**: Due to the global scarcity of rare disease data, the Marfan-cluster models are trained on **principled synthetic data**. Patients are generated using precise Bernoulli sampling from real published HPO annotation-frequency distributions (sourced from `phenotype.hpoa`).

---

## 🚀 Quickstart

### Prerequisites
Ensure you have Python 3.9+ installed. For the quantum module, Qiskit is required.

```bash
# Clone the repository
git clone <your-repo-link>
cd qresolve-dx

# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline End-to-End
To generate the datasets, train the classical models, run the confusion detector, execute the quantum benchmark, and generate the final report:

```bash
python run_pipeline.py
```
*Results and the Markdown benchmark report will be saved in the `benchmarks/` and `data/processed/` directories.*

### Starting the API Server
Launch the FastAPI backend to interact with the models:

```bash
cd backend
python main.py
```
*The API will be available at `http://localhost:8000`. You can view the interactive Swagger UI at `http://localhost:8000/docs`.*

### Starting the Frontend UI
Launch the Vite React frontend to interact with the diagnostic pipeline visually:

```bash
cd frontend
npm install
npm run dev
```
*The application UI will be available at `http://localhost:5173`.*

---

## 🌐 API Endpoints

* `POST /diagnose`: Submit patient symptoms (HPO terms or text) and receive ranked differential diagnoses. Automatically routes to the quantum resolver if the case is flagged as "hard".
* `GET /explain/{case_id}`: Retrieve SHAP evidence and Information-Gain-based next-test recommendations for a specific diagnosis.
* `GET /benchmark/report`: Fetch the latest statistical benchmark comparing the Quantum vs. Classical models.
* `GET /diseases`: Retrieve the catalog of common and rare diseases available in the system.
* `POST /nlp/extract`: Extract Human Phenotype Ontology (HPO) terms from unstructured clinical text.
* `POST /graph`: Generate D3-compatible Knowledge Graph data linking patient symptoms to matching diseases.
* `GET /diseases/{disease_type}/features`: Fetch the required clinical feature metadata and schema for common disease ML inputs.

## 📚 Documentation
All detailed documentation is located in the `docs/` folder:
- **[Technical Architecture](docs/TECHNICAL_EXPLANATION.md):** Deep-dive into the mathematics, Information Theory, and QSVM implementation.
- **[Pitch Deck Data](docs/PITCH_DECK_CONTENT.md):** Cloud compute economics, scale strategies, and the 13TB dataset handling architecture.
- **[Prototype Roadmap](docs/PROTOTYPE_ROADMAP.md):** Roadmap for FHIR/HL7 integration, PDF clinical reports, and medical imaging CV.
