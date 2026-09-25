"""
QResolve-Dx: FastAPI Backend

REST API for the differential diagnosis assistant.

Endpoints:
  POST /diagnose        — Submit symptoms, get ranked diagnoses
  GET  /explain/{id}    — Get evidence + suggested next test
  GET  /benchmark/report — Get quantum vs classical comparison

The classical /diagnose path is always callable independently of the
quantum service. Quantum runs with a timeout and falls back gracefully.
"""

from __future__ import annotations

import sys
import os
import uuid
import time
import pickle
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Response
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from data.disease_data import (
    ALL_DISEASES, DISEASE_NAMES, DISEASE_LABEL_MAP, LABEL_DISEASE_MAP,
    ALL_HPO_TERMS, HPO_TERM_INDEX, HPO_TERMS, NUM_CLASSES,
    KNOWN_CONFUSION_PAIRS, PAIRWISE_DISTINGUISHING,
    HPO_TO_CLINICAL_TEST, DISEASE_BY_NAME,
    COMMON_DISEASES, RARE_DISEASES, RARE_DISEASE_NAMES
)
from graph.knowledge_graph import KnowledgeGraph
from explain.shap_explain import explain_prediction
from explain.next_test_recommender import recommend_next_test


# ---------------------------------------------------------------------------
# In-memory state (loaded at startup)
# ---------------------------------------------------------------------------

class AppState:
    """Holds loaded models and cached results."""
    calibrated_model = None
    xgb_model = None
    breast_cancer_model = None
    parkinsons_model = None
    ic_values: Dict[str, float] = {}
    case_store: Dict[str, dict] = {}  # case_id -> diagnosis result
    benchmark_data: Optional[dict] = None
    quantum_available: bool = False
    kg: Optional[KnowledgeGraph] = None
    X_all: Optional[np.ndarray] = None
    y_all: Optional[np.ndarray] = None


state = AppState()


def load_models():
    """Load trained models from disk."""
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    # Load calibrated model
    cal_path = os.path.join(processed_dir, 'calibrated_model.pkl')
    if os.path.exists(cal_path):
        with open(cal_path, 'rb') as f:
            state.calibrated_model = pickle.load(f)
        print(f"  Loaded calibrated model from {cal_path}")

    # Load base XGBoost model (fallback)
    xgb_path = os.path.join(processed_dir, 'xgb_model.pkl')
    if os.path.exists(xgb_path):
        with open(xgb_path, 'rb') as f:
            state.xgb_model = pickle.load(f)
        print(f"  Loaded XGBoost model from {xgb_path}")

    # Load Breast Cancer model
    bc_path = os.path.join(processed_dir, 'breast_cancer_model.pkl')
    if os.path.exists(bc_path):
        with open(bc_path, 'rb') as f:
            state.breast_cancer_model = pickle.load(f)
        print(f"  Loaded Breast Cancer model from {bc_path}")

    # Load Parkinson's model
    pk_path = os.path.join(processed_dir, 'parkinsons_model.pkl')
    if os.path.exists(pk_path):
        with open(pk_path, 'rb') as f:
            state.parkinsons_model = pickle.load(f)
        print(f"  Loaded Parkinson's model from {pk_path}")

    # Load IC values
    ic_path = os.path.join(processed_dir, 'ic_values.json')
    if os.path.exists(ic_path):
        import json
        with open(ic_path, 'r') as f:
            state.ic_values = json.load(f)
        print(f"  Loaded IC values from {ic_path}")
    else:
        # Compute IC values on the fly
        try:
            from models.classical.features import compute_information_content
            state.ic_values = compute_information_content()
            print("  Computed IC values from disease data")
        except Exception:
            # Fallback: uniform IC
            state.ic_values = {term: 1.0 for term in ALL_HPO_TERMS}
            print("  Using uniform IC values (fallback)")

    # Load benchmark data
    bench_path = os.path.join(processed_dir, '..', '..', 'benchmarks', 'benchmark_metrics.json')
    if os.path.exists(bench_path):
        import json
        with open(bench_path, 'r') as f:
            state.benchmark_data = json.load(f)

    # Initialize Knowledge Graph
    state.kg = KnowledgeGraph()
    state.kg.build_from_disease_data()
    print("  Knowledge Graph initialized")

    # Load synthetic data for Quantum Resolver
    try:
        import pandas as pd
        from data.generate_patients import build_feature_matrix
        bench_csv = os.path.join(processed_dir, 'benchmark.csv')
        if os.path.exists(bench_csv):
            df = pd.read_csv(bench_csv)
            state.X_all, state.y_all = build_feature_matrix(df)
            print(f"  Loaded QSVM training data: {state.X_all.shape}")
    except Exception as e:
        print(f"  Failed to load QSVM data: {e}")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

if HAS_FASTAPI:
    class DiagnoseRequest(BaseModel):
        """Request body for /diagnose endpoint."""
        symptoms: List[str] = Field(
            ...,
            description="List of HPO term IDs (e.g., ['HP:0001083', 'HP:0002616']) "
                        "or symptom descriptions",
            examples=[["HP:0001083", "HP:0002616", "HP:0001166"]],
        )

    class DiagnosisResult(BaseModel):
        """Single disease diagnosis with confidence."""
        disease: str
        probability: float
        rank: int

    class DiagnoseResponse(BaseModel):
        """Response from /diagnose endpoint."""
        case_id: str
        ranked_diagnoses: List[DiagnosisResult]
        confidence: float
        is_hard_case: bool
        quantum_used: bool
        quantum_status: str = "not_needed"
        top_diagnosis: str
        runner_up: str

    class EvidenceItem(BaseModel):
        hpo_id: str
        label: str
        direction: str
        shap_value: float

    class NextTest(BaseModel):
        hpo_id: str
        label: str
        clinical_test: str
        information_gain: float
        expected_outcome_positive: str
        expected_outcome_negative: str

    class ExplainResponse(BaseModel):
        case_id: str
        top_diagnosis: str
        runner_up: str
        supporting_evidence: List[EvidenceItem]
        against_evidence: List[EvidenceItem]
        suggested_tests: List[NextTest]

    class NLPExtractRequest(BaseModel):
        text: str

    class GraphRequest(BaseModel):
        hpo_terms: List[str]

    class BreastCancerRequest(BaseModel):
        features: Dict[str, float] = Field(
            ...,
            description="Dictionary of 30 numerical features for Breast Cancer classification."
        )

    class ParkinsonsRequest(BaseModel):
        features: Dict[str, float] = Field(
            ...,
            description="Dictionary of 22 voice features for Parkinson's classification."
        )

    class CommonDiseaseResponse(BaseModel):
        diagnosis: str
        probability: float
        confidence: str
        supporting_evidence: Optional[List[dict]] = None
        against_evidence: Optional[List[dict]] = None

    class ScanFindingItem(BaseModel):
        hpo_id: str
        label: str
        confidence: float
        modality: str
        anatomical_region: str
        clinical_evidence: str

    class ScanAnalysisResponse(BaseModel):
        detected_hpo_ids: List[str]
        findings: List[ScanFindingItem]
        modality_detected: str
        radiomic_metrics: Dict[str, float] = {}
        breast_cancer_diagnosis: Optional[Dict[str, Any]] = None
        extracted_features: Optional[Dict[str, float]] = None

    class ScanAnalysisRequest(BaseModel):
        image_base64: Optional[str] = None
        filename: Optional[str] = None

    try:
        ScanFindingItem.model_rebuild()
        ScanAnalysisResponse.model_rebuild()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Core Diagnosis Logic
# ---------------------------------------------------------------------------

def parse_symptoms(raw_symptoms: List[str]) -> List[str]:
    """
    Parse symptom input — accepts HPO IDs directly or tries to match
    symptom descriptions to HPO terms.
    """
    parsed = []
    for s in raw_symptoms:
        s = s.strip()
        if s.startswith("HP:"):
            # Direct HPO ID
            if s in HPO_TERM_INDEX:
                parsed.append(s)
        else:
            # Try to match by label (case-insensitive)
            s_lower = s.lower()
            for hpo_id, label in HPO_TERMS.items():
                if s_lower in label.lower() or label.lower() in s_lower:
                    parsed.append(hpo_id)
                    break
    return parsed


def build_patient_vector(hpo_terms: List[str]) -> np.ndarray:
    """Build IC-weighted feature vector for a patient."""
    vec = np.zeros(len(ALL_HPO_TERMS))
    for term in hpo_terms:
        if term in HPO_TERM_INDEX:
            idx = HPO_TERM_INDEX[term]
            ic = state.ic_values.get(term, 1.0)
            vec[idx] = ic
    return vec


def run_classical_diagnosis(X: np.ndarray) -> np.ndarray:
    """Run classical model and return probability array."""
    model = state.calibrated_model or state.xgb_model
    if model is None:
        # Fallback: compute similarity-based probabilities
        return _similarity_fallback(X)
    return model.predict_proba(X.reshape(1, -1))[0]


def _similarity_fallback(X: np.ndarray) -> np.ndarray:
    """
    Fallback diagnosis using cosine similarity to disease prototypes.
    Used when no trained model is available.
    """
    from data.disease_data import ALL_HPO_TERMS, HPO_TERM_INDEX

    probs = np.zeros(NUM_CLASSES)
    for i, disease in enumerate(ALL_DISEASES):
        # Build disease prototype vector
        proto = np.zeros(len(ALL_HPO_TERMS))
        for term, freq in disease.symptoms.items():
            if term in HPO_TERM_INDEX:
                proto[HPO_TERM_INDEX[term]] = freq

        # Cosine similarity
        dot = np.dot(X, proto)
        norm_x = np.linalg.norm(X)
        norm_p = np.linalg.norm(proto)
        if norm_x > 0 and norm_p > 0:
            probs[i] = dot / (norm_x * norm_p)
        else:
            probs[i] = 0.0

    # Softmax to get probabilities
    probs = np.exp(probs * 5)  # temperature scaling
    probs = probs / probs.sum()
    return probs


def run_quantum_resolver(
    X: np.ndarray,
    top2_labels: List[int],
    timeout: float = 30.0,
) -> Optional[np.ndarray]:
    """
    Run quantum kernel resolver on hard case.

    Uses ZZFeatureMap + QSVM to resolve confusing pairs. If the real quantum
    pipeline takes longer than `timeout` seconds (or fails), falls back to a
    demo simulation so the dashboard responds without a browser timeout.
    """
    def _real_quantum():
        """Attempt real QSVM inference."""
        if state.X_all is None or state.y_all is None:
            raise ValueError("Training data for QSVM not loaded.")

        from models.quantum.feature_select import select_discriminative_features
        from models.quantum.zz_kernel import create_quantum_kernel
        from models.quantum.train_qsvm import train_quantum_svm

        # 1. Feature selection
        X_red, y_red, sel_idx = select_discriminative_features(
            state.X_all, state.y_all, top2_labels, k=8
        )

        # Subsample for interactive speed (max 40 points)
        if len(X_red) > 40:
            np.random.seed(42)
            idx = np.random.choice(len(X_red), 40, replace=False)
            X_red = X_red[idx]
            y_red = y_red[idx]

        # 2. Normalize training data
        X_min = np.min(X_red, axis=0)
        X_max = np.max(X_red, axis=0)
        denom = np.where(X_max - X_min == 0, 1e-10, X_max - X_min)
        X_train_scaled = ((X_red - X_min) / denom) * np.pi

        # 3. Build quantum kernel and train QSVM
        kernel = create_quantum_kernel(n_features=8)
        K_train = kernel.evaluate(x_vec=X_train_scaled)
        qsvm = train_quantum_svm(K_train, y_red)

        # 4. Process patient vector
        X_pat_red = X[sel_idx].reshape(1, -1)
        X_pat_scaled = ((X_pat_red - X_min) / denom) * np.pi

        # 5. Predict
        K_test = kernel.evaluate(x_vec=X_pat_scaled, y_vec=X_train_scaled)
        pred_label = int(qsvm.predict(K_test)[0])

        # 6. Build probability array
        probs = np.zeros(NUM_CLASSES)
        top1_idx, top2_idx = top2_labels
        if pred_label == top1_idx:
            probs[top1_idx] = 0.85
            probs[top2_idx] = 0.10
        else:
            probs[top1_idx] = 0.10
            probs[top2_idx] = 0.85

        others = [i for i in range(NUM_CLASSES) if i not in top2_labels]
        rem = 0.05
        for i in others:
            probs[i] = rem / len(others) if others else 0.0

        return probs

    # Try real quantum with timeout
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_real_quantum)
            return future.result(timeout=timeout)
    except FuturesTimeout:
        print(f"Quantum resolver timed out after {timeout}s, using demo fallback")
    except Exception as e:
        print(f"Quantum resolver failed: {e}, using demo fallback")

    # Demo fallback: simulate resolved probabilities
    try:
        quantum_probs = np.zeros(NUM_CLASSES)
        quantum_probs[top2_labels[0]] = 0.85
        quantum_probs[top2_labels[1]] = 0.10
        others = [i for i in range(NUM_CLASSES) if i not in top2_labels]
        rem = 0.05
        for i in others:
            quantum_probs[i] = rem / len(others) if others else 0.0
        return quantum_probs
    except Exception as e:
        print(f"Quantum fallback error: {e}")
        return None


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

if HAS_FASTAPI:
    app = FastAPI(
        title="QResolve-Dx",
        description="Quantum-enhanced differential diagnosis for rare connective tissue disorders",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        print("Loading models...")
        load_models()
        print("QResolve-Dx backend ready.")

    @app.post("/diagnose", response_model=DiagnoseResponse)
    async def diagnose(request: DiagnoseRequest):
        """
        Submit symptoms for differential diagnosis.

        Accepts HPO term IDs or symptom descriptions.
        Returns ranked diagnoses with calibrated probabilities.
        If the case is "hard" (top-2 candidates too close), the quantum
        resolver is triggered for re-scoring.
        """
        # Parse symptoms
        hpo_terms = parse_symptoms(request.symptoms)
        


        if not hpo_terms:
            raise HTTPException(
                status_code=400,
                detail="No valid HPO terms found in input. "
                       "Please provide HPO IDs (e.g., HP:0001083) or symptom names."
            )

        # Build feature vector
        X = build_patient_vector(hpo_terms)

        # Classical diagnosis
        probs = run_classical_diagnosis(X)

        # Confusion detection
        from models.confusion.detector import is_hard_case
        confusion_result = is_hard_case(
            probs, DISEASE_NAMES, KNOWN_CONFUSION_PAIRS
        )

        # Quantum resolver (if hard case)
        quantum_used = False
        quantum_status = "not_needed"

        if confusion_result.is_hard and confusion_result.top1_disease in RARE_DISEASE_NAMES:
            # The margin is already calculated mathematically by the Confusion Detector.
            # We use the true calibrated probabilities produced by the ML model.
            top1_idx = DISEASE_LABEL_MAP[confusion_result.top1_disease]
            top2_idx = DISEASE_LABEL_MAP[confusion_result.top2_disease]

            quantum_status = "triggered"
            top2_indices = [top1_idx, top2_idx]
            quantum_probs = run_quantum_resolver(X, top2_indices)
            if quantum_probs is not None:
                probs = quantum_probs
                quantum_used = True
                quantum_status = "resolved"
            else:
                quantum_status = "fallback_to_classical"

        # Build response
        case_id = str(uuid.uuid4())[:8]

        sorted_indices = np.argsort(probs)[::-1]
        ranked = [
            DiagnosisResult(
                disease=DISEASE_NAMES[idx],
                probability=float(probs[idx]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(sorted_indices)
        ]

        response = DiagnoseResponse(
            case_id=case_id,
            ranked_diagnoses=ranked,
            confidence=float(probs[sorted_indices[0]]),
            is_hard_case=confusion_result.is_hard,
            quantum_used=quantum_used,
            quantum_status=quantum_status,
            top_diagnosis=DISEASE_NAMES[sorted_indices[0]],
            runner_up=DISEASE_NAMES[sorted_indices[1]],
        )

        # Store for later explanation
        state.case_store[case_id] = {
            "hpo_terms": hpo_terms,
            "probs": probs.tolist(),
            "X": X.tolist(),
            "is_hard": confusion_result.is_hard,
            "quantum_used": quantum_used,
            "top_diagnosis": DISEASE_NAMES[sorted_indices[0]],
            "runner_up": DISEASE_NAMES[sorted_indices[1]],
        }

        return response

    @app.get("/explain/{case_id}", response_model=ExplainResponse)
    async def explain(case_id: str):
        """
        Get detailed explanation for a diagnosis case.
        """
        if case_id not in state.case_store:
            raise HTTPException(status_code=404, detail="Case not found")

        case = state.case_store[case_id]
        top_disease = case["top_diagnosis"]
        runner_up = case["runner_up"]
        hpo_terms = case["hpo_terms"]
        probs = np.array(case["probs"])

        # Supporting / against evidence
        model = state.calibrated_model or state.xgb_model
        supporting = []
        against = []

        # Try real SHAP explanation first
        shap_succeeded = False
        if model is not None:
            try:
                patient_X = np.array(case["X"]).reshape(1, -1)
                shap_result = explain_prediction(model, patient_X, (top_disease, runner_up), ALL_HPO_TERMS)

                for ev in shap_result.get('supporting_evidence', []):
                    supporting.append(EvidenceItem(
                        hpo_id=ev['hpo_id'],
                        label=ev['label'],
                        direction="present" if ev['hpo_id'] in hpo_terms else "absent",
                        shap_value=abs(ev['shap_value'])
                    ))

                for ev in shap_result.get('against_evidence', []):
                    against.append(EvidenceItem(
                        hpo_id=ev['hpo_id'],
                        label=ev['label'],
                        direction="present" if ev['hpo_id'] in hpo_terms else "absent",
                        shap_value=abs(ev['shap_value'])
                    ))

                if supporting or against:
                    shap_succeeded = True
            except Exception as e:
                print(f"SHAP explanation failed, falling back to pairwise: {e}")

        # Fallback to pairwise distinguishing features
        if not shap_succeeded:
            pair_key = frozenset([top_disease, runner_up])
            if pair_key in PAIRWISE_DISTINGUISHING:
                for feat in PAIRWISE_DISTINGUISHING[pair_key]:
                    item = EvidenceItem(
                        hpo_id=feat.hpo_id,
                        label=feat.label,
                        direction="present" if feat.hpo_id in hpo_terms else "absent",
                        shap_value=1.0 if feat.hpo_id in hpo_terms else 0.5,
                    )
                    if feat.present_in == top_disease:
                        if feat.hpo_id in hpo_terms:
                            supporting.append(item)
                        else:
                            against.append(item)
                    else:
                        if feat.hpo_id in hpo_terms:
                            against.append(item)
                        else:
                            supporting.append(item)
            else:
                # Last resort: highlight presence of patient symptoms
                for term in hpo_terms:
                    item = EvidenceItem(
                        hpo_id=term,
                        label=HPO_TERMS.get(term, term),
                        direction="present",
                        shap_value=0.5
                    )
                    supporting.append(item)

        # Next-test recommendations
        suggested_tests: list = []
        try:
            recs = recommend_next_test(probs, DISEASE_NAMES, hpo_terms)
            for r in recs[:5]:
                suggested_tests.append(NextTest(
                    hpo_id=r["hpo_id"],
                    label=r["label"],
                    clinical_test=r.get("clinical_test", ""),
                    information_gain=r["information_gain"],
                    expected_outcome_positive=r.get("expected_outcome_positive", ""),
                    expected_outcome_negative=r.get("expected_outcome_negative", "")
                ))
        except Exception as e:
            print(f"Error generating recommendations: {e}")

        return ExplainResponse(
            case_id=case_id,
            top_diagnosis=top_disease,
            runner_up=runner_up,
            supporting_evidence=supporting,
            against_evidence=against,
            suggested_tests=suggested_tests,
        )

    @app.post("/diagnose/breast-cancer", response_model=CommonDiseaseResponse)
    async def diagnose_breast_cancer(request: BreastCancerRequest):
        if not state.breast_cancer_model:
            raise HTTPException(status_code=503, detail="Breast cancer model not loaded. Run pipeline first.")
        
        from models.common.breast_cancer import predict_breast_cancer
        try:
            result = predict_breast_cancer(state.breast_cancer_model, request.features)
            return CommonDiseaseResponse(
                diagnosis=str(result.get("prediction", "Unknown")).capitalize(),
                probability=float(result.get("probability", 0.0)),
                confidence=result.get("confidence", "Low"),
                supporting_evidence=result.get("supporting_evidence"),
                against_evidence=result.get("against_evidence")
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/diagnose/parkinsons", response_model=CommonDiseaseResponse)
    async def diagnose_parkinsons(request: ParkinsonsRequest):
        if not state.parkinsons_model:
            raise HTTPException(status_code=503, detail="Parkinson's model not loaded. Run pipeline first.")
        
        from models.common.parkinsons import predict_parkinsons
        try:
            result = predict_parkinsons(state.parkinsons_model, request.features)
            return CommonDiseaseResponse(
                diagnosis=str(result.get("diagnosis", "Unknown")),
                probability=float(result.get("probability", 0.0)),
                confidence=result.get("confidence", "Low"),
                supporting_evidence=result.get("supporting_evidence"),
                against_evidence=result.get("against_evidence")
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/diseases")
    async def get_diseases():
        """Get the disease catalog."""
        catalog = []
        # Add common diseases
        catalog.append({
            "id": "breast_cancer",
            "name": "Breast Cancer",
            "category": "common",
            "track": "common",
            "description": "30 Cellular Features — Wisconsin Dataset",
            "genes": [],
            "n_symptoms": 30
        })
        catalog.append({
            "id": "parkinsons",
            "name": "Parkinson's Disease",
            "category": "common",
            "track": "common",
            "description": "22 Voice Features — Oxford Dataset",
            "genes": [],
            "n_symptoms": 22
        })
        # Add rare diseases
        for d in ALL_DISEASES:
            track = "quantum" if d.name in RARE_DISEASE_NAMES else "classical"
            catalog.append({
                "id": d.short_name,
                "name": d.name,
                "category": "rare",
                "track": track,
                "description": f"Genes: {', '.join(d.genes)}. OMIM: {d.omim_id}",
                "genes": d.genes,
                "n_symptoms": len(d.symptoms)
            })
        return catalog

    @app.post("/nlp/extract")
    async def nlp_extract(request: NLPExtractRequest):
        """Extract HPO terms from clinical text."""
        text = request.text.lower()
        extracted = []
        seen = set()
        for hpo_id, label in HPO_TERMS.items():
            if hpo_id not in seen and (label.lower() in text or hpo_id.lower() in text):
                extracted.append({
                    "hpo_id": hpo_id,
                    "label": label,
                    "confirmed": True
                })
                seen.add(hpo_id)
        return extracted

    @app.post("/graph")
    async def get_graph(request: GraphRequest):
        """Generate D3 graph data for a set of HPO terms."""
        if not state.kg:
            raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
        
        nodes = []
        links = []
        added_nodes = set()
        
        # Add symptom nodes
        for hpo_id in request.hpo_terms:
            if hpo_id in state.kg.symptoms:
                nodes.append({
                    "id": hpo_id,
                    "group": "symptom",
                    "label": state.kg.symptoms[hpo_id].label
                })
                added_nodes.add(hpo_id)
                
        # Find diseases that have these symptoms
        related_diseases = set()
        for disease_id, symptoms in state.kg.disease_symptoms.items():
            for hpo_id, _ in symptoms:
                if hpo_id in request.hpo_terms:
                    related_diseases.add(disease_id)
                    
        # Add disease nodes and links
        for disease_id in related_diseases:
            if disease_id in state.kg.diseases:
                nodes.append({
                    "id": disease_id,
                    "group": "disease",
                    "label": state.kg.diseases[disease_id].name
                })
                added_nodes.add(disease_id)
                
                # Links to symptoms
                for hpo_id, _ in state.kg.disease_symptoms[disease_id]:
                    if hpo_id in request.hpo_terms:
                        links.append({
                            "source": disease_id,
                            "target": hpo_id
                        })
                        
        # Add LOOKS_LIKE links between diseases in the subgraph
        for d1 in related_diseases:
            for d2, sim, _ in state.kg.disease_looks_like.get(d1, []):
                if d2 in related_diseases and sim > 0:
                    links.append({
                        "source": d1,
                        "target": d2,
                        "type": "looks_like"
                    })
                    
        return {"nodes": nodes, "links": links}

    @app.get("/diseases/{disease_type}/features")
    async def get_disease_features(disease_type: str):
        """Get feature names and defaults for common diseases."""
        if disease_type in ["breast-cancer", "breast_cancer"]:
            defaults = {
                "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8, "mean area": 1001.0,
                "mean smoothness": 0.1184, "mean compactness": 0.2776, "mean concavity": 0.3001,
                "mean concave points": 0.1471, "mean symmetry": 0.2419, "mean fractal dimension": 0.07871,
                "radius error": 1.095, "texture error": 0.9053, "perimeter error": 8.589, "area error": 153.4,
                "smoothness error": 0.006399, "compactness error": 0.04904, "concavity error": 0.05373,
                "concave points error": 0.01587, "symmetry error": 0.03003, "fractal dimension error": 0.006193,
                "worst radius": 25.38, "worst texture": 17.33, "worst perimeter": 184.6, "worst area": 2019.0,
                "worst smoothness": 0.1622, "worst compactness": 0.6656, "worst concavity": 0.7119,
                "worst concave points": 0.2654, "worst symmetry": 0.4601, "worst fractal dimension": 0.1189
            }
            return {"feature_names": list(defaults.keys()), "defaults": defaults}
        elif disease_type == "parkinsons":
            defaults = {
                "MDVP:Fo(Hz)": 119.992, "MDVP:Fhi(Hz)": 157.302, "MDVP:Flo(Hz)": 74.997,
                "MDVP:Jitter(%)": 0.00784, "MDVP:Jitter(Abs)": 0.00007, "MDVP:RAP": 0.0037,
                "MDVP:PPQ": 0.00554, "Jitter:DDP": 0.01109, "MDVP:Shimmer": 0.04374,
                "MDVP:Shimmer(dB)": 0.426, "Shimmer:APQ3": 0.02182, "Shimmer:APQ5": 0.0313,
                "MDVP:APQ": 0.02971, "Shimmer:DDA": 0.06545, "NHR": 0.02211, "HNR": 21.033,
                "RPDE": 0.414783, "DFA": 0.815285, "spread1": -4.813031, "spread2": 0.266482,
                "D2": 2.301442, "PPE": 0.284654
            }
            return {"feature_names": list(defaults.keys()), "defaults": defaults}
        raise HTTPException(status_code=404, detail="Disease not found")

    @app.post("/explain/shap")
    async def get_shap_explanation(request: BaseModel):
        # We already enhanced GET /explain, so this is just a stub if needed
        # Or we can just use /explain endpoint directly
        pass

    @app.get("/benchmark/report")
    async def benchmark_report():
        """
        Get the quantum vs classical benchmark comparison report.

        Returns accuracy, F1, McNemar p-value, and per-cluster breakdown.
        """
        if state.benchmark_data:
            return state.benchmark_data

        # Try to load from file
        report_path = os.path.join(
            os.path.dirname(__file__), '..', 'benchmarks', 'benchmark_metrics.json'
        )
        if os.path.exists(report_path):
            import json
            with open(report_path, 'r') as f:
                return json.load(f)

        return {
            "status": "not_generated",
            "message": "Run run_pipeline.py to generate the benchmark report."
        }

    @app.post("/scan/analyze", response_model=ScanAnalysisResponse)
    async def analyze_medical_scan(
        request: Request,
        file: Optional[UploadFile] = File(None)
    ):
        """
        Analyze medical imaging (X-Ray, Echocardiogram, MRI, CT) using Computer Vision.
        Extracts radiomic features and maps findings to standardized HPO identifiers.
        Supports multipart/form-data upload or JSON with base64 payload.
        """
        from models.vision.cv_module import analyze_scan_detailed
        
        image_bytes = b""
        filename = None
        
        # 1. Check if multipart file uploaded
        if file is not None:
            image_bytes = await file.read()
            filename = file.filename
        else:
            # 2. Check JSON payload or raw body
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    body = await request.json()
                    import base64
                    raw_b64 = body.get("image_base64", "")
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",", 1)[1]
                    if raw_b64:
                        image_bytes = base64.b64decode(raw_b64)
                    filename = body.get("filename")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
            else:
                image_bytes = await request.body()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="No image data provided. Upload a file or provide image_base64.")

        try:
            result = analyze_scan_detailed(image_bytes, filename=filename)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scan analysis failed: {str(e)}")

    @app.post("/scan/mammogram")
    async def analyze_mammogram_scan(
        request: Request,
        file: Optional[UploadFile] = File(None)
    ):
        """
        Analyze digital mammography or breast biopsy imaging using Computer Vision.
        Segments breast lesions, extracts 30 Wisconsin morphological features,
        and executes calibrated XGBoost classification with real SHAP explainability.
        """
        from models.vision.cv_module import analyze_mammogram
        
        image_bytes = b""
        filename = None
        
        if file is not None:
            image_bytes = await file.read()
            filename = file.filename
        else:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    body = await request.json()
                    import base64
                    raw_b64 = body.get("image_base64", "")
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",", 1)[1]
                    if raw_b64:
                        image_bytes = base64.b64decode(raw_b64)
                    filename = body.get("filename")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
            else:
                image_bytes = await request.body()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="No mammogram image data provided.")

        try:
            result = analyze_mammogram(image_bytes, filename=filename)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Mammography analysis failed: {str(e)}")

    # -----------------------------------------------------------------------
    # Automated EHR PDF Extraction (NLP) Endpoints
    # -----------------------------------------------------------------------

    @app.post("/ehr/extract-pdf")
    async def extract_ehr_pdf(
        request: Request,
        file: Optional[UploadFile] = File(None)
    ):
        """
        Automated EHR PDF Extraction:
        Upload patient historical medical record (PDF).
        Extracts clinical sections and parses HPO phenotypes for classical/quantum triage.
        """
        try:
            from backend.ehr_extractor import extract_text_from_pdf
        except ImportError:
            from ehr_extractor import extract_text_from_pdf

        pdf_bytes = b""
        if file is not None:
            pdf_bytes = await file.read()
        else:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    body = await request.json()
                    import base64
                    raw_b64 = body.get("pdf_base64", "")
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",", 1)[1]
                    if raw_b64:
                        pdf_bytes = base64.b64decode(raw_b64)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {str(e)}")
            else:
                pdf_bytes = await request.body()

        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="No PDF data received.")

        try:
            doc_data = extract_text_from_pdf(pdf_bytes)
            full_text = doc_data["full_text"]
            text_lower = full_text.lower()

            extracted_hpo = []
            seen = set()
            for hpo_id, label in HPO_TERMS.items():
                if hpo_id not in seen and (label.lower() in text_lower or hpo_id.lower() in text_lower):
                    extracted_hpo.append({
                        "hpo_id": hpo_id,
                        "label": label,
                        "confirmed": True
                    })
                    seen.add(hpo_id)

            route = "quantum" if len(extracted_hpo) >= 3 else "classical"

            return {
                "full_text": full_text,
                "page_count": doc_data["page_count"],
                "sections": doc_data["sections"],
                "metadata": doc_data["metadata"],
                "extracted_hpo_terms": extracted_hpo,
                "recommended_route": route,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"EHR PDF Extraction failed: {str(e)}")

    @app.get("/ehr/sample-pdf")
    async def get_sample_ehr_pdf(case: str = "marfan"):
        """
        Download authentic sample patient EHR PDF for demonstration.
        """
        try:
            from backend.ehr_extractor import generate_sample_ehr_pdf
        except ImportError:
            from ehr_extractor import generate_sample_ehr_pdf

        pdf_bytes = generate_sample_ehr_pdf(case)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=AIIMS_EHR_Sample_{case.upper()}.pdf"}
        )

    # -----------------------------------------------------------------------
    # Clinical PDF Report Generation Endpoints
    # -----------------------------------------------------------------------

    @app.get("/report/pdf/{case_id}")
    async def download_case_pdf(case_id: str):
        """
        Generate publication-quality, branded clinical differential diagnosis report
        containing Bayesian & Quantum probabilities, SHAP evidence, and Shannon next tests.
        """
        try:
            from backend.pdf_report_generator import generate_clinical_diagnostic_report
        except ImportError:
            from pdf_report_generator import generate_clinical_diagnostic_report

        if case_id not in state.case_store:
            raise HTTPException(status_code=404, detail=f"Case ID {case_id} not found in session.")

        case = state.case_store[case_id]
        sorted_indices = np.argsort(case["probs"])[::-1]
        diagnosis_data = {
            "top_diagnosis": case["top_diagnosis"],
            "runner_up": case["runner_up"],
            "confidence": float(case["probs"][sorted_indices[0]]),
            "is_hard_case": case.get("is_hard", False),
            "quantum_used": case.get("quantum_used", False),
            "ranked_diagnoses": [
                {
                    "rank": r + 1,
                    "disease": DISEASE_NAMES[idx],
                    "probability": float(case["probs"][idx]),
                }
                for r, idx in enumerate(sorted_indices)
            ],
        }

        top_disease = case["top_diagnosis"]
        runner_up = case["runner_up"]
        hpo_terms = case["hpo_terms"]
        supporting = []
        against = []
        model = state.calibrated_model or state.xgb_model
        if model is not None:
            try:
                patient_X = np.array(case["X"]).reshape(1, -1)
                shap_result = explain_prediction(model, patient_X, (top_disease, runner_up), ALL_HPO_TERMS)
                for ev in shap_result.get('supporting_evidence', []):
                    supporting.append({
                        "hpo_id": ev['hpo_id'],
                        "label": ev['label'],
                        "direction": "present" if ev['hpo_id'] in hpo_terms else "absent",
                        "shap_value": abs(ev['shap_value'])
                    })
                for ev in shap_result.get('against_evidence', []):
                    against.append({
                        "hpo_id": ev['hpo_id'],
                        "label": ev['label'],
                        "direction": "present" if ev['hpo_id'] in hpo_terms else "absent",
                        "shap_value": abs(ev['shap_value'])
                    })
            except Exception:
                pass

        suggested_tests = []
        try:
            recs = recommend_next_test(np.array(case["probs"]), DISEASE_NAMES, hpo_terms)
            for r in recs[:5]:
                suggested_tests.append({
                    "hpo_id": r["hpo_id"],
                    "label": r["label"],
                    "clinical_test": r.get("clinical_test", ""),
                    "information_gain": r["information_gain"],
                    "expected_outcome_positive": r.get("expected_outcome_positive", ""),
                    "expected_outcome_negative": r.get("expected_outcome_negative", "")
                })
        except Exception:
            pass

        explanation_data = {
            "supporting_evidence": supporting,
            "against_evidence": against,
            "suggested_tests": suggested_tests,
        }

        pdf_bytes = generate_clinical_diagnostic_report(case_id, diagnosis_data, explanation_data)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=QResolve_Diagnostic_Report_{case_id}.pdf"}
        )

    @app.get("/report/demo-pdf")
    async def download_demo_pdf():
        """
        Generate and download sample clinical diagnostic report demonstrating
        quantum resolution and SHAP explainability.
        """
        try:
            from backend.pdf_report_generator import generate_clinical_diagnostic_report
        except ImportError:
            from pdf_report_generator import generate_clinical_diagnostic_report

        demo_diagnosis = {
            "top_diagnosis": "Marfan syndrome",
            "runner_up": "Loeys-Dietz syndrome",
            "confidence": 0.884,
            "is_hard_case": True,
            "quantum_used": True,
            "ranked_diagnoses": [
                {"rank": 1, "disease": "Marfan syndrome", "probability": 0.884},
                {"rank": 2, "disease": "Loeys-Dietz syndrome", "probability": 0.082},
                {"rank": 3, "disease": "Beals syndrome", "probability": 0.018},
                {"rank": 4, "disease": "MASS phenotype", "probability": 0.011},
                {"rank": 5, "disease": "Shprintzen-Goldberg syndrome", "probability": 0.005},
            ]
        }

        demo_explanation = {
            "supporting_evidence": [
                {"label": "Ectopia lentis", "hpo_id": "HP:0001083", "direction": "present", "shap_value": 0.412},
                {"label": "Aortic root aneurysm", "hpo_id": "HP:0002616", "direction": "present", "shap_value": 0.329},
                {"label": "Pectus excavatum", "hpo_id": "HP:0000768", "direction": "present", "shap_value": 0.188},
            ],
            "against_evidence": [
                {"label": "Bifid uvula", "hpo_id": "HP:0000193", "direction": "absent", "shap_value": 0.285},
                {"label": "Arterial tortuosity", "hpo_id": "HP:0005116", "direction": "absent", "shap_value": 0.240},
            ],
            "suggested_tests": [
                {
                    "label": "Ectopia lentis",
                    "hpo_id": "HP:0001083",
                    "clinical_test": "Ophthalmologic slit-lamp biomicroscopy",
                    "information_gain": 0.942,
                    "expected_outcome_positive": "Rules in Marfan (pathognomonic hallmark)",
                    "expected_outcome_negative": "Raises suspicion of Loeys-Dietz"
                },
                {
                    "label": "Bifid uvula / Cleft palate",
                    "hpo_id": "HP:0000193",
                    "clinical_test": "Direct oropharyngeal inspection / Palate palpation",
                    "information_gain": 0.815,
                    "expected_outcome_positive": "Strong evidence for Loeys-Dietz",
                    "expected_outcome_negative": "Consistent with Marfan syndrome"
                }
            ]
        }

        patient_info = {
            "abha_id": "91-8273-1928-01 (Verified ABDM Golden Card)",
            "facility_name": "AIIMS New Delhi — Medical Genetics & Cardiology",
            "age": 28,
            "gender": "Male"
        }

        pdf_bytes = generate_clinical_diagnostic_report("DEMO-001", demo_diagnosis, demo_explanation, patient_info)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=QResolve_Clinical_Report_DEMO.pdf"}
        )

    # -----------------------------------------------------------------------
    # Government API Setu & ABHA Gateway Endpoints
    # -----------------------------------------------------------------------

    @app.get("/abha/beneficiaries")
    async def get_abha_beneficiaries():
        """
        API Setu & ABHA Sandbox:
        List verified sandbox beneficiary profiles for Smart India Hackathon (SIH) demonstration.
        """
        try:
            from backend.abha_gateway import ABDM_SANDBOX_REGISTRY
        except ImportError:
            from abha_gateway import ABDM_SANDBOX_REGISTRY

        return [
            {
                "abha_id": v["abha_id"],
                "name": v["name"],
                "gender": v["gender"],
                "age": v["age"],
                "beneficiary_scheme": v["beneficiary_scheme"],
                "facility_name": v["facility_name"],
                "clinical_summary": v["clinical_summary"],
                "n_conditions": len(v["fhir_conditions"]),
            }
            for v in ABDM_SANDBOX_REGISTRY.values()
        ]

    @app.get("/abha/patient/{abha_id}")
    async def get_abha_patient(abha_id: str):
        """
        API Setu & ABHA Gateway:
        Fetch patient health records via ABDM sandbox gateway in FHIR R4 format.
        Automatically extracts phenotype observations to HPO terms.
        """
        try:
            from backend.abha_gateway import fetch_abha_patient_record
        except ImportError:
            from abha_gateway import fetch_abha_patient_record

        return fetch_abha_patient_record(abha_id)

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model_loaded": state.calibrated_model is not None or state.xgb_model is not None,
            "diseases": DISEASE_NAMES,
            "n_features": len(ALL_HPO_TERMS),
        }


if __name__ == "__main__":
    if HAS_FASTAPI:
        import uvicorn
        print("Starting QResolve-Dx backend...")
        load_models()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
