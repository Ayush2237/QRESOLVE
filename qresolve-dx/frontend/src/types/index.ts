export type DiagnoseRequest = {
  symptoms: string[];
  case_id?: string;
};

export type DiagnosisResult = {
  disease: string;
  probability: number;
  rank: number;
};

export type DiagnoseResponse = {
  case_id: string;
  ranked_diagnoses: DiagnosisResult[];
  confidence: number;
  is_hard_case: boolean;
  quantum_used: boolean;
  quantum_status: string;
  top_diagnosis: string;
  runner_up: string;
};

export type EvidenceItem = {
  hpo_id: string;
  label: string;
  direction: string;
  shap_value: number;
};

export type NextTest = {
  hpo_id: string;
  label: string;
  clinical_test: string;
  information_gain: number;
};

export type ExplainResponse = {
  case_id: string;
  top_diagnosis: string;
  runner_up: string;
  supporting_evidence: EvidenceItem[];
  against_evidence: EvidenceItem[];
  suggested_tests: NextTest[];
};

export type PatientCase = {
  id: string;
  patientName: string;
  age: number;
  sex: string;
  currentStage: number; // 1-9
  diagnoseResponse?: DiagnoseResponse;
  explainResponse?: ExplainResponse;
};
