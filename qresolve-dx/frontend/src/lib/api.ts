import type {
  DiagnoseRequest, DiagnoseResponse, ExplainResponse,
  DiseaseInfo, NLPResult, GraphData,
  BreastCancerRequest, ParkinsonsRequest, CommonDiseaseResponse,
  FeatureMetadata,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Request failed: ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  /**
   * Get the full disease catalog (common + rare)
   */
  getDiseases: async (): Promise<DiseaseInfo[]> => {
    return request<DiseaseInfo[]>(`${API_BASE_URL}/diseases`);
  },

  /**
   * Submit patient symptoms to get a diagnosis
   */
  diagnose: async (req: DiagnoseRequest): Promise<DiagnoseResponse> => {
    return request<DiagnoseResponse>(`${API_BASE_URL}/diagnose`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  },

  /**
   * Get explainability (SHAP & next test) for a specific case
   */
  explain: async (caseId: string): Promise<ExplainResponse> => {
    return request<ExplainResponse>(`${API_BASE_URL}/explain/${caseId}`);
  },

  /**
   * Extract HPO terms from clinical free text via NLP
   */
  extractNLP: async (text: string): Promise<NLPResult[]> => {
    return request<NLPResult[]>(`${API_BASE_URL}/nlp/extract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
  },

  /**
   * Get knowledge graph data for a set of HPO terms
   */
  getGraph: async (hpoTerms: string[]): Promise<GraphData> => {
    return request<GraphData>(`${API_BASE_URL}/graph`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hpo_terms: hpoTerms }),
    });
  },

  /**
   * Get feature names and defaults for common diseases
   */
  getFeatures: async (diseaseType: string): Promise<FeatureMetadata> => {
    return request<FeatureMetadata>(`${API_BASE_URL}/diseases/${diseaseType}/features`);
  },

  /**
   * Diagnose breast cancer from 30 cellular features
   */
  diagnoseBreastCancer: async (req: BreastCancerRequest): Promise<CommonDiseaseResponse> => {
    return request<CommonDiseaseResponse>(`${API_BASE_URL}/diagnose/breast-cancer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  },

  /**
   * Diagnose Parkinson's from 22 voice features
   */
  diagnoseParkinsons: async (req: ParkinsonsRequest): Promise<CommonDiseaseResponse> => {
    return request<CommonDiseaseResponse>(`${API_BASE_URL}/diagnose/parkinsons`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  },

  /**
   * Get benchmark report (quantum vs classical comparison)
   */
  getBenchmarkReport: async (): Promise<any> => {
    return request<any>(`${API_BASE_URL}/benchmark/report`);
  },
};
