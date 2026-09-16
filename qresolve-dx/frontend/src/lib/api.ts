import type { DiagnoseRequest, DiagnoseResponse, ExplainResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export const api = {
  /**
   * Submit patient symptoms to get a diagnosis
   */
  diagnose: async (request: DiagnoseRequest): Promise<DiagnoseResponse> => {
    const response = await fetch(`${API_BASE_URL}/diagnose`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Failed to get diagnosis');
    }

    return response.json();
  },

  /**
   * Get explainability (SHAP & next test) for a specific case
   */
  explain: async (caseId: string): Promise<ExplainResponse> => {
    const response = await fetch(`${API_BASE_URL}/explain/${caseId}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Failed to get explanation');
    }

    return response.json();
  },
  
  /**
   * Get neo4j graph data
   */
  getGraph: async (_disease: string): Promise<any> => {
      // Mock for now until graph endpoint is confirmed
      return { nodes: [], edges: [] };
  }
};
