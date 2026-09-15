import React, { useState, useEffect } from 'react';
import { api } from '../../lib/api';
import type { CommonDiseaseResponse, FeatureMetadata } from '../../types';

export const CommonDiseaseInput = ({ diseaseType }: { diseaseType: 'breast-cancer' | 'parkinsons' }) => {
  const [result, setResult] = useState<CommonDiseaseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [metadata, setMetadata] = useState<FeatureMetadata | null>(null);
  const [features, setFeatures] = useState<Record<string, number>>({});
  const [loadingMetadata, setLoadingMetadata] = useState(true);

  useEffect(() => {
    const fetchMetadata = async () => {
      setLoadingMetadata(true);
      setError(null);
      setResult(null);
      try {
        const data = await api.getFeatures(diseaseType);
        setMetadata(data);
        setFeatures({ ...data.defaults });
      } catch (err: any) {
        setError(err.message || 'Failed to load feature metadata.');
      } finally {
        setLoadingMetadata(false);
      }
    };
    fetchMetadata();
  }, [diseaseType]);

  const handleInputChange = (key: string, value: string) => {
    setFeatures(prev => ({
      ...prev,
      [key]: value === '' ? 0 : parseFloat(value) || 0
    }));
  };

  const handleDiagnose = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      if (diseaseType === 'breast-cancer') {
        const res = await api.diagnoseBreastCancer({ features });
        setResult(res);
      } else {
        const res = await api.diagnoseParkinsons({ features });
        setResult(res);
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred during diagnosis.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-8 py-10">
      <div className="mb-8">
        <h2 className="text-2xl font-semibold text-ink">
          {diseaseType === 'breast-cancer' ? 'Breast Cancer Diagnosis' : "Parkinson's Disease Diagnosis"}
        </h2>
        <p className="text-sm text-ink-muted mt-2">
          Submit clinical measurements to receive a classical ML prediction.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-8">
        <div className="col-span-2">
          <div className="bg-surface border border-border p-6 rounded-md shadow-sm">
            <h3 className="font-medium text-ink mb-1">Patient Data Input</h3>
            <p className="text-xs text-ink-muted mb-6">
              The form is pre-filled with measurements from a sample case. Adjust the values below to test the model.
            </p>
            
            {loadingMetadata ? (
              <div className="py-12 text-center text-ink-muted font-serif">Loading features...</div>
            ) : metadata ? (
              <form onSubmit={handleDiagnose}>
                <div className="grid grid-cols-2 gap-x-6 gap-y-4 mb-6">
                  {metadata.feature_names.map((key) => (
                    <div key={key} className="flex flex-col">
                      <label className="text-[11px] font-semibold text-ink-muted uppercase tracking-wider mb-1" htmlFor={key}>
                        {key}
                      </label>
                      <input
                        id={key}
                        type="number"
                        step="any"
                        required
                        value={features[key] ?? ''}
                        onChange={(e) => handleInputChange(key, e.target.value)}
                        className="bg-bg border border-border rounded-sm px-3 py-1.5 text-sm font-mono text-ink focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
                      />
                    </div>
                  ))}
                </div>

                <div className="pt-4 border-t border-border flex items-center justify-between">
                  <button
                    type="button"
                    onClick={() => setFeatures({ ...metadata.defaults })}
                    className="text-sm text-ink-muted hover:text-ink transition-colors"
                  >
                    Reset Defaults
                  </button>
                  <button
                    type="submit"
                    disabled={loading}
                    className="bg-primary text-white px-6 py-2 rounded-sm text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
                  >
                    {loading ? 'Running Inference...' : 'Run Prediction'}
                  </button>
                </div>
              </form>
            ) : null}
            
            {error && (
              <div className="mt-4 p-4 bg-red-50 text-red-700 text-sm rounded-sm">
                {error}
              </div>
            )}
          </div>
        </div>

        <div className="col-span-1">
          <div className="sticky top-6">
            <div className="bg-surface border border-border p-6 rounded-md shadow-sm transition-all duration-300">
              <h3 className="font-medium text-ink mb-6 text-lg flex justify-between items-center">
                Diagnostic Result
                {loading && <span className="text-[10px] uppercase font-bold text-primary animate-pulse bg-primary-soft px-2 py-1 rounded">Computing</span>}
              </h3>
              
              {!result && !error && (
                <div className="py-8 text-center text-sm text-ink-muted">
                  Submit the form to view the model's prediction.
                </div>
              )}

              {result && (
                <div className={`space-y-4 transition-opacity duration-300 ${loading ? 'opacity-50' : 'opacity-100'}`}>
                  <div className="p-4 bg-bg border border-border rounded-sm text-center">
                    <div className="text-xs text-ink-muted uppercase tracking-wider mb-1 font-semibold">Prediction</div>
                    <div className={`text-xl font-semibold ${result.diagnosis.toLowerCase() === 'malignant' || result.diagnosis.toLowerCase().includes('parkinson') ? 'text-red-600' : 'text-green-600'}`}>
                      {result.diagnosis}
                    </div>
                  </div>
                  
                  <div className="p-4 bg-bg border border-border rounded-sm text-center">
                    <div className="text-xs text-ink-muted uppercase tracking-wider mb-1 font-semibold">Model Confidence</div>
                    <div className="text-xl font-mono text-ink">{(result.confidence * 100).toFixed(1)}%</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
