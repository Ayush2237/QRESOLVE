import React, { useState, useEffect, useRef } from 'react';
import { api } from '../../lib/api';
import type { CommonDiseaseResponse, FeatureMetadata, MammogramAnalysisResult } from '../../types';
import { Badge } from '../common/Badge';
import { Button } from '../common/Button';

export const CommonDiseaseInput = ({ diseaseType }: { diseaseType: 'breast-cancer' | 'parkinsons' }) => {
  const [result, setResult] = useState<CommonDiseaseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [metadata, setMetadata] = useState<FeatureMetadata | null>(null);
  const [features, setFeatures] = useState<Record<string, number>>({});
  const [loadingMetadata, setLoadingMetadata] = useState(true);

  // Mammogram upload state
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [mammoResult, setMammoResult] = useState<MammogramAnalysisResult | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetadata = async () => {
      setLoadingMetadata(true);
      setError(null);
      setResult(null);
      setMammoResult(null);
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

  const handleMammogramUpload = async (file: File) => {
    setIsScanning(true);
    setScanError(null);
    try {
      const res = await api.analyzeMammogram(file, file.name);
      setMammoResult(res);
      // Auto-populate all 30 Wisconsin features from the segmented mass contour
      setFeatures(res.extracted_features);
      // Set the live prediction
      setResult({
        diagnosis: res.diagnosis,
        probability: res.probability,
        confidence: res.confidence,
        supporting_evidence: res.supporting_evidence,
        against_evidence: res.against_evidence,
      });
    } catch (err: any) {
      setScanError(err.message || 'Failed to analyze digital mammogram');
    } finally {
      setIsScanning(false);
    }
  };

  const handlePresetMammogram = async (presetType: 'malignant' | 'benign') => {
    setIsScanning(true);
    setScanError(null);
    try {
      const dummyPngBase64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
      const filename = presetType === 'malignant' ? "mammogram_spiculated_malignant.png" : "mammogram_circumscribed_benign.png";
      const res = await api.analyzeMammogram(dummyPngBase64, filename);
      setMammoResult(res);
      setFeatures(res.extracted_features);
      setResult({
        diagnosis: res.diagnosis,
        probability: res.probability,
        confidence: res.confidence,
        supporting_evidence: res.supporting_evidence,
        against_evidence: res.against_evidence,
      });
    } catch (err: any) {
      setScanError(err.message || 'Failed to run mammogram analysis');
    } finally {
      setIsScanning(false);
    }
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
          {diseaseType === 'breast-cancer' ? 'Breast Cancer Diagnosis & Mammography' : "Parkinson's Disease Diagnosis"}
        </h2>
        <p className="text-sm text-ink-muted mt-2">
          {diseaseType === 'breast-cancer'
            ? 'Upload digital mammography scans for automated lesion segmentation and Wisconsin 30-feature extraction, or adjust measurements manually.'
            : 'Submit clinical measurements to receive a classical ML prediction.'}
        </p>
      </div>

      {/* Hidden file input for mammogram upload */}
      {diseaseType === 'breast-cancer' && (
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept="image/*,.dcm,.dicom"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleMammogramUpload(e.target.files[0]);
            }
          }}
        />
      )}

      {/* Digital Mammography CV Upload Banner */}
      {diseaseType === 'breast-cancer' && (
        <div className="mb-8 bg-surface border border-border p-5 rounded-md shadow-sm">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-600 inline-block"></span>
                <span className="font-semibold text-sm text-ink">Digital Mammography (Computer Vision Module)</span>
                <Badge variant="primary" className="text-[10px]">AI SEGMENTATION</Badge>
              </div>
              <p className="text-xs text-ink-muted mt-1 max-w-xl">
                Upload a mammogram scan (DICOM/PNG/JPG). The computer vision engine will segment the suspicious mass contour, compute all 30 Wisconsin morphological features, and run calibrated XGBoost inference.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                disabled={isScanning || loading}
                className="text-xs py-1.5 px-3 border-blue-600/40 hover:border-blue-600 text-blue-600 font-medium"
              >
                {isScanning ? (
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-3 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></span>
                    Segmenting Mass...
                  </span>
                ) : (
                  '📁 Upload Mammogram'
                )}
              </Button>

              <div className="flex items-center gap-1 border-l border-border pl-2">
                <button
                  type="button"
                  onClick={() => handlePresetMammogram('malignant')}
                  disabled={isScanning}
                  className="text-[11px] text-red-700 bg-red-50 hover:bg-red-100 px-2 py-1 rounded border border-red-200 transition-colors"
                  title="Test Spiculated Malignant Mass"
                >
                  + Malignant Scan
                </button>
                <button
                  type="button"
                  onClick={() => handlePresetMammogram('benign')}
                  disabled={isScanning}
                  className="text-[11px] text-green-700 bg-green-50 hover:bg-green-100 px-2 py-1 rounded border border-green-200 transition-colors"
                  title="Test Circumscribed Benign Fibroadenoma"
                >
                  + Benign Scan
                </button>
              </div>
            </div>
          </div>

          {scanError && (
            <div className="mt-3 p-3 bg-red-50 text-red-700 text-xs rounded border border-red-200">
              Mammography Scan Error: {scanError}
            </div>
          )}

          {mammoResult && (
            <div className="mt-4 p-4 bg-blue-50/50 border border-blue-200 rounded-sm text-xs animate-fade-in">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <span className="font-semibold text-blue-900 uppercase tracking-wide">
                    Computer Vision Morphological Analysis Complete
                  </span>
                  <div className="text-ink-muted text-[11px] mt-0.5">{mammoResult.birads_score}</div>
                </div>
                <Badge variant={mammoResult.diagnosis.toLowerCase() === 'malignant' ? 'warning' : 'primary'}>
                  {(mammoResult.probability * 100).toFixed(1)}% {mammoResult.diagnosis}
                </Badge>
              </div>

              <div className="text-[11px] text-ink-muted leading-relaxed mb-3">
                {mammoResult.radiological_summary}
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-blue-200/60 font-mono text-[10px] text-ink-muted">
                <div>Mass Radius: {mammoResult.lesion_metrics.mean_radius_px}px</div>
                <div>Contour Perimeter: {mammoResult.lesion_metrics.boundary_perimeter_px}px</div>
                <div>Compactness: {mammoResult.lesion_metrics.lesion_compactness}</div>
                <div>Concavity Index: {mammoResult.lesion_metrics.concavity_index}</div>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-3 gap-8">
        <div className="col-span-2">
          <div className="bg-surface border border-border p-6 rounded-md shadow-sm">
            <h3 className="font-medium text-ink mb-1">
              {diseaseType === 'breast-cancer' ? '30 Extracted Morphological Features (Wisconsin Standard)' : 'Patient Data Input'}
            </h3>
            <p className="text-xs text-ink-muted mb-6">
              {diseaseType === 'breast-cancer'
                ? 'These 30 clinical metrics are extracted directly from the digitized mass contour. Adjust values below or run inference.'
                : 'The form is pre-filled with measurements from a sample case. Adjust the values below to test the model.'}
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
                    onClick={() => {
                      setFeatures({ ...metadata.defaults });
                      setMammoResult(null);
                    }}
                    className="text-sm text-ink-muted hover:text-ink transition-colors"
                  >
                    Reset Defaults
                  </button>
                  <button
                    type="submit"
                    disabled={loading || isScanning}
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
                  Upload a mammogram or submit the form to view the model's prediction.
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
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-bg border border-border rounded-sm text-center">
                      <div className="text-xs text-ink-muted uppercase tracking-wider mb-1 font-semibold">Probability</div>
                      <div className="text-xl font-mono text-ink">{(result.probability * 100).toFixed(1)}%</div>
                    </div>
                    <div className="p-4 bg-bg border border-border rounded-sm text-center">
                      <div className="text-xs text-ink-muted uppercase tracking-wider mb-1 font-semibold">Confidence</div>
                      <div className="text-xl font-mono text-ink">{result.confidence}</div>
                    </div>
                  </div>

                  {result.supporting_evidence && result.supporting_evidence.length > 0 && (
                    <div className="mt-6 pt-6 border-t border-border">
                      <h4 className="text-xs font-semibold text-green-700 uppercase tracking-wider mb-3">Supporting Features (SHAP)</h4>
                      <div className="space-y-3">
                        {result.supporting_evidence.map((ev, i) => {
                          const allVals = [...(result.supporting_evidence||[]), ...(result.against_evidence||[])].map(x => Math.abs(x.shap_value));
                          const maxVal = Math.max(0.01, ...allVals);
                          const width = `${Math.min(100, (Math.abs(ev.shap_value) / maxVal) * 100)}%`;
                          return (
                            <div key={i} className="flex flex-col gap-1">
                              <div className="flex justify-between text-xs">
                                <span className="font-medium text-ink truncate mr-2" title={ev.feature}>{ev.feature}</span>
                                <span className="text-green-600 font-mono font-bold">+{ev.shap_value.toFixed(3)}</span>
                              </div>
                              <div className="w-full bg-gray-200 h-1.5 rounded-full overflow-hidden">
                                <div className="bg-green-500 h-full rounded-full" style={{ width }}></div>
                              </div>
                              <div className="text-[10px] text-ink-muted">Value: {ev.value.toFixed(2)}</div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {result.against_evidence && result.against_evidence.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-border">
                      <h4 className="text-xs font-semibold text-red-700 uppercase tracking-wider mb-3">Counter Features (SHAP)</h4>
                      <div className="space-y-3">
                        {result.against_evidence.map((ev, i) => {
                          const allVals = [...(result.supporting_evidence||[]), ...(result.against_evidence||[])].map(x => Math.abs(x.shap_value));
                          const maxVal = Math.max(0.01, ...allVals);
                          const width = `${Math.min(100, (Math.abs(ev.shap_value) / maxVal) * 100)}%`;
                          return (
                            <div key={i} className="flex flex-col gap-1">
                              <div className="flex justify-between text-xs">
                                <span className="font-medium text-ink truncate mr-2" title={ev.feature}>{ev.feature}</span>
                                <span className="text-red-600 font-mono font-bold">-{Math.abs(ev.shap_value).toFixed(3)}</span>
                              </div>
                              <div className="w-full bg-gray-200 h-1.5 rounded-full overflow-hidden">
                                <div className="bg-red-500 h-full rounded-full" style={{ width }}></div>
                              </div>
                              <div className="text-[10px] text-ink-muted">Value: {ev.value.toFixed(2)}</div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
