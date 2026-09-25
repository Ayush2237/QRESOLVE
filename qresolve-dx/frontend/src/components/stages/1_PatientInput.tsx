import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';
import { api } from '../../lib/api';
import type {
  DiseaseInfo, ScanAnalysisResponse, EhrExtractionResult,
  AbhaBeneficiarySummary, AbhaPatientProfile,
} from '../../types';

export const PatientInput = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [symptoms, setSymptoms] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [disease, setDisease] = useState<DiseaseInfo | null>(null);

  // Multi-Modal CV Scan Analysis state
  const scanFileInputRef = useRef<HTMLInputElement>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState<ScanAnalysisResponse | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);

  // Automated EHR PDF Extraction state
  const ehrFileInputRef = useRef<HTMLInputElement>(null);
  const [isExtractingEhr, setIsExtractingEhr] = useState(false);
  const [ehrResult, setEhrResult] = useState<EhrExtractionResult | null>(null);
  const [ehrError, setEhrError] = useState<string | null>(null);

  // ABHA / API Setu state
  const [abhaIdInput, setAbhaIdInput] = useState('');
  const [isFetchingAbha, setIsFetchingAbha] = useState(false);
  const [abhaPatient, setAbhaPatient] = useState<AbhaPatientProfile | null>(null);
  const [abhaBeneficiaries, setAbhaBeneficiaries] = useState<AbhaBeneficiarySummary[]>([]);
  const [activeTab, setActiveTab] = useState<'ehr' | 'scan' | 'abha'>('ehr');

  useEffect(() => {
    const loadDisease = async () => {
      try {
        const diseases = await api.getDiseases();
        const d = diseases.find(x => x.id === id);
        if (d) setDisease(d);
      } catch (e) {
        console.error("Failed to load diseases", e);
      }
    };
    if (id && id !== 'new') {
      loadDisease();
    }

    // Load available ABDM beneficiaries for quick selection
    const loadBeneficiaries = async () => {
      try {
        const list = await api.getAbhaBeneficiaries();
        setAbhaBeneficiaries(list);
      } catch (e) {
        console.warn("Could not preload ABHA beneficiaries", e);
      }
    };
    loadBeneficiaries();
  }, [id]);

  const handleDemo = () => {
    let demoText = "28yo male presenting with pectus excavatum, arachnodactyly, and recent echocardiogram showing aortic root dilation (Z-score 2.9). No family history of sudden cardiac death.";
    
    if (id === 'beals') {
      demoText = "14yo male presenting with congenital flexion contractures of the elbows and knees. Examination shows crumpled ears and arachnodactyly. No ectopia lentis or aortic root dilation.";
    } else if (id === 'loeys_dietz') {
      demoText = "19yo male with arterial tortuosity, hypertelorism, and bifid uvula. Echocardiogram reveals ascending aortic aneurysm.";
    } else if (id === 'mass') {
      demoText = "Patient exhibits myopia and mitral valve prolapse, along with striae distensae and mild aortic root dilation.";
    } else if (id === 'shprintzen_goldberg') {
      demoText = "Patient presents with craniosynostosis, intellectual disability, and arachnodactyly.";
    } else if (id === 'marfan') {
      demoText = "28yo male presenting with pectus excavatum, arachnodactyly, and recent echocardiogram showing aortic root aneurysm (Z-score 2.9). No ectopia lentis.";
    }

    setSymptoms(demoText);
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      navigate(`/case/${id || 'new'}/nlp`, { state: { rawText: demoText } });
    }, 400);
  };

  // 1. EHR PDF Extraction Handlers
  const handleEhrUpload = async (file: File) => {
    setIsExtractingEhr(true);
    setEhrError(null);
    try {
      const result = await api.extractEhrPdf(file);
      setEhrResult(result);
      if (result.full_text) {
        setSymptoms(result.full_text);
      }
    } catch (err: any) {
      setEhrError(err.message || 'Failed to extract text from EHR PDF');
    } finally {
      setIsExtractingEhr(false);
    }
  };

  const handleSampleEhr = async (caseType: string = 'marfan') => {
    setIsExtractingEhr(true);
    setEhrError(null);
    try {
      // Fetch sample PDF from backend endpoint
      const response = await fetch(api.getSampleEhrPdfUrl(caseType));
      if (!response.ok) throw new Error("Could not fetch sample EHR PDF from backend");
      const blob = await response.blob();
      const file = new File([blob], `AIIMS_Record_${caseType}.pdf`, { type: 'application/pdf' });
      await handleEhrUpload(file);
    } catch (err: any) {
      setEhrError(err.message || 'Failed to load sample EHR PDF');
      setIsExtractingEhr(false);
    }
  };

  // 2. Computer Vision Scan Handlers
  const handleScanUpload = async (file: File) => {
    setIsScanning(true);
    setScanError(null);
    try {
      const result = await api.analyzeScan(file, file.name);
      setScanResult(result);

      const findingsSummary = result.findings
        .map(f => `${f.label} (${f.hpo_id})`)
        .join(', ');
      
      const appendText = symptoms.trim()
        ? `\n\n[CV Imaging Findings - ${result.modality_detected}]: ${findingsSummary}. ${result.findings.map(f => f.clinical_evidence).join(' ')}`
        : `Patient imaging study (${result.modality_detected}) reveals: ${findingsSummary}. ${result.findings.map(f => f.clinical_evidence).join(' ')}`;
      
      setSymptoms(prev => (prev.trim() ? prev + appendText : appendText));
    } catch (err: any) {
      setScanError(err.message || 'Failed to analyze imaging scan');
    } finally {
      setIsScanning(false);
    }
  };

  const handlePresetScan = async (presetType: 'chest_xray' | 'echo' | 'mra') => {
    setIsScanning(true);
    setScanError(null);
    try {
      const dummyPngBase64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
      const filenameMap = {
        chest_xray: "chest_radiograph_pectus.png",
        echo: "echocardiogram_aortic_dilation.png",
        mra: "mra_angiography_tortuosity.png",
      };
      const result = await api.analyzeScan(dummyPngBase64, filenameMap[presetType]);
      setScanResult(result);

      const findingsSummary = result.findings
        .map(f => `${f.label} (${f.hpo_id})`)
        .join(', ');

      const appendText = symptoms.trim()
        ? `\n\n[CV Imaging Findings - ${result.modality_detected}]: ${findingsSummary}. ${result.findings.map(f => f.clinical_evidence).join(' ')}`
        : `Patient imaging study (${result.modality_detected}) reveals: ${findingsSummary}. ${result.findings.map(f => f.clinical_evidence).join(' ')}`;

      setSymptoms(prev => (prev.trim() ? prev + appendText : appendText));
    } catch (err: any) {
      setScanError(err.message || 'Failed to analyze imaging scan');
    } finally {
      setIsScanning(false);
    }
  };

  // 3. ABHA / API Setu Gateway Handlers
  const handleFetchAbha = async (idToFetch?: string) => {
    const targetId = idToFetch || abhaIdInput;
    if (!targetId.trim()) return;

    setIsFetchingAbha(true);
    try {
      const profile = await api.getAbhaPatient(targetId);
      setAbhaPatient(profile);
      setAbhaIdInput(profile.abha_id);

      // Extract clinical summary and FHIR conditions
      const conditionNames = profile.fhir_conditions.map(c => `${c.display} (${c.code})`).join(', ');
      const formattedNotes = `[ABHA ID: ${profile.abha_id}] Patient: ${profile.name} (${profile.age}yo ${profile.gender})\nFacility: ${profile.facility_name}\nScheme: ${profile.beneficiary_scheme}\n\nClinical Summary:\n${profile.clinical_summary}\n\nVerified FHIR Conditions:\n${conditionNames}`;
      
      setSymptoms(formattedNotes);
    } catch (err: any) {
      alert(`ABHA Fetch failed: ${err.message || 'Beneficiary record not found'}`);
    } finally {
      setIsFetchingAbha(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symptoms.trim()) return;

    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      navigate(`/case/${id || 'new'}/nlp`, { state: { rawText: symptoms, abhaId: abhaPatient?.abha_id } });
    }, 400);
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div>
        <h2 className="text-xl font-semibold text-ink">1. Patient Input & Data Intake</h2>
        <p className="text-ink-muted text-sm mt-1">
          Automated EHR PDF extraction, Ayushman Bharat (ABHA) record ingestion, or multi-modal imaging CV analysis.
        </p>
        {disease && (
          <div className="mt-2 text-sm text-primary">Testing pathway for: {disease.name}</div>
        )}
      </div>

      {/* Hidden file inputs */}
      <input
        type="file"
        ref={ehrFileInputRef}
        className="hidden"
        accept=".pdf,application/pdf"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            handleEhrUpload(e.target.files[0]);
          }
        }}
      />
      <input
        type="file"
        ref={scanFileInputRef}
        className="hidden"
        accept="image/*,.dcm,.dicom"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            handleScanUpload(e.target.files[0]);
          }
        }}
      />

      {/* Mode Switcher Tabs */}
      <div className="border border-border bg-surface p-1 flex gap-1 text-xs">
        <button
          type="button"
          onClick={() => setActiveTab('ehr')}
          className={`flex-1 py-2 px-3 font-semibold transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'ehr'
              ? 'bg-primary text-white shadow-sm'
              : 'text-ink-muted hover:text-ink hover:bg-gray-100'
          }`}
        >
          <span>📑</span>
          <span>EHR PDF Extractor (NLP)</span>
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('abha')}
          className={`flex-1 py-2 px-3 font-semibold transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'abha'
              ? 'bg-primary text-white shadow-sm'
              : 'text-ink-muted hover:text-ink hover:bg-gray-100'
          }`}
        >
          <span>🇮🇳</span>
          <span>Ayushman Bharat (ABHA)</span>
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('scan')}
          className={`flex-1 py-2 px-3 font-semibold transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'scan'
              ? 'bg-primary text-white shadow-sm'
              : 'text-ink-muted hover:text-ink hover:bg-gray-100'
          }`}
        >
          <span>🔬</span>
          <span>Imaging Scan (CV)</span>
        </button>
      </div>

      {/* Tab 1: EHR PDF Extractor */}
      {activeTab === 'ehr' && (
        <Card className="p-4 border-l-4 border-l-blue-600 bg-surface space-y-3">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm text-ink">Automated EHR PDF Ingestion</span>
                <Badge variant="default" className="text-[10px] font-mono uppercase bg-blue-100 text-blue-800">PyPDF & NLP</Badge>
              </div>
              <p className="text-xs text-ink-muted mt-1">
                Upload historical hospital clinical record PDF to extract HPI, phenotypic markers, and clinical sections.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => ehrFileInputRef.current?.click()}
                disabled={isExtractingEhr || isLoading}
                className="text-xs py-1.5 px-3 border-blue-400 text-blue-700 hover:bg-blue-50"
              >
                {isExtractingEhr ? 'Extracting PDF...' : '📄 Upload EHR PDF'}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => handleSampleEhr('marfan')}
                disabled={isExtractingEhr || isLoading}
                className="text-xs py-1.5 px-3 text-ink-muted hover:text-primary"
              >
                Sample AIIMS PDF
              </Button>
            </div>
          </div>

          {ehrError && (
            <div className="p-2.5 bg-red-50 border border-red-200 text-red-700 text-xs">
              EHR Extraction Error: {ehrError}
            </div>
          )}

          {ehrResult && (
            <div className="p-3 bg-blue-50/50 border border-blue-200 space-y-2 text-xs">
              <div className="flex justify-between items-center">
                <span className="font-semibold text-blue-900">
                  Extracted {ehrResult.page_count} Page Document • Route: {ehrResult.recommended_route.toUpperCase()}
                </span>
                <span className="font-mono text-ink-muted">
                  {ehrResult.extracted_hpo_terms.length} HPO Phenotypes Recognized
                </span>
              </div>
              {ehrResult.extracted_hpo_terms.length > 0 && (
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {ehrResult.extracted_hpo_terms.map(h => (
                    <span key={h.hpo_id} className="bg-white px-2 py-0.5 border border-blue-300 text-blue-900 rounded text-[11px] font-mono">
                      {h.label} ({h.hpo_id})
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </Card>
      )}

      {/* Tab 2: ABHA / API Setu */}
      {activeTab === 'abha' && (
        <Card className="p-4 border-l-4 border-l-green-600 bg-surface space-y-3">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm text-ink">Ayushman Bharat Health Account (ABHA)</span>
                <Badge variant="default" className="text-[10px] font-mono uppercase bg-green-100 text-green-800">API Setu / ABDM</Badge>
              </div>
              <p className="text-xs text-ink-muted mt-1">
                Direct integration with National Health Authority sandbox to retrieve FHIR R4 clinical records.
              </p>
            </div>
          </div>

          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Enter 14-digit ABHA ID (e.g. 91-8273-1928-01)"
              value={abhaIdInput}
              onChange={(e) => setAbhaIdInput(e.target.value)}
              className="flex-1 border border-border px-3 py-1.5 text-xs font-mono focus:ring-1 focus:ring-green-600 outline-none"
            />
            <Button
              type="button"
              variant="outline"
              onClick={() => handleFetchAbha()}
              disabled={isFetchingAbha || !abhaIdInput.trim()}
              className="text-xs py-1.5 px-3 border-green-600 text-green-800 hover:bg-green-50"
            >
              {isFetchingAbha ? 'Fetching...' : 'Fetch Record'}
            </Button>
          </div>

          {/* Quick preset verified sandbox beneficiaries */}
          <div className="pt-1">
            <div className="text-[11px] text-ink-muted mb-1.5">SIH Sandbox Test Profiles (Click to load):</div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              {abhaBeneficiaries.map((b) => (
                <button
                  key={b.abha_id}
                  type="button"
                  onClick={() => handleFetchAbha(b.abha_id)}
                  className="p-2 border border-border hover:border-green-600 bg-white hover:bg-green-50/50 text-left transition-colors flex flex-col justify-between"
                >
                  <div className="font-semibold text-[11px] text-ink truncate">{b.name}</div>
                  <div className="text-[10px] font-mono text-ink-muted">{b.abha_id}</div>
                  <div className="text-[10px] text-green-700 font-medium truncate mt-1">{b.facility_name.split('—')[0]}</div>
                </button>
              ))}
            </div>
          </div>

          {abhaPatient && (
            <div className="p-3 bg-green-50/60 border border-green-300 text-xs space-y-1.5">
              <div className="flex justify-between items-center font-semibold text-green-950">
                <span>✓ {abhaPatient.name} ({abhaPatient.age}y, {abhaPatient.gender})</span>
                <span className="font-mono text-[10px] bg-green-200 px-1.5 py-0.5 rounded text-green-900">{abhaPatient.beneficiary_scheme}</span>
              </div>
              <div className="text-ink-muted text-[11px]">{abhaPatient.facility_name}</div>
              <div className="text-green-900 text-[11px]">
                <b>Verified FHIR Conditions:</b> {abhaPatient.fhir_conditions.map(c => c.display).join(', ')}
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Tab 3: Imaging Scan (CV Module) */}
      {activeTab === 'scan' && (
        <Card className="p-4 border-l-4 border-l-quantum bg-surface space-y-3">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm text-ink">Multi-Modal Imaging (CV Module)</span>
                <Badge variant="default" className="text-[10px] font-mono uppercase bg-purple-100 text-purple-800">X-Ray / Echo / MRI</Badge>
              </div>
              <p className="text-xs text-ink-muted mt-1">
                Extract radiomic features directly from imaging studies into standardized HPO terms.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => scanFileInputRef.current?.click()}
                disabled={isScanning || isLoading}
                className="text-xs py-1.5 px-3 border-quantum/40 hover:border-quantum text-quantum-text"
              >
                {isScanning ? 'Analyzing Scan...' : '📁 Upload Scan'}
              </Button>
            </div>
          </div>

          {/* Quick preset scan triggers */}
          <div className="flex items-center gap-2 pt-1 border-t border-border">
            <span className="text-[11px] text-ink-muted">Quick Presets:</span>
            <button
              type="button"
              onClick={() => handlePresetScan('chest_xray')}
              disabled={isScanning}
              className="text-[11px] text-ink-muted hover:text-primary px-2 py-0.5 bg-gray-100 hover:bg-gray-200 rounded"
            >
              + Chest X-Ray (Pectus)
            </button>
            <button
              type="button"
              onClick={() => handlePresetScan('echo')}
              disabled={isScanning}
              className="text-[11px] text-ink-muted hover:text-primary px-2 py-0.5 bg-gray-100 hover:bg-gray-200 rounded"
            >
              + Echocardiogram (Aortic Root)
            </button>
            <button
              type="button"
              onClick={() => handlePresetScan('mra')}
              disabled={isScanning}
              className="text-[11px] text-ink-muted hover:text-primary px-2 py-0.5 bg-gray-100 hover:bg-gray-200 rounded"
            >
              + MRA (Arterial Tortuosity)
            </button>
          </div>

          {scanError && (
            <div className="p-2.5 bg-red-50 border border-red-200 text-red-700 text-xs">
              Imaging Scan Error: {scanError}
            </div>
          )}

          {scanResult && scanResult.findings.length > 0 && (
            <div className="p-3 bg-purple-50/50 border border-purple-200 space-y-2 text-xs">
              <div className="flex justify-between items-center text-purple-900 font-semibold">
                <span>Radiomic Findings Detected ({scanResult.modality_detected})</span>
                <span className="font-mono text-ink-muted">Appended to Symptom Vector</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                {scanResult.findings.map((f, i) => (
                  <div key={i} className="bg-white p-2 border border-purple-200 flex flex-col justify-between">
                    <div className="flex justify-between items-start mb-0.5">
                      <span className="font-medium text-xs text-ink">{f.label}</span>
                      <Badge variant="primary" className="text-[10px]">
                        {(f.confidence * 100).toFixed(0)}%
                      </Badge>
                    </div>
                    <div className="text-[10px] text-ink-muted font-mono">{f.hpo_id} • {f.anatomical_region}</div>
                    <div className="text-[10px] text-ink-muted mt-0.5">{f.clinical_evidence}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Main Clinical Notes & Ingested Phenotypes Form */}
      <Card className="p-6 border-l-4 border-l-primary">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <div className="flex justify-between items-center mb-2">
              <label htmlFor="symptoms" className="block text-sm font-semibold tracking-wide uppercase text-ink-muted">
                Clinical Notes, Symptoms & Extracted Phenotypes
              </label>
              {symptoms.length > 0 && (
                <button
                  type="button"
                  onClick={() => setSymptoms('')}
                  className="text-xs text-red-600 hover:underline"
                >
                  Clear Notes
                </button>
              )}
            </div>
            <textarea
              id="symptoms"
              rows={8}
              className="w-full border border-border p-4 rounded-none focus:ring-2 focus:ring-primary focus:border-primary outline-none text-[14px] font-serif leading-relaxed text-ink"
              placeholder="e.g. 28yo male presenting with pectus excavatum, arachnodactyly, and recent echocardiogram showing aortic root dilation (Z-score 2.9). Slit-lamp biomicroscopy reveals ectopia lentis."
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
            />
          </div>

          <div className="flex justify-between items-center pt-2">
            <Button type="button" variant="outline" onClick={handleDemo} disabled={isLoading || isScanning || isExtractingEhr}>
              Fast-Track Demo Case
            </Button>
            <Button type="submit" disabled={isLoading || isScanning || isExtractingEhr || !symptoms.trim()}>
              {isLoading ? 'Processing...' : 'Process with NLP →'}
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
};
