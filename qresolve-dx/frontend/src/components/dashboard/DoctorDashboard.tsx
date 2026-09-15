import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../../lib/api';
import type { DiseaseInfo } from '../../types';

export const DoctorDashboard = () => {
  const navigate = useNavigate();
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDiseases = async () => {
      try {
        const data = await api.getDiseases();
        setDiseases(data);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch diseases');
      } finally {
        setLoading(false);
      }
    };
    fetchDiseases();
  }, []);

  return (
    <div className="min-h-screen bg-bg">
      {/* Header */}
      <header className="border-b border-border bg-surface">
        <div className="max-w-5xl mx-auto px-8 py-6 flex justify-between items-end">
          <div>
            <h1 className="text-2xl font-semibold text-ink tracking-tight">QResolve</h1>
            <p className="text-sm text-ink-muted mt-1">Hybrid Quantum-Classical Diagnostic Assistant</p>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-8 py-10">
        {/* Intro */}
        <div className="mb-10">
          <h2 className="text-lg font-semibold text-ink mb-2">Benchmark Cases & Diagnostics</h2>
          <p className="text-sm text-ink-muted leading-relaxed max-w-2xl">
            Select a disease category to walk through the full diagnostic pipeline.
            <strong className="text-ink"> Classical ML</strong> cases resolve directly via XGBoost / LightGBM.
            <strong className="text-primary"> Rare disease</strong> cases with ambiguous margins are escalated to the Quantum SVM resolver.
          </p>
        </div>

        {loading && <div className="text-ink-muted">Loading available diseases from backend...</div>}
        {error && <div className="text-red-600">Error: {error}</div>}

        {!loading && !error && (
          <>
            {/* Common Diseases Section */}
            <section className="mb-10">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                <h3 className="text-sm font-semibold uppercase tracking-wide text-blue-600">Common Diseases — Classical ML</h3>
                <div className="flex-1 h-px bg-border"></div>
                <span className="text-[11px] text-ink-muted font-mono">STANDALONE PIPELINE</span>
              </div>

              <div className="grid grid-cols-2 gap-4">
                {diseases.filter(d => d.category === 'common').map(disease => (
                  <button
                    key={disease.id}
                    onClick={() => navigate(`/common/${disease.id}`)}
                    className="group text-left bg-surface border border-border p-5 hover:border-blue-500 hover:shadow-[0_0_0_1px_var(--color-blue-500)] transition-all duration-150"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <div className="font-semibold text-ink text-[15px] group-hover:text-blue-600 transition-colors">{disease.name}</div>
                        <div className="text-xs text-ink-muted mt-0.5">{disease.description}</div>
                      </div>
                      <div className="flex flex-col items-end gap-1">
                        <span className="text-xs font-mono px-2 py-0.5 bg-blue-50 text-blue-600 font-medium">Common</span>
                      </div>
                    </div>
                    <p className="text-sm text-ink-muted font-serif leading-relaxed">Features: {disease.n_symptoms}</p>
                    <div className="mt-3 pt-3 border-t border-border flex items-center gap-2 text-xs text-ink-muted">
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-500 inline-block"></span>
                      Classical ML (Sklearn / XGBoost)
                    </div>
                  </button>
                ))}
              </div>
            </section>

            {/* Rare Diseases Section */}
            <section>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-3 h-3 rounded-full bg-quantum"></div>
                <h3 className="text-sm font-semibold uppercase tracking-wide text-quantum-text">Rare Diseases — Hybrid Pipeline</h3>
                <div className="flex-1 h-px bg-border"></div>
                <span className="text-[11px] text-ink-muted font-mono">5-DISEASE CLUSTER</span>
              </div>

              <div className="grid grid-cols-3 gap-4">
                {diseases.filter(d => d.category === 'rare').map(disease => (
                  <button
                    key={disease.id}
                    onClick={() => navigate(`/case/${disease.id}/input`)}
                    className={`group text-left bg-surface border border-border p-5 transition-all duration-150 ${disease.track === 'quantum' ? 'hover:border-quantum hover:shadow-[0_0_0_1px_var(--color-quantum)]' : 'hover:border-primary hover:shadow-[0_0_0_1px_var(--color-primary)]'}`}
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <div className={`font-semibold text-[15px] transition-colors ${disease.track === 'quantum' ? 'text-ink group-hover:text-quantum-text' : 'text-ink group-hover:text-primary'}`}>{disease.name}</div>
                        <div className="text-xs text-ink-muted mt-0.5">{disease.description}</div>
                      </div>
                      <span className={`text-xs font-mono px-2 py-0.5 font-medium ${disease.track === 'quantum' ? 'bg-quantum-soft text-quantum-text' : 'bg-primary-soft text-primary'}`}>
                        {disease.track === 'quantum' ? 'Quantum' : 'Classical'}
                      </span>
                    </div>
                    <p className="text-sm text-ink-muted font-serif leading-relaxed">Known HPO Symptoms: {disease.n_symptoms}</p>
                    <div className="mt-3 pt-3 border-t border-border flex justify-between items-center text-xs text-ink-muted">
                      <div className="flex items-center gap-2">
                        <span className={`w-1.5 h-1.5 rounded-full inline-block ${disease.track === 'quantum' ? 'bg-quantum' : 'bg-primary'}`}></span>
                        {disease.track === 'quantum' ? 'Classical → QSVM → XAI' : 'XGBoost → XAI'}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
};
