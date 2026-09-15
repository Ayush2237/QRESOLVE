import { useNavigate } from 'react-router-dom';

const BENCHMARKS = [
  {
    id: 'CASE-BC01',
    label: 'Breast Cancer',
    subtitle: 'Ductal Carcinoma Screening',
    description: '52F — Irregular mass, microcalcifications on mammogram.',
    track: 'classical' as const,
    confidence: '92%',
  },
  {
    id: 'CASE-PD01',
    label: "Parkinson's Disease",
    subtitle: 'Movement Disorder Assessment',
    description: '68M — Unilateral resting tremor, bradykinesia, rigidity.',
    track: 'classical' as const,
    confidence: '88%',
  },
  {
    id: 'CASE-MF01',
    label: 'Marfan Syndrome',
    subtitle: 'Connective Tissue Disorder',
    description: '28M — Pectus excavatum, arachnodactyly, aortic root dilation.',
    track: 'quantum' as const,
    confidence: '49%',
  },
  {
    id: 'CASE-ED01',
    label: 'Ehlers-Danlos Syndrome',
    subtitle: 'Hypermobility Spectrum',
    description: '41F — Joint hypermobility, translucent skin, dislocations.',
    track: 'quantum' as const,
    confidence: '49%',
  },
  {
    id: 'CASE-LD01',
    label: 'Loeys-Dietz Syndrome',
    subtitle: 'Vascular / Skeletal Overlap',
    description: '19M — Arterial tortuosity, hypertelorism, bifid uvula.',
    track: 'quantum' as const,
    confidence: '48%',
  },
];

export const DoctorDashboard = () => {
  const navigate = useNavigate();

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
          <h2 className="text-lg font-semibold text-ink mb-2">Benchmark Cases</h2>
          <p className="text-sm text-ink-muted leading-relaxed max-w-2xl">
            Select a demo case to walk through the full diagnostic pipeline.
            <strong className="text-ink"> Classical ML</strong> cases resolve directly via XGBoost / LightGBM.
            <strong className="text-primary"> Rare disease</strong> cases with ambiguous margins are escalated to the Quantum SVM resolver.
          </p>
        </div>

        {/* Classical ML Section */}
        <section className="mb-10">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-3 h-3 rounded-full bg-primary"></div>
            <h3 className="text-sm font-semibold uppercase tracking-wide text-primary">Classical ML — High Confidence</h3>
            <div className="flex-1 h-px bg-border"></div>
            <span className="text-[11px] text-ink-muted font-mono">QUANTUM NOT REQUIRED</span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {BENCHMARKS.filter(b => b.track === 'classical').map(bench => (
              <button
                key={bench.id}
                onClick={() => navigate(`/case/${bench.id}/input`)}
                className="group text-left bg-surface border border-border p-5 hover:border-primary hover:shadow-[0_0_0_1px_var(--color-primary)] transition-all duration-150"
              >
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <div className="font-semibold text-ink text-[15px] group-hover:text-primary transition-colors">{bench.label}</div>
                    <div className="text-xs text-ink-muted mt-0.5">{bench.subtitle}</div>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <span className="text-xs font-mono px-2 py-0.5 bg-primary-soft text-primary font-medium">Classical</span>
                    <span className="text-[11px] text-ink-muted font-mono">{bench.confidence} conf.</span>
                  </div>
                </div>
                <p className="text-sm text-ink-muted font-serif leading-relaxed">{bench.description}</p>
                <div className="mt-3 pt-3 border-t border-border flex items-center gap-2 text-xs text-ink-muted">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary inline-block"></span>
                  XGBoost / LightGBM → Explainability → Report
                </div>
              </button>
            ))}
          </div>
        </section>

        {/* Quantum ML Section */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-3 h-3 rounded-full bg-quantum"></div>
            <h3 className="text-sm font-semibold uppercase tracking-wide text-quantum-text">Rare Disease — Quantum Escalation</h3>
            <div className="flex-1 h-px bg-border"></div>
            <span className="text-[11px] text-ink-muted font-mono">QSVM REQUIRED</span>
          </div>

          <div className="grid grid-cols-3 gap-4">
            {BENCHMARKS.filter(b => b.track === 'quantum').map(bench => (
              <button
                key={bench.id}
                onClick={() => navigate(`/case/${bench.id}/input`)}
                className="group text-left bg-surface border border-border p-5 hover:border-quantum hover:shadow-[0_0_0_1px_var(--color-quantum)] transition-all duration-150"
              >
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <div className="font-semibold text-ink text-[15px] group-hover:text-quantum-text transition-colors">{bench.label}</div>
                    <div className="text-xs text-ink-muted mt-0.5">{bench.subtitle}</div>
                  </div>
                  <span className="text-xs font-mono px-2 py-0.5 bg-quantum-soft text-quantum-text font-medium">Quantum</span>
                </div>
                <p className="text-sm text-ink-muted font-serif leading-relaxed">{bench.description}</p>
                <div className="mt-3 pt-3 border-t border-border flex justify-between items-center text-xs text-ink-muted">
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-quantum inline-block"></span>
                    Classical → Confusion → QSVM → XAI
                  </div>
                  <span className="font-mono">{bench.confidence}</span>
                </div>
              </button>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
};
