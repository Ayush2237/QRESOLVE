import { Outlet, useNavigate, useParams, useLocation } from 'react-router-dom';

const STAGES = [
  { path: 'input', label: '1. Patient Input', icon: '📋' },
  { path: 'nlp', label: '2. NLP & Normalization', icon: '🔤' },
  { path: 'graph', label: '3. Knowledge Graph', icon: '🕸️' },
  { path: 'triage', label: '4. Classical Triage', icon: '⚙️' },
  { path: 'confusion', label: '5. Confusion Detection', icon: '⚠️' },
  { path: 'quantum', label: '6. Quantum Resolver', icon: '⚛️' },
  { path: 'explain', label: '7. Explainable Output', icon: '📊' },
  { path: 'test', label: '8. Next Test', icon: '🧪' },
];

export const MainLayout = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const location = useLocation();
  const currentIndex = STAGES.findIndex(s => location.pathname.includes(s.path));

  return (
    <div className="min-h-screen bg-bg">
      {/* Top nav bar */}
      <header className="border-b border-border bg-surface sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-8 py-3 flex items-center justify-between">
          <button
            onClick={() => navigate('/')}
            className="text-sm font-semibold text-primary hover:text-blue-700 transition-colors flex items-center gap-2"
          >
            ← QResolve
          </button>
          <span className="text-xs font-mono text-ink-muted bg-bg px-3 py-1 border border-border">
            {id}
          </span>
        </div>

        {/* Pipeline progress bar */}
        <div className="max-w-6xl mx-auto px-8 pb-3 flex items-center gap-1 overflow-x-auto">
          {STAGES.map((stage, index) => {
            const isPast = index < currentIndex;
            const isCurrent = index === currentIndex;
            const isQuantum = stage.path === 'quantum' || stage.path === 'confusion';

            return (
              <div key={stage.path} className="flex items-center">
                <div
                  className={`
                    text-[11px] font-medium px-2.5 py-1 whitespace-nowrap transition-all
                    ${isCurrent
                      ? isQuantum
                        ? 'bg-quantum-soft text-quantum-text border border-quantum/30'
                        : 'bg-primary-soft text-primary border border-primary/30'
                      : isPast
                        ? 'text-ink-muted bg-gray-50'
                        : 'text-gray-300 bg-transparent'
                    }
                  `}
                >
                  {stage.label}
                </div>
                {index < STAGES.length - 1 && (
                  <div className={`w-4 h-px mx-0.5 ${isPast ? 'bg-primary/40' : 'bg-border'}`} />
                )}
              </div>
            );
          })}
        </div>
      </header>

      {/* Content area — full width, no sidebar */}
      <div className="max-w-4xl mx-auto p-10">
        <Outlet />
      </div>
    </div>
  );
};
