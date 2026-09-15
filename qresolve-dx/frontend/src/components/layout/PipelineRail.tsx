import { useLocation } from 'react-router-dom';

const STAGES = [
  { path: '/input', label: 'Patient Input' },
  { path: '/nlp', label: 'NLP & Normalization' },
  { path: '/graph', label: 'Knowledge Graph' },
  { path: '/triage', label: 'Classical Triage' },
  { path: '/confusion', label: 'Confusion Detection' },
  { path: '/quantum', label: 'Quantum Resolver' },
  { path: '/explain', label: 'Explainable Output' },
  { path: '/test', label: 'Next Test' }
];

export const PipelineRail = () => {
  const location = useLocation();
  const currentIndex = STAGES.findIndex(s => location.pathname.includes(s.path));

  return (
    <div className="w-64 border-r border-border h-screen bg-bg p-6 flex flex-col fixed left-0 top-0">
      <h2 className="text-sm font-semibold text-ink mb-8 uppercase tracking-wide">QResolve Pipeline</h2>
      <div className="flex flex-col gap-4">
        {STAGES.map((step, index) => {
          const isPast = index < currentIndex;
          const isCurrent = index === currentIndex;
          const isQuantum = step.path === '/quantum';
          
          return (
            <div key={step.path} className={`flex items-center gap-3 ${isCurrent ? 'text-ink font-medium' : isPast ? 'text-ink-muted' : 'text-gray-300'}`}>
              <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${isCurrent ? (isQuantum ? 'bg-quantum' : 'bg-primary') : isPast ? 'bg-gray-300' : 'bg-gray-200'}`} />
              <span className={`text-[13px] ${isCurrent && isQuantum ? 'text-quantum-text' : ''}`}>{step.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
