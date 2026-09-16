import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { api } from '../../lib/api';
import type { ExplainResponse } from '../../types';

export const ExplainableOutput = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const triageResult = location.state?.triageResult;
  const [explainData, setExplainData] = useState<ExplainResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!triageResult) return;
    
    const fetchExplain = async () => {
      try {
        const data = await api.explain(triageResult.case_id);
        setExplainData(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchExplain();
  }, [triageResult]);

  if (!triageResult) return <div className="p-8">No case data found. <Button onClick={() => navigate(-1)}>Go Back</Button></div>;
  if (isLoading) return <div className="p-8 text-ink-muted font-serif">Loading SHAP explanations...</div>;
  if (error) return <div className="p-8 text-red-600 bg-red-50">Error fetching explanation: {error}</div>;
  if (!explainData) return null;

  const isQuantum = triageResult.quantum_used;
  
  // Calculate max importance for scaling bars
  const allEvidences = [...explainData.supporting_evidence, ...explainData.against_evidence];
  const maxImportance = Math.max(0.1, ...allEvidences.map(e => e.importance));

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div>
        <h2 className="text-xl font-semibold text-ink">7. Explainable Output</h2>
        <p className="text-ink-muted text-sm mt-1">SHAP value feature contributions and causal rationale.</p>
      </div>

      <Card className={`p-0 border-t-4 overflow-hidden ${isQuantum ? 'border-quantum' : 'border-primary'}`}>
        <div className="px-8 py-6 bg-surface">
          <div className="text-[13px] font-semibold text-ink uppercase tracking-wide mb-6">
            Evidence for {explainData.top_diagnosis} vs {explainData.runner_up}
          </div>
          
          <div className="space-y-8">
            <div>
              <h4 className="text-xs font-semibold text-green-700 uppercase tracking-wider mb-2">Supporting Evidence (SHAP Values)</h4>
              <div className="divide-y divide-border border border-border bg-gray-50/50">
                {explainData.supporting_evidence.map((ev, i) => (
                  <div key={i} className="py-3 px-4 flex justify-between items-center text-sm">
                    <span className="font-medium text-ink">{ev.label}</span>
                    <div className="flex items-center gap-3">
                      <span className="text-green-700 font-mono text-xs font-bold">+{ev.shap_value.toFixed(4)}</span>
                      <span className="text-ink-muted bg-white border border-border px-2 py-0.5 text-xs font-mono">{ev.direction}</span>
                    </div>
                  </div>
                ))}
                {explainData.supporting_evidence.length === 0 && <div className="py-3 px-4 text-sm text-ink-muted">No explicit supporting evidence found.</div>}
              </div>
            </div>

            <div>
              <h4 className="text-xs font-semibold text-red-700 uppercase tracking-wider mb-2">Counter Evidence (SHAP Values)</h4>
              <div className="divide-y divide-border border border-border bg-gray-50/50">
                {explainData.against_evidence.map((ev, i) => (
                  <div key={i} className="py-3 px-4 flex justify-between items-center text-sm">
                    <span className="font-medium text-ink">{ev.label}</span>
                    <div className="flex items-center gap-3">
                      <span className="text-red-700 font-mono text-xs font-bold">{ev.shap_value.toFixed(4)}</span>
                      <span className="text-ink-muted bg-white border border-border px-2 py-0.5 text-xs font-mono">{ev.direction}</span>
                    </div>
                  </div>
                ))}
                {explainData.against_evidence.length === 0 && <div className="py-3 px-4 text-sm text-ink-muted">No counter evidence found.</div>}
              </div>
            </div>
          </div>
        </div>
      </Card>

      <div className="flex justify-between items-center pt-4">
        <Button variant="outline" onClick={() => navigate(-1)}>← Back</Button>
        <Button onClick={() => navigate(`/case/${triageResult.case_id}/test`, { state: { explainData, triageResult } })}>
          View Next Test Recommendation →
        </Button>
      </div>
    </div>
  );
};
