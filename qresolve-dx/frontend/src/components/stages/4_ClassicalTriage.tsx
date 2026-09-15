import { useLocation, useNavigate } from 'react-router-dom';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';

export const ClassicalTriage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const triageResult = location.state?.triageResult;

  if (!triageResult) {
    return <div className="p-8">No triage data. <Button onClick={() => navigate(-1)}>Go Back</Button></div>;
  }

  const { ranked_diagnoses, is_hard_case, case_id } = triageResult;

  const handleNext = () => {
    if (is_hard_case) {
      navigate(`/case/${case_id}/confusion`, { state: { triageResult } });
    } else {
      navigate(`/case/${case_id}/explain`, { state: { triageResult } });
    }
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div>
        <h2 className="text-xl font-semibold text-ink">4. Classical Triage Result</h2>
        <p className="text-ink-muted text-sm mt-1">LightGBM ranked differential diagnosis.</p>
      </div>

      <Card className="p-0 border-l-4 border-l-primary overflow-hidden shadow-sm">
        <div className="bg-primary-soft/30 border-b border-border px-6 py-4 flex justify-between items-center">
          <h3 className="text-sm font-semibold text-primary uppercase tracking-wide">Ranked Candidates</h3>
        </div>
        <div className="divide-y divide-border bg-surface">
          {ranked_diagnoses.slice(0, 5).map((diag: any, idx: number) => {
            const isTop = idx === 0;
            const isRunnerUp = idx === 1;
            return (
              <div key={idx} className={`p-4 px-6 flex items-center justify-between ${isTop ? 'bg-primary-soft/10' : ''}`}>
                <div className="flex items-center gap-4">
                  <span className="text-ink-muted font-mono w-4">{diag.rank}.</span>
                  <span className={`font-semibold ${isTop ? 'text-primary' : 'text-ink'}`}>{diag.disease}</span>
                  {isTop && <Badge variant="primary">Lead Hypothesis</Badge>}
                  {isRunnerUp && <Badge variant="default">Close Rival</Badge>}
                </div>
                <div className="font-mono text-sm text-ink-muted">
                  {(diag.probability * 100).toFixed(1)}%
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      {!is_hard_case && (
        <Card className="p-5 border-l-4 border-l-primary bg-primary-soft/40 shadow-sm mt-4">
          <div className="flex items-start gap-4">
            <div className="text-primary text-xl mt-0.5 font-bold">✓</div>
            <div>
              <h4 className="font-semibold text-ink text-[15px]">Classical ML Sufficient</h4>
              <p className="text-[14px] text-ink-muted mt-1.5 leading-relaxed font-serif">
                The classical XGBoost/LightGBM models have achieved a high confidence diagnostic margin. 
                As demonstrated in our methodology, <strong className="text-primary">Quantum SVM resolution is not required</strong> when the probability gap is this large. 
                Proceeding directly to the Explainability layer.
              </p>
            </div>
          </div>
        </Card>
      )}

      <div className="flex justify-between items-center pt-4">
        <Button variant="outline" onClick={() => navigate(-1)}>← Back</Button>
        <Button onClick={handleNext}>
          {is_hard_case ? 'Review Confusion Detection →' : 'View Explainable Output →'}
        </Button>
      </div>
    </div>
  );
};
