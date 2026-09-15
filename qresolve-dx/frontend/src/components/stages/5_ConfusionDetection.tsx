import { useLocation, useNavigate } from 'react-router-dom';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';
import { useState } from 'react';

export const ConfusionDetection = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const triageResult = location.state?.triageResult;
  const [isEscalating, setIsEscalating] = useState(false);

  if (!triageResult || !triageResult.is_hard_case) {
    return <div className="p-8">No confusion detected for this case. <Button onClick={() => navigate(-1)}>Go Back</Button></div>;
  }

  const handleEscalate = () => {
    setIsEscalating(true);
    // Simulate quantum escalation time then navigate to quantum view
    setTimeout(() => {
      navigate(`/case/${triageResult.case_id}/quantum`, { state: { triageResult } });
    }, 1500);
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div>
        <h2 className="text-xl font-semibold text-ink flex items-center gap-3">
          5. Confusion Detection
          <Badge variant="warning">Ambiguity Flagged</Badge>
        </h2>
        <p className="text-ink-muted text-sm mt-1">The classical model cannot confidently separate the top two diagnoses.</p>
      </div>

      <Card className="p-6 border-l-4 border-l-warning bg-warning-soft/20">
        <h3 className="font-semibold text-warning-text mb-6">Diagnostic Conflict</h3>
        
        <div className="flex items-center gap-6 justify-center py-6 bg-surface border border-border shadow-sm">
          <div className="text-center min-w-[140px]">
            <div className="text-xs text-ink-muted uppercase tracking-wide font-semibold mb-1">Rank 1</div>
            <div className="font-semibold text-ink text-lg">{triageResult.top_diagnosis}</div>
            <div className="text-sm font-mono text-ink-muted mt-1">{(triageResult.ranked_diagnoses[0].probability * 100).toFixed(1)}%</div>
          </div>
          
          <div className="text-warning-text font-bold text-xl px-6 border-x border-border/50">VS</div>
          
          <div className="text-center min-w-[140px]">
            <div className="text-xs text-ink-muted uppercase tracking-wide font-semibold mb-1">Rank 2</div>
            <div className="font-semibold text-ink text-lg">{triageResult.runner_up}</div>
            <div className="text-sm font-mono text-ink-muted mt-1">{(triageResult.ranked_diagnoses[1].probability * 100).toFixed(1)}%</div>
          </div>
        </div>

        <p className="text-[15px] text-ink-muted mt-6 font-serif leading-relaxed">
          The probability margin is critically narrow. Classical features mapped to these phenotypes exhibit high cross-correlation. Escalation to the Quantum Kernel Resolver is required to compute nonlinear feature separation in a higher-dimensional Hilbert space.
        </p>
      </Card>

      <div className="flex justify-between items-center pt-4">
        <Button variant="outline" onClick={() => navigate(-1)}>← Back</Button>
        <Button onClick={handleEscalate} variant="quantum" disabled={isEscalating}>
          {isEscalating ? 'Initializing QSVM Kernel...' : 'Escalate to Quantum Resolver →'}
        </Button>
      </div>
    </div>
  );
};
