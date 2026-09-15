import { useLocation, useNavigate } from 'react-router-dom';
import { Card } from '../common/Card';
import { Button } from '../common/Button';

export const NextTestRecommendation = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const explainData = location.state?.explainData;

  if (!explainData) return <div className="p-8">No explain data available. <Button onClick={() => navigate(-1)}>Go Back</Button></div>;

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div>
        <h2 className="text-xl font-semibold text-ink">8. Next Test Recommendation</h2>
        <p className="text-ink-muted text-sm mt-1">Suggested clinical tests optimized for information gain to finalize diagnosis.</p>
      </div>

      <div className="space-y-4">
        {explainData.suggested_tests.map((test: any, idx: number) => (
          <Card key={idx} className={`p-6 ${idx === 0 ? 'border-l-4 border-l-primary bg-primary-soft/10 shadow-sm' : 'opacity-80'}`}>
            <div className="flex justify-between items-start">
              <div>
                <div className="text-xs text-primary uppercase tracking-wider font-semibold mb-2">
                  {idx === 0 ? 'Highest Info Gain' : `Alternative ${idx}`}
                </div>
                <h3 className="font-semibold text-ink text-lg">{test.clinical_test || test.label}</h3>
                <p className="text-ink-muted text-[15px] mt-2 font-serif">To confirm presence/absence of: <strong>{test.label}</strong></p>
              </div>
              <div className="text-right">
                <div className="text-xs text-ink-muted uppercase font-semibold mb-1">Gain</div>
                <div className="font-mono text-primary font-bold bg-white border border-border px-2 py-1">
                  {test.information_gain.toFixed(3)}
                </div>
              </div>
            </div>
          </Card>
        ))}
        
        {explainData.suggested_tests.length === 0 && (
          <Card className="p-8 text-center bg-gray-50 border-dashed border-2">
            <p className="text-ink-muted font-serif">No further tests mathematically recommended for this specific differential.</p>
          </Card>
        )}
      </div>

      <div className="flex justify-between items-center pt-4">
        <Button variant="outline" onClick={() => navigate(-1)}>← Back</Button>
        <Button onClick={() => navigate('/')}>Complete Case & Return to Dashboard</Button>
      </div>
    </div>
  );
};
