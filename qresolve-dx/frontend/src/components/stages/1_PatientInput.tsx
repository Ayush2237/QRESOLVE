import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '../common/Card';
import { Button } from '../common/Button';

export const PatientInput = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [symptoms, setSymptoms] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const getDemoText = (caseId?: string) => {
    switch (caseId) {
      case 'CASE-BC01': return "52yo female presenting with a new irregular breast mass. Mammogram shows clustered microcalcifications. No family history of breast cancer.";
      case 'CASE-PD01': return "68yo male with unilateral resting tremor in right hand, bradykinesia, and rigidity. No history of neuroleptic medication use.";
      case 'CASE-MF01': return "28yo male presenting with pectus excavatum, arachnodactyly, and recent echocardiogram showing aortic root dilation (Z-score 2.9). No family history of sudden cardiac death.";
      case 'CASE-ED01': return "41yo female with severe joint hypermobility, translucent skin, and history of joint dislocations. Beighton score of 8/9.";
      case 'CASE-LD01': return "19yo male with arterial tortuosity, hypertelorism, and bifid uvula. Echocardiogram reveals ascending aortic aneurysm.";
      default: return "28yo male presenting with pectus excavatum, arachnodactyly, and recent echocardiogram showing aortic root dilation (Z-score 2.9). No family history of sudden cardiac death.";
    }
  };

  const handleDemo = () => {
    const demoText = getDemoText(id);
    setSymptoms(demoText);
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      navigate(`/case/${id || 'new'}/nlp`, { state: { rawText: demoText } });
    }, 400);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symptoms.trim()) return;

    setIsLoading(true);
    // Simulate short delay then pass raw text to NLP stage
    setTimeout(() => {
      setIsLoading(false);
      navigate(`/case/${id || 'new'}/nlp`, { state: { rawText: symptoms } });
    }, 400);
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div>
        <h2 className="text-xl font-semibold text-ink">1. Patient Input</h2>
        <p className="text-ink-muted text-sm mt-1">Enter clinical notes or free-text symptoms to begin the diagnostic pipeline.</p>
      </div>

      <Card className="p-6 border-l-4 border-l-primary">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label htmlFor="symptoms" className="block text-sm font-semibold tracking-wide uppercase text-ink-muted mb-2">Clinical Notes & Symptoms</label>
            <textarea
              id="symptoms"
              rows={8}
              className="w-full border border-border p-4 rounded-none focus:ring-2 focus:ring-primary focus:border-primary outline-none text-[15px] font-serif leading-relaxed text-ink"
              placeholder="e.g. 28yo male presenting with pectus excavatum, arachnodactyly, and recent echocardiogram showing aortic root dilation (Z-score 2.9). No family history of sudden cardiac death."
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
            />
          </div>

          <div className="flex justify-between items-center pt-2">
            <Button type="button" variant="outline" onClick={handleDemo} disabled={isLoading}>
              Fast-Track Demo Case
            </Button>
            <Button type="submit" disabled={isLoading || !symptoms.trim()}>
              {isLoading ? 'Processing...' : 'Process with NLP →'}
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
};
