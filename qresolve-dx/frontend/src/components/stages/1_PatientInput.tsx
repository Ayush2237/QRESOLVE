import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { api } from '../../lib/api';
import type { DiseaseInfo } from '../../types';

export const PatientInput = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [symptoms, setSymptoms] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [disease, setDisease] = useState<DiseaseInfo | null>(null);

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symptoms.trim()) return;

    setIsLoading(true);
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
        {disease && (
          <div className="mt-2 text-sm text-primary">Testing pathway for: {disease.name}</div>
        )}
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
