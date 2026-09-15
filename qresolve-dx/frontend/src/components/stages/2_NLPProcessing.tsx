import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';

export const NLPProcessing = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [extractedTerms, setExtractedTerms] = useState<{ term: string, confirmed: boolean }[]>([]);

  useEffect(() => {
    // Simulate NLP extraction based on case type
    const termsByCase: Record<string, { term: string, confirmed: boolean }[]> = {
      'CASE-BC01': [
        { term: 'Irregular breast mass', confirmed: true },
        { term: 'Microcalcifications on mammogram', confirmed: true },
        { term: 'No family history of breast cancer', confirmed: true },
        { term: 'Skin dimpling', confirmed: false },
      ],
      'CASE-PD01': [
        { term: 'Unilateral resting tremor', confirmed: true },
        { term: 'Bradykinesia', confirmed: true },
        { term: 'Rigidity', confirmed: true },
        { term: 'Postural instability', confirmed: false },
      ],
      'CASE-MF01': [
        { term: 'Aortic root aneurysm', confirmed: true },
        { term: 'Pectus excavatum', confirmed: true },
        { term: 'Arachnodactyly', confirmed: true },
        { term: 'Ectopia lentis', confirmed: true },
      ],
      'CASE-ED01': [
        { term: 'Joint hypermobility', confirmed: true },
        { term: 'Skin translucency', confirmed: true },
        { term: 'Bruising susceptibility', confirmed: true },
        { term: 'Striae distensae', confirmed: true },
      ],
      'CASE-LD01': [
        { term: 'Arterial tortuosity', confirmed: true },
        { term: 'Hypertelorism', confirmed: true },
        { term: 'Bifid uvula', confirmed: true },
        { term: 'Aortic aneurysm', confirmed: true },
      ],
    };
    setExtractedTerms(termsByCase[id || ''] || termsByCase['CASE-MF01']);
  }, [id]);

  const handleConfirm = (index: number) => {
    const newTerms = [...extractedTerms];
    newTerms[index].confirmed = !newTerms[index].confirmed;
    setExtractedTerms(newTerms);
  };

  const handleGenerateGraph = () => {
    const confirmedTerms = extractedTerms.filter(t => t.confirmed).map(t => t.term);
    navigate(`/case/${id}/graph`, { state: { confirmedTerms } });
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-xl font-semibold text-ink">2. NLP & Normalization</h2>
          <p className="text-ink-muted text-sm mt-1">Review and confirm extracted HPO terms from clinical notes.</p>
        </div>
      </div>

      <Card className="p-0 border-t-4 border-primary overflow-hidden">
        <div className="bg-primary-soft/30 border-b border-border px-6 py-4 flex justify-between items-center">
          <h3 className="text-sm font-semibold text-primary uppercase tracking-wide">Extracted Phenotypes</h3>
          <Badge variant="primary">High Confidence</Badge>
        </div>
        <div className="divide-y divide-border bg-surface">
          {extractedTerms.map((term, idx) => (
            <div key={idx} className={`p-4 px-6 flex items-center justify-between transition-colors ${!term.confirmed ? 'bg-gray-50/50 opacity-75' : ''}`}>
              <label className="flex items-center gap-4 cursor-pointer flex-1">
                <input
                  type="checkbox"
                  checked={term.confirmed}
                  onChange={() => handleConfirm(idx)}
                  className="w-4 h-4 text-primary rounded border-gray-300 focus:ring-primary cursor-pointer"
                />
                <span className={`font-medium ${term.confirmed ? 'text-ink' : 'text-ink-muted line-through'}`}>
                  {term.term}
                </span>
              </label>
              {term.confirmed && <Badge variant="default" className="bg-gray-100 text-ink-muted text-[10px]">HPO Mapped</Badge>}
            </div>
          ))}
        </div>
      </Card>


      <div className="flex justify-between items-center pt-4">
        <Button variant="outline" onClick={() => navigate(-1)}>← Back to Input</Button>
        <Button onClick={handleGenerateGraph} disabled={!extractedTerms.some(t => t.confirmed)}>
          Generate Knowledge Graph →
        </Button>
      </div>
    </div>
  );
};
