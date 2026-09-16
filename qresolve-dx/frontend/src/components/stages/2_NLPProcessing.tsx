import { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';
import { api } from '../../lib/api';
import type { NLPResult } from '../../types';

export const NLPProcessing = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const rawText = location.state?.rawText || '';
  
  const [extractedTerms, setExtractedTerms] = useState<NLPResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const extractTerms = async () => {
      try {
        if (!rawText) {
          setLoading(false);
          return;
        }
        const terms = await api.extractNLP(rawText);
        setExtractedTerms(terms);
      } catch (err: any) {
        setError(err.message || 'Failed to extract terms');
      } finally {
        setLoading(false);
      }
    };
    
    extractTerms();
  }, [rawText]);

  const handleConfirm = (index: number) => {
    const newTerms = [...extractedTerms];
    newTerms[index].confirmed = !newTerms[index].confirmed;
    setExtractedTerms(newTerms);
  };

  const handleGenerateGraph = () => {
    const confirmedTerms = extractedTerms.filter(t => t.confirmed).map(t => t.hpo_id);
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
          {loading && <div className="p-6 text-ink-muted">Extracting terms from text...</div>}
          {error && <div className="p-6 text-red-600">Error: {error}</div>}
          {!loading && !error && extractedTerms.length === 0 && (
            <div className="p-6 text-ink-muted">No HPO terms identified in the text.</div>
          )}
          {!loading && !error && extractedTerms.map((term, idx) => (
            <div key={idx} className={`p-4 px-6 flex items-center justify-between transition-colors ${!term.confirmed ? 'bg-gray-50/50 opacity-75' : ''}`}>
              <label className="flex items-center gap-4 cursor-pointer flex-1">
                <input
                  type="checkbox"
                  checked={term.confirmed}
                  onChange={() => handleConfirm(idx)}
                  className="w-4 h-4 text-primary rounded border-gray-300 focus:ring-primary cursor-pointer"
                />
                <span className={`font-medium ${term.confirmed ? 'text-ink' : 'text-ink-muted line-through'}`}>
                  {term.label} <span className="text-xs text-ink-muted ml-2">({term.hpo_id})</span>
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
