import sys, os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.disease_data import (
    ALL_DISEASES, HPO_TERMS, GENE_DISEASE_MAP,
    PAIRWISE_DISTINGUISHING, KNOWN_CONFUSION_PAIRS,
    DiseaseProfile
)

@dataclass
class DiseaseNode:
    id: str
    name: str
    omim_id: str
    orpha_id: str

@dataclass
class SymptomNode:
    hpo_id: str
    label: str

@dataclass
class GeneNode:
    symbol: str
    hgnc_id: str

class KnowledgeGraph:
    def __init__(self):
        self.diseases: Dict[str, DiseaseNode] = {}
        self.symptoms: Dict[str, SymptomNode] = {}
        self.genes: Dict[str, GeneNode] = {}
        
        # Edges
        self.disease_symptoms: Dict[str, List[Tuple[str, float]]] = {}
        self.disease_genes: Dict[str, List[str]] = {}
        self.gene_diseases: Dict[str, List[str]] = {}
        self.disease_looks_like: Dict[str, List[Tuple[str, float, List[str]]]] = {}

    def build_from_disease_data(self):
        """Populates the knowledge graph from the disease_data module."""
        # Create symptoms
        for hpo_id, label in HPO_TERMS.items():
            self.symptoms[hpo_id] = SymptomNode(hpo_id=hpo_id, label=label)
            
        # Create genes
        for gene_symbol, diseases in GENE_DISEASE_MAP.items():
            self.genes[gene_symbol] = GeneNode(symbol=gene_symbol, hgnc_id="")
            
        # Create diseases and has_symptom / caused_by
        for disease in ALL_DISEASES:
            disease_id = disease.disease_id
            self.diseases[disease_id] = DiseaseNode(
                id=disease_id,
                name=disease.name,
                omim_id=disease.omim_id,
                orpha_id=getattr(disease, "orpha_id", "") or ""
            )
            
            self.disease_symptoms[disease_id] = []
            for hpo_id, freq in disease.symptoms.items():
                self.disease_symptoms[disease_id].append((hpo_id, freq))
                
            self.disease_genes[disease_id] = []
            
        for gene_symbol, diseases in GENE_DISEASE_MAP.items():
            for disease_name in diseases:
                # Need to map disease_name to disease_id
                disease_id = None
                for d in ALL_DISEASES:
                    if d.name == disease_name:
                        disease_id = d.disease_id
                        break
                
                if disease_id and disease_id in self.diseases:
                    self.disease_genes[disease_id].append(gene_symbol)
                    if gene_symbol not in self.gene_diseases:
                        self.gene_diseases[gene_symbol] = []
                    self.gene_diseases[gene_symbol].append(disease_id)
                    
        # Compute Looks_Like
        self.disease_looks_like = {d: [] for d in self.diseases}
        disease_ids = list(self.diseases.keys())
        for i in range(len(disease_ids)):
            for j in range(i + 1, len(disease_ids)):
                d1 = disease_ids[i]
                d2 = disease_ids[j]
                sim = self.compute_jaccard_similarity(d1, d2)
                
                dist_features = self.get_distinguishing_features(d1, d2)
                    
                self.disease_looks_like[d1].append((d2, sim, dist_features))
                self.disease_looks_like[d2].append((d1, sim, dist_features))

    def get_disease_symptoms(self, disease_id: str) -> List[Tuple[str, str, float]]:
        """Returns symptoms for a disease as (hpo_id, label, frequency)."""
        res = []
        for hpo_id, freq in self.disease_symptoms.get(disease_id, []):
            label = self.symptoms[hpo_id].label if hpo_id in self.symptoms else "Unknown"
            res.append((hpo_id, label, freq))
        return res

    def get_similar_diseases(self, disease_id: str) -> List[Tuple[str, float]]:
        """Returns diseases sorted by Jaccard similarity."""
        similar = []
        for d2, sim, _ in self.disease_looks_like.get(disease_id, []):
            similar.append((d2, sim))
        return sorted(similar, key=lambda x: x[1], reverse=True)

    def get_distinguishing_features(self, d1: str, d2: str) -> List[dict]:
        """Returns the distinguishing features between two diseases."""
        pair_key1 = f"{d1}-{d2}"
        pair_key2 = f"{d2}-{d1}"
        if pair_key1 in PAIRWISE_DISTINGUISHING:
            return PAIRWISE_DISTINGUISHING[pair_key1]
        elif pair_key2 in PAIRWISE_DISTINGUISHING:
            return PAIRWISE_DISTINGUISHING[pair_key2]
        return []

    def compute_jaccard_similarity(self, d1: str, d2: str) -> float:
        """Jaccard similarity of symptom sets, weighted by frequency (min/max)."""
        symptoms1 = dict(self.disease_symptoms.get(d1, []))
        symptoms2 = dict(self.disease_symptoms.get(d2, []))
        
        all_symptoms = set(symptoms1.keys()).union(set(symptoms2.keys()))
        if not all_symptoms:
            return 0.0
            
        intersection_weight = 0.0
        union_weight = 0.0
        
        for hpo in all_symptoms:
            f1 = symptoms1.get(hpo, 0.0)
            f2 = symptoms2.get(hpo, 0.0)
            intersection_weight += min(f1, f2)
            union_weight += max(f1, f2)
            
        return intersection_weight / union_weight if union_weight > 0 else 0.0

    def get_gene_diseases(self, gene: str) -> List[str]:
        """Returns diseases caused by a gene."""
        return self.gene_diseases.get(gene, [])

    def summary(self) -> str:
        """Returns graph statistics."""
        return (f"Knowledge Graph Summary:\n"
                f"- Diseases: {len(self.diseases)}\n"
                f"- Symptoms: {len(self.symptoms)}\n"
                f"- Genes: {len(self.genes)}\n"
                f"- Has Symptom Edges: {sum(len(v) for v in self.disease_symptoms.values())}\n"
                f"- Caused By Edges: {sum(len(v) for v in self.disease_genes.values())}\n"
                f"- Looks Like Edges: {sum(len(v) for v in self.disease_looks_like.values())}")

if __name__ == '__main__':
    kg = KnowledgeGraph()
    kg.build_from_disease_data()
    print(kg.summary())
    
    print("\nTop Confusion Pairs (by Jaccard similarity):")
    seen = set()
    confusion_pairs = []
    for d1, similarities in kg.disease_looks_like.items():
        for d2, sim, _ in similarities:
            pair = tuple(sorted([d1, d2]))
            if pair not in seen:
                seen.add(pair)
                confusion_pairs.append((pair, sim))
                
    confusion_pairs = sorted(confusion_pairs, key=lambda x: x[1], reverse=True)
    for pair, sim in confusion_pairs[:5]:
        d1_name = kg.diseases[pair[0]].name
        d2_name = kg.diseases[pair[1]].name
        print(f"{d1_name} vs {d2_name}: {sim:.4f}")
