// Nodes
// (:Disease {id, name, omim_id, orpha_id})
// (:Symptom {hpo_id, label})
// (:Gene {symbol, hgnc_id})

// Edges
// (:Disease)-[:HAS_SYMPTOM {frequency: float}]->(:Symptom)
// (:Disease)-[:CAUSED_BY]->(:Gene)
// (:Disease)-[:LOOKS_LIKE {similarity: float, distinguishing_features: [hpo_id]}]->(:Disease)

// Constraints
CREATE CONSTRAINT FOR (d:Disease) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT FOR (s:Symptom) REQUIRE s.hpo_id IS UNIQUE;
CREATE CONSTRAINT FOR (g:Gene) REQUIRE g.symbol IS UNIQUE;
