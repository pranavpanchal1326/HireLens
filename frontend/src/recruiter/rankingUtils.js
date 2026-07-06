import { FEATURE_LABELS } from '../lib/apertureConfidence';

// Maps weight keys → API feature_vector field names. Weights let a recruiter tilt
// the ranking (§10.7, e.g. skills 2× experience). Applied client-side over the
// returned feature vectors — honest, since every feature is already present.
const FIELD = {
  tfidf: 'tfidf_score',
  embedding: 'embedding_score',
  skillOverlap: 'skill_overlap_pct',
  expMatch: 'exp_match',
  eduMatch: 'edu_match',
};

export function weightedScore(fv, weights) {
  let num = 0, den = 0;
  for (const key of Object.keys(FIELD)) {
    const w = weights[key] ?? 1;
    num += w * (fv[FIELD[key]] ?? 0);
    den += w;
  }
  return den ? Math.round((num / den) * 100) : 0;
}

// Top-N driving features (highest weighted contribution) for the table's
// "why this rank" column and the SR summary.
export function topDrivers(fv, weights, n = 2) {
  return Object.keys(FIELD)
    .map((key) => ({
      key,
      label: FEATURE_LABELS[key],
      contribution: (weights[key] ?? 1) * (fv[FIELD[key]] ?? 0),
      value: fv[FIELD[key]] ?? 0,
    }))
    .sort((a, b) => b.contribution - a.contribution)
    .slice(0, n);
}

// Re-rank the candidate list; reassign 1..N. With UNIFORM weights we defer to the
// backend's authoritative final_score (so the default order matches the Score
// column). Once a recruiter tilts weights, we re-rank by weighted contribution.
export function reRank(candidates, weights) {
  const vals = Object.values(weights);
  const uniform = vals.every((w) => w === vals[0]);
  return [...candidates]
    .map((c) => ({ ...c, weighted: weightedScore(c.score_result.feature_vector, weights) }))
    .sort((a, b) =>
      uniform
        ? b.score_result.final_score - a.score_result.final_score
        : b.weighted - a.weighted || b.score_result.final_score - a.score_result.final_score,
    )
    .map((c, i) => ({ ...c, rank: i + 1 }));
}
