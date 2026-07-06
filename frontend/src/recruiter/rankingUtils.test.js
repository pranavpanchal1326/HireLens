import { describe, it, expect } from 'vitest';
import { weightedScore, topDrivers, reRank } from './rankingUtils';
import { DEFAULT_WEIGHTS } from './RecruiterContext';

const fv = (o) => ({ tfidf_score: 0, embedding_score: 0, skill_overlap_pct: 0, exp_match: 0, edu_match: 0, ...o });
const cand = (id, final, feat) => ({ candidate_id: id, rank: 0, score_result: { final_score: final, feature_vector: fv(feat) } });

describe('weightedScore', () => {
  it('averages features under uniform weights and scales to 0-100', () => {
    expect(weightedScore(fv({ skill_overlap_pct: 1 }), DEFAULT_WEIGHTS)).toBe(20); // 1 of 5 features full
    expect(weightedScore(fv({ tfidf_score: 1, embedding_score: 1, skill_overlap_pct: 1, exp_match: 1, edu_match: 1 }), DEFAULT_WEIGHTS)).toBe(100);
    expect(weightedScore(fv({}), DEFAULT_WEIGHTS)).toBe(0);
  });

  it('respects tilted weights', () => {
    const w = { ...DEFAULT_WEIGHTS, skillOverlap: 3, embedding: 0, tfidf: 0, expMatch: 0, eduMatch: 0 };
    // only skills counts → weighted score == skill value * 100
    expect(weightedScore(fv({ skill_overlap_pct: 0.5, embedding_score: 1 }), w)).toBe(50);
  });
});

describe('topDrivers', () => {
  it('returns the highest-contribution features first', () => {
    const d = topDrivers(fv({ skill_overlap_pct: 0.9, embedding_score: 0.2 }), DEFAULT_WEIGHTS, 2);
    expect(d[0].key).toBe('skillOverlap');
    expect(d).toHaveLength(2);
  });
});

describe('reRank', () => {
  const pool = [
    cand('A', 40, { skill_overlap_pct: 0.9 }),
    cand('B', 60, { skill_overlap_pct: 0.1, embedding_score: 0.9 }),
    cand('C', 50, { embedding_score: 0.5 }),
  ];

  it('orders by authoritative final_score under uniform weights', () => {
    const r = reRank(pool, DEFAULT_WEIGHTS);
    expect(r.map((c) => c.candidate_id)).toEqual(['B', 'C', 'A']); // 60,50,40
    expect(r.map((c) => c.rank)).toEqual([1, 2, 3]);
  });

  it('re-ranks by weighted score once weights are tilted', () => {
    const w = { ...DEFAULT_WEIGHTS, skillOverlap: 3, embedding: 0, tfidf: 0, expMatch: 0, eduMatch: 0 };
    const r = reRank(pool, w);
    expect(r[0].candidate_id).toBe('A'); // A has the highest skill overlap
  });
});
