import { describe, it, expect } from 'vitest';
import { bandVisuals, toPetals, strongestFeature } from './apertureConfidence';

describe('bandVisuals', () => {
  it('maps each confidence band to distinct treatment', () => {
    expect(bandVisuals('high').ring).toBe('var(--fit-500)');
    expect(bandVisuals('medium').ring).toBe('var(--gap-500)');
    expect(bandVisuals('low').ring).toBe('var(--lowconf-500)');
  });

  it('renders low confidence with the lighter numeral and honest label', () => {
    const low = bandVisuals('low');
    expect(low.number).toBe('var(--ember-300)');
    expect(low.label.toLowerCase()).toContain('low confidence');
  });

  it('falls back to the cautious low treatment for unknown bands', () => {
    expect(bandVisuals('nonsense').ring).toBe('var(--lowconf-500)');
    expect(bandVisuals(undefined).ring).toBe('var(--lowconf-500)');
  });
});

describe('toPetals', () => {
  it('normalizes the 5-feature vector in petal order and clamps to 0-1', () => {
    const petals = toPetals({ tfidf_score: 0.5, embedding_score: 2, skill_overlap_pct: -1, exp_match: 0.3, edu_match: 'x' });
    expect(petals.map((p) => p.key)).toEqual(['tfidf', 'embedding', 'skillOverlap', 'expMatch', 'eduMatch']);
    expect(petals[1].value).toBe(1);   // clamped from 2
    expect(petals[2].value).toBe(0);   // clamped from -1
    expect(petals[4].value).toBe(0);   // non-numeric → 0
  });
});

describe('strongestFeature', () => {
  it('names the dominant feature', () => {
    expect(strongestFeature({ tfidf_score: 0.1, embedding_score: 0.2, skill_overlap_pct: 0.95, exp_match: 0.3, edu_match: 0.4 })).toBe('skills');
  });
});
