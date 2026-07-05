// Shared confidence-band → visual mapping for the Aperture Bloom (Design Blueprint
// §6.3 / §6.4). BOTH the hero and micro tiers import this so the two tiers can
// never visually disagree about what "medium confidence" means.
//
// All colors are R1 CSS-variable references — no hardcoded hex lives here.

export const FEATURE_LABELS = {
  tfidf: 'lexical',
  embedding: 'semantic',
  skillOverlap: 'skills',
  expMatch: 'experience',
  eduMatch: 'education',
};

/**
 * Visual treatment for a confidence band (§6.3).
 * @param {'high'|'medium'|'low'} band
 */
export function bandVisuals(band) {
  switch ((band || '').toLowerCase()) {
    case 'high':
      return {
        ring: 'var(--fit-500)',
        number: 'var(--ember-score)',
        label: 'High confidence',
      };
    case 'medium':
      return {
        ring: 'var(--gap-500)',
        number: 'var(--ember-score)',
        label: 'Medium confidence',
      };
    default: // low / unknown → honest, cautious treatment
      return {
        ring: 'var(--lowconf-500)',
        // §6.3: the number is rendered slightly lighter at low confidence.
        number: 'var(--ember-300)',
        label: 'Low confidence — we could only read part of this resume',
      };
  }
}

/** Normalize an API feature_vector (0-1) into the bloom's petal order. */
export function toPetals(fv = {}) {
  return [
    { key: 'tfidf', value: clamp01(fv.tfidf_score) },
    { key: 'embedding', value: clamp01(fv.embedding_score) },
    { key: 'skillOverlap', value: clamp01(fv.skill_overlap_pct) },
    { key: 'expMatch', value: clamp01(fv.exp_match) },
    { key: 'eduMatch', value: clamp01(fv.edu_match) },
  ];
}

/** Name of the strongest feature, for the §15 screen-reader label. */
export function strongestFeature(fv = {}) {
  const petals = toPetals(fv);
  const top = petals.reduce((a, b) => (b.value > a.value ? b : a), petals[0]);
  return FEATURE_LABELS[top.key] || top.key;
}

function clamp01(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}
