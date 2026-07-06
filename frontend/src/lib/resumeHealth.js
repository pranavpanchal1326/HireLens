// §11.1-D Resume health check — rule-based, no JD, same warm coaching frame (§12).
// Pure client-side heuristics over the resume text: ATS readability, weak-verb
// flags, and missing quantifiable achievements. Each check returns a status
// ('good' | 'warn') plus forward-looking, action-framed copy.

const WEAK_VERBS = [
  'responsible for', 'worked on', 'helped', 'assisted', 'was involved in',
  'participated in', 'duties included', 'in charge of', 'handled', 'dealt with',
];

const STRONG_VERB_HINT = ['led', 'built', 'shipped', 'designed', 'launched', 'drove', 'increased', 'reduced', 'owned', 'created'];

export function analyzeResumeHealth(text) {
  const t = (text || '').trim();
  const words = t.split(/\s+/).filter(Boolean);

  // 1) Quantifiable achievements — count distinct metric mentions across the whole
  // text (robust to paragraph-vs-bullet formatting, unlike a per-line ratio):
  // percentages, currency, multipliers, and counts of people/time/scale.
  const metricMatches = t.match(/\d[\d,.]*\s*%|\$\s?\d[\d,.]*|\b\d+\s*(?:x|k|m)\b|\b\d+\s*(?:users|customers|engineers|people|hours|days|weeks|months|years|projects|teams)\b/gi) || [];
  const metricCount = metricMatches.length;
  const quant = {
    id: 'quantify',
    title: 'Quantified achievements',
    status: metricCount >= 3 ? 'good' : 'warn',
    detail: metricCount >= 3
      ? `${metricCount} concrete metrics — that's exactly what recruiters scan for.`
      : `Only ${metricCount} metric${metricCount === 1 ? '' : 's'} found. Add numbers — "cut build time 40%", "led 4 engineers" — to make impact legible.`,
    metric: metricCount === 1 ? '1 metric' : `${metricCount} metrics`,
  };

  // 2) Weak verbs — passive/vague phrasing that dilutes impact.
  const lower = t.toLowerCase();
  const foundWeak = WEAK_VERBS.filter((v) => lower.includes(v));
  const verbs = {
    id: 'verbs',
    title: 'Strong action verbs',
    status: foundWeak.length === 0 ? 'good' : 'warn',
    detail: foundWeak.length === 0
      ? 'No weak phrasing spotted — your bullets lead with action.'
      : `Swap weak phrasing (${foundWeak.slice(0, 3).map((v) => `"${v}"`).join(', ')}) for strong verbs like ${STRONG_VERB_HINT.slice(0, 3).join(', ')}.`,
    metric: foundWeak.length ? `${foundWeak.length} to fix` : 'clear',
  };

  // 3) ATS readability — length + section signals that machines parse reliably.
  const hasSections = /\b(experience|education|skills|projects|summary)\b/i.test(t);
  const lengthOk = words.length >= 120 && words.length <= 900;
  const atsScore = (hasSections ? 0.5 : 0) + (lengthOk ? 0.5 : 0);
  const ats = {
    id: 'ats',
    title: 'ATS readability',
    status: atsScore >= 0.75 ? 'good' : 'warn',
    detail: [
      hasSections ? 'Clear section headers detected.' : 'Add standard headers (Experience, Skills, Education) so parsers find your content.',
      lengthOk ? 'Length is in the ATS-friendly range.' : words.length < 120 ? 'This looks short — add detail so there\'s enough to parse.' : 'This runs long — tighten to keep the strongest signal up top.',
    ].join(' '),
    metric: `${Math.round(atsScore * 100)}%`,
  };

  const checks = [ats, verbs, quant];
  const goodCount = checks.filter((c) => c.status === 'good').length;
  const overall = Math.round((goodCount / checks.length) * 100);

  return { checks, overall, wordCount: words.length };
}
