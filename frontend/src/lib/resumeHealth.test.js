import { describe, it, expect } from 'vitest';
import { analyzeResumeHealth } from './resumeHealth';

describe('analyzeResumeHealth', () => {
  it('flags weak verbs and rewards strong ones', () => {
    const weak = analyzeResumeHealth('Responsible for maintaining systems. Helped the team. Worked on features across the codebase here.');
    const verbs = weak.checks.find((c) => c.id === 'verbs');
    expect(verbs.status).toBe('warn');

    const strong = analyzeResumeHealth('Led the platform team. Built and shipped three services. Designed the data model for the org.');
    expect(strong.checks.find((c) => c.id === 'verbs').status).toBe('good');
  });

  it('counts quantified achievements robustly across formatting', () => {
    const r = analyzeResumeHealth('Cut build time 40%. Led 4 engineers. Grew revenue $2M over 3 years across the platform.');
    const quant = r.checks.find((c) => c.id === 'quantify');
    expect(quant.status).toBe('good');
    expect(quant.metric).toMatch(/metric/);
  });

  it('checks ATS readability via sections and length', () => {
    const r = analyzeResumeHealth('Experience: built things. Skills: python, sql. Education: B.S. '.repeat(4));
    expect(r.checks.find((c) => c.id === 'ats')).toBeTruthy();
    expect(r.overall).toBeGreaterThanOrEqual(0);
    expect(r.overall).toBeLessThanOrEqual(100);
  });

  it('handles empty input without throwing', () => {
    const r = analyzeResumeHealth('');
    expect(r.wordCount).toBe(0);
    expect(r.checks).toHaveLength(3);
  });
});
