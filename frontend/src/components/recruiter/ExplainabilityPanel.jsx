import { useState, useEffect, useRef } from 'react';
import { X, Copy, Check } from 'lucide-react';
import ApertureBloom from '../ApertureBloom';
import { FEATURE_LABELS, toPetals } from '../../lib/apertureConfidence';

// §10.8 Explainability panel — slide-over from the right. The physical embodiment
// of P1 (show the why): full bloom, feature-by-feature breakdown with the trained
// model's feature importance, the RAG skill matches, and an editable auto-drafted
// feedback note. Blind mode swaps the name for the anonymized display name.
export default function ExplainabilityPanel({ candidate, displayName, onClose }) {
  const sr = candidate?.score_result;
  const [note, setNote] = useState('');
  const [copied, setCopied] = useState(false);
  const noteRef = useRef(null);

  useEffect(() => {
    if (sr) setNote(draftNote(sr, displayName));
  }, [sr, displayName]);

  useEffect(() => {
    const onEsc = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onEsc);
    return () => window.removeEventListener('keydown', onEsc);
  }, [onClose]);

  if (!candidate) return null;
  const petals = toPetals(sr.feature_vector);
  const importance = sr.feature_importance || null;

  const copyNote = async () => {
    let ok = false;
    try {
      await navigator.clipboard.writeText(note);
      ok = true;
    } catch {
      // Fallback for contexts where the async Clipboard API is blocked (iframes,
      // non-focused windows): a transient textarea + execCommand still copies.
      try {
        const ta = document.createElement('textarea');
        ta.value = note;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        ok = document.execCommand('copy');
        document.body.removeChild(ta);
      } catch { ok = false; }
    }
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } else if (noteRef.current) {
      // Total fallback (locked-down contexts): select the note so the user can copy
      // it manually. Honest — we never claim success we didn't achieve.
      noteRef.current.focus();
      noteRef.current.select();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true" aria-label={`Explainability for ${displayName}`}>
      <div className="absolute inset-0 bg-black/25 animate-[fade-in_var(--dur-fast)_var(--ease-settle)]" onClick={onClose} />
      <aside
        data-theme="recruiter"
        className="relative w-full max-w-[480px] bg-canvas h-full overflow-y-auto shadow-[var(--shadow-md)] animate-[slide-in_var(--dur-base)_var(--ease-settle)]"
        style={{ animationName: 'slide-in' }}
      >
        <div className="sticky top-0 bg-canvas/95 backdrop-blur border-b border-border px-6 py-4 flex items-center justify-between z-10">
          <div>
            <p className="text-caption uppercase tracking-wider text-muted">Rank #{candidate.rank}</p>
            <h2 className="text-h3 text-ink">{displayName}</h2>
          </div>
          <button onClick={onClose} className="p-2 rounded-md text-muted hover:text-ink hover:bg-surface focus-ember" aria-label="Close">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Full bloom. */}
          <div className="rounded-[var(--card-radius)] bg-surface border border-border">
            <ApertureBloom
              featureVector={sr.feature_vector}
              score={sr.final_score}
              confidence={sr.scoring_confidence}
              confidenceBand={sr.confidence_level}
              showLegend={false}
            />
          </div>

          {/* Feature breakdown. */}
          <section>
            <h3 className="text-small font-semibold text-ink mb-3">Why this score</h3>
            <ul className="space-y-2.5">
              {petals.map((p) => {
                const imp = importance?.[p.key] ?? importance?.[FEATURE_LABELS[p.key]];
                return (
                  <li key={p.key}>
                    <div className="flex items-center justify-between text-small">
                      <span className="capitalize text-ink">{FEATURE_LABELS[p.key]}</span>
                      <span className="tabular-nums text-muted">{Math.round(p.value * 100)}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-border mt-1 overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${Math.round(p.value * 100)}%`, background: 'var(--ember-500)' }} />
                    </div>
                    {imp != null && (
                      <p className="text-caption text-muted mt-0.5 tabular-nums">model weight {(imp * 100).toFixed(0)}%</p>
                    )}
                  </li>
                );
              })}
            </ul>
          </section>

          {/* RAG skill matches (§10.6). */}
          {sr.matched_skills?.length > 0 && (
            <section>
              <h3 className="text-small font-semibold text-ink mb-2">Skill matches</h3>
              <div className="flex flex-wrap gap-1.5">
                {sr.matched_skills.map((m, i) => (
                  <span key={i} className="inline-flex items-center gap-1 rounded-full bg-fit-fill text-fit-text px-2.5 py-1 text-caption"
                    title={m.match_type === 'semantic' ? `"${m.resume_skill}" ≈ "${m.jd_skill}"` : undefined}>
                    {m.match_type === 'semantic' && <span aria-hidden className="font-semibold">≈</span>}
                    {m.jd_skill || m.resume_skill}
                  </span>
                ))}
              </div>
            </section>
          )}

          {/* Editable auto-drafted note. */}
          <section>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-small font-semibold text-ink">Feedback note</h3>
              <button onClick={copyNote} className="inline-flex items-center gap-1 text-caption text-ember-700 hover:underline focus-ember">
                {copied ? <><Check className="w-3.5 h-3.5" /> Copied</> : <><Copy className="w-3.5 h-3.5" /> Copy</>}
              </button>
            </div>
            <textarea
              ref={noteRef}
              value={note}
              onChange={(e) => setNote(e.target.value)}
              rows={7}
              className="w-full px-3.5 py-3 rounded-md bg-surface border border-border text-ink text-[15px] leading-relaxed resize-y focus:border-ember-300 focus-ember"
            />
            <p className="text-caption text-muted mt-1.5">Auto-drafted from the evidence above — edit before sending.</p>
          </section>
        </div>
      </aside>
    </div>
  );
}

// Draft a concise, justification-first note (§12 recruiter voice). Decisive, cites
// the evidence, frames gaps factually — never a verdict on the person.
function draftNote(sr, name) {
  const strengths = [...sr.matched_skills].slice(0, 3).map((m) => m.jd_skill || m.resume_skill);
  const gaps = sr.gaps.slice(0, 3).map((g) => g.missing_skill);
  const lines = [
    `Re: ${name} — fit score ${sr.final_score} (${sr.confidence_level} confidence).`,
    strengths.length ? `Strengths: ${strengths.join(', ')}.` : '',
    gaps.length ? `Gaps vs. this role: ${gaps.join(', ')}.` : 'Covers the required skills for this role.',
    'Recommendation: ' + (sr.final_score >= 70 ? 'advance to a screen.' : sr.final_score >= 50 ? 'consider if pipeline is thin; probe the gaps above.' : 'likely not a match for this specific role.'),
  ];
  return lines.filter(Boolean).join('\n\n');
}
