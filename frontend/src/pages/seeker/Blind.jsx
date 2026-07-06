import { useState } from 'react';
import { ShieldCheck, EyeOff, GraduationCap, UserX, ImageOff, ArrowRight, Scale, Trash2 } from 'lucide-react';
import Button from '../../components/ui/Button';
import Card from '../../components/ui/Card';
import { Textarea } from '../../components/ui/Field';
import { EmptyState, ErrorState, ApertureLoader } from '../../components/ui/states';
import { scoreFit, humanizeError } from '../../lib/api';
import { useAnalysis } from '../../seeker/AnalysisContext';

const EXAMPLE_RESUME =
  'Jane Doe. She is a Senior Backend Engineer with 6 years building Python microservices, REST APIs, and PostgreSQL. She led a team of 4 engineers. Graduated from Stanford University with a B.S. in Computer Science. Skilled in AWS, Docker, Kubernetes, Terraform.';
const EXAMPLE_JD =
  'Senior Backend Engineer. Requirements: 5+ years Python, REST API design, PostgreSQL, Kubernetes, Terraform, team leadership. AWS preferred. Bachelor degree in Computer Science required.';

const TYPE_META = {
  gender_term: { icon: UserX, label: 'Gender term' },
  institution: { icon: GraduationCap, label: 'University / institution' },
  photo: { icon: ImageOff, label: 'Photo' },
};

// §11.3-I / §13 — blind mode as a VISIBLE, one-tap trust feature that shows what
// was stripped and why, and doubles as a bias check: if removing identity signals
// changes the score, that limitation is surfaced honestly (P3), never hidden.
export default function Blind() {
  const { resumeText, setResumeText, jdText, setJdText } = useAnalysis();
  const [status, setStatus] = useState('idle'); // idle | running | done | error
  const [openScore, setOpenScore] = useState(null);   // blind-off result
  const [blindResult, setBlindResult] = useState(null); // { score, anonymization }
  const [error, setError] = useState(null);

  const canRun = resumeText.trim().length > 40 && jdText.trim().length > 40 && status !== 'running';

  function loadExample() {
    setResumeText(EXAMPLE_RESUME);
    setJdText(EXAMPLE_JD);
  }

  async function runBiasCheck() {
    setStatus('running');
    setError(null);
    setOpenScore(null);
    setBlindResult(null);
    try {
      // Two scores on identical input — one open, one blind — so any score shift
      // attributable purely to identity signals is visible (§13 bias check).
      const open = await scoreFit({ resumeText, jdText, blindMode: false });
      const blind = await scoreFit({ resumeText, jdText, blindMode: true });
      setOpenScore(open.score_result.final_score);
      setBlindResult({
        score: blind.score_result.final_score,
        anonymization: blind.anonymization,
      });
      setStatus('done');
    } catch (err) {
      const e = humanizeError(err);
      setError(e);
      setStatus('error');
    }
  }

  const delta = blindResult != null && openScore != null ? blindResult.score - openScore : null;

  return (
    <div className="max-w-[1120px] mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-12 gap-8">
      <section className="lg:col-span-5 space-y-5">
        <div>
          <span className="inline-flex items-center gap-1.5 text-caption font-medium bg-ember-50 text-ember-700 rounded-full px-3 py-1 mb-3">
            <ShieldCheck className="w-3.5 h-3.5" /> Trust feature
          </span>
          <h1 className="text-h1 text-ink font-editorial">Blind mode & bias check.</h1>
          <p className="text-body text-muted mt-1.5">
            We strip your name, photo, gender terms, and university before scoring — then show you
            exactly what was removed, and whether it moved your score at all.
          </p>
        </div>

        <div className="flex justify-between items-center">
          <h2 className="text-small font-semibold text-ink">Your resume & the role</h2>
          <button onClick={loadExample} className="text-caption text-ember-700 hover:underline focus-ember">Load an example</button>
        </div>
        <Textarea id="blind-resume" label="Resume text" rows={7} value={resumeText}
          onChange={(e) => setResumeText(e.target.value)} placeholder="Paste your resume text…" />
        <Textarea id="blind-jd" label="Job description" rows={4} value={jdText}
          onChange={(e) => setJdText(e.target.value)} placeholder="Paste the job description…" />
        <Button variant="primary" size="lg" className="w-full" icon={Scale} disabled={!canRun} loading={status === 'running'} onClick={runBiasCheck}>
          {status === 'running' ? 'Running blind & open scores…' : 'Run blind mode + bias check'}
        </Button>

        {/* Data retention (§13) — plain language, always present. */}
        <div className="flex gap-2.5 rounded-[var(--card-radius)] bg-sunken border border-border p-4">
          <Trash2 className="w-4 h-4 text-muted shrink-0 mt-0.5" />
          <p className="text-caption text-muted leading-relaxed">
            <span className="text-ink font-medium">Your data isn't stored.</span> Resumes are read for
            this session only and never kept beyond it without your explicit consent. Stripped text is
            recorded as a one-way hash — never the original.
          </p>
        </div>
      </section>

      <section className="lg:col-span-7">
        <Card pad="lg" className="min-h-[520px] flex flex-col">
          {status === 'running' ? (
            <div className="flex-1 flex items-center justify-center"><ApertureLoader label="Scoring open and blind…" /></div>
          ) : status === 'error' ? (
            <div className="flex-1 flex items-center justify-center"><ErrorState title="We couldn't run the check" body={error?.message} /></div>
          ) : status === 'done' && blindResult ? (
            <BiasResult openScore={openScore} blindResult={blindResult} delta={delta} />
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <EmptyState
                icon={EyeOff}
                title="See yourself the way the model does — blind"
                body="Run the check to see what identity signals get removed before scoring, and whether they were affecting your result."
              />
            </div>
          )}
        </Card>
      </section>
    </div>
  );
}

function BiasResult({ openScore, blindResult, delta }) {
  const stripped = blindResult.anonymization?.stripped_items || [];
  const biasFlag = delta !== null && Math.abs(delta) >= 2;
  return (
    <div className="flex-1 flex flex-col">
      {/* Open vs blind scores. */}
      <h3 className="text-h3 text-ink mb-3">Open vs. blind score</h3>
      <div className="flex items-center justify-center gap-5 rounded-[var(--card-radius)] bg-sunken border border-border p-5">
        <div className="text-center">
          <p className="text-display tabular-nums text-ink" style={{ fontFamily: 'var(--font-sans)', fontSize: 40 }}>{openScore}</p>
          <p className="text-caption text-muted">with identity</p>
        </div>
        <ArrowRight className="w-5 h-5 text-muted" />
        <div className="text-center">
          <p className="text-display tabular-nums text-ember-score" style={{ fontFamily: 'var(--font-sans)', fontSize: 40 }}>{blindResult.score}</p>
          <p className="text-caption text-muted">blind</p>
        </div>
      </div>

      {/* Bias verdict — honest either way (§13). */}
      <div className="mt-4 rounded-[var(--card-radius)] p-4"
        style={{ background: biasFlag ? 'var(--gap-fill)' : 'var(--fit-fill)' }}>
        <p className="text-small font-medium" style={{ color: biasFlag ? 'var(--gap-text)' : 'var(--fit-text)' }}>
          {biasFlag
            ? `Heads up: removing identity signals changed the score by ${delta > 0 ? '+' : ''}${delta}. We surface this as a limitation to investigate, not hide.`
            : delta === 0
              ? 'No change. Removing identity signals did not move your score — a good sign for fairness on this input.'
              : `Only a ${Math.abs(delta)}-point difference — identity signals had minimal effect on this result.`}
        </p>
      </div>

      {/* What was stripped and why. */}
      <h3 className="text-h3 text-ink mt-6 mb-3">What we removed before scoring</h3>
      {stripped.length === 0 ? (
        <p className="text-small text-muted">Nothing identity-revealing was detected to strip in this resume.</p>
      ) : (
        <ul className="space-y-2.5">
          {stripped.map((item, i) => {
            const meta = TYPE_META[item.type] || { icon: EyeOff, label: item.type };
            const Icon = meta.icon;
            return (
              <li key={i} className="flex items-center gap-3 rounded-[var(--card-radius)] border border-border bg-surface p-3">
                <span className="w-9 h-9 rounded-lg bg-ember-50 text-ember-700 flex items-center justify-center shrink-0">
                  <Icon className="w-4 h-4" strokeWidth={1.75} />
                </span>
                <div className="flex-1">
                  <p className="text-small font-medium text-ink">{meta.label}</p>
                  <p className="text-caption text-muted">
                    Replaced with <span className="font-mono bg-sunken rounded px-1">{item.replaced_with}</span>
                    {item.occurrences > 1 && ` · ${item.occurrences}×`}
                  </p>
                </div>
              </li>
            );
          })}
        </ul>
      )}
      <p className="text-caption text-muted mt-4 pt-4 border-t border-border">
        We show categories and placeholders only — the original text is never stored or re-displayed.
      </p>
    </div>
  );
}
