import { useState } from 'react';
import { HeartPulse, CheckCircle2, AlertCircle, Sparkles } from 'lucide-react';
import Button from '../../components/ui/Button';
import Card from '../../components/ui/Card';
import { Textarea } from '../../components/ui/Field';
import Dropzone from '../../components/ui/Dropzone';
import ArcMeter from '../../components/ui/ArcMeter';
import { EmptyState, ErrorState } from '../../components/ui/states';
import { parseResume, humanizeError } from '../../lib/api';
import { analyzeResumeHealth } from '../../lib/resumeHealth';
import { useAnalysis } from '../../seeker/AnalysisContext';

// §11.1-D — resume health check, no comparison target. Same warm coaching frame:
// ATS readability, weak-verb flags, missing quantifiable achievements. Rule-based.
export default function Health() {
  const { resumeText, setResumeText } = useAnalysis();
  const [pasteMode, setPasteMode] = useState(true);
  const [parsing, setParsing] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  const canCheck = resumeText.trim().length > 60;

  async function handleFile(file) {
    setParsing(true);
    setError(null);
    try {
      const parsed = await parseResume(file);
      setResumeText(parsed.raw_text || '');
      setReport(analyzeResumeHealth(parsed.raw_text || ''));
    } catch (err) {
      setError(humanizeError(err));
    } finally {
      setParsing(false);
    }
  }

  function handleCheck() {
    setError(null);
    setReport(analyzeResumeHealth(resumeText));
  }

  return (
    <div className="max-w-[1120px] mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-12 gap-8">
      <section className="lg:col-span-5 space-y-5">
        <div>
          <h1 className="text-h1 text-ink font-editorial">A quick health check.</h1>
          <p className="text-body text-muted mt-1.5">
            No job description needed — just a read on how your resume lands with recruiters and ATS.
          </p>
        </div>

        {pasteMode ? (
          <Textarea
            id="health-resume"
            label="Resume text"
            rows={12}
            value={resumeText}
            onChange={(e) => setResumeText(e.target.value)}
            placeholder="Paste your resume text here…"
          />
        ) : (
          <Dropzone onFile={handleFile} parsing={parsing} />
        )}
        <button
          type="button"
          onClick={() => setPasteMode((p) => !p)}
          className="text-caption text-ember-700 hover:underline focus-ember"
        >
          {pasteMode ? 'or upload a file instead' : '← paste text instead'}
        </button>

        {pasteMode && (
          <Button variant="primary" size="lg" className="w-full" icon={HeartPulse} disabled={!canCheck} onClick={handleCheck}>
            Check my resume
          </Button>
        )}
      </section>

      <section className="lg:col-span-7">
        <Card pad="lg" className="min-h-[520px] flex flex-col">
          {error ? (
            <div className="flex-1 flex items-center justify-center"><ErrorState body={error.message} /></div>
          ) : report ? (
            <HealthReport report={report} />
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <EmptyState
                icon={Sparkles}
                title="Your resume's health, at a glance"
                body="Add your resume to see ATS readability, verb strength, and whether your wins are quantified."
              />
            </div>
          )}
        </Card>
      </section>
    </div>
  );
}

function HealthReport({ report }) {
  const tone = report.overall >= 75 ? 'fit' : report.overall >= 50 ? 'gap' : 'lowconf';
  return (
    <div className="flex-1 flex flex-col">
      <div className="flex items-center gap-4 pb-5 border-b border-border">
        <ArcMeter value={report.overall / 100} tone={tone} size={72} />
        <div>
          <h3 className="text-h3 text-ink">Resume health</h3>
          <p className="text-small text-muted">
            {report.overall >= 75 ? "Strong shape — a few polish moves below." : "Some quick wins below to sharpen it."}
            {' '}· {report.wordCount} words
          </p>
        </div>
      </div>

      <ul className="mt-5 space-y-4">
        {report.checks.map((c) => (
          <li key={c.id} className="flex gap-3">
            {c.status === 'good' ? (
              <CheckCircle2 className="w-5 h-5 text-fit-500 shrink-0 mt-0.5" strokeWidth={2} aria-hidden />
            ) : (
              <AlertCircle className="w-5 h-5 text-gap-500 shrink-0 mt-0.5" strokeWidth={2} aria-hidden />
            )}
            <div>
              <div className="flex items-center gap-2">
                <p className="text-small font-semibold text-ink">{c.title}</p>
                <span
                  className={`text-caption tabular-nums rounded-full px-2 py-0.5 ${c.status === 'good' ? 'bg-fit-fill text-fit-text' : 'bg-gap-fill text-gap-text'}`}
                >
                  {c.metric}
                </span>
              </div>
              <p className="text-small text-muted mt-0.5">{c.detail}</p>
            </div>
          </li>
        ))}
      </ul>

      <p className="text-caption text-muted mt-6 pt-4 border-t border-border">
        A health check is a coaching read, not a score against a specific role — for that, add a job description on the Analyze screen.
      </p>
    </div>
  );
}
