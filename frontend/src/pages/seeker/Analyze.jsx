import { useState } from 'react';
import { Sparkles, ShieldCheck, RotateCcw, Lightbulb, FileSearch } from 'lucide-react';
import Button from '../../components/ui/Button';
import Card from '../../components/ui/Card';
import { Textarea } from '../../components/ui/Field';
import Dropzone from '../../components/ui/Dropzone';
import { EmptyState, ErrorState, ApertureLoader } from '../../components/ui/states';
import ApertureBloom from '../../components/ApertureBloom';
import GapReport from '../../components/GapReport';
import { parseResume, scoreFit, humanizeError } from '../../lib/api';
import { useAnalysis } from '../../seeker/AnalysisContext';

// §11.1-B — the flagship. Input (resume + JD + blind toggle) resolves into the
// hero aperture bloom, the gap-report diff, and forward-looking suggestions.
// Everything reads "here's what to fix", never "here's why you failed" (§12).
export default function Analyze() {
  // Inputs live in shared session state so the rescan screen can pick up where
  // this one left off (§11.1-C). File-parse + result state stay local to this view.
  const { resumeText, setResumeText, jdText, setJdText, blindMode, setBlindMode, pushResult } = useAnalysis();
  const [resume, setResume] = useState(null);   // ParsedResume from /parse
  const [parsing, setParsing] = useState(false);
  const [pasteMode, setPasteMode] = useState(false);

  const [status, setStatus] = useState('idle');  // idle | scoring | done | error
  const [result, setResult] = useState(null);    // score_result
  const [maturity, setMaturity] = useState(null);
  const [error, setError] = useState(null);

  const resumeReady = pasteMode ? resumeText.trim().length > 40 : !!resume?.raw_text;
  const canAnalyze = resumeReady && jdText.trim().length > 40 && status !== 'scoring';

  async function handleFile(file) {
    setParsing(true);
    setError(null);
    try {
      const parsed = await parseResume(file);
      setResume({ ...parsed, fileName: file.name });
    } catch (err) {
      setError(humanizeError(err));
    } finally {
      setParsing(false);
    }
  }

  async function handleAnalyze() {
    setStatus('scoring');
    setError(null);
    setResult(null);
    try {
      const sourceText = pasteMode ? resumeText : resume.raw_text;
      // Mirror the resume text into shared state so rescan can prefill + re-run it.
      if (!pasteMode && resume?.raw_text) setResumeText(resume.raw_text);
      const data = await scoreFit({ resumeText: sourceText, jdText, blindMode });
      setResult(data.score_result);
      setMaturity(data.pipeline_maturity);
      pushResult(data.score_result);   // record for the rescan delta / momentum trail
      setStatus('done');
    } catch (err) {
      setError(humanizeError(err));
      setStatus('error');
    }
  }

  // Top suggestions (§11.1-B) — derived, forward-looking. Gap actions first
  // (highest leverage), then any confidence-rationale nudges about input quality.
  const suggestions = result
    ? [
        ...result.gaps.slice(0, 3).map((g) => g.suggested_action),
        ...(result.gaps.length < 3 ? result.confidence_reasons.slice(0, 3 - result.gaps.length) : []),
      ].slice(0, 3)
    : [];

  return (
    <div className="max-w-[1120px] mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-12 gap-8">
      {/* ── Input column ─────────────────────────────────────────────────── */}
      <section className="lg:col-span-5 space-y-5">
        <div>
          <h1 className="text-h1 text-ink font-editorial">Let's see your fit.</h1>
          <p className="text-body text-muted mt-1.5">
            Add your resume and the job you're targeting. We'll show what's working and what to fix.
          </p>
        </div>

        <div>
          {pasteMode ? (
            <Textarea
              id="resume-text"
              label="Resume text"
              rows={7}
              value={resumeText}
              onChange={(e) => setResumeText(e.target.value)}
              placeholder="Paste your resume text here…"
            />
          ) : (
            <Dropzone
              onFile={handleFile}
              parsing={parsing}
              parsed={
                resume
                  ? {
                      fileName: resume.fileName,
                      parseConfidence: resume.parsing_confidence,
                      fieldsFound: Math.round((resume.parsing_confidence || 0) * 12),
                      fieldsExpected: 12,
                    }
                  : null
              }
            />
          )}
          <button
            type="button"
            onClick={() => setPasteMode((p) => !p)}
            className="text-caption text-ember-700 hover:underline mt-2 focus-ember"
          >
            {pasteMode ? '← Upload a file instead' : 'or paste resume text instead'}
          </button>
        </div>

        <Textarea
          id="jd"
          label="Job description"
          rows={8}
          value={jdText}
          onChange={(e) => setJdText(e.target.value)}
          placeholder="Paste the full job description here…"
          hint={jdText.trim().length > 0 && jdText.trim().length <= 40 ? 'A little more detail gives a sharper read.' : 'The more complete, the sharper the read.'}
        />

        {/* Blind-mode toggle — a trust feature, visible here (§11.1-A / §13). */}
        <button
          type="button"
          role="switch"
          aria-checked={blindMode}
          onClick={() => setBlindMode((b) => !b)}
          className="w-full flex items-center gap-3 rounded-xl border border-border bg-surface p-3.5 text-left focus-ember hover:bg-canvas transition-colors"
        >
          <span className={`w-9 h-5 rounded-full relative transition-colors shrink-0 ${blindMode ? 'bg-ember-500' : 'bg-border'}`}>
            <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${blindMode ? 'left-4.5' : 'left-0.5'}`} />
          </span>
          <span className="flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-ember-700" strokeWidth={1.75} />
            <span className="text-small text-ink font-medium">Blind mode</span>
          </span>
          <span className="text-caption text-muted ml-auto">strips name, photo & school before scoring</span>
        </button>

        <Button
          variant="primary"
          size="lg"
          className="w-full"
          icon={Sparkles}
          disabled={!canAnalyze}
          loading={status === 'scoring'}
          onClick={handleAnalyze}
        >
          {status === 'scoring' ? 'Bringing your fit into focus…' : 'Analyze fit'}
        </Button>
        <p className="text-caption text-muted text-center">Free · 3 scans a month · resumes aren't stored beyond your session</p>
      </section>

      {/* ── Result column ────────────────────────────────────────────────── */}
      <section className="lg:col-span-7">
        <Card pad="lg" className="min-h-[520px] flex flex-col">
          {status === 'scoring' ? (
            <div className="flex-1 flex items-center justify-center">
              <ApertureLoader />
            </div>
          ) : status === 'error' ? (
            <div className="flex-1 flex items-center justify-center">
              <ErrorState
                title={error?.kind === 'freemium' ? 'That was your last free scan this month' : "We couldn't score this clearly"}
                body={error?.message}
              />
            </div>
          ) : status === 'done' && result ? (
            <ResultView result={result} maturity={maturity} suggestions={suggestions} onRescan={() => setStatus('idle')} />
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <EmptyState
                icon={FileSearch}
                title="Your fit, brought into focus"
                body="Add a resume and a job description, then analyze. The score blooms open and shows its reasoning."
              />
            </div>
          )}
        </Card>
      </section>
    </div>
  );
}

function ResultView({ result, maturity, suggestions, onRescan }) {
  return (
    <div className="flex-1 flex flex-col animate-[slide-up_var(--dur-base)_var(--ease-settle)]">
      {/* Hero bloom — the signature reveal. */}
      <div className="rounded-[var(--card-radius)] bg-canvas border border-border">
        <ApertureBloom
          featureVector={result.feature_vector}
          score={result.final_score}
          confidence={result.scoring_confidence}
          confidenceBand={result.confidence_level}
        />
      </div>

      {/* Maturity honesty chip (§14 — never hide a provisional model). */}
      {maturity?.overall_status && maturity.overall_status !== 'tuned' && (
        <p className="text-caption text-muted mt-3 text-center">
          {maturity.description || 'Provisional weights — calibrating against ground truth.'}
        </p>
      )}

      {/* Gap diff. */}
      <div className="mt-6">
        <h3 className="text-h3 text-ink mb-3">What to work on</h3>
        <GapReport matchedSkills={result.matched_skills} gaps={result.gaps} />
      </div>

      {/* Forward-looking suggestions. */}
      {suggestions.length > 0 && (
        <div className="mt-6 rounded-[var(--card-radius)] bg-ember-50 border border-ember-100 p-5">
          <h4 className="text-small font-semibold text-ember-700 flex items-center gap-1.5 mb-2.5">
            <Lightbulb className="w-4 h-4" strokeWidth={1.75} /> Top suggestions
          </h4>
          <ol className="space-y-2">
            {suggestions.map((s, i) => (
              <li key={i} className="flex gap-2.5 text-small text-ink">
                <span className="tabular-nums font-semibold text-ember-700">{i + 1}.</span>
                <span>{s}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="mt-6 pt-4 border-t border-border flex items-center justify-between">
        <span className="text-caption text-muted tabular-nums">
          {result.pipeline_version} · trace {result.score_id.slice(0, 8)}
        </span>
        <Button variant="ghost" size="sm" icon={RotateCcw} onClick={onRescan}>
          Edit & rescan
        </Button>
      </div>
    </div>
  );
}
