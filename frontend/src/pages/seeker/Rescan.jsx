import { useState } from 'react';
import { Link } from 'react-router-dom';
import { RefreshCw, TrendingUp, ArrowRight } from 'lucide-react';
import Button from '../../components/ui/Button';
import Card from '../../components/ui/Card';
import { Textarea } from '../../components/ui/Field';
import { EmptyState, ErrorState, ApertureLoader } from '../../components/ui/states';
import ApertureBloom from '../../components/ApertureBloom';
import { useAnalysis } from '../../seeker/AnalysisContext';
import { scoreFit, humanizeError } from '../../lib/api';

// §11.1-C — the retention surface. Edit → rescan → score DELTA shown before/after
// with the bloom morphing and momentum celebrated ("62 → 78 over 3 revisions").
// Turns a one-shot novelty into a returned-to tool.
export default function Rescan() {
  const { resumeText, setResumeText, jdText, setJdText, blindMode, history, pushResult } = useAnalysis();
  const [status, setStatus] = useState('idle'); // idle | scoring | error
  const [error, setError] = useState(null);

  const latest = history[history.length - 1] || null;
  const prev = history.length >= 2 ? history[history.length - 2] : null;
  const delta = latest && prev ? latest.score - prev.score : null;
  const canRescan = resumeText.trim().length > 40 && jdText.trim().length > 40 && status !== 'scoring';

  async function handleRescan() {
    setStatus('scoring');
    setError(null);
    try {
      const data = await scoreFit({ resumeText, jdText, blindMode });
      pushResult(data.score_result);
      setStatus('idle');
    } catch (err) {
      setError(humanizeError(err));
      setStatus('error');
    }
  }

  // No history yet → send them to the flagship first.
  if (history.length === 0) {
    return (
      <div className="max-w-[720px] mx-auto px-6 py-16">
        <Card>
          <EmptyState
            icon={RefreshCw}
            title="Rescan lives here"
            body="Run your first analysis, then come back to edit your resume and watch your fit climb."
            action={<Link to="/seeker/analyze"><Button variant="primary">Analyze fit first</Button></Link>}
          />
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-[1120px] mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-12 gap-8">
      {/* Edit column */}
      <section className="lg:col-span-5 space-y-5">
        <div>
          <h1 className="text-h1 text-ink font-editorial">Edit, then rescan.</h1>
          <p className="text-body text-muted mt-1.5">
            Make a change below and rescan — we'll show exactly how much it moved your fit.
          </p>
        </div>
        <Textarea
          id="rescan-resume"
          label="Resume text"
          rows={10}
          value={resumeText}
          onChange={(e) => setResumeText(e.target.value)}
          placeholder="Your resume text…"
        />
        <Textarea
          id="rescan-jd"
          label="Job description"
          rows={5}
          value={jdText}
          onChange={(e) => setJdText(e.target.value)}
        />
        <Button
          variant="primary" size="lg" className="w-full"
          icon={RefreshCw} disabled={!canRescan} loading={status === 'scoring'}
          onClick={handleRescan}
        >
          {status === 'scoring' ? 'Rescanning…' : 'Rescan my fit'}
        </Button>
      </section>

      {/* Delta column */}
      <section className="lg:col-span-7">
        <Card pad="lg" className="min-h-[520px] flex flex-col">
          {status === 'scoring' ? (
            <div className="flex-1 flex items-center justify-center"><ApertureLoader label="Re-focusing your fit…" /></div>
          ) : status === 'error' ? (
            <div className="flex-1 flex items-center justify-center">
              <ErrorState title="We couldn't rescan just now" body={error?.message} />
            </div>
          ) : (
            <div className="flex-1 flex flex-col">
              {/* Delta headline. */}
              {delta !== null && (
                <DeltaBanner prev={prev.score} next={latest.score} delta={delta} />
              )}

              {/* Current bloom. */}
              <div className="rounded-[var(--card-radius)] bg-canvas border border-border mt-4">
                <ApertureBloom
                  featureVector={latest.feature_vector}
                  score={latest.score}
                  confidence={latest.scoring_confidence}
                  confidenceBand={latest.confidence_level}
                  showLegend={false}
                />
              </div>

              {/* Momentum trail (§11.1-C). */}
              <MomentumTrail history={history} />
            </div>
          )}
        </Card>
      </section>
    </div>
  );
}

function DeltaBanner({ prev, next, delta }) {
  const up = delta > 0, flat = delta === 0;
  const color = up ? 'var(--fit-text)' : flat ? 'var(--lowconf-text)' : 'var(--gap-text)';
  const fill = up ? 'var(--fit-fill)' : flat ? 'var(--lowconf-fill)' : 'var(--gap-fill)';
  return (
    <div className="flex items-center justify-center gap-4 rounded-[var(--card-radius)] p-4" style={{ background: fill }}>
      <span className="text-h2 tabular-nums text-muted">{prev}</span>
      <ArrowRight className="w-5 h-5 text-muted" />
      <span className="text-display tabular-nums" style={{ color, fontFamily: 'var(--font-sans)' }}>{next}</span>
      <span className="inline-flex items-center gap-1 text-small font-semibold tabular-nums rounded-full px-2.5 py-1"
        style={{ color, background: 'var(--surface)' }}>
        <TrendingUp className={`w-4 h-4 ${up ? '' : flat ? 'opacity-40' : 'rotate-180'}`} />
        {delta > 0 ? `+${delta}` : delta}
      </span>
    </div>
  );
}

function MomentumTrail({ history }) {
  if (history.length < 2) {
    return (
      <p className="text-small text-muted text-center mt-6">
        Make an edit and rescan to see your momentum build here.
      </p>
    );
  }
  const first = history[0].score;
  const last = history[history.length - 1].score;
  const net = last - first;
  return (
    <div className="mt-6 pt-5 border-t border-border">
      <div className="flex items-center gap-2 mb-3">
        <TrendingUp className="w-4 h-4 text-fit-500" />
        <h4 className="text-small font-semibold text-ink">Your momentum</h4>
        {net > 0 && (
          <span className="text-caption font-semibold text-fit-text bg-fit-fill rounded-full px-2 py-0.5 tabular-nums ml-auto">
            +{net} over {history.length} revisions
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 flex-wrap">
        {history.map((h, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-small tabular-nums font-medium text-ink bg-canvas border border-border rounded-md px-2 py-1">
              {h.score}
            </span>
            {i < history.length - 1 && <ArrowRight className="w-3.5 h-3.5 text-muted" />}
          </div>
        ))}
      </div>
    </div>
  );
}
