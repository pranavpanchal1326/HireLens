import { useEffect, useState } from 'react';
import { Activity, TrendingUp, Target, Info, CheckCircle2, Clock } from 'lucide-react';
import Card from '../../components/ui/Card';
import { SkeletonCard, ErrorState } from '../../components/ui/states';
import { useRecruiter } from '../../recruiter/RecruiterContext';
import { getMetrics, humanizeError } from '../../lib/api';

// §11.2-H / §14 — the MLOps trust surface. Spearman correlation + Precision@k over
// time, ALWAYS with sample size and confidence intervals (std). A small ground-truth
// set is visibly caveated, never hidden (P3). Renders the honest readiness state:
// unready → collecting; provisional → placeholder weights; tuned → calibrated.
export default function Dashboard() {
  const { auth, signOut } = useRecruiter();
  const [state, setState] = useState('loading');
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const d = await getMetrics({ auth });
        if (alive) { setData(d); setState('done'); }
      } catch (err) {
        const e = humanizeError(err);
        if (alive) { setError(e); setState('error'); if (e.kind === 'auth') signOut(); }
      }
    })();
    return () => { alive = false; };
  }, [auth, signOut]);

  return (
    <div className="max-w-[1120px] mx-auto px-6 py-8">
      <div className="flex items-center gap-2.5 mb-1">
        <Activity className="w-5 h-5 text-ember-500" />
        <h1 className="text-h1 text-ink">Accuracy dashboard</h1>
      </div>
      <p className="text-body text-muted mb-6">
        How well HireLens ranks against human judgement — measured honestly, with sample sizes and confidence intervals.
      </p>

      {state === 'loading' && <Card pad="lg"><SkeletonCard /></Card>}
      {state === 'error' && <ErrorState title="We couldn't load metrics" body={error?.message} />}
      {state === 'done' && data && <DashboardBody data={data} />}
    </div>
  );
}

function DashboardBody({ data }) {
  const cm = data.current_metrics;
  return (
    <div className="space-y-5">
      <ReadinessBanner state={data.readiness_state} details={data.status_details} progress={data.progress} />

      {/* Metric cards — real values with CI when ready; pending otherwise. */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          icon={TrendingUp}
          title="Spearman ρ"
          blurb="Rank correlation vs. human ratings (−1 … 1). Higher is better."
          stat={cm?.spearman}
          format={(v) => v.toFixed(3)}
        />
        <MetricCard
          icon={Target}
          title="Precision@5"
          blurb="Of the top 5 ranked, the fraction humans also rated strong."
          stat={cm?.precision_at_5}
          format={(v) => `${Math.round(v * 100)}%`}
        />
        <MetricCard
          icon={Target}
          title="Precision@10"
          blurb="Same idea, over the top 10."
          stat={cm?.precision_at_10}
          format={(v) => `${Math.round(v * 100)}%`}
        />
      </div>

      {cm?.small_sample_caveat && (
        <div className="flex gap-2.5 rounded-[var(--card-radius)] bg-lowconf-fill text-lowconf-text p-4">
          <Info className="w-4 h-4 shrink-0 mt-0.5" />
          <p className="text-small">{cm.small_sample_caveat}</p>
        </div>
      )}

      {/* Trend over pipeline versions. */}
      <Card pad="lg">
        <h2 className="text-h3 text-ink mb-1">Spearman ρ over pipeline versions</h2>
        <p className="text-caption text-muted mb-4">Each point is a model version; the band is ±1 std.</p>
        {data.trend?.length > 1 ? (
          <TrendChart trend={data.trend} />
        ) : (
          <div className="flex items-center gap-2 text-small text-muted py-6 justify-center">
            <Clock className="w-4 h-4" />
            Trend appears once two or more evaluated versions exist.
          </div>
        )}
      </Card>

      {/* Feature importance (trained model). */}
      {data.feature_importance && (
        <Card pad="lg">
          <h2 className="text-h3 text-ink mb-3">What the model weights most</h2>
          <FeatureImportance importance={data.feature_importance} />
        </Card>
      )}
    </div>
  );
}

function ReadinessBanner({ state, details, progress }) {
  const map = {
    tuned: { icon: CheckCircle2, fill: 'var(--fit-fill)', text: 'var(--fit-text)', label: 'Calibrated' },
    provisional: { icon: Info, fill: 'var(--gap-fill)', text: 'var(--gap-text)', label: 'Provisional weights' },
    unready: { icon: Clock, fill: 'var(--lowconf-fill)', text: 'var(--lowconf-text)', label: 'Collecting ground truth' },
  };
  const s = map[state] || map.unready;
  const Icon = s.icon;
  const done = progress?.pairs_with_full_rater_coverage ?? 0;
  const total = progress?.total_target ?? 0;
  const pct = total ? Math.round((done / total) * 100) : 0;
  return (
    <div className="rounded-[var(--card-radius)] p-5" style={{ background: s.fill }}>
      <div className="flex items-start gap-3">
        <Icon className="w-5 h-5 shrink-0 mt-0.5" style={{ color: s.text }} />
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="text-small font-semibold" style={{ color: s.text }}>{s.label}</span>
            <span className="text-caption rounded-full px-2 py-0.5 tabular-nums" style={{ background: 'var(--surface)', color: s.text }}>
              {state}
            </span>
          </div>
          <p className="text-small mt-1" style={{ color: s.text }}>{details}</p>
          {total > 0 && (
            <div className="mt-3">
              <div className="flex items-center justify-between text-caption tabular-nums mb-1" style={{ color: s.text }}>
                <span>Ground-truth pairs rated</span>
                <span>{done}/{total}</span>
              </div>
              <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--surface)' }}>
                <div className="h-full rounded-full" style={{ width: `${pct}%`, background: s.text }} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricCard({ icon: Icon, title, blurb, stat, format }) {
  const ready = stat && stat.mean != null;
  return (
    <Card pad="lg">
      <div className="flex items-center gap-2 mb-1">
        <Icon className="w-4 h-4 text-slate-700" />
        <h3 className="text-small font-semibold text-ink">{title}</h3>
      </div>
      {ready ? (
        <>
          <p className="text-display tabular-nums text-ink" style={{ fontFamily: 'var(--font-sans)', fontSize: 40 }}>
            {format(stat.mean)}
          </p>
          <p className="text-caption text-muted tabular-nums">
            {stat.std != null ? `± ${format(stat.std)} std` : 'std n/a'} · n = {stat.sample_size}
          </p>
        </>
      ) : (
        <>
          <p className="text-display tabular-nums text-muted/50" style={{ fontFamily: 'var(--font-sans)', fontSize: 40 }}>—</p>
          <p className="text-caption text-muted">Pending ground truth</p>
        </>
      )}
      <p className="text-caption text-muted mt-2 leading-relaxed">{blurb}</p>
    </Card>
  );
}

function TrendChart({ trend }) {
  const pts = trend.filter((t) => t.spearman_mean != null);
  if (pts.length < 2) return null;
  const W = 640, H = 180, pad = 30;
  const xs = (i) => pad + (i * (W - 2 * pad)) / (pts.length - 1);
  const ys = (v) => H - pad - ((v + 1) / 2) * (H - 2 * pad); // spearman −1..1
  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${xs(i)} ${ys(p.spearman_mean)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <line x1={pad} y1={ys(0)} x2={W - pad} y2={ys(0)} stroke="var(--border)" strokeDasharray="3 3" />
      <path d={line} fill="none" stroke="var(--ember-500)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      {pts.map((p, i) => (
        <g key={i}>
          <circle cx={xs(i)} cy={ys(p.spearman_mean)} r="4" fill="var(--ember-500)" />
          <text x={xs(i)} y={H - 8} textAnchor="middle" fontSize="10" fill="var(--muted)">{p.pipeline_version}</text>
        </g>
      ))}
    </svg>
  );
}

function FeatureImportance({ importance }) {
  const entries = Object.entries(importance).filter(([, v]) => typeof v === 'number').sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, v]) => v), 1);
  return (
    <ul className="space-y-2.5">
      {entries.map(([k, v]) => (
        <li key={k}>
          <div className="flex justify-between text-small">
            <span className="capitalize text-ink">{k.replace(/_/g, ' ')}</span>
            <span className="tabular-nums text-muted">{(v * 100).toFixed(0)}%</span>
          </div>
          <div className="h-2 rounded-full bg-border mt-1 overflow-hidden">
            <div className="h-full rounded-full" style={{ width: `${(v / max) * 100}%`, background: 'var(--slate-700)' }} />
          </div>
        </li>
      ))}
    </ul>
  );
}
