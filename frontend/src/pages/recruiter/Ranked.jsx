import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { ListOrdered, SlidersHorizontal, EyeOff, RotateCcw, ChevronUp, ChevronDown } from 'lucide-react';
import Button from '../../components/ui/Button';
import Card from '../../components/ui/Card';
import { EmptyState } from '../../components/ui/states';
import ApertureBloomMicro from '../../components/ApertureBloomMicro';
import ExplainabilityPanel from '../../components/recruiter/ExplainabilityPanel';
import { useRecruiter, DEFAULT_WEIGHTS } from '../../recruiter/RecruiterContext';
import { reRank, topDrivers } from '../../recruiter/rankingUtils';

const WEIGHT_KEYS = [
  { key: 'skillOverlap', label: 'Skills' },
  { key: 'embedding', label: 'Semantic' },
  { key: 'expMatch', label: 'Experience' },
  { key: 'eduMatch', label: 'Education' },
  { key: 'tfidf', label: 'Lexical' },
];

export default function Ranked() {
  const { ranking, weights, setWeights, blindMode, setBlindMode } = useRecruiter();
  const [showWeights, setShowWeights] = useState(false);
  const [open, setOpen] = useState(null); // candidate for panel
  const [sortKey, setSortKey] = useState('rank'); // rank | score | confidence

  // Re-rank by the current weights, then apply any column sort override.
  const rows = useMemo(() => {
    const candidates = ranking?.ranking_result?.ranked_candidates || [];
    const ranked = reRank(candidates, weights);
    const sorted = [...ranked];
    if (sortKey === 'score') sorted.sort((a, b) => b.score_result.final_score - a.score_result.final_score);
    else if (sortKey === 'confidence') sorted.sort((a, b) => b.score_result.scoring_confidence - a.score_result.scoring_confidence);
    return sorted;
  }, [ranking, weights, sortKey]);

  const displayNameFor = (c, i) =>
    blindMode ? `Candidate ${String.fromCharCode(65 + i)}` : (c.candidate_id || c.anonymized_display_name || `Candidate ${c.rank}`);

  if (!ranking) {
    return (
      <div className="max-w-[560px] mx-auto px-6 py-16">
        <Card>
          <EmptyState
            icon={ListOrdered}
            title="No ranking yet"
            body="Upload a job description and a pool of candidates to see them ranked and explained."
            action={<Link to="/recruiter/batch"><Button variant="primary">Start a batch</Button></Link>}
          />
        </Card>
      </div>
    );
  }

  const setW = (key, v) => setWeights((w) => ({ ...w, [key]: Math.max(0, Math.min(3, v)) }));

  return (
    <div className="max-w-[1280px] mx-auto px-6 py-8">
      <div className="flex flex-wrap items-end justify-between gap-4 mb-5">
        <div>
          <h1 className="text-h1 text-ink">Ranked candidates</h1>
          <p className="text-small text-muted mt-1 tabular-nums">
            {ranking.total_successful} ranked
            {ranking.total_failed > 0 && ` · ${ranking.total_failed} unreadable`}
            {' · '}{ranking.ranking_result.pipeline_version}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant={blindMode ? 'primary' : 'secondary'} size="sm" icon={EyeOff} onClick={() => setBlindMode((b) => !b)}>
            {blindMode ? 'Blind on' : 'Blind mode'}
          </Button>
          <Button variant="secondary" size="sm" icon={SlidersHorizontal} onClick={() => setShowWeights((s) => !s)}>
            Weights
          </Button>
        </div>
      </div>

      {/* Weight controls — calm, not a wall of sliders (§10.7). */}
      {showWeights && (
        <Card pad="sm" className="mb-4">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
            {WEIGHT_KEYS.map(({ key, label }) => (
              <div key={key} className="flex items-center gap-2">
                <span className="text-caption text-muted w-16">{label}</span>
                <div className="flex items-center gap-1.5">
                  <button onClick={() => setW(key, (weights[key] ?? 1) - 0.5)} className="w-6 h-6 rounded bg-sunken border border-border text-muted hover:text-ink focus-ember">−</button>
                  <span className="text-small tabular-nums w-8 text-center text-ink">{(weights[key] ?? 1).toFixed(1)}×</span>
                  <button onClick={() => setW(key, (weights[key] ?? 1) + 0.5)} className="w-6 h-6 rounded bg-sunken border border-border text-muted hover:text-ink focus-ember">+</button>
                </div>
              </div>
            ))}
            <Button variant="ghost" size="sm" icon={RotateCcw} onClick={() => setWeights(DEFAULT_WEIGHTS)}>Reset</Button>
          </div>
        </Card>
      )}

      {/* Mobile: dense table collapses to stacked cards below md (§16). */}
      <div className="md:hidden space-y-2.5">
        {rows.map((c, i) => {
          const sr = c.score_result;
          const drivers = topDrivers(sr.feature_vector, weights, 2);
          return (
            <Card key={c.candidate_id + i} pad="sm" as="button" interactive
              className="w-full text-left flex items-center gap-3"
              onClick={() => setOpen({ ...c, _name: displayNameFor(c, i) })}
            >
              <span className="text-h3 tabular-nums font-semibold text-ink w-7 text-center shrink-0">{c.rank}</span>
              <ApertureBloomMicro score={sr.final_score} confidence={sr.scoring_confidence} confidenceBand={sr.confidence_level} size={34} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-small font-medium text-ink truncate">{displayNameFor(c, i)}</p>
                  <span className="text-small tabular-nums font-semibold text-ink">{sr.final_score}</span>
                </div>
                <div className="flex items-center gap-1.5 mt-1">
                  <ConfidencePill level={sr.confidence_level} />
                  <span className="text-caption text-muted capitalize truncate">{drivers.map((d) => d.label).join(', ')}</span>
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Desktop: dense table (§10.7). */}
      <Card pad="none" className="overflow-hidden hidden md:block">
        <table className="w-full text-small">
          <thead>
            <tr className="text-left" style={{ background: 'var(--slate-700)' }}>
              <Th className="w-14 text-white/90">#</Th>
              <Th className="text-white/90">Candidate</Th>
              <Th className="w-16 text-white/90">Fit</Th>
              <SortableTh label="Score" active={sortKey === 'score'} onClick={() => setSortKey(sortKey === 'score' ? 'rank' : 'score')} />
              <Th className="text-white/90">Top drivers</Th>
              <SortableTh label="Confidence" active={sortKey === 'confidence'} onClick={() => setSortKey(sortKey === 'confidence' ? 'rank' : 'confidence')} />
            </tr>
          </thead>
          <tbody>
            {rows.map((c, i) => {
              const sr = c.score_result;
              const drivers = topDrivers(sr.feature_vector, weights, 2);
              return (
                <tr
                  key={c.candidate_id + i}
                  onClick={() => setOpen({ ...c, _name: displayNameFor(c, i) })}
                  className="border-t border-border cursor-pointer transition-colors hover:bg-ember-50"
                  style={{ background: i % 2 ? 'var(--canvas)' : 'var(--surface)' }}
                >
                  <Td className="tabular-nums font-semibold text-ink">{c.rank}</Td>
                  <Td className="font-medium text-ink">{displayNameFor(c, i)}</Td>
                  <Td><ApertureBloomMicro score={sr.final_score} confidence={sr.scoring_confidence} confidenceBand={sr.confidence_level} size={30} /></Td>
                  <Td className="tabular-nums font-semibold text-ink">{sr.final_score}</Td>
                  <Td>
                    <span className="flex flex-wrap gap-1">
                      {drivers.map((d) => (
                        <span key={d.key} className="capitalize text-caption bg-sunken border border-border rounded-full px-2 py-0.5 text-muted">
                          {d.label} {Math.round(d.value * 100)}%
                        </span>
                      ))}
                    </span>
                  </Td>
                  <Td>
                    <ConfidencePill level={sr.confidence_level} />
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>

      <p className="text-caption text-muted mt-3">Click any row for the full justification and an editable feedback note.</p>

      {open && (
        <ExplainabilityPanel candidate={open} displayName={open._name} onClose={() => setOpen(null)} />
      )}
    </div>
  );
}

function Th({ children, className = '' }) {
  return <th className={`px-4 py-2.5 text-caption font-semibold uppercase tracking-wide ${className}`}>{children}</th>;
}
function SortableTh({ label, active, onClick }) {
  return (
    <th className="px-4 py-2.5">
      <button onClick={onClick} className="flex items-center gap-1 text-caption font-semibold uppercase tracking-wide text-white/90 hover:text-white">
        {label}
        {active ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronUp className="w-3 h-3 opacity-40" />}
      </button>
    </th>
  );
}
function Td({ children, className = '' }) {
  return <td className={`px-4 py-3 ${className}`}>{children}</td>;
}
function ConfidencePill({ level }) {
  const map = {
    high: 'bg-fit-fill text-fit-text',
    medium: 'bg-gap-fill text-gap-text',
    low: 'bg-lowconf-fill text-lowconf-text',
  };
  return <span className={`text-caption font-medium rounded-full px-2 py-0.5 capitalize ${map[level] || map.low}`}>{level}</span>;
}
