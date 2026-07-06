import { useState } from 'react';
import { Link } from 'react-router-dom';
import { FileSearch } from 'lucide-react';
import ApertureBloom from '../components/ApertureBloom';
import ApertureBloomMicro from '../components/ApertureBloomMicro';
import Button from '../components/ui/Button';
import Card from '../components/ui/Card';
import Chip from '../components/ui/Chip';
import { Input, Textarea } from '../components/ui/Field';
import ArcMeter from '../components/ui/ArcMeter';
import Dropzone from '../components/ui/Dropzone';
import { EmptyState, ErrorState, SkeletonCard, ApertureLoader } from '../components/ui/states';

// Design-system reference ("the kit"). Grows across D2 (signature) and D3 (all
// component states) into a living gallery — the proof that every band, tier, and
// state is deliberate, not accidental. Reachable at /kit.

// Three representative results — deliberately different feature vectors so the
// blooms are visibly distinct flowers (§6.2 "generative identity").
const SAMPLES = {
  high: {
    score: 86, confidence: 0.94, band: 'high',
    fv: { tfidf_score: 0.82, embedding_score: 0.88, skill_overlap_pct: 0.9, exp_match: 0.8, edu_match: 0.72 },
  },
  medium: {
    score: 64, confidence: 0.66, band: 'medium',
    fv: { tfidf_score: 0.55, embedding_score: 0.7, skill_overlap_pct: 0.48, exp_match: 0.75, edu_match: 0.4 },
  },
  low: {
    score: 41, confidence: 0.28, band: 'low',
    fv: { tfidf_score: 0.3, embedding_score: 0.35, skill_overlap_pct: 0.22, exp_match: 0.4, edu_match: 0.15 },
  },
};

export default function Kit() {
  // Bumping this key remounts the blooms → replays the reveal motion (§6.5), so
  // the animation can be inspected on demand.
  const [replay, setReplay] = useState(0);

  return (
    <div data-density="seeker" className="min-h-screen bg-canvas text-ink">
      <div className="max-w-[1120px] mx-auto px-6 py-12">
        <div className="flex items-center justify-between mb-2">
          <p className="text-caption uppercase tracking-wider text-muted">Design system · living reference</p>
          <Link to="/" className="text-small text-muted hover:text-ink">← Start</Link>
        </div>
        <h1 className="text-display text-ink">The kit</h1>
        <p className="text-body text-muted mt-2 max-w-lg">
          Every confidence band, both signature tiers, and (from D3) every component
          state — proof the system is reasoned, not decorated.
        </p>

        {/* ── Signature · hero tier ────────────────────────────────────────── */}
        <SectionTitle
          title="Signature — hero tier"
          action={
            <button
              onClick={() => setReplay((r) => r + 1)}
              className="text-small font-medium text-ember-700 bg-ember-50 rounded-md px-3 py-1.5 hover:bg-ember-100 transition-colors focus-ember"
            >
              Replay reveal
            </button>
          }
        />
        <div key={replay} className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {Object.entries(SAMPLES).map(([k, s]) => (
            <div key={k} className="rounded-2xl border border-border bg-surface p-5" style={{ boxShadow: 'var(--shadow-sm)' }}>
              <p className="text-caption uppercase tracking-wider text-muted mb-1 capitalize">{k} confidence</p>
              <ApertureBloom
                featureVector={s.fv}
                score={s.score}
                confidence={s.confidence}
                confidenceBand={s.band}
              />
            </div>
          ))}
        </div>

        {/* ── Signature · micro tier ───────────────────────────────────────── */}
        <SectionTitle title="Signature — micro tier (recruiter lists)" />
        <div className="rounded-2xl border border-border bg-surface p-6" style={{ boxShadow: 'var(--shadow-sm)' }}>
          <div className="flex flex-wrap items-center gap-8">
            {Object.entries(SAMPLES).map(([k, s]) => (
              <div key={k} className="flex items-center gap-3">
                <ApertureBloomMicro score={s.score} confidence={s.confidence} confidenceBand={s.band} size={36} />
                <div>
                  <p className="text-small text-ink font-medium capitalize">{k}</p>
                  <p className="text-caption text-muted tabular-nums">ring {Math.round(s.confidence * 100)}%</p>
                </div>
              </div>
            ))}
          </div>
          <p className="text-caption text-muted mt-4">
            No petals; confidence is carried by ring completeness alone. Legible at 24–32px in a 50-row table.
          </p>
        </div>

        {/* ── Buttons (§10.1) ──────────────────────────────────────────────── */}
        <SectionTitle title="Buttons" />
        <Card>
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary">Analyze fit</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="ghost">Ghost action</Button>
            <Button variant="destructive">Delete</Button>
            <Button variant="primary" loading>Scoring</Button>
            <Button variant="primary" disabled>Disabled</Button>
          </div>
        </Card>

        {/* ── Chips (§10.4 / §10.6) ────────────────────────────────────────── */}
        <SectionTitle title="Skill chips — matched · missing · present · semantic ≈" />
        <Card>
          <div className="flex flex-wrap gap-2">
            <Chip kind="matched">Python</Chip>
            <Chip kind="matched">REST APIs</Chip>
            <Chip kind="missing">Kubernetes</Chip>
            <Chip kind="missing">Terraform</Chip>
            <Chip kind="present">Excel</Chip>
            <Chip kind="semantic" title='"led a team" ≈ people management'>people management</Chip>
          </div>
          <p className="text-caption text-muted mt-3">
            Every chip pairs color with an icon, so meaning survives grayscale. The ≈ chip shows the system
            understood intent (RAG match), not just a keyword.
          </p>
        </Card>

        {/* ── Inputs (§10.2) ───────────────────────────────────────────────── */}
        <SectionTitle title="Inputs" />
        <Card>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <Input id="k-role" label="Target role" placeholder="e.g. Senior Backend Engineer" hint="16px text — never zooms on mobile." />
            <Textarea id="k-jd" label="Job description" rows={4} placeholder="Paste the job description…" />
          </div>
        </Card>

        {/* ── Dual meters (§10.9) ──────────────────────────────────────────── */}
        <SectionTitle title="Confidence meters — parsing vs scoring, never conflated" />
        <Card>
          <div className="flex flex-wrap gap-10">
            <ArcMeter value={0.92} tone="fit" label="Parsing confidence" sublabel="11/12 fields read" />
            <ArcMeter value={0.66} tone="gap" label="Scoring confidence" sublabel="Model certainty" />
          </div>
          <p className="text-caption text-muted mt-3">
            Two distinct meters. A low score can always be traced to “we couldn’t read it” vs “genuine mismatch.”
          </p>
        </Card>

        {/* ── Dropzone (§10.2) ─────────────────────────────────────────────── */}
        <SectionTitle title="Resume dropzone — resting · parsing · parsed" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <Dropzone onFile={() => {}} />
          <Dropzone onFile={() => {}} parsing />
          <Dropzone
            onFile={() => {}}
            parsed={{ fileName: 'jane_doe_resume.pdf', parseConfidence: 0.92, fieldsFound: 11, fieldsExpected: 12 }}
          />
        </div>

        {/* ── States (§10.10) ──────────────────────────────────────────────── */}
        <SectionTitle title="States — empty · loading · error · low-confidence" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <Card pad="none" className="overflow-hidden">
            <EmptyState
              icon={FileSearch}
              title="No assessment yet"
              body="Add your resume and a job description, then analyze your fit."
              action={<Button variant="primary">Analyze fit</Button>}
            />
          </Card>
          <Card>
            <ApertureLoader />
          </Card>
          <Card>
            <ErrorState
              body="This looks like a scanned image. Try exporting a text-based PDF, or paste the text directly."
              action={<Button variant="secondary" size="sm">Paste text instead</Button>}
            />
          </Card>
          <Card>
            <p className="text-caption uppercase tracking-wider text-muted mb-3">Loading skeleton</p>
            <SkeletonCard />
          </Card>
        </div>

        <div className="h-16" />
      </div>
    </div>
  );
}

function SectionTitle({ title, action }) {
  return (
    <div className="flex items-center justify-between mt-12 mb-4 pb-2 border-b border-border">
      <h2 className="text-h3 text-ink">{title}</h2>
      {action}
    </div>
  );
}
