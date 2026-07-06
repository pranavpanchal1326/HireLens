import { Link } from 'react-router-dom';
import { ArrowRight, ArrowUpRight, User, Users, ScanLine, Sparkles, ShieldCheck } from 'lucide-react';
import BrandMark from '../components/BrandMark';
import ModeToggle from '../components/ModeToggle';
import ApertureBloom from '../components/ApertureBloom';
import ApertureRings from '../components/ApertureRings';
import Reveal from '../components/Reveal';

// "Into Focus" landing — cinematic minimal hero, Enhancv warmth. The world is
// graphite + paper; the aperture bloom is the one warm light. On load the headline
// resolves from blur (focus-pull) and the bloom breathes. Everything below is calm.
const HERO_FV = { tfidf_score: 0.78, embedding_score: 0.88, skill_overlap_pct: 0.92, exp_match: 0.7, edu_match: 0.66 };

export default function Landing() {
  return (
    <div data-density="seeker" className="min-h-screen bg-canvas text-ink">
      {/* minimal top bar */}
      <header className="absolute top-0 inset-x-0 z-20">
        <div className="max-w-[1200px] mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <BrandMark size={28} />
            <span className="text-h3 font-editorial">HireLens</span>
          </div>
          <div className="flex items-center gap-3">
            <ModeToggle />
            <Link to="/recruiter" className="text-caption text-muted hover:text-ink transition-colors">For recruiters →</Link>
          </div>
        </div>
      </header>

      {/* ── HERO ─────────────────────────────────────────────────────────── */}
      <section className="relative min-h-screen flex items-center overflow-hidden">
        <div className="max-w-[1200px] mx-auto px-6 w-full grid lg:grid-cols-2 gap-12 items-center pt-24 pb-16">
          {/* Left: the words, resolving from blur */}
          <div className="relative z-10">
            <p className="reveal text-caption uppercase tracking-[0.24em] text-muted mb-5" style={{ animationDelay: '0ms' }}>
              Resume intelligence
            </p>
            <h1 className="font-editorial text-ink leading-[1.02] tracking-[-0.02em]" style={{ fontSize: 'clamp(40px, 6.5vw, 76px)' }}>
              <span className="reveal block" style={{ animationDelay: '80ms' }}>See your fit</span>
              <span className="reveal block" style={{ animationDelay: '220ms' }}>come into <span className="text-ember-500">focus.</span></span>
            </h1>
            <p className="reveal text-body text-muted mt-6 max-w-md" style={{ animationDelay: '380ms' }}>
              Upload once. See exactly what's working, what to fix, and watch your score
              climb — with the reasoning shown, never a black box.
            </p>
            <div className="reveal flex flex-wrap items-center gap-3 mt-8" style={{ animationDelay: '480ms' }}>
              <Link
                to="/seeker"
                className="inline-flex items-center gap-2 h-12 px-6 rounded-xl bg-ember-500 text-white font-medium text-small shadow-sm hover:brightness-[0.94] transition-all focus-ember glow-ember"
              >
                Improve my resume <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                to="/recruiter"
                className="inline-flex items-center gap-2 h-12 px-5 rounded-xl border border-border text-ink font-medium text-small hover:bg-veil transition-all focus-ember"
              >
                I'm hiring
              </Link>
            </div>
            <p className="reveal text-caption text-muted mt-5" style={{ animationDelay: '560ms' }}>
              Free to use · resumes read for your session only · no dark patterns
            </p>
          </div>

          {/* Right: the living bloom inside a drifting aperture-ring backdrop */}
          <div className="relative flex justify-center lg:justify-end">
            {/* Optical art — concentric focus rings drifting behind the bloom. */}
            <div aria-hidden className="pointer-events-none absolute inset-0 flex items-center justify-center lg:justify-end">
              <div className="w-[460px] h-[460px] max-w-[110vw] -mr-4 opacity-90">
                <ApertureRings />
              </div>
            </div>
            <div className="reveal-slow relative z-10">
              <ApertureBloom featureVector={HERO_FV} score={86} confidence={0.94} confidenceBand="high" showLegend={false} showLabel={false} alive />
              {/* product tease — a score-result card edge */}
              <div className="hidden md:block absolute -bottom-6 -left-10 w-56 rounded-2xl border border-border bg-surface p-4 shadow-[var(--shadow-md)] rotate-[-4deg]">
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-6 h-6 rounded-md bg-fit-fill text-fit-text flex items-center justify-center text-[11px] font-bold">✓</span>
                  <span className="text-caption text-muted">You already have</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {['Python', 'REST APIs', 'AWS'].map((s) => (
                    <span key={s} className="text-[11px] rounded-full bg-fit-fill text-fit-text px-2 py-0.5">{s}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* scroll cue */}
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 text-caption text-muted flex flex-col items-center gap-1 reveal" style={{ animationDelay: '800ms' }}>
          <span>See how it works</span>
          <span className="w-px h-6 bg-border" />
        </div>
      </section>

      {/* ── TWO DOORS ────────────────────────────────────────────────────── */}
      <section className="max-w-[1120px] mx-auto px-6 py-20">
        <Reveal className="grid sm:grid-cols-2 gap-5">
          <Door to="/seeker" icon={<User className="w-5 h-5" />} eyebrow="For job seekers" title="Improve my resume"
            body="Upload once, see what to fix, and watch your fit climb with each revision." />
          <Door to="/recruiter" icon={<Users className="w-5 h-5" />} eyebrow="For recruiters" title="Rank a candidate pool"
            body="One job, many resumes — ranked, explained, and defensible in seconds." graphite />
        </Reveal>
      </section>

      {/* ── THE HUMAN BEAT ───────────────────────────────────────────────── */}
      <Reveal as="section" className="max-w-[760px] mx-auto px-6 py-16 text-center">
        <p className="text-caption uppercase tracking-[0.2em] text-muted mb-4">Why it's different</p>
        <h2 className="font-editorial text-ink" style={{ fontSize: 'clamp(26px, 4vw, 40px)', lineHeight: 1.15 }}>
          You're not a keyword. You're a person a machine finally read properly.
        </h2>
        <p className="text-body text-muted mt-5 max-w-lg mx-auto">
          Keyword-blind filters reject people for saying "led a team" instead of "people management."
          HireLens reads meaning, shows its reasoning, and tells you what to do next — warmly, honestly.
        </p>
      </Reveal>

      {/* ── HOW IT WORKS ─────────────────────────────────────────────────── */}
      <section className="max-w-[1120px] mx-auto px-6 py-16">
        <Reveal className="grid md:grid-cols-3 gap-5">
          <Frame n="1" icon={ScanLine} title="Read, not scanned" body="We parse your resume into skills, experience, and education — and tell you how much we could read." />
          <Frame n="2" icon={Sparkles} title="Scored, then explained" body="A hybrid model scores fit and blooms it open — five petals for the five signals that drove it." />
          <Frame n="3" icon={ShieldCheck} title="Honest by design" body="Confidence shown, gaps framed as to-dos, identity strippable before scoring. No black box." />
        </Reveal>
        <Reveal className="text-center mt-14">
          <Link to="/seeker" className="inline-flex items-center gap-2 h-13 px-8 rounded-xl bg-ember-500 text-white font-medium text-body shadow-sm hover:brightness-[0.94] transition-all focus-ember glow-ember">
            See your fit come into focus <ArrowRight className="w-5 h-5" />
          </Link>
        </Reveal>
      </section>

      <footer className="border-t border-border">
        <div className="max-w-[1120px] mx-auto px-6 py-8 flex flex-wrap justify-between gap-3 text-caption text-muted">
          <span>HireLens — resume intelligence, in focus.</span>
          <span>Resumes aren't stored beyond your session. Every score shows its reasoning.</span>
        </div>
      </footer>
    </div>
  );
}

function Door({ to, icon, eyebrow, title, body, graphite = false }) {
  return (
    <Link
      to={to}
      className="group relative rounded-2xl border border-border bg-surface p-7 flex flex-col transition-all duration-[320ms] hover:-translate-y-1"
      style={{ boxShadow: 'var(--shadow-sm)' }}
    >
      <span className="w-11 h-11 rounded-xl flex items-center justify-center text-white mb-5"
        style={{ background: graphite ? 'var(--slate-700)' : 'var(--ember-500)' }}>
        {icon}
      </span>
      <span className="text-caption uppercase tracking-wider text-muted">{eyebrow}</span>
      <h3 className="text-h2 text-ink mt-1">{title}</h3>
      <p className="text-small text-muted mt-2 flex-1">{body}</p>
      <span className="mt-6 inline-flex items-center gap-1.5 text-small font-medium text-ember-700 group-hover:gap-2.5 transition-all"
        style={{ color: graphite ? 'var(--ink)' : 'var(--ember-700)' }}>
        Continue <ArrowUpRight className="w-4 h-4" />
      </span>
    </Link>
  );
}

function Frame({ n, icon: Icon, title, body }) {
  return (
    <div className="rounded-2xl border border-border bg-surface p-6" style={{ boxShadow: 'var(--shadow-sm)' }}>
      <div className="flex items-center gap-3 mb-3">
        <span className="w-9 h-9 rounded-lg bg-ember-50 text-ember-700 flex items-center justify-center">
          <Icon className="w-4.5 h-4.5" strokeWidth={1.75} />
        </span>
        <span className="text-caption tabular-nums text-muted">Step {n}</span>
      </div>
      <h3 className="text-h3 text-ink">{title}</h3>
      <p className="text-small text-muted mt-1.5">{body}</p>
    </div>
  );
}
