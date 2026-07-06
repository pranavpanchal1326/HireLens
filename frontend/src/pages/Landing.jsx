import { Link } from 'react-router-dom';
import { ArrowRight, User, Users } from 'lucide-react';

// Landing — the two-temperament idea, stated in the first three seconds (§1.1,
// §4). One engine, two doors. Seeker door is warm/spacious; recruiter door is
// cool/slate. The single aperture motif overhead is the shared signature (§4
// handshake). Sentence case throughout (§7.3).
export default function Landing() {
  return (
    <div
      data-theme="seeker"
      className="min-h-screen bg-canvas text-ink flex flex-col items-center justify-center px-6 py-16 relative overflow-hidden"
    >
      {/* Ambient warmth — one soft ember wash, no chartjunk (P4). */}
      <div
        aria-hidden
        className="pointer-events-none absolute -top-56 left-1/2 -translate-x-1/2 w-[560px] h-[560px] rounded-full opacity-[0.14]"
        style={{ background: 'radial-gradient(circle, var(--ember-500), transparent 60%)' }}
      />

      <header className="relative z-10 flex flex-col items-center text-center max-w-2xl">
        {/* Aperture mark — a still, resolved bloom (§6). */}
        <LandingMark />

        <p className="text-caption uppercase tracking-[0.2em] text-muted mt-8 mb-3">
          Resume intelligence
        </p>
        <h1 className="text-display text-ink" style={{ fontSize: 'clamp(34px, 6vw, 52px)' }}>
          See your fit come into focus.
        </h1>
        <p className="text-body text-muted mt-5 max-w-md">
          One engine, two temperaments — a calm coaching space for job seekers, a
          fast, honest cockpit for recruiters. Every score shows its reasoning.
        </p>
      </header>

      {/* Two doors. */}
      <div className="relative z-10 mt-14 w-full max-w-3xl grid grid-cols-1 sm:grid-cols-2 gap-5">
        <DoorCard
          to="/seeker"
          theme="seeker"
          icon={<User className="w-5 h-5" />}
          eyebrow="For job seekers"
          title="Improve my resume"
          body="Upload once, see what to fix, and watch your fit climb with each revision."
        />
        <DoorCard
          to="/recruiter"
          theme="recruiter"
          icon={<Users className="w-5 h-5" />}
          eyebrow="For recruiters"
          title="Rank a candidate pool"
          body="One job, many resumes — ranked, explained, and defensible in seconds."
        />
      </div>

      <p className="relative z-10 text-caption text-muted mt-12">
        Free to use · resumes read for your session only · every score shows its reasoning
      </p>
    </div>
  );
}

function DoorCard({ to, theme, icon, eyebrow, title, body }) {
  const cool = theme === 'recruiter';
  return (
    <Link
      to={to}
      className="group relative rounded-2xl border p-7 flex flex-col transition-all duration-[320ms] hover:-translate-y-1"
      style={{
        background: cool ? 'var(--rec-surface, #FCFBF8)' : 'var(--surface)',
        borderColor: 'var(--border)',
        boxShadow: 'var(--shadow-sm)',
      }}
      onMouseEnter={(e) => (e.currentTarget.style.boxShadow = 'var(--shadow-md)')}
      onMouseLeave={(e) => (e.currentTarget.style.boxShadow = 'var(--shadow-sm)')}
    >
      <span
        className="w-11 h-11 rounded-xl flex items-center justify-center text-white mb-5"
        style={{ background: cool ? 'var(--slate-700)' : 'var(--ember-500)' }}
      >
        {icon}
      </span>
      <span className="text-caption uppercase tracking-wider text-muted">{eyebrow}</span>
      <h2 className="text-h2 text-ink mt-1">{title}</h2>
      <p className="text-small text-muted mt-2 flex-1">{body}</p>
      <span
        className="mt-6 inline-flex items-center gap-1.5 text-small font-medium"
        style={{ color: cool ? 'var(--slate-700)' : 'var(--ember-700)' }}
      >
        Continue
        <ArrowRight className="w-4 h-4 transition-transform duration-200 group-hover:translate-x-1" />
      </span>
    </Link>
  );
}

// A resolved, still aperture bloom used purely as brand art on the landing.
function LandingMark() {
  const petals = [0.9, 0.62, 0.78, 0.5, 0.7];
  const CX = 60, CY = 64;
  const path = (v) => {
    const tip = 20 + v * 26, w = 9 + v * 4;
    return `M${CX} ${CY} Q ${CX - w} ${CY - tip * 0.55} ${CX} ${CY - tip} Q ${CX + w} ${CY - tip * 0.55} ${CX} ${CY} Z`;
  };
  return (
    <svg viewBox="0 0 120 128" width="112" height="120" role="img" aria-label="HireLens">
      <g opacity="0.92">
        {petals.map((v, k) => (
          <path
            key={k}
            d={path(v)}
            fill={k % 3 === 0 ? 'var(--ember-500)' : 'var(--ember-300)'}
            transform={`rotate(${k * 72} ${CX} ${CY})`}
            opacity="0.72"
          />
        ))}
      </g>
      <circle cx={CX} cy={CY} r="42" fill="none" stroke="var(--fit-500)" strokeWidth="3.5"
        strokeLinecap="round" pathLength="100" strokeDasharray="82 100" transform={`rotate(-90 ${CX} ${CY})`} />
      <circle cx={CX} cy={CY} r="16" fill="var(--surface)" stroke="var(--border)" strokeWidth="1.5" />
      <circle cx={CX} cy={CY} r="4" fill="var(--ember-500)" />
    </svg>
  );
}
