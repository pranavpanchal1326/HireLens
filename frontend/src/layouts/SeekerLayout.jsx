import { Outlet, NavLink, Link } from 'react-router-dom';
import { Sparkles, RefreshCw, HeartPulse, ShieldCheck } from 'lucide-react';
import BrandMark from '../components/BrandMark';
import { AnalysisProvider } from '../seeker/AnalysisContext';

// Seeker chrome (§4): warm, spacious, encouraging. Single-column reading width,
// generous air. Nav is calm — one active accent, never a busy toolbar (P4).
const nav = [
  { to: '/seeker/analyze', label: 'Analyze fit', icon: Sparkles },
  { to: '/seeker/rescan', label: 'Rescan', icon: RefreshCw },
  { to: '/seeker/health', label: 'Resume health', icon: HeartPulse },
  { to: '/seeker/blind', label: 'Blind mode', icon: ShieldCheck },
];

export default function SeekerLayout() {
  return (
    // §4: this subtree IS the seeker temperament — scope data-theme here so every
    // CSS variable below (canvas/surface/ink/radii) resolves to the warm side.
    <div data-theme="seeker" className="min-h-screen bg-canvas text-ink flex flex-col">
      <header className="border-b border-border bg-surface/80 backdrop-blur sticky top-0 z-30">
        <div className="max-w-[1120px] mx-auto px-6 h-16 flex items-center justify-between">
          <Link to="/seeker/analyze" className="flex items-center gap-2.5">
            <BrandMark size={30} />
            <span className="text-h3 font-editorial text-ink">HireLens</span>
          </Link>
          {/* Nav collapses to icon-only below md so it stays usable on mobile
              (never disappears — §16). Labels return at md+. */}
          <nav className="flex items-center gap-0.5 sm:gap-1">
            {nav.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                title={label}
                className={({ isActive }) =>
                  `flex items-center gap-1.5 px-2.5 sm:px-3 h-11 rounded-md text-small font-medium transition-colors duration-200 ${
                    isActive
                      ? 'bg-ember-50 text-ember-700'
                      : 'text-muted hover:text-ink hover:bg-canvas'
                  }`
                }
              >
                <Icon className="w-4 h-4 shrink-0" strokeWidth={1.75} />
                <span className="hidden md:inline">{label}</span>
              </NavLink>
            ))}
          </nav>
          <Link
            to="/recruiter"
            className="hidden sm:block text-caption text-muted hover:text-ink transition-colors"
          >
            I'm a recruiter →
          </Link>
        </div>
      </header>

      {/* Reading-width canvas (720px seeker cap, §8/§16), roomy vertical rhythm.
          AnalysisProvider shares session-only inputs + score history across the
          seeker screens (Analyze ↔ Rescan). */}
      <main className="flex-1 w-full">
        <AnalysisProvider>
          <Outlet />
        </AnalysisProvider>
      </main>

      <footer className="border-t border-border py-6">
        <div className="max-w-[1120px] mx-auto px-6 text-caption text-muted flex flex-wrap gap-x-6 gap-y-1 justify-between">
          <span>Resumes aren't stored beyond your session without consent.</span>
          <span>Every score shows its reasoning.</span>
        </div>
      </footer>
    </div>
  );
}
