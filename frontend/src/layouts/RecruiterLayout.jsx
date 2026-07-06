import { useState } from 'react';
import { Outlet, NavLink, Link, useLocation } from 'react-router-dom';
import { LayoutGrid, ListOrdered, Activity, LogOut, Lock, ArrowLeft } from 'lucide-react';
import BrandMark from '../components/BrandMark';
import Button from '../components/ui/Button';
import Card from '../components/ui/Card';
import { Input } from '../components/ui/Field';
import ModeToggle from '../components/ModeToggle';
import { RecruiterProvider, useRecruiter } from '../recruiter/RecruiterContext';

// Recruiter chrome (§4): cool greige, dense, volume-first. Slate-anchored header
// (§5.4). Recruiter endpoints need HTTP Basic auth (PRD §9), so the shell gates
// behind a sign-in until credentials are held (in memory only).
const nav = [
  { to: '/recruiter/batch', label: 'Batch upload', icon: LayoutGrid },
  { to: '/recruiter/ranked', label: 'Ranked list', icon: ListOrdered },
  { to: '/recruiter/dashboard', label: 'Accuracy', icon: Activity },
];

export default function RecruiterLayout() {
  return (
    <RecruiterProvider>
      <RecruiterShell />
    </RecruiterProvider>
  );
}

function RecruiterShell() {
  const { isAuthed, signOut, auth } = useRecruiter();
  const location = useLocation();
  return (
    <div data-density="recruiter" className="min-h-screen bg-canvas text-ink flex flex-col">
      <header className="sticky top-0 z-30" style={{ background: 'var(--slate-700)' }}>
        <div className="max-w-[1280px] mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link
              to="/"
              className="flex items-center gap-1.5 text-caption text-white/60 hover:text-white transition-colors mr-1"
              title="Back to Landing Page"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Back</span>
            </Link>
            <div className="h-4 w-px bg-white/20 hidden sm:block mr-1" />
            <Link to="/recruiter/ranked" className="flex items-center gap-2.5">
              <BrandMark size={26} onDark />
              <span className="text-h3 text-white/95">HireLens</span>
              <span className="text-caption text-white/55 border-l border-white/20 pl-2.5 ml-1">Recruiter</span>
            </Link>
          </div>
          {isAuthed && (
            <nav className="flex items-center gap-0.5 sm:gap-1">
              {nav.map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  title={label}
                  className={({ isActive }) =>
                    `flex items-center gap-1.5 px-2.5 sm:px-3 h-10 rounded-md text-small font-medium transition-colors duration-200 ${
                      isActive ? 'bg-white/15 text-white' : 'text-white/70 hover:text-white hover:bg-white/10'
                    }`
                  }
                >
                  <Icon className="w-4 h-4 shrink-0" strokeWidth={1.75} />
                  <span className="hidden md:inline">{label}</span>
                </NavLink>
              ))}
            </nav>
          )}
          <div className="flex items-center gap-2">
            <ModeToggle onDark />
            {isAuthed && (
              <button onClick={signOut} className="flex items-center gap-1.5 text-caption text-white/70 hover:text-white transition-colors">
                <LogOut className="w-3.5 h-3.5" /> {auth?.username}
              </button>
            )}
            <Link to="/seeker" className="text-caption text-white/60 hover:text-white transition-colors">← Seeker view</Link>
          </div>
        </div>
      </header>

      <main className="flex-1 w-full">
        {isAuthed ? (
          <div key={location.pathname} className="page-enter">
            <Outlet />
          </div>
        ) : (
          <RecruiterSignIn />
        )}
      </main>

      <footer className="border-t border-border py-4">
        <div className="max-w-[1280px] mx-auto px-6 text-caption text-muted flex justify-between">
          <span>Per-account data isolation — no cross-company visibility.</span>
          <span>Ranks are explainable and exportable.</span>
        </div>
      </footer>
    </div>
  );
}

function RecruiterSignIn() {
  const { signIn } = useRecruiter();
  const [username, setUsername] = useState('recruiter_one');
  const [password, setPassword] = useState('password123');
  return (
    <div className="max-w-[440px] mx-auto px-6 py-16">
      <Card pad="lg">
        <span className="w-11 h-11 rounded-xl flex items-center justify-center text-white mb-4" style={{ background: 'var(--slate-700)' }}>
          <Lock className="w-5 h-5" strokeWidth={1.75} />
        </span>
        <h1 className="text-h2 text-ink">Recruiter sign in</h1>
        <p className="text-small text-muted mt-1.5">
          Recruiter tools are account-scoped. Your data never crosses company boundaries.
        </p>
        <form
          className="mt-6 space-y-4"
          onSubmit={(e) => { e.preventDefault(); signIn(username.trim(), password); }}
        >
          <Input id="rec-user" label="Username" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" />
          <Input id="rec-pass" label="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" />
          <Button type="submit" variant="primary" size="lg" className="w-full">Sign in</Button>
        </form>
        <p className="text-caption text-muted mt-4">
          Demo account prefilled (<span className="tabular-nums">recruiter_one</span>) — this is a portfolio build.
        </p>
      </Card>
    </div>
  );
}
