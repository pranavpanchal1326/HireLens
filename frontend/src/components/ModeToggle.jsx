import { useEffect, useState } from 'react';
import { Sun, Moon } from 'lucide-react';
import { getMode, setMode } from '../lib/atmosphere';

// Lightbox ⇄ Darkroom toggle. Poetic reframe: you're not switching a theme, you're
// stepping between the lightbox and the darkroom. Persisted; respects OS on first visit.
export default function ModeToggle({ onDark = false }) {
  const [mode, setLocal] = useState(getMode());

  useEffect(() => { setLocal(getMode()); }, []);

  const toggle = () => {
    const next = mode === 'dark' ? 'light' : 'dark';
    setMode(next);
    setLocal(next);
  };

  const dark = mode === 'dark';
  const base = onDark
    ? 'text-white/70 hover:text-white hover:bg-white/10'
    : 'text-muted hover:text-ink hover:bg-veil';

  return (
    <button
      onClick={toggle}
      className={`flex items-center justify-center w-9 h-9 rounded-md transition-colors duration-200 focus-ember ${base}`}
      aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
      title={dark ? 'Lightbox' : 'Darkroom'}
    >
      {dark ? <Sun className="w-4 h-4" strokeWidth={1.75} /> : <Moon className="w-4 h-4" strokeWidth={1.75} />}
    </button>
  );
}
