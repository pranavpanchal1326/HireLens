import React, { useEffect, useState } from 'react';
import { User, Users } from 'lucide-react';

// R1: repurposed from light/dark (blueprint has no dark axis) to the blueprint's
// seeker <-> recruiter theme layers, toggled via the data-theme attribute.
export default function ThemeToggle() {
  const [theme, setTheme] = useState('seeker');

  useEffect(() => {
    const current = document.documentElement.getAttribute('data-theme') || 'seeker';
    setTheme(current);
  }, []);

  const toggleTheme = () => {
    const next = theme === 'seeker' ? 'recruiter' : 'seeker';
    document.documentElement.setAttribute('data-theme', next);
    setTheme(next);
  };

  return (
    <button
      onClick={toggleTheme}
      className="flex items-center gap-1.5 px-3 py-2 rounded-md border border-border bg-surface text-muted hover:text-ink hover:bg-canvas transition-all duration-200 shadow-sm text-xs font-medium capitalize"
      aria-label="Toggle seeker or recruiter theme"
    >
      {theme === 'seeker' ? <User className="w-4 h-4" /> : <Users className="w-4 h-4" />}
      {theme}
    </button>
  );
}
