import { createContext, useContext, useState, useCallback, useMemo } from 'react';

// Recruiter session state: Basic-auth creds (in memory only, never persisted),
// the last ranking result, weight overrides (§10.7), and blind mode (§11.3).
const RecruiterCtx = createContext(null);

// Default feature weights — equal spine; recruiters tilt these (e.g. skills 2×).
export const DEFAULT_WEIGHTS = { skillOverlap: 1, embedding: 1, tfidf: 1, expMatch: 1, eduMatch: 1 };

export function RecruiterProvider({ children }) {
  const [auth, setAuth] = useState(null);         // { username, password } | null
  const [ranking, setRanking] = useState(null);   // RankResponse
  const [weights, setWeights] = useState(DEFAULT_WEIGHTS);
  const [blindMode, setBlindMode] = useState(false);

  const signIn = useCallback((username, password) => setAuth({ username, password }), []);
  const signOut = useCallback(() => { setAuth(null); setRanking(null); }, []);

  const value = useMemo(
    () => ({
      auth, signIn, signOut, isAuthed: !!auth,
      ranking, setRanking,
      weights, setWeights,
      blindMode, setBlindMode,
    }),
    [auth, signIn, signOut, ranking, weights, blindMode],
  );
  return <RecruiterCtx.Provider value={value}>{children}</RecruiterCtx.Provider>;
}

export function useRecruiter() {
  const ctx = useContext(RecruiterCtx);
  if (!ctx) throw new Error('useRecruiter must be used within <RecruiterProvider>');
  return ctx;
}
