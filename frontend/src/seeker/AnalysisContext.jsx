import { createContext, useContext, useState, useCallback, useMemo } from 'react';

// Session-only analysis state shared across seeker screens (Analyze ↔ Rescan).
// IN-MEMORY ONLY — never persisted (privacy §13: resumes aren't stored beyond the
// session). Holds the working inputs and a history of scores so the rescan screen
// can show the before/after delta and momentum trail (§11.1-C).
const AnalysisCtx = createContext(null);

export function AnalysisProvider({ children }) {
  const [resumeText, setResumeText] = useState('');
  const [jdText, setJdText] = useState('');
  const [blindMode, setBlindMode] = useState(false);
  const [history, setHistory] = useState([]); // [{ score, feature_vector, confidence_level, scoring_confidence, created_at }]

  const pushResult = useCallback((result) => {
    setHistory((h) => [
      ...h,
      {
        score: result.final_score,
        feature_vector: result.feature_vector,
        confidence_level: result.confidence_level,
        scoring_confidence: result.scoring_confidence,
        created_at: result.created_at,
      },
    ]);
  }, []);

  const reset = useCallback(() => setHistory([]), []);

  const value = useMemo(
    () => ({
      resumeText, setResumeText,
      jdText, setJdText,
      blindMode, setBlindMode,
      history, pushResult, reset,
    }),
    [resumeText, jdText, blindMode, history, pushResult, reset],
  );

  return <AnalysisCtx.Provider value={value}>{children}</AnalysisCtx.Provider>;
}

export function useAnalysis() {
  const ctx = useContext(AnalysisCtx);
  if (!ctx) throw new Error('useAnalysis must be used within <AnalysisProvider>');
  return ctx;
}
