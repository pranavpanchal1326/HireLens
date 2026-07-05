import React, { useState } from 'react';
import axios from 'axios';
import { Target, Sparkles, RefreshCw, AlertTriangle, ShieldCheck } from 'lucide-react';
import ThemeToggle from './components/ThemeToggle';
import ResumeParser from './components/ResumeParser';
import JobDescriptionInput from './components/JobDescriptionInput';
import ApertureBloom from './components/ApertureBloom';
import SkillAlignment from './components/SkillAlignment';

export default function App() {
  const [parsedResume, setParsedResume] = useState(null);
  const [parsedJd, setParsedJd] = useState(null);
  const [scoreResult, setScoreResult] = useState(null);
  const [pipelineMaturity, setPipelineMaturity] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const calculateFitScore = async () => {
    if (!parsedResume || !parsedJd) return;

    setLoading(true);
    setError(null);
    setScoreResult(null);

    const payload = {
      raw_resume_text: parsedResume.raw_text,
      raw_jd_text: parsedJd.raw_text,
    };

    try {
      const response = await axios.post('http://localhost:8000/api/v1/score', payload);
      setScoreResult(response.data.score_result);
      setPipelineMaturity(response.data.pipeline_maturity);
    } catch (err) {
      console.error(err);
      if (err.response && err.response.data && err.response.data.message) {
        setError(err.response.data.message);
      } else {
        setError('Failed to compute score. Please check your backend connection.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Confidence pill colors — mapped to the blueprint fit/ember/gap/lowconf families.
  const getConfidenceColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'high':
        return 'bg-fit-fill text-fit-text';
      case 'medium':
        return 'bg-ember-50 text-ember-700';
      case 'low':
        return 'bg-gap-fill text-gap-text';
      default:
        return 'bg-lowconf-fill text-lowconf-text';
    }
  };

  // Large score numeral (≥24px) — ember-500 fill is permitted here per §5.1.
  const getScoreColor = (score) => {
    if (score >= 80) return 'text-fit-text';
    if (score >= 60) return 'text-ember-score';
    return 'text-gap-text';
  };

  return (
    <div className="min-h-screen bg-canvas text-ink flex flex-col">
      {/* Header */}
      <header className="border-b border-border bg-surface py-4 shadow-sm">
        <div className="max-w-[1600px] mx-auto px-6 flex justify-between items-center">
          <div className="flex items-center gap-2.5">
            <div className="bg-ember-500 text-white p-2 rounded-lg shadow-md">
              <Target className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight m-0 text-ink">HireLens</h1>
              <p className="text-[10px] text-muted font-medium">AI-POWERED CANDIDATE MATCHING & SKILL GAP ANALYSIS</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="flex-1 max-w-[1600px] w-full mx-auto px-6 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Side: Inputs & Uploads */}
        <section className="lg:col-span-5 flex flex-col">
          <ResumeParser onParsed={setParsedResume} parsedData={parsedResume} />
          <JobDescriptionInput onParsed={setParsedJd} parsedData={parsedJd} />

          {/* Action Trigger */}
          <div className="mt-5">
            <button
              onClick={calculateFitScore}
              disabled={!parsedResume || !parsedJd || loading}
              className={`w-full py-3.5 rounded-xl font-medium text-sm transition-all duration-200 shadow-sm flex items-center justify-center gap-2 cursor-pointer
                ${(!parsedResume || !parsedJd)
                  ? 'bg-canvas text-muted cursor-not-allowed border border-border'
                  : 'bg-ember-500 hover:bg-ember-700 text-white font-semibold hover:shadow-lg'
                }`}
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Calculating Match Vector...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Calculate Match Alignment
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="mt-4 bg-gap-fill border border-gap-500 text-gap-text rounded-lg p-4 flex gap-3 text-xs">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold block">Scoring Engine Error</span>
                {error}
              </div>
            </div>
          )}
        </section>

        {/* Right Side: Analysis & Visualization */}
        <section className="lg:col-span-7 flex flex-col bg-surface border border-border rounded-2xl p-6 shadow-sm min-h-[500px]">
          {loading ? (
            <div className="flex-1 flex flex-col items-center justify-center">
              <RefreshCw className="w-10 h-10 text-ember-500 animate-spin mb-4" />
              <h3 className="text-sm font-semibold text-ink">Evaluating Profile Alignment...</h3>
              <p className="text-xs text-muted mt-1 max-w-sm text-center">
                Generating TF-IDF matrices, computing embedding cosines, and executing RAG-based semantic skill matcher.
              </p>
            </div>
          ) : scoreResult ? (
            <div className="flex-1 flex flex-col justify-between animate-slide-up">
              <div>
                {/* Score & Profile Header */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 pb-5 border-b border-border">
                  <div className="flex items-center gap-4">
                    {/* Score Dial */}
                    <div className="relative w-18 h-18 rounded-full border-4 border-border flex items-center justify-center bg-canvas">
                      <span className={`text-2xl font-bold tabular-nums ${getScoreColor(scoreResult.final_score)}`}>
                        {scoreResult.final_score}
                      </span>
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-ink m-0">Fit Score Report</h2>
                      <div className="flex flex-wrap gap-2 mt-1.5">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-medium tracking-wide uppercase ${getConfidenceColor(scoreResult.confidence_level)}`}>
                          Confidence: {scoreResult.confidence_level}
                        </span>
                        <span className="bg-canvas text-muted px-2 py-0.5 rounded text-[10px] font-medium">
                          Maturity: {pipelineMaturity || 'Uncalibrated'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Version tag */}
                  <div className="text-right">
                    <span className="bg-ember-50 text-ember-700 border border-ember-100 px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wide tabular-nums">
                      {scoreResult.pipeline_version}
                    </span>
                  </div>
                </div>

                {/* Aperture bloom radar */}
                <div className="border border-border rounded-xl bg-canvas mt-5">
                  <ApertureBloom
                    featureVector={scoreResult.feature_vector}
                    score={scoreResult.final_score}
                    confidence={scoreResult.scoring_confidence}
                    confidenceBand={scoreResult.confidence_level}
                  />
                </div>

                {/* Skill alignment exact vs semantic + gaps */}
                <SkillAlignment
                  matchedSkills={scoreResult.matched_skills}
                  gaps={scoreResult.gaps}
                />

                {/* Explainability Panels / Confidence Reasons */}
                {scoreResult.confidence_reasons && scoreResult.confidence_reasons.length > 0 && (
                  <div className="mt-6 bg-canvas border border-border rounded-xl p-4">
                    <h4 className="text-[10px] font-semibold uppercase tracking-wider text-muted mb-2.5 flex items-center gap-1.5">
                      <ShieldCheck className="w-4 h-4 text-ember-500" /> Explainability & Confidence Rationale
                    </h4>
                    <ul className="space-y-1.5 text-xs text-muted list-disc list-inside">
                      {scoreResult.confidence_reasons.map((reason, i) => (
                        <li key={i}>{reason}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="mt-8 pt-4 border-t border-border text-[10px] text-muted font-medium flex justify-between items-center">
                <span>Calculated on {new Date(scoreResult.created_at).toLocaleDateString()}</span>
                <span className="tabular-nums">Trace ID: {scoreResult.score_id.slice(0, 8)}...</span>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
              <div className="w-12 h-12 rounded-full bg-canvas flex items-center justify-center text-muted mb-4 border border-border">
                <Target className="w-6 h-6" />
              </div>
              <h3 className="text-sm font-semibold text-ink">No Assessment Loaded</h3>
              <p className="text-xs text-muted mt-2 max-w-sm leading-relaxed">
                Ingest your resume and supply the target job description details in the side panel, then calculate the match vector to view full analytics.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
