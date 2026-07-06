# Changelog

All notable changes to HireLens are documented here. This project follows the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

### Added
- Phase 0.1: Repo scaffold, environment config, and CI-ready test harness established.
- Phase D1 (Design Blueprint): React Router shell with `/seeker/*` and `/recruiter/*`
  subtrees, each scoping `data-theme` so the two temperaments (§4) resolve from one
  token spine. Two-door landing page (Fraunces headline + resolved aperture mark),
  seeker & recruiter chrome (slate-anchored recruiter header, §5.4), typography-scale
  utilities (§7.2), non-suppressible ember focus rings (§5.6), and a global
  reduced-motion damp (§9). Retired the monolithic single-screen `App.jsx`.
- Phase D2 (Design Blueprint): Signature completed top-to-bottom. Hero `ApertureBloom`
  gains configurable heading/legend props; both hero and micro tiers verified across
  all three confidence bands (§6.3) — high (green full ring, ember-score numeral),
  medium (ochre ⅔ ring), low (grey-teal short ring, lighter numeral, honest label).
  Added `/kit` living design-system reference showcasing the signature (grows in D3).
- Phase D3 (Design Blueprint): Shared component library under `components/ui/` —
  Button (primary/secondary/ghost/destructive, §10.1), Card (§10.3), Chip with
  matched/missing/present/semantic-≈ variants (§10.4/§10.6), Input/Textarea (16px,
  §10.2), dual ArcMeter for parsing-vs-scoring confidence (§10.9), Dropzone with
  parse-confidence readout (§10.2), and all states — Empty/Error(blameless)/Skeleton/
  ApertureLoader (§10.10). The aperture-as-loader suppresses its verdict label while
  waiting. All showcased in `/kit`.
- Phase D4 (Design Blueprint): Seeker flagship score-result screen at `/seeker/analyze`
  (§11.1-B), wired to live `POST /score`. Resume upload (or paste fallback) + JD entry
  + blind-mode toggle resolve into the hero aperture bloom, the gap-report "diff"
  (§10.6 — matched skills, semantic ≈ RAG matches, gaps framed as to-dos), and
  forward-looking top suggestions (§12). Added disclosed `X-Anon-Id` first-party token
  for honest freemium counting, and warm/blameless error + freemium states. Retired the
  dead pre-router components (App.jsx, ResumeParser, JobDescriptionInput, SkillAlignment,
  ThemeToggle).

- Phase D5 (Design Blueprint): Seeker rescan + health screens. `AnalysisProvider`
  shares session-only (never persisted, §13) inputs + score history across seeker
  screens. Rescan (`/seeker/rescan`, §11.1-C) is the retention surface — edit → rescan
  → before/after delta (43→64, +21 fit-green), bloom morph, and a momentum trail.
  Resume health (`/seeker/health`, §11.1-D) is a no-JD rule-based coaching read — ATS
  readability, weak-verb flags, quantified-achievement count — via `lib/resumeHealth.js`.

- Phase D6 (Design Blueprint): Recruiter side. HTTP Basic sign-in gate (PRD §9,
  in-memory creds), batch upload (`/recruiter/batch`) with sample-pool loader wired to
  live `POST /rank`. Dense ranked table (§10.7) — micro-signatures, top-2 driving
  features, confidence pills, sortable columns, zebra + slate header. Configurable
  weights re-rank client-side over the returned feature vectors (§10.7); blind mode
  anonymizes names (§11.3). Explainability slide-over (§10.8) — full bloom, per-feature
  breakdown, RAG skill matches, and an editable auto-drafted, justification-first
  feedback note (§12) with copy-to-clipboard.

- Phase D7 (Design Blueprint): Recruiter accuracy dashboard (`/recruiter/dashboard`,
  §11.2-H / §14), wired to live `GET /metrics`. Honest readiness banner
  (unready/provisional/tuned) with ground-truth collection progress; Spearman ρ and
  Precision@5/@10 metric cards that show mean ± std (confidence interval) and sample
  size when ready, "pending ground truth" otherwise (P3 — never faked); Spearman trend
  chart over pipeline versions; trained-model feature-importance bars. Currently renders
  the true `unready` state (0/28 pairs rated).

- Phase D8 (Design Blueprint): Trust surfaces. Blind-mode screen (`/seeker/blind`,
  §11.3-I / §13) runs an open score and a blind score on identical input, showing (a)
  the anonymization disclosure — what was stripped and why (institution, gender terms,
  photo), placeholders only, never raw PII — and (b) a bias check: whether removing
  identity signals moved the score, surfaced honestly either way (P3). Plain-language
  data-retention copy. Honest freemium messaging (§12, no dark patterns) is handled in
  the Analyze/rescan 429 states.

- Phase D9 (Design Blueprint): Responsive, accessibility, and polish pass. Nav no longer
  disappears on mobile — collapses to icon-only on both sides (§16); recruiter ranked
  table collapses to stacked micro-signature cards below `md` (§16); explainability panel
  is a full-screen sheet on mobile. WCAG AA audit (§15): ink/canvas 14.6:1, semantic text
  tones 5.3–6.8:1, muted ~4.5–4.9:1, non-suppressible ember focus-visible rings. Primary
  ember-500 CTAs retain the blueprint-sanctioned large-text/UI contrast (§5.1/§10.1, per
  §18 governance). Motion budget respected — expressive motion only on the score reveal.
  Sentence-case, forward-looking voice throughout (§12).

### Changed
- Removed the freemium scan cap. The anonymous seeker `/score` path is now unlimited
  by default via a new `FREEMIUM_SCAN_LIMIT` setting (0 = unlimited); the limiter code
  and tests are retained so a paid tier can be re-enabled without code changes. All
  "3 scans a month" UI copy removed from the landing and analyze screens.
- Removed internal `(§x.y)` blueprint section references from user-visible `/kit` titles
  (kept in code comments as documentation).

### Added
- Frontend test suite (Vitest + Testing Library): 22 tests across ranking utils,
  resume-health heuristics, aperture confidence bands, API error humanization, and a
  Chip component render — closing the prior zero-frontend-test gap. `npm test` to run.

### Fixed
- Explainability "Copy note" failed silently in contexts where the async Clipboard API
  is blocked (sandboxed iframes, unfocused windows) — the catch swallowed the error and
  gave no feedback. Now falls back to `execCommand('copy')`, and if that's also blocked,
  selects the note text so it can be copied manually (never a dead-end, never a false
  "Copied").
- Recruiter ranked list showed a non-monotonic default order (rank 3 scoring below
  rank 4) because the client-side weighted average diverged from the backend's trained
  `final_score`. `reRank` now defers to `final_score` under uniform weights and only
  re-ranks by weighted contribution once a recruiter actively tilts the weights.
- Resume-health "Quantified achievements" badge showed a misleading line-ratio (e.g.
  "100%" while the copy said "Only 1 line") — now counts distinct metric mentions across
  the whole text and reports a plain count ("3 metrics"), robust to paragraph vs bullet
  formatting.
- Scoring endpoint crashed on every request (`EmbeddingScorer.score() takes 3 positional
  arguments but 5 were given`): `get_orchestrator_tools()` injected a raw `EmbeddingScorer`
  into `HybridScorer`, which requires the cache-aware `CachedEmbeddingScorer` (4-arg
  `score(resume_id, resume_text, jd_id, jd_text)`). Now wraps the base scorer in
  `CachedEmbeddingScorer(EmbeddingCache())` at the endpoint wiring. Live `/score` verified
  end-to-end from the browser.
