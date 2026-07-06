import { Sparkles } from 'lucide-react';
import ApertureBloom from '../ApertureBloom';

// §10.10 — the states that separate a real product from a happy-path demo.
// Every one is warm and gives one clear next action; never a blank void, never a
// red stack-trace.

// Empty: inviting, one CTA.
export function EmptyState({ icon: Icon = Sparkles, title, body, action }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-14 px-6">
      <span className="w-14 h-14 rounded-2xl bg-ember-50 text-ember-700 flex items-center justify-center mb-4">
        <Icon className="w-6 h-6" strokeWidth={1.75} aria-hidden />
      </span>
      <h3 className="text-h3 text-ink">{title}</h3>
      {body && <p className="text-small text-muted mt-2 max-w-sm">{body}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

// Error: blameless copy — the SYSTEM couldn't read it, not the user (P3, §12).
// A trust-defining moment, so it gets first-class design, never a red trace.
export function ErrorState({ title = "We couldn't read this resume clearly", body, action }) {
  return (
    <div className="rounded-[var(--card-radius)] border border-gap-500/30 bg-gap-fill/50 p-6 flex gap-4">
      <span className="w-10 h-10 shrink-0 rounded-xl bg-gap-fill text-gap-text flex items-center justify-center" aria-hidden>
        {/* warm, not alarm-red */}
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
        </svg>
      </span>
      <div>
        <p className="text-small font-semibold text-gap-text">{title}</p>
        {body && <p className="text-small text-gap-text/90 mt-1">{body}</p>}
        {action && <div className="mt-3">{action}</div>}
      </div>
    </div>
  );
}

// Loading: calm skeleton — no spinner-of-doom. (On the score path the aperture
// bloom IS the loader; see ApertureLoader below.)
export function Skeleton({ className = '' }) {
  return <div className={`animate-pulse rounded-md bg-border/60 ${className}`} />;
}

export function SkeletonCard() {
  return (
    <div className="space-y-3">
      <Skeleton className="h-5 w-1/3" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-5/6" />
      <Skeleton className="h-4 w-2/3" />
    </div>
  );
}

// The wait IS the reveal (§10.10): while scoring, the aperture blooms in with a
// gentle placeholder vector, so the loading state becomes the result.
export function ApertureLoader({ label = 'Bringing your fit into focus…' }) {
  return (
    <div className="flex flex-col items-center justify-center py-10" aria-live="polite" aria-busy="true">
      <ApertureBloom
        featureVector={{ tfidf_score: 0.5, embedding_score: 0.55, skill_overlap_pct: 0.5, exp_match: 0.5, edu_match: 0.45 }}
        score={0}
        confidence={0}
        confidenceBand="medium"
        showLegend={false}
        showLabel={false}
      />
      <p className="text-small text-muted mt-2">{label}</p>
    </div>
  );
}
