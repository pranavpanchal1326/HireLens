import { Check, Plus } from 'lucide-react';

// §10.6 Gap report — "the diff". Two columns: what you HAVE (matched, fit-green
// check; semantic matches marked ≈ so the user sees the system understood intent,
// not just keywords) and what's MISSING (gap-ochre plus, each phrased as an ACTION
// — a to-do, never a deficiency, per §12).
export default function GapReport({ matchedSkills = [], gaps = [] }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* HAVE */}
      <div className="rounded-[var(--card-radius)] border border-border bg-surface p-5">
        <div className="flex items-center gap-2 mb-3">
          <span className="w-6 h-6 rounded-md bg-fit-fill text-fit-text flex items-center justify-center">
            <Check className="w-3.5 h-3.5" strokeWidth={2.5} aria-hidden />
          </span>
          <h4 className="text-small font-semibold text-ink">You already have</h4>
          <span className="text-caption text-muted tabular-nums ml-auto">{matchedSkills.length}</span>
        </div>
        {matchedSkills.length === 0 ? (
          <p className="text-small text-muted">No direct matches yet — the gaps opposite are your fastest wins.</p>
        ) : (
          <ul className="space-y-2">
            {matchedSkills.map((m, i) => (
              <li key={i} className="flex items-center gap-2 text-small text-ink">
                <Check className="w-4 h-4 text-fit-500 shrink-0" strokeWidth={2.25} aria-hidden />
                <span>{m.jd_skill || m.resume_skill}</span>
                {m.match_type === 'semantic' && (
                  <span
                    className="text-caption text-fit-text bg-fit-fill rounded-full px-1.5 py-0.5"
                    title={`"${m.resume_skill}" ≈ "${m.jd_skill}" — matched by meaning`}
                  >
                    ≈ {m.resume_skill}
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* MISSING — framed as to-dos */}
      <div className="rounded-[var(--card-radius)] border border-border bg-surface p-5">
        <div className="flex items-center gap-2 mb-3">
          <span className="w-6 h-6 rounded-md bg-gap-fill text-gap-text flex items-center justify-center">
            <Plus className="w-3.5 h-3.5" strokeWidth={2.5} aria-hidden />
          </span>
          <h4 className="text-small font-semibold text-ink">Add these to close the gap</h4>
          <span className="text-caption text-muted tabular-nums ml-auto">{gaps.length}</span>
        </div>
        {gaps.length === 0 ? (
          <p className="text-small text-fit-text">Nothing missing — you cover every required skill. 🎯</p>
        ) : (
          <ul className="space-y-3">
            {gaps.map((g, i) => (
              <li key={i} className="flex gap-2.5">
                <Plus className="w-4 h-4 text-gap-500 shrink-0 mt-0.5" strokeWidth={2.25} aria-hidden />
                <div>
                  <p className="text-small font-medium text-ink">{g.missing_skill}</p>
                  <p className="text-small text-muted">{g.suggested_action}</p>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
