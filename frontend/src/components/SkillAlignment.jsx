import React, { useState } from 'react';
import { Check, ClipboardList, AlertCircle, ArrowUpRight } from 'lucide-react';

export default function SkillAlignment({ matchedSkills = [], gaps = [] }) {
  const [checkedGaps, setCheckedGaps] = useState({});

  const toggleGap = (index) => {
    setCheckedGaps(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full mt-6">
      {/* Matched Skills Card */}
      <div className="bg-surface border border-border rounded-xl p-5 shadow-sm flex flex-col">
        <div className="flex items-center gap-2 mb-4">
          <Check className="w-5 h-5 text-fit-500" />
          <h3 className="text-sm font-semibold uppercase tracking-wider text-ink">
            Aligned Skills (<span className="tabular-nums">{matchedSkills.length}</span>)
          </h3>
        </div>

        {matchedSkills.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center border border-dashed border-border rounded-lg bg-canvas">
            <AlertCircle className="w-8 h-8 text-muted mb-2" />
            <p className="text-sm text-muted">No matching skills identified.</p>
          </div>
        ) : (
          <div className="flex flex-wrap gap-2.5 overflow-y-auto max-h-80 pr-1">
            {matchedSkills.map((match, idx) => {
              const isSemantic = match.match_type === 'semantic';
              return (
                <div
                  key={idx}
                  className={`group relative flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-all duration-200 hover:scale-[1.02] cursor-default
                    ${isSemantic
                      ? 'bg-ember-50 text-ember-700 border-ember-100'
                      : 'bg-fit-fill text-fit-text border-fit-500/40'
                    }`}
                >
                  <span>{match.resume_skill}</span>
                  {isSemantic && (
                    <>
                      <span className="text-ember-500 font-bold">≈</span>
                      <span className="text-[10px] text-muted group-hover:text-ember-700 transition-colors">
                        {match.jd_skill}
                      </span>
                      {/* Hover Tooltip with Similarity Score */}
                      <span className="absolute -top-8 left-1/2 -translate-x-1/2 scale-0 group-hover:scale-100 transition-transform duration-150 bg-ink text-surface text-[10px] px-2 py-1 rounded shadow-md z-30 pointer-events-none whitespace-nowrap tabular-nums">
                        Semantic Match: {Math.round(match.similarity_score * 100)}%
                      </span>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Actionable Gaps Card */}
      <div className="bg-surface border border-border rounded-xl p-5 shadow-sm flex flex-col">
        <div className="flex items-center gap-2 mb-4">
          <ClipboardList className="w-5 h-5 text-gap-500" />
          <h3 className="text-sm font-semibold uppercase tracking-wider text-ink">
            Career Path Checklist (<span className="tabular-nums">{gaps.length}</span>)
          </h3>
        </div>

        {gaps.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center border border-dashed border-border rounded-lg bg-canvas">
            <Check className="w-8 h-8 text-fit-500 mb-2" />
            <p className="text-sm text-muted">All skills matched! No skill gaps identified.</p>
          </div>
        ) : (
          <div className="flex flex-col gap-2.5 overflow-y-auto max-h-80 pr-1">
            <p className="text-[11px] text-muted mb-1 italic">
              "Gaps are to-dos, not verdicts." Checked items reflect your target skills.
            </p>
            {gaps.map((gap, idx) => {
              const isChecked = !!checkedGaps[idx];
              return (
                <div
                  key={idx}
                  onClick={() => toggleGap(idx)}
                  className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all duration-200 hover:border-ember-300
                    ${isChecked
                      ? 'bg-canvas border-border opacity-60'
                      : 'bg-canvas border-border'
                    }`}
                >
                  <div className="mt-0.5">
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => {}} // handled by parent onClick
                      className="w-4 h-4 rounded border-border text-ember-500 focus:ring-ember-500 focus:ring-offset-0 bg-transparent cursor-pointer"
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-start gap-2">
                      <h4 className={`text-xs font-semibold truncate ${isChecked ? 'line-through text-muted' : 'text-ink'}`}>
                        {gap.missing_skill}
                      </h4>
                      <span className="flex items-center gap-0.5 text-[9px] text-ember-700 font-medium">
                        Actionable <ArrowUpRight className="w-2.5 h-2.5" />
                      </span>
                    </div>
                    <p className={`text-[11px] mt-1 leading-relaxed ${isChecked ? 'line-through text-muted' : 'text-muted'}`}>
                      {gap.suggested_action}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
