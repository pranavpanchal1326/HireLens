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
      <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm flex flex-col">
        <div className="flex items-center gap-2 mb-4">
          <Check className="w-5 h-5 text-emerald-500" />
          <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-800 dark:text-zinc-200">
            Aligned Skills ({matchedSkills.length})
          </h3>
        </div>

        {matchedSkills.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center border border-dashed border-zinc-200 dark:border-zinc-800 rounded-lg bg-zinc-50/50 dark:bg-zinc-900/10">
            <AlertCircle className="w-8 h-8 text-zinc-400 mb-2" />
            <p className="text-sm text-zinc-500">No matching skills identified.</p>
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
                      ? 'bg-blue-50/55 dark:bg-blue-900/10 text-blue-700 dark:text-blue-400 border-blue-200/60 dark:border-blue-900/40' 
                      : 'bg-emerald-50/55 dark:bg-emerald-900/10 text-emerald-700 dark:text-emerald-400 border-emerald-200/60 dark:border-emerald-900/40'
                    }`}
                >
                  <span>{match.resume_skill}</span>
                  {isSemantic && (
                    <>
                      <span className="text-blue-500 dark:text-blue-400/80 font-bold">≈</span>
                      <span className="text-[10px] text-zinc-400 dark:text-zinc-550 group-hover:text-blue-600 dark:group-hover:text-blue-300 transition-colors">
                        {match.jd_skill}
                      </span>
                      {/* Hover Tooltip with Similarity Score */}
                      <span className="absolute -top-8 left-1/2 -translate-x-1/2 scale-0 group-hover:scale-100 transition-transform duration-150 bg-zinc-900 dark:bg-zinc-800 text-white dark:text-zinc-200 text-[10px] px-2 py-1 rounded shadow-md z-30 pointer-events-none whitespace-nowrap font-mono">
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
      <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm flex flex-col">
        <div className="flex items-center gap-2 mb-4">
          <ClipboardList className="w-5 h-5 text-amber-500" />
          <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-800 dark:text-zinc-200">
            Career Path Checklist ({gaps.length})
          </h3>
        </div>

        {gaps.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center border border-dashed border-zinc-200 dark:border-zinc-800 rounded-lg bg-zinc-50/50 dark:bg-zinc-900/10">
            <Check className="w-8 h-8 text-emerald-500 mb-2" />
            <p className="text-sm text-zinc-500">All skills matched! No skill gaps identified.</p>
          </div>
        ) : (
          <div className="flex flex-col gap-2.5 overflow-y-auto max-h-80 pr-1">
            <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mb-1 italic">
              "Gaps are to-dos, not verdicts." Checked items reflect your target skills.
            </p>
            {gaps.map((gap, idx) => {
              const isChecked = !!checkedGaps[idx];
              return (
                <div
                  key={idx}
                  onClick={() => toggleGap(idx)}
                  className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all duration-200 hover:border-zinc-300 dark:hover:border-zinc-700
                    ${isChecked 
                      ? 'bg-zinc-50 dark:bg-zinc-900/30 border-zinc-200 dark:border-zinc-800 opacity-60' 
                      : 'bg-zinc-50/50 dark:bg-[#0c0c0f] border-zinc-200/80 dark:border-zinc-800/80'
                    }`}
                >
                  <div className="mt-0.5">
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => {}} // handled by parent onClick
                      className="w-4 h-4 rounded border-zinc-300 dark:border-zinc-850 text-blue-600 focus:ring-blue-500 focus:ring-offset-0 bg-transparent cursor-pointer"
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-start gap-2">
                      <h4 className={`text-xs font-semibold font-mono truncate ${isChecked ? 'line-through text-zinc-400 dark:text-zinc-550' : 'text-zinc-800 dark:text-zinc-200'}`}>
                        {gap.missing_skill}
                      </h4>
                      <span className="flex items-center gap-0.5 text-[9px] text-blue-500 font-medium">
                        Actionable <ArrowUpRight className="w-2.5 h-2.5" />
                      </span>
                    </div>
                    <p className={`text-[11px] mt-1 leading-relaxed ${isChecked ? 'line-through text-zinc-400 dark:text-zinc-550' : 'text-zinc-500 dark:text-zinc-400'}`}>
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
