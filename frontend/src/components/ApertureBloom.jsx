import React from 'react';
import ReactECharts from 'echarts-for-react';

export default function ApertureBloom({ featureVector }) {
  if (!featureVector) return null;

  // Values in backend are 0.0 - 1.0; scale to 100 for percentage visualization
  const dataValues = [
    Math.round(featureVector.tfidf_score * 100),
    Math.round(featureVector.embedding_score * 100),
    Math.round(featureVector.skill_overlap_pct * 100),
    Math.round(featureVector.exp_match * 100),
    Math.round(featureVector.edu_match * 100)
  ];

  // Radar indicator labels (LOCKED field order)
  // [tfidf_score, embedding_score, skill_overlap_pct, exp_match, edu_match]
  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(9, 9, 11, 0.95)',
      borderColor: '#27272a',
      borderWidth: 1,
      textStyle: {
        color: '#fafafa',
        fontFamily: 'DM Sans, sans-serif',
        fontSize: 12
      },
      formatter: (params) => {
        const indicators = ['Lexical (TF-IDF)', 'Semantic (Embedding)', 'Skill Overlap', 'Experience Match', 'Education Match'];
        let html = '<div class="p-1 font-sans"><span class="font-bold text-zinc-400 block mb-1">Aperture Dimension</span>';
        params.value.forEach((val, idx) => {
          html += `<div class="flex justify-between gap-4 py-0.5"><span class="text-zinc-300">${indicators[idx]}:</span><span class="font-mono text-blue-400 font-bold">${val}%</span></div>`;
        });
        html += '</div>';
        return html;
      }
    },
    radar: {
      shape: 'circle',
      indicator: [
        { name: 'Lexical Match (TF-IDF)', max: 100 },
        { name: 'Semantic Match (Embedding)', max: 100 },
        { name: 'Skill Overlap (RAG)', max: 100 },
        { name: 'Experience Level', max: 100 },
        { name: 'Education Level', max: 100 }
      ],
      axisName: {
        color: '#a1a1aa',
        fontFamily: 'DM Sans, sans-serif',
        fontWeight: 500,
        fontSize: 11
      },
      splitLine: {
        lineStyle: {
          color: [
            'rgba(63, 63, 70, 0.1)',
            'rgba(63, 63, 70, 0.2)',
            'rgba(63, 63, 70, 0.4)',
            'rgba(63, 63, 70, 0.6)',
            'rgba(63, 63, 70, 0.8)'
          ].reverse()
        }
      },
      splitArea: {
        show: false
      },
      axisLine: {
        lineStyle: {
          color: 'rgba(63, 63, 70, 0.3)'
        }
      }
    },
    series: [
      {
        name: 'Fit Vector',
        type: 'radar',
        data: [
          {
            value: dataValues,
            name: 'Candidate Profile Match',
            symbol: 'circle',
            symbolSize: 6,
            itemStyle: {
              color: '#3b82f6'
            },
            areaStyle: {
              color: 'rgba(59, 130, 246, 0.25)'
            },
            lineStyle: {
              width: 2,
              color: '#3b82f6'
            }
          }
        ]
      }
    ]
  };

  return (
    <div className="w-full flex flex-col items-center justify-center p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-4 self-start">
        Aperture-Bloom Alignment Signature
      </h3>
      <div className="w-full h-80 relative">
        <ReactECharts
          option={option}
          style={{ height: '100%', width: '100%' }}
          opts={{ renderer: 'svg' }}
        />
      </div>
      <div className="mt-2 grid grid-cols-5 gap-2 w-full text-center text-[10px] text-zinc-400">
        <div>
          <span className="block font-mono text-sm text-zinc-800 dark:text-zinc-200 font-bold">{dataValues[0]}%</span>
          <span>Lexical</span>
        </div>
        <div>
          <span className="block font-mono text-sm text-zinc-800 dark:text-zinc-200 font-bold">{dataValues[1]}%</span>
          <span>Semantic</span>
        </div>
        <div>
          <span className="block font-mono text-sm text-zinc-800 dark:text-zinc-200 font-bold">{dataValues[2]}%</span>
          <span>Skills</span>
        </div>
        <div>
          <span className="block font-mono text-sm text-zinc-800 dark:text-zinc-200 font-bold">{dataValues[3]}%</span>
          <span>Experience</span>
        </div>
        <div>
          <span className="block font-mono text-sm text-zinc-800 dark:text-zinc-200 font-bold">{dataValues[4]}%</span>
          <span>Education</span>
        </div>
      </div>
    </div>
  );
}
