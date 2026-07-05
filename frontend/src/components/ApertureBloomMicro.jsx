import React from 'react';
import { bandVisuals } from '../lib/apertureConfidence';

// Design Blueprint §6.4 — the micro/list tier: ring + number ONLY, no petals,
// legible at 24-32px, for the recruiter ranked list. Confidence is shown by ring
// completeness ONLY. Reuses the SAME bandVisuals() as the hero so the two tiers
// can never disagree about what a confidence band looks like.

const CX = 18;
const CY = 18;
const RING_R = 15;

export default function ApertureBloomMicro({
  score = 0,
  confidence = 0, // 0-1
  confidenceBand = 'low',
  size = 32,
}) {
  const confPct = Math.max(0, Math.min(100, Math.round(confidence * 100)));
  const visuals = bandVisuals(confidenceBand);
  const dashRest = 100 - confPct;
  const srLabel = `Fit score ${score}, ${visuals.label.toLowerCase()}.`;

  return (
    <svg viewBox="0 0 36 36" width={size} height={size} role="img" aria-label={srLabel}>
      <title>{srLabel}</title>
      {/* Track. */}
      <circle cx={CX} cy={CY} r={RING_R} fill="none" stroke="var(--border)" strokeWidth="3" />
      {/* Confidence arc — completeness = confidence. */}
      <circle
        cx={CX}
        cy={CY}
        r={RING_R}
        fill="none"
        stroke={visuals.ring}
        strokeWidth="3"
        strokeLinecap="round"
        pathLength="100"
        strokeDasharray="100"
        strokeDashoffset={dashRest}
        transform={`rotate(-90 ${CX} ${CY})`}
      />
      <text
        className="tabular-nums"
        x={CX}
        y={CY}
        textAnchor="middle"
        dominantBaseline="central"
        fontFamily="Hanken Grotesk, sans-serif"
        fontWeight="700"
        fontSize="13"
        fill="var(--ink)"
      >
        {score}
      </text>
    </svg>
  );
}
