import React from 'react';
import { bandVisuals, toPetals, strongestFeature, FEATURE_LABELS } from '../lib/apertureConfidence';

// Design Blueprint §6 — the signature "aperture bloom". Five petals fan from a
// central pupil (the score); a ring reports confidence. Petals are the model's
// REAL feature vector (§6.2) — the art IS the data. Pure SVG per §6.6 geometry,
// coloured entirely from R1 tokens (no hardcoded hex).

const CX = 110;
const CY = 118;
const PUPIL_R = 26;
const RING_R = 80;

// Leaf path pointing straight up from the aperture centre, per §6.6:
// tip radius = 34 + value*40, width control = 15 + value*6.
function petalPath(value) {
  const tip = 34 + value * 40;
  const w = 15 + value * 6;
  return `M${CX} ${CY} Q ${CX - w} ${CY - tip * 0.55} ${CX} ${CY - tip} Q ${CX + w} ${CY - tip * 0.55} ${CX} ${CY} Z`;
}

// §6.6 tonal pattern: ember-500 / ember-300 / ember-300 for tonal depth.
const petalTone = (k) => (k % 3 === 0 ? 'var(--ember-500)' : 'var(--ember-300)');

export default function ApertureBloom({
  featureVector,
  score = 0,
  confidence = 0, // 0-1 (scoring_confidence)
  confidenceBand = 'low',
}) {
  if (!featureVector) return null;

  const petals = toPetals(featureVector);
  const confPct = Math.max(0, Math.min(100, Math.round(confidence * 100)));
  const visuals = bandVisuals(confidenceBand);
  const dashRest = 100 - confPct; // resolved-state dash offset for the ring

  const srLabel = `Fit score ${score}, ${visuals.label.toLowerCase()}, strongest in ${strongestFeature(featureVector)}.`;

  return (
    <div className="w-full flex flex-col items-center justify-center p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-4 self-start">
        Aperture-Bloom Alignment Signature
      </h3>

      <svg
        viewBox="0 0 220 236"
        width="220"
        height="236"
        role="img"
        aria-label={srLabel}
        style={{ maxWidth: '100%' }}
      >
        <title>{srLabel}</title>

        {/* Petals — the live feature vector. */}
        <g className="ab-petals">
          {petals.map((p, k) => (
            <path
              key={p.key}
              className="ab-petal"
              d={petalPath(p.value)}
              fill={petalTone(k)}
              transform={`rotate(${k * 72} ${CX} ${CY})`}
              style={{ animationDelay: `${k * 0.065}s` }}
            />
          ))}
        </g>

        {/* Quiet single pulse at ~1s, then gone (§6.5). */}
        <circle
          className="ab-pulse"
          cx={CX}
          cy={CY}
          r={RING_R}
          fill="none"
          stroke="var(--ember-300)"
          strokeWidth="2"
        />

        {/* Confidence ring — partial arc, drawn clockwise from top (§6.6). */}
        <circle
          className="ab-ring"
          cx={CX}
          cy={CY}
          r={RING_R}
          fill="none"
          stroke={visuals.ring}
          strokeWidth="6"
          strokeLinecap="round"
          pathLength="100"
          strokeDasharray="100"
          strokeDashoffset={dashRest}
          transform={`rotate(-90 ${CX} ${CY})`}
          style={{ '--ab-dash-rest': dashRest }}
        />

        {/* Pupil + score. */}
        <circle cx={CX} cy={CY} r={PUPIL_R} fill="var(--surface)" stroke="var(--border)" strokeWidth="1.5" />
        <text
          className="ab-number tabular-nums"
          x={CX}
          y={CY}
          textAnchor="middle"
          dominantBaseline="central"
          fontFamily="Hanken Grotesk, sans-serif"
          fontWeight="700"
          fontSize="30"
          fill={visuals.number}
        >
          {score}
        </text>
      </svg>

      {/* Confidence label (§6.3) — honest, band-specific. */}
      <p className="text-[11px] font-medium mt-1 text-center" style={{ color: visuals.ring }}>
        {visuals.label}
      </p>

      {/* Per-feature legend (explainability, §10.8). */}
      <div className="mt-3 grid grid-cols-5 gap-2 w-full text-center text-[10px] text-muted">
        {petals.map((p) => (
          <div key={p.key}>
            <span className="block tabular-nums text-sm text-ink font-bold">{Math.round(p.value * 100)}%</span>
            <span className="capitalize">{FEATURE_LABELS[p.key]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
