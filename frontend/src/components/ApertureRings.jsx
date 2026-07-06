// Decorative optical backdrop — faint concentric aperture rings + tick marks, like
// a lens barrel or camera focus scale. Slowly drifts. Pure art, very low opacity,
// sits behind the hero bloom (§4). Never carries information; aria-hidden.
export default function ApertureRings({ className = '' }) {
  const C = 200;
  const ticks = Array.from({ length: 48 });
  return (
    <svg
      aria-hidden
      viewBox="0 0 400 400"
      className={`aperture-rings ${className}`}
      style={{ width: '100%', height: '100%' }}
    >
      <g fill="none" stroke="var(--ink)" strokeOpacity="0.06">
        <circle cx={C} cy={C} r="190" strokeWidth="1" />
        <circle cx={C} cy={C} r="150" strokeWidth="1" strokeDasharray="2 6" />
        <circle cx={C} cy={C} r="112" strokeWidth="1" />
        <circle cx={C} cy={C} r="74" strokeWidth="1" strokeDasharray="1 5" />
      </g>
      {/* Focus-scale ticks around the outer ring. */}
      <g stroke="var(--ink)" strokeOpacity="0.08" strokeWidth="1">
        {ticks.map((_, i) => {
          const a = (i / ticks.length) * Math.PI * 2;
          const r1 = 190, r2 = i % 4 === 0 ? 178 : 184;
          return (
            <line
              key={i}
              x1={C + Math.cos(a) * r1} y1={C + Math.sin(a) * r1}
              x2={C + Math.cos(a) * r2} y2={C + Math.sin(a) * r2}
            />
          );
        })}
      </g>
      {/* Faint ember focus dot — the one warm point. */}
      <circle cx={C} cy={C} r="3" fill="var(--ember-500)" fillOpacity="0.5" />
    </svg>
  );
}
