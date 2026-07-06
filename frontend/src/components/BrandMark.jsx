// BrandMark — the micro, resolved aperture used in app chrome (§6.4 micro tier,
// stripped to a stable logo). No animation: at rest it reads as a logo, not a
// screensaver (§6.5). `onDark` swaps the pupil fill for the slate header.
export default function BrandMark({ size = 28, onDark = false }) {
  const CX = 30, CY = 32;
  const petals = [0.95, 0.6, 0.8, 0.55, 0.72];
  const path = (v) => {
    const tip = 11 + v * 13, w = 4.5 + v * 2;
    return `M${CX} ${CY} Q ${CX - w} ${CY - tip * 0.55} ${CX} ${CY - tip} Q ${CX + w} ${CY - tip * 0.55} ${CX} ${CY} Z`;
  };
  return (
    <svg viewBox="0 0 60 64" width={size} height={size * (64 / 60)} role="img" aria-label="HireLens logo">
      <g>
        {petals.map((v, k) => (
          <path
            key={k}
            d={path(v)}
            fill={k % 3 === 0 ? 'var(--ember-500)' : 'var(--ember-300)'}
            transform={`rotate(${k * 72} ${CX} ${CY})`}
            opacity="0.85"
          />
        ))}
      </g>
      <circle
        cx={CX} cy={CY} r="22" fill="none"
        stroke={onDark ? 'var(--ember-300)' : 'var(--fit-500)'}
        strokeWidth="2.5" strokeLinecap="round"
        pathLength="100" strokeDasharray="80 100" transform={`rotate(-90 ${CX} ${CY})`}
      />
      <circle cx={CX} cy={CY} r="8" fill={onDark ? 'var(--slate-700)' : 'var(--surface)'}
        stroke="var(--border)" strokeWidth="1" />
      <circle cx={CX} cy={CY} r="2.4" fill="var(--ember-500)" />
    </svg>
  );
}
