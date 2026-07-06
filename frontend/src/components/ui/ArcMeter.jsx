// §10.9 Confidence & parsing meters. Two DISTINCT, always-visible meters that must
// never be conflated: parsing confidence (% expected fields extracted) and scoring
// confidence (model certainty). Both reuse the signature ring language (§14
// consistency) so the whole product feels of one hand.
//
// `tone` picks the arc color; `kind` only changes the default label/semantics.
const TONE = {
  fit: 'var(--fit-500)',
  gap: 'var(--gap-500)',
  lowconf: 'var(--lowconf-500)',
  ember: 'var(--ember-500)',
  slate: 'var(--slate-700)',
};

export default function ArcMeter({
  value = 0,            // 0-1
  label,
  sublabel,
  tone = 'ember',
  size = 88,
  strokeWidth = 7,
}) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  const r = 18;
  const cx = 20, cy = 20;
  const dashRest = 100 - pct;
  const stroke = TONE[tone] || TONE.ember;

  return (
    <div className="flex items-center gap-3">
      <svg viewBox="0 0 40 40" width={size} height={size} role="img"
        aria-label={`${label || 'Meter'}: ${pct} percent`}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--border)" strokeWidth={strokeWidth} />
        <circle
          cx={cx} cy={cy} r={r} fill="none" stroke={stroke} strokeWidth={strokeWidth}
          strokeLinecap="round" pathLength="100" strokeDasharray="100"
          strokeDashoffset={dashRest} transform={`rotate(-90 ${cx} ${cy})`}
        />
        <text x={cx} y={cy} textAnchor="middle" dominantBaseline="central"
          className="tabular-nums" fontFamily="Hanken Grotesk, sans-serif"
          fontWeight="700" fontSize="11" fill="var(--ink)">
          {pct}
        </text>
      </svg>
      {(label || sublabel) && (
        <div>
          {label && <p className="text-small font-medium text-ink">{label}</p>}
          {sublabel && <p className="text-caption text-muted mt-0.5">{sublabel}</p>}
        </div>
      )}
    </div>
  );
}
