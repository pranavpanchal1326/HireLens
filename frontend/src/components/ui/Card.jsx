// §10.3 Card. Surface fill, theme card-radius, border + low warm shadow. Seeker
// gets roomier padding (24px), recruiter tighter (16px) — a carrier of the two
// temperaments (§8). `interactive` adds a smooth hover-lift for clickable cards.
export default function Card({ as: Tag = 'div', pad = 'theme', interactive = false, className = '', children, ...props }) {
  const padding =
    pad === 'none' ? '' : pad === 'sm' ? 'p-4' : pad === 'lg' ? 'p-6' : 'p-5 md:p-6';
  return (
    <Tag
      className={`bg-surface border border-border rounded-[var(--card-radius)] ${padding} ${interactive ? 'lift cursor-pointer hover:border-ember-300/60' : ''} ${className}`}
      style={{ boxShadow: 'var(--shadow-sm)' }}
      {...props}
    >
      {children}
    </Tag>
  );
}
