// §10.3 Card. Surface fill, theme card-radius, border + low warm shadow. Seeker
// gets roomier padding (24px), recruiter tighter (16px) — a carrier of the two
// temperaments (§8). Padding follows the active theme unless overridden.
export default function Card({ as: Tag = 'div', pad = 'theme', className = '', children, ...props }) {
  const padding =
    pad === 'none' ? '' : pad === 'sm' ? 'p-4' : pad === 'lg' ? 'p-6' : 'p-5 md:p-6';
  return (
    <Tag
      className={`bg-surface border border-border rounded-[var(--card-radius)] ${padding} ${className}`}
      style={{ boxShadow: 'var(--shadow-sm)' }}
      {...props}
    >
      {children}
    </Tag>
  );
}
