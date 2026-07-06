// §10.2 Inputs. Surface fill, border, r-md, 16px text (prevents mobile zoom).
// Focus: ember ring + border→ember-300. Labels always present (§15).
export function Input({ label, hint, id, className = '', ...props }) {
  return (
    <label htmlFor={id} className="block">
      {label && <span className="block text-small font-medium text-ink mb-1.5">{label}</span>}
      <input
        id={id}
        className={`w-full h-11 px-3.5 rounded-md bg-surface border border-border text-ink text-[16px] placeholder:text-muted transition-colors duration-200 focus:border-ember-300 focus-ember ${className}`}
        {...props}
      />
      {hint && <span className="block text-caption text-muted mt-1.5">{hint}</span>}
    </label>
  );
}

export function Textarea({ label, hint, id, rows = 6, className = '', ...props }) {
  return (
    <label htmlFor={id} className="block">
      {label && <span className="block text-small font-medium text-ink mb-1.5">{label}</span>}
      <textarea
        id={id}
        rows={rows}
        className={`w-full px-3.5 py-3 rounded-md bg-surface border border-border text-ink text-[16px] leading-relaxed placeholder:text-muted transition-colors duration-200 resize-y focus:border-ember-300 focus-ember ${className}`}
        {...props}
      />
      {hint && <span className="block text-caption text-muted mt-1.5">{hint}</span>}
    </label>
  );
}
