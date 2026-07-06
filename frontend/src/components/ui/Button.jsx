// §10.1 Buttons. One high-emphasis primary per view (P4). All variants themed via
// the token spine, so a button reads correctly under either data-theme.
const VARIANTS = {
  // Primary: ember-500 fill, white label. The only high-emphasis action per view.
  primary:
    'bg-ember-500 text-white hover:brightness-[0.94] shadow-sm disabled:bg-border disabled:text-muted disabled:shadow-none',
  // Secondary: surface fill + border + ink label.
  secondary:
    'bg-surface text-ink border border-border hover:bg-canvas disabled:text-muted',
  // Ghost: no fill, ember-700 label — low-emphasis inline actions.
  ghost:
    'bg-transparent text-ember-700 hover:bg-ember-50 disabled:text-muted',
  // Destructive: clay/gap tones, never pure red (§10.1); always confirm elsewhere.
  destructive:
    'bg-gap-fill text-gap-text border border-gap-500/40 hover:brightness-[0.97]',
};

const SIZES = {
  sm: 'h-9 px-3 text-small gap-1.5 rounded-md',
  md: 'h-11 px-5 text-small gap-2 rounded-md',
  lg: 'h-13 px-6 text-body font-medium gap-2 rounded-xl',
};

export default function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  icon: Icon = null,
  className = '',
  children,
  ...props
}) {
  return (
    <button
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={`inline-flex items-center justify-center font-medium transition-all duration-200 cursor-pointer disabled:cursor-not-allowed focus-ember ${VARIANTS[variant]} ${SIZES[size]} ${className}`}
      {...props}
    >
      {loading ? (
        // Calm inline spinner only — never a spinner-of-doom on a full view (§10.10).
        <span
          className="w-4 h-4 rounded-full border-2 border-current border-t-transparent animate-spin"
          aria-hidden
        />
      ) : (
        Icon && <Icon className="w-4 h-4" strokeWidth={1.75} aria-hidden />
      )}
      {children}
    </button>
  );
}
