import { Check, Plus, Minus } from 'lucide-react';

// §10.4 Chips / tags. Skill chips: matched / missing / present-not-required, plus
// a semantic-match ("≈") variant for RAG matches (§10.6). ALWAYS paired with an
// icon so meaning survives grayscale (§5.6 color-blind safety).
const KIND = {
  matched:  { cls: 'bg-fit-fill text-fit-text',       Icon: Check },
  missing:  { cls: 'bg-gap-fill text-gap-text',        Icon: Plus },   // "add this", a to-do not a failure
  present:  { cls: 'bg-canvas text-muted border border-border', Icon: Minus },
  semantic: { cls: 'bg-fit-fill text-fit-text',        Icon: null },   // shows a ≈ glyph
};

export default function Chip({ kind = 'matched', children, title }) {
  const { cls, Icon } = KIND[kind] || KIND.matched;
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-caption font-medium ${cls}`}
    >
      {kind === 'semantic' ? (
        <span aria-hidden className="font-semibold leading-none">≈</span>
      ) : (
        Icon && <Icon className="w-3 h-3" strokeWidth={2.25} aria-hidden />
      )}
      {children}
    </span>
  );
}
