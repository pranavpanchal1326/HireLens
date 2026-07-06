import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Chip from './Chip';

describe('Chip', () => {
  it('renders its label', () => {
    render(<Chip kind="matched">Python</Chip>);
    expect(screen.getByText('Python')).toBeInTheDocument();
  });

  it('shows the ≈ glyph for semantic matches (meaning survives grayscale)', () => {
    render(<Chip kind="semantic">people management</Chip>);
    expect(screen.getByText('≈')).toBeInTheDocument();
    expect(screen.getByText('people management')).toBeInTheDocument();
  });

  it('applies the missing (gap) treatment for missing skills', () => {
    const { container } = render(<Chip kind="missing">Kubernetes</Chip>);
    expect(container.firstChild.className).toMatch(/gap/);
  });
});
