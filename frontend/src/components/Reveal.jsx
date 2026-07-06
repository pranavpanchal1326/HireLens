import { useEffect, useRef, useState } from 'react';

// Sections "come into focus" as they scroll into view — a fade + rise (never a
// text blur; law #3). Honors reduced-motion by showing immediately.
export default function Reveal({ children, className = '', delay = 0, as: Tag = 'div' }) {
  const ref = useRef(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { setShown(true); return; }
    const io = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setShown(true); io.disconnect(); } },
      { threshold: 0.14, rootMargin: '0px 0px -8% 0px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <Tag ref={ref} className={`reveal-scroll ${shown ? 'reveal-in' : ''} ${className}`} style={{ transitionDelay: `${delay}ms` }}>
      {children}
    </Tag>
  );
}
