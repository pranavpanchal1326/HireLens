// The cursor-as-light engine (§8). A soft pool of Ember warmth trails the pointer
// — it lifts light, never touches text sharpness. Runs a rAF lerp so the light
// *follows* you with a little lag (alive, not rigid). Fully disabled for touch-only
// devices and reduced-motion users; the site is 100% usable without it.
export function initCursorLight() {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finePointer = window.matchMedia('(pointer: fine)').matches;
  if (reduce || !finePointer) return () => {};

  const root = document.documentElement;
  let tx = window.innerWidth * 0.5;
  let ty = window.innerHeight * 0.35;
  let cx = tx;
  let cy = ty;
  let raf = 0;
  let active = false;

  const onMove = (e) => {
    tx = e.clientX;
    ty = e.clientY;
    if (!active) { active = true; loop(); }
  };

  const loop = () => {
    cx += (tx - cx) * 0.12; // lerp — the light trails the pointer
    cy += (ty - cy) * 0.12;
    root.style.setProperty('--mx', `${(cx / window.innerWidth) * 100}%`);
    root.style.setProperty('--my', `${(cy / window.innerHeight) * 100}%`);
    if (Math.abs(tx - cx) > 0.3 || Math.abs(ty - cy) > 0.3) {
      raf = requestAnimationFrame(loop);
    } else {
      active = false;
    }
  };

  window.addEventListener('pointermove', onMove, { passive: true });
  return () => {
    window.removeEventListener('pointermove', onMove);
    cancelAnimationFrame(raf);
  };
}

// Read/apply the light|dark room, persisted so the choice survives reloads.
export function getMode() {
  return document.documentElement.getAttribute('data-mode') || 'light';
}
export function setMode(mode) {
  document.documentElement.setAttribute('data-mode', mode);
  try { localStorage.setItem('hirelens_mode', mode); } catch { /* storage blocked */ }
}
