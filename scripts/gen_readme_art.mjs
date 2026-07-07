// Generates the README hero art — the aperture bloom, in light and dark, as
// standalone SVGs (hardcoded colors; GitHub sanitizes CSS vars/scripts). The
// geometry matches the live component (§6.6) so the README shows the real signature.
import { writeFileSync, mkdirSync } from 'node:fs';

const CX = 300, CY = 190, RING_R = 128, PUPIL_R = 44;

// Petal geometry, scaled ~1.6x from the component's 34+val*40 / 15+val*6.
function petal(value) {
  const tip = (34 + value * 40) * 1.6;
  const w = (15 + value * 6) * 1.6;
  return `M${CX} ${CY} Q ${CX - w} ${CY - tip * 0.55} ${CX} ${CY - tip} Q ${CX + w} ${CY - tip * 0.55} ${CX} ${CY} Z`;
}

// A representative "strong fit" flower (skills-forward, balanced).
const VALUES = [0.82, 0.9, 0.94, 0.72, 0.68];

function ringsBackdrop(stroke) {
  const ticks = Array.from({ length: 48 }).map((_, i) => {
    const a = (i / 48) * Math.PI * 2;
    const r1 = 176, r2 = i % 4 === 0 ? 162 : 169;
    return `<line x1="${(CX + Math.cos(a) * r1).toFixed(1)}" y1="${(CY + Math.sin(a) * r1).toFixed(1)}" x2="${(CX + Math.cos(a) * r2).toFixed(1)}" y2="${(CY + Math.sin(a) * r2).toFixed(1)}"/>`;
  }).join('');
  return `
  <g fill="none" stroke="${stroke}" stroke-opacity="0.10">
    <circle cx="${CX}" cy="${CY}" r="176"/>
    <circle cx="${CX}" cy="${CY}" r="140" stroke-dasharray="2 7"/>
    <circle cx="${CX}" cy="${CY}" r="104"/>
  </g>
  <g stroke="${stroke}" stroke-opacity="0.14" stroke-width="1.2">${ticks}</g>`;
}

function bloom({ pupilFill, pupilStroke, numberFill, ring, ringStroke, backdrop, glow }) {
  const petals = VALUES.map((v, k) => {
    const fill = k % 3 === 0 ? '#E85A2C' : '#F2896A';
    return `<path d="${petal(v)}" fill="${fill}" fill-opacity="0.72" transform="rotate(${k * 72} ${CX} ${CY})"/>`;
  }).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 380" width="600" height="380" role="img" aria-label="HireLens — the aperture bloom, a fit score resolving into focus">
  <defs>
    <radialGradient id="glow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#E85A2C" stop-opacity="${glow}"/>
      <stop offset="60%" stop-color="#E85A2C" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect x="${CX - 180}" y="${CY - 180}" width="360" height="360" fill="url(#glow)"/>
  ${ringsBackdrop(backdrop)}
  <g>${petals}</g>
  <circle cx="${CX}" cy="${CY}" r="${RING_R}" fill="none" stroke="${ringStroke}" stroke-opacity="0.25" stroke-width="8"/>
  <circle cx="${CX}" cy="${CY}" r="${RING_R}" fill="none" stroke="${ring}" stroke-width="8" stroke-linecap="round"
    pathLength="100" stroke-dasharray="86 100" transform="rotate(-90 ${CX} ${CY})"/>
  <circle cx="${CX}" cy="${CY}" r="${PUPIL_R}" fill="${pupilFill}" stroke="${pupilStroke}" stroke-width="2"/>
  <text x="${CX}" y="${CY + 1}" text-anchor="middle" dominant-baseline="central"
    font-family="Menlo, Consolas, monospace" font-weight="700" font-size="46" fill="${numberFill}"
    style="font-variant-numeric:tabular-nums">86</text>
</svg>`;
}

const light = bloom({
  pupilFill: '#FFFFFF', pupilStroke: '#E7E7EA', numberFill: '#C7431A',
  ring: '#E85A2C', ringStroke: '#0A0A0B', backdrop: '#0A0A0B', glow: 0.14,
});
const dark = bloom({
  pupilFill: '#151517', pupilStroke: '#262629', numberFill: '#FF6B3D',
  ring: '#FF6B3D', ringStroke: '#F4F4F2', backdrop: '#F4F4F2', glow: 0.30,
});

mkdirSync('docs', { recursive: true });
writeFileSync('docs/hero-bloom-light.svg', light);
writeFileSync('docs/hero-bloom-dark.svg', dark);
console.log('Wrote docs/hero-bloom-light.svg and docs/hero-bloom-dark.svg');
