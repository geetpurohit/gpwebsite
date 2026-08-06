#!/usr/bin/env node
// Build the quasar layer from the Million Quasars Catalogue (Milliquas, VII/294).
//
// This is what takes the map past its current 600 Mpc edge. Quasars are the most
// luminous persistent objects known, so they are visible almost to the edge of
// the observable universe: the catalogue reaches z ~ 7, roughly 8,900 Mpc
// comoving, versus 14,200 Mpc to the surface of last scattering.
//
// Distances are comoving, integrated properly rather than using a linear Hubble
// law — at these redshifts v = cz is wrong by a factor of several.
//
// Usage: node scripts/build-quasars.mjs <rawDir> <outDir>

import fs from 'node:fs';
import path from 'node:path';

const rawDir = process.argv[2];
const outDir = process.argv[3];
if (!rawDir || !outDir) {
  console.error('usage: node scripts/build-quasars.mjs <rawDir> <outDir>');
  process.exit(1);
}

// Planck 2018 flat LCDM
const H0 = 67.66, OM = 0.3111, OL = 1 - OM;
const C_KMS = 299792.458;
const D_H = C_KMS / H0;                 // Hubble distance, Mpc

// Comoving distance by Simpson integration of dz/E(z).
const Ez = (z) => Math.sqrt(OM * (1 + z) ** 3 + OL);
function comovingMpc(z) {
  const n = 256, h = z / n;
  let s = 1 / Ez(0) + 1 / Ez(z);
  for (let i = 1; i < n; i++) s += (i % 2 ? 4 : 2) / Ez(i * h);
  return D_H * (h / 3) * s;
}
// sanity anchors printed below so the cosmology is visibly right
for (const z of [0.1, 1, 3, 7]) {
  console.log(`  z=${String(z).padEnd(4)} -> ${comovingMpc(z).toFixed(0).padStart(5)} Mpc comoving`);
}

const D = Math.PI / 180;
const num = (s) => { const v = parseFloat(s); return Number.isFinite(v) ? v : null; };

const lines = fs.readFileSync(path.join(rawDir, 'milliquas.tsv'), 'latin1').split('\n');
const pos = [], col = [];
let n = 0, noZ = 0, zmax = 0, dmax = 0;
const byKind = { quasar: 0, agn: 0, other: 0 };
const zHist = new Array(16).fill(0);

// Colour by redshift: a perceptual ramp from cool (nearby) to deep red (early
// universe). Redshift IS the colour here, so this encodes lookback time directly.
function zColor(z) {
  const t = Math.min(1, z / 5);
  if (t < 0.5) { const u = t / 0.5; return [0.45 + 0.35 * u, 0.80 - 0.10 * u, 1.0 - 0.35 * u]; }
  const u = (t - 0.5) / 0.5;
  return [0.80 + 0.20 * u, 0.70 - 0.48 * u, 0.65 - 0.50 * u];
}

for (const L of lines) {
  if (!L || L[0] === '#') continue;
  const f = L.split('\t');
  if (f.length < 4) continue;
  const ra = num(f[0]), de = num(f[1]), z = num(f[2]);
  const type = (f[3] || '').trim();
  if (ra === null || de === null) continue;
  if (z === null || !(z > 0.001)) { noZ++; continue; }

  const dMpc = comovingMpc(z);
  const dPc = dMpc * 1e6;
  const rr = ra * D, dd = de * D;
  pos.push(
    dPc * Math.cos(dd) * Math.cos(rr),
    dPc * Math.cos(dd) * Math.sin(rr),
    dPc * Math.sin(dd),
  );
  const c = zColor(z);
  // brighter marker for spectroscopically confirmed type-I quasars
  const isQ = type.startsWith('Q');
  col.push(
    Math.round(255 * c[0]), Math.round(255 * c[1]), Math.round(255 * c[2]),
    isQ ? 64 : 46,
  );
  if (isQ) byKind.quasar++;
  else if (/^[AN]/.test(type)) byKind.agn++;
  else byKind.other++;
  if (z > zmax) zmax = z;
  if (dMpc > dmax) dmax = dMpc;
  zHist[Math.min(15, Math.floor(z))]++;
  n++;
}

fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, 'qso_pos.bin'), Buffer.from(new Float32Array(pos).buffer));
fs.writeFileSync(path.join(outDir, 'qso_col.bin'), Buffer.from(new Uint8Array(col).buffer));
const bytes = pos.length * 4 + col.length;

console.log('\nquasars placed :', n.toLocaleString(), '| dropped (no redshift):', noZ.toLocaleString());
console.log('  type-I quasars:', byKind.quasar.toLocaleString(),
  '| AGN/Seyfert:', byKind.agn.toLocaleString(), '| other:', byKind.other.toLocaleString());
console.log('  max redshift  : z =', zmax.toFixed(2), '->', (dmax / 1000).toFixed(2), 'Gpc comoving');
console.log('  on disk       :', (bytes / 1e6).toFixed(1), 'MB');
console.log('\nredshift distribution:');
for (let i = 0; i < 8; i++) {
  if (!zHist[i]) continue;
  console.log(`  z ${i}-${i + 1}`.padEnd(10), String(zHist[i]).padStart(8),
    '#'.repeat(Math.round(40 * zHist[i] / Math.max(...zHist))));
}

fs.writeFileSync(path.join(outDir, 'quasars.json'), JSON.stringify({
  count: n,
  maxRedshift: +zmax.toFixed(3),
  maxComovingMpc: Math.round(dmax),
  source: 'Million Quasars Catalogue (Milliquas, Flesch — VizieR VII/294)',
  cosmology: `flat LCDM, H0=${H0}, Om=${OM}; comoving distance by Simpson integration`,
  frame: 'equatorial ICRS cartesian, parsecs, Sun at origin',
  retrieved: new Date().toISOString().slice(0, 10),
}, null, 2));
