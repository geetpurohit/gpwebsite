#!/usr/bin/env node
// Globular clusters as real objects rather than uniform dots.
//
// Source: Harris 1996 (2010 edition), VizieR VII/202. Each cluster carries a
// measured half-light radius, absolute magnitude and King concentration, so a
// sparse outer-halo cluster no longer looks identical to Omega Centauri.
//
// Note on identifiers: the catalogue's `Name` column holds the COMMON name and is
// usually blank; the NGC designation lives in `ID`. Matching on `Name` silently
// matched nothing, which is why labels came out empty the first time.
//
// Usage: node scripts/build-globulars.mjs <rawDir> <outDir>

import fs from 'node:fs';
import path from 'node:path';

const rawDir = process.argv[2], outDir = process.argv[3];
if (!rawDir || !outDir) { console.error('usage: build-globulars.mjs <rawDir> <outDir>'); process.exit(1); }

const D = Math.PI / 180;
const sex = (s, isRA) => {
  const m = (s || '').trim().split(/\s+/).map(parseFloat);
  if (m.length < 2 || m.some((v) => !Number.isFinite(v))) return null;
  const sg = /^-/.test(s.trim()) ? -1 : 1;
  return sg * (Math.abs(m[0]) + (m[1] || 0) / 60 + (m[2] || 0) / 3600) * (isRA ? 15 : 1);
};
const num = (s) => { const v = parseFloat(s); return Number.isFinite(v) ? v : null; };

// Messier designations for the clusters people actually recognise
const MESSIER = {
  'NGC 104': '47 Tucanae', 'NGC 5139': 'Omega Centauri', 'NGC 6656': 'M22', 'NGC 6205': 'M13',
  'NGC 7078': 'M15', 'NGC 5272': 'M3', 'NGC 6121': 'M4', 'NGC 6341': 'M92', 'NGC 6254': 'M10',
  'NGC 6218': 'M12', 'NGC 6715': 'M54', 'NGC 6626': 'M28', 'NGC 6266': 'M62', 'NGC 6093': 'M80',
  'NGC 5904': 'M5', 'NGC 6333': 'M9', 'NGC 6402': 'M14', 'NGC 7089': 'M2', 'NGC 6864': 'M75',
  'NGC 6397': 'NGC 6397', 'NGC 6752': 'NGC 6752', 'NGC 362': 'NGC 362',
};

const lines = fs.readFileSync(path.join(rawDir, 'gc.tsv'), 'latin1').split('\n');
const pos = [], col = [], labels = [];
let n = 0, nP = 0, rhMin = Infinity, rhMax = 0, noRh = 0;

for (const L of lines) {
  if (!L || L[0] === '#') continue;
  const f = L.split('\t');
  if (f.length < 8) continue;
  const id = (f[0] || '').trim();          // ID, e.g. "NGC 104"
  const common = (f[1] || '').trim();      // common name, often blank
  const ra = sex(f[2], true), de = sex(f[3], false);
  const dkpc = num(f[4]), MV = num(f[5]), conc = num(f[6]), Rh = num(f[7]);
  if (ra === null || de === null || dkpc === null || !(dkpc > 0)) continue;

  const d = dkpc * 1000;
  if (Rh === null) noRh++;
  const rh = Rh !== null ? d * (Rh / 60) * D : 4;   // arcmin -> parsecs
  if (!(rh > 0.3 && rh < 200)) continue;
  if (rh < rhMin) rhMin = rh;
  if (rh > rhMax) rhMax = rh;

  // luminosity sets the star count; King concentration sets how cuspy the core is
  const cnt = Math.max(30, Math.min(4000, Math.round(Math.pow(10, (-(MV ?? -6) - 1.5) / 2.6))));
  const cc = conc !== null ? Math.max(0.7, Math.min(2.5, conc)) : 1.5;
  const pw = 0.85 + (cc - 0.7) * 0.55;

  const c = [Math.cos(de * D) * Math.cos(ra * D), Math.cos(de * D) * Math.sin(ra * D), Math.sin(de * D)];
  for (let i = 0; i < cnt; i++) {
    const rr = rh * 3.2 * Math.pow(Math.random(), pw);
    const th = Math.random() * Math.PI * 2, cph = 2 * Math.random() - 1;
    const sph = Math.sqrt(1 - cph * cph);
    pos.push(
      +(c[0] * d + rr * sph * Math.cos(th)).toFixed(1),
      +(c[1] * d + rr * sph * Math.sin(th)).toFixed(1),
      +(c[2] * d + rr * cph).toFixed(1),
    );
    col.push(1.0, 0.88, 0.62);   // old, metal-poor: warm yellow
    nP++;
  }

  // Collect every candidate, then rank at the end. Capping inside the loop let
  // obscure clusters (E 1, AvdB, ESO452-SC11) fill the quota in RA order and crowd
  // out M13, M15 and M22.
  const messier = MESSIER[id];
  const tidy = (common || '').replace(/^M\s+(\d)/, 'M$1');
  const nice = messier || (/^M\d+$/.test(tidy) ? tidy : null);
  if (nice) {
    labels.push({ name: nice, prio: messier ? 0 : 1, mv: MV ?? 0,
      x: +(c[0] * d).toFixed(0), y: +(c[1] * d).toFixed(0), z: +(c[2] * d).toFixed(0) });
  }
  n++;
}

// rank: curated names first, then brightest, and keep the best 16
labels.sort((a, b) => (a.prio - b.prio) || (a.mv - b.mv));
labels.splice(16);
for (const l of labels) { delete l.prio; delete l.mv; }

fs.mkdirSync(outDir, { recursive: true });
const size = new Array(nP).fill(0.5);
fs.writeFileSync(path.join(outDir, 'globulars.json'), JSON.stringify({
  count: n, points: nP, pos, col, size, labels,
  source: 'Harris 1996 (2010 ed.), VizieR VII/202 — half-light radius, luminosity, King concentration',
  note: 'each cluster is a King-like sphere at its measured half-light radius and concentration',
  frame: 'equatorial ICRS cartesian, parsecs',
  retrieved: new Date().toISOString().slice(0, 10),
}));

console.log('globulars:', n, '| points:', nP.toLocaleString(),
  '| r_h', rhMin.toFixed(1), '-', rhMax.toFixed(1), 'pc',
  '| no R_h:', noRh, '| labels:', labels.length);
console.log('labelled:', labels.map((l) => l.name).join(', '));
