#!/usr/bin/env node
// Build the Cepheid layer: real young-population tracers of the Milky Way's
// spiral arms, replacing reliance on a synthetic two-arm model.
//
//   Gaia DR3 Cepheids (I/358/vcep), ~15k
//
// Distances use the reddening-free Wesenheit magnitude
//     W = G - 1.90 (BP - RP)
// which is extinction-free by construction. That matters enormously here: unlike
// the halo RR Lyrae, Cepheids sit in the dusty disc where an extinction guess
// would dominate the error, and this catalogue carries no A_G column.
//
// The period-luminosity zero point is calibrated on LMC Cepheids (mu = 18.477),
// fitted separately per pulsation mode, then VALIDATED against the SMC, which is
// deliberately excluded from the fit.
//
// Usage: node scripts/build-cepheids.mjs <rawDir> <outDir>

import fs from 'node:fs';
import path from 'node:path';

const rawDir = process.argv[2];
const outDir = process.argv[3];
if (!rawDir || !outDir) {
  console.error('usage: node scripts/build-cepheids.mjs <rawDir> <outDir>');
  process.exit(1);
}

const D = Math.PI / 180;
const unit = (ra, dec) => [Math.cos(dec) * Math.cos(ra), Math.cos(dec) * Math.sin(ra), Math.sin(dec)];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const num = (s) => { const v = parseFloat(s); return Number.isFinite(v) ? v : null; };

const MU_LMC = 18.477;          // Pietrzynski+ 2019 eclipsing-binary distance, ~1%
const MU_SMC = 18.977;          // held back for validation only
const LMC = unit(80.89 * D, -69.756 * D);
const SMC = unit(13.19 * D, -72.829 * D);
const EZ = unit(192.85948 * D, 27.12825 * D);   // north galactic pole

// ------------------------------------------------------------------- parse
const lines = fs.readFileSync(path.join(rawDir, 'cepheids.tsv'), 'latin1').split('\n');
const stars = [];
for (const L of lines) {
  if (!L || L[0] === '#') continue;
  const f = L.split('\t');
  if (f.length < 9) continue;
  const PF = num(f[0]), P1O = num(f[1]);
  const G = num(f[2]), BP = num(f[3]), RP = num(f[4]);
  const cls = (f[5] || '').trim(), mode = (f[6] || '').trim();
  const ra = num(f[7]), de = num(f[8]);
  if (G === null || BP === null || RP === null || ra === null || de === null) continue;
  const P = PF !== null ? PF : P1O;
  if (P === null || !(P > 0)) continue;
  if (!/^(DCEP|T2CEP|ACEP)$/.test(cls)) continue;
  const W = G - 1.90 * (BP - RP);              // reddening-free
  const v = unit(ra * D, de * D);
  // group key: mode matters, first overtones are brighter at fixed period
  const key = cls + '/' + (mode === 'FIRST_OVERTONE' ? '1O' : 'F');
  stars.push({ P, W, key, v, logP: Math.log10(P) });
}
console.log('parsed Cepheids:', stars.length.toLocaleString());

// -------------------------------------------------- calibrate on the LMC
// Least-squares W = A + B logP for LMC members of each mode; the absolute
// zero point is then a = A - mu_LMC.
function fitLMC(key) {
  const s = stars.filter((x) => x.key === key && dot(x.v, LMC) > Math.cos(5 * D));
  if (s.length < 30) return null;
  // one sigma-clip pass to drop blends and non-members
  const fit = (arr) => {
    let n = arr.length, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const p of arr) { sx += p.logP; sy += p.W; sxx += p.logP * p.logP; sxy += p.logP * p.W; }
    const B = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const A = (sy - B * sx) / n;
    return { A, B, n };
  };
  let r = fit(s);
  const resid = s.map((p) => Math.abs(p.W - (r.A + r.B * p.logP)));
  const sorted = [...resid].sort((a, b) => a - b);
  const cut = sorted[Math.floor(sorted.length * 0.90)] || 1;
  const kept = s.filter((p) => Math.abs(p.W - (r.A + r.B * p.logP)) <= cut);
  r = fit(kept);
  return { a: r.A - MU_LMC, b: r.B, n: r.n };
}

const CAL = {};
for (const key of ['DCEP/F', 'DCEP/1O', 'T2CEP/F', 'ACEP/F', 'ACEP/1O']) {
  const c = fitLMC(key);
  if (c) { CAL[key] = c; console.log(`  ${key.padEnd(9)} n=${String(c.n).padStart(5)}  M_W = ${c.a.toFixed(3)} + ${c.b.toFixed(3)} logP`); }
}
// modes without enough LMC members fall back to the classical fundamental relation
const FALLBACK = CAL['DCEP/F'];
if (!FALLBACK) { console.error('LMC calibration failed'); process.exit(1); }

const distOf = (s) => {
  const c = CAL[s.key] || FALLBACK;
  const MW = c.a + c.b * s.logP;
  return Math.pow(10, (s.W - MW + 5) / 5);
};

// ------------------------------------- INDEPENDENT validation on the SMC
{
  const smc = stars.filter((x) => dot(x.v, SMC) > Math.cos(3 * D)).map(distOf).filter((d) => d > 20000 && d < 200000);
  smc.sort((a, b) => a - b);
  const med = smc[Math.floor(smc.length / 2)];
  const truth = Math.pow(10, (MU_SMC + 5) / 5);
  console.log(`\nSMC validation (NOT used in the fit): n=${smc.length}`);
  console.log(`  recovered median ${(med / 1000).toFixed(1)} kpc  vs true ${(truth / 1000).toFixed(1)} kpc` +
    `   error ${(100 * (med - truth) / truth).toFixed(1)}%`);
}

// ------------------------------------------------------------------ build
const pos = [], col = [];
let n = 0, nMW = 0, zsum = 0;
for (const s of stars) {
  const d = distOf(s);
  if (!(d > 30 && d < 200000)) continue;
  const p = [s.v[0] * d, s.v[1] * d, s.v[2] * d];
  pos.push(p[0], p[1], p[2]);
  const old = s.key.startsWith('T2CEP');
  if (old) col.push(255, 176, 112, 78);           // Type II: old population, amber
  else col.push(214, 238, 255, 96);               // classical: young, hot, blue-white
  if (d < 30000) { nMW++; zsum += Math.abs(dot(p, EZ)); }
  n++;
}
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, 'cep_pos.bin'), Buffer.from(new Float32Array(pos).buffer));
fs.writeFileSync(path.join(outDir, 'cep_col.bin'), Buffer.from(new Uint8Array(col).buffer));

console.log('\nCepheids written:', n.toLocaleString());
console.log('mean |z| of Milky Way Cepheids:', (zsum / nMW).toFixed(0), 'pc',
  '(young disc scale height is ~70-100 pc)');

fs.writeFileSync(path.join(outDir, 'cepheids.json'), JSON.stringify({
  count: n,
  source: 'Gaia DR3 Cepheids (I/358/vcep, Ripepi et al. 2023)',
  distances: 'reddening-free Wesenheit W = G - 1.90(BP-RP); PL zero point calibrated on LMC (mu=18.477), validated on SMC',
  calibration: CAL,
  frame: 'equatorial ICRS cartesian, parsecs, Sun at origin',
  retrieved: new Date().toISOString().slice(0, 10),
}, null, 2));
