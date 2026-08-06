#!/usr/bin/env node
// Build the real solar-system layer from Minor Planet Center catalogues.
//
//   MPCORB.DAT.gz  ~1.55M numbered + unnumbered minor planets
//   Distant.txt    ~8.1k TNOs / Centaurs (merged for anything MPCORB lags on)
//   CometEls.txt   ~950 comets, including long-period ones whose measured
//                  aphelia genuinely reach the Oort cloud
//
// Output is orbital ELEMENTS, not positions: Kepler is solved in the vertex
// shader, so 1.5M objects cost 16 bytes each and actually orbit.
//
// Usage: node scripts/build-solar.mjs <rawDir> <outDir>

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

const rawDir = process.argv[2];
const outDir = process.argv[3];
if (!rawDir || !outDir) {
  console.error('usage: node scripts/build-solar.mjs <rawDir> <outDir>');
  process.exit(1);
}

// ---------------------------------------------------------------- date utils
// MPC packed epoch: century letter, 2-digit year, month char, day char.
// e.g. "K2669" -> 2026-06-09  (K=20xx, 26, month 6, day 9)
const unpackChar = (c) => {
  const n = c.charCodeAt(0);
  if (n >= 48 && n <= 57) return n - 48;        // '0'-'9'
  if (n >= 65 && n <= 90) return n - 65 + 10;   // 'A'-'Z' -> 10-35
  if (n >= 97 && n <= 122) return n - 97 + 36;  // 'a'-'z' -> 36-61
  return 0;
};
const CENTURY = { I: 18, J: 19, K: 20, L: 21 };

function packedToJD(s) {
  if (!s || s.length < 5) return NaN;
  const cent = CENTURY[s[0]];
  if (cent === undefined) return NaN;
  const year = cent * 100 + parseInt(s.slice(1, 3), 10);
  const month = unpackChar(s[3]);
  const day = unpackChar(s[4]);
  if (!Number.isFinite(year) || !month || !day) return NaN;
  return gregorianToJD(year, month, day);
}

// Standard Gregorian -> Julian Day (integer day, 0h TT).
function gregorianToJD(y, m, d) {
  let Y = y, M = m;
  if (M <= 2) { Y -= 1; M += 12; }
  const A = Math.floor(Y / 100);
  const B = 2 - A + Math.floor(A / 4);
  return Math.floor(365.25 * (Y + 4716)) + Math.floor(30.6001 * (M + 1)) + d + B - 1524.5;
}

// Common epoch every object is propagated to, so the shader needs no per-object
// epoch: mean anomaly is linear in time for an unperturbed two-body orbit.
const EPOCH_JD = gregorianToJD(2026, 8, 5);

// --------------------------------------------------------------- classifying
// Dynamical class drives colour in the renderer.
const CLS = { NEA: 0, INNER: 1, MAIN: 2, OUTER: 3, TROJAN: 4, CENTAUR: 5, TNO: 6, SCATTERED: 7, COMET: 8 };

function classify(a, e, q) {
  if (q < 1.3) return CLS.NEA;                 // near-Earth by IAU definition
  if (a < 2.0) return CLS.INNER;               // Hungarias, Mars-crossers
  if (a < 2.82) return CLS.MAIN;               // inner + middle main belt
  if (a < 3.7) return CLS.OUTER;               // outer belt, Cybeles
  if (a < 4.6) return CLS.OUTER;               // Hildas
  if (a < 5.5) return CLS.TROJAN;              // Jupiter Trojans
  if (a < 30.1) return CLS.CENTAUR;            // Centaurs
  if (a < 50 && e < 0.24) return CLS.TNO;      // classical Kuiper belt
  return CLS.SCATTERED;                        // scattered disc, detached, sednoids
}

// ------------------------------------------------------------------- packing
// 16 bytes per object:
//   0  Float32  a   semi-major axis (AU)
//   4  Uint16   e   eccentricity      (e * 65535, bound orbits only)
//   6  Uint16   i   inclination       (i / 180 * 65535)
//   8  Uint16   Om  longitude of ascending node (deg / 360 * 65535)
//  10  Uint16   w   argument of perihelion      (deg / 360 * 65535)
//  12  Uint16   M   mean anomaly at EPOCH_JD    (deg / 360 * 65535)
//  14  Uint8    H   absolute magnitude, stored as (H + 2) * 8
//  15  Uint8    cls dynamical class
const REC = 16;
const u16deg = (d) => {
  let x = d % 360;
  if (x < 0) x += 360;
  return Math.min(65535, Math.round((x / 360) * 65535));
};

function makeWriter(capacity) {
  const buf = Buffer.alloc(capacity * REC);
  let n = 0;
  return {
    push(a, e, inc, Om, w, M, H, cls) {
      const o = n * REC;
      buf.writeFloatLE(a, o);
      buf.writeUInt16LE(Math.min(65535, Math.max(0, Math.round(e * 65535))), o + 4);
      buf.writeUInt16LE(Math.min(65535, Math.max(0, Math.round((inc / 180) * 65535))), o + 6);
      buf.writeUInt16LE(u16deg(Om), o + 8);
      buf.writeUInt16LE(u16deg(w), o + 10);
      buf.writeUInt16LE(u16deg(M), o + 12);
      buf.writeUInt8(Math.min(255, Math.max(0, Math.round((H + 2) * 8))), o + 14);
      buf.writeUInt8(cls, o + 15);
      n++;
    },
    get count() { return n; },
    slice() { return buf.subarray(0, n * REC); },
  };
}

// --------------------------------------------------------- MPCORB-style lines
// Fixed-width columns, verified against Ceres (a=2.7655526, e=0.0796923,
// i=10.58803) and Pluto. parseFloat tolerates the leading pad spaces.
function parseMPCOrbLine(line) {
  if (line.length < 103) return null;
  const H = parseFloat(line.slice(8, 13));
  const epoch = line.slice(20, 25);
  const M = parseFloat(line.slice(26, 35));
  const w = parseFloat(line.slice(37, 46));
  const Om = parseFloat(line.slice(48, 57));
  const inc = parseFloat(line.slice(59, 68));
  const e = parseFloat(line.slice(70, 79));
  const n = parseFloat(line.slice(80, 91));
  const a = parseFloat(line.slice(92, 103));
  if (!Number.isFinite(a) || !Number.isFinite(e) || !Number.isFinite(inc)) return null;
  if (!Number.isFinite(M) || !Number.isFinite(w) || !Number.isFinite(Om)) return null;
  if (a <= 0 || e < 0 || e >= 1) return null;   // bound elliptical orbits only
  return { H: Number.isFinite(H) ? H : 18, epoch, M, w, Om, inc, e, a, n };
}

// Propagate mean anomaly from the object's own epoch to the common epoch.
function propagate(rec) {
  const jd = packedToJD(rec.epoch);
  // Gaussian mean motion (deg/day) if the file's n is missing.
  const n = Number.isFinite(rec.n) && rec.n > 0 ? rec.n : 0.9856076686 / Math.pow(rec.a, 1.5);
  if (!Number.isFinite(jd)) return rec.M;
  return rec.M + n * (EPOCH_JD - jd);
}

// ------------------------------------------------------------------ read raw
function readLines(file) {
  let buf = fs.readFileSync(file);
  if (file.endsWith('.gz')) buf = zlib.gunzipSync(buf);
  return buf.toString('latin1').split('\n');
}

console.log('epoch: JD', EPOCH_JD, '(2026-08-05)');

const writer = makeWriter(1_700_000);
const seen = new Set();
const stats = {};
let skipped = 0;
const bump = (k) => { stats[k] = (stats[k] || 0) + 1; };

function ingestMPCOrb(file, label) {
  const lines = readLines(path.join(rawDir, file));
  let added = 0;
  for (const line of lines) {
    // Header block and the blank separators between the three sections.
    if (line.length < 103 || line[0] === '-' || line.trimStart().startsWith('Des')) continue;
    const rec = parseMPCOrbLine(line);
    if (!rec) { skipped++; continue; }
    const desig = line.slice(0, 7).trim();
    if (desig && seen.has(desig)) continue;
    if (desig) seen.add(desig);
    const q = rec.a * (1 - rec.e);
    const cls = classify(rec.a, rec.e, q);
    writer.push(rec.a, rec.e, rec.inc, rec.Om, rec.w, propagate(rec), rec.H, cls);
    bump(cls);
    added++;
  }
  console.log(`${label}: +${added.toLocaleString()}`);
}

ingestMPCOrb('MPCORB.DAT.gz', 'MPCORB');
ingestMPCOrb('Distant.txt', 'Distant (TNOs/Centaurs)');

// ---------------------------------------------------------------- the comets
// Different fixed-width layout, and given by perihelion distance q + e rather
// than a. Long-period comets are the only objects with *measured* orbits that
// physically reach the Oort cloud.
{
  const lines = readLines(path.join(rawDir, 'CometEls.txt'));
  let added = 0;
  const aphelia = [];
  for (const line of lines) {
    if (line.length < 80) continue;
    const q = parseFloat(line.slice(30, 39));
    const e = parseFloat(line.slice(41, 50));
    const w = parseFloat(line.slice(51, 60));
    const Om = parseFloat(line.slice(61, 70));
    const inc = parseFloat(line.slice(71, 80));
    if (![q, e, w, Om, inc].every(Number.isFinite)) continue;
    // Parabolic/hyperbolic fits (e >= 1) have no bound orbit to draw.
    if (e >= 0.9999 || q <= 0) continue;
    const a = q / (1 - e);
    if (!Number.isFinite(a) || a <= 0) continue;

    const py = parseInt(line.slice(14, 18), 10);
    const pm = parseInt(line.slice(19, 21), 10);
    const pd = parseFloat(line.slice(22, 29));
    let M = 0;
    if (Number.isFinite(py) && Number.isFinite(pm) && Number.isFinite(pd)) {
      const perihJD = gregorianToJD(py, pm, Math.floor(pd)) + (pd - Math.floor(pd));
      const n = 0.9856076686 / Math.pow(a, 1.5);
      M = n * (EPOCH_JD - perihJD);   // M = 0 at perihelion passage
    }
    const H = parseFloat(line.slice(91, 95));
    writer.push(a, e, inc, Om, w, M, Number.isFinite(H) ? H : 12, CLS.COMET);
    bump(CLS.COMET);
    aphelia.push(a * (1 + e));
    added++;
  }
  aphelia.sort((x, y) => y - x);
  console.log(`Comets: +${added.toLocaleString()}`);
  console.log(`  measured aphelia > 1,000 AU : ${aphelia.filter((x) => x > 1000).length}`);
  console.log(`  measured aphelia > 10,000 AU: ${aphelia.filter((x) => x > 10000).length}`);
  console.log(`  farthest measured aphelion  : ${Math.round(aphelia[0]).toLocaleString()} AU`);
}

// ------------------------------------------------------------------- outputs
fs.mkdirSync(outDir, { recursive: true });
const out = writer.slice();
fs.writeFileSync(path.join(outDir, 'sso.bin'), out);

const NAMES = ['NEA', 'inner', 'mainBelt', 'outerBelt', 'trojan', 'centaur', 'kuiper', 'scattered', 'comet'];
const byClass = {};
for (const [k, v] of Object.entries(stats)) byClass[NAMES[k]] = v;

const meta = {
  count: writer.count,
  epochJD: EPOCH_JD,
  record: 16,
  bytes: out.length,
  byClass,
  source: 'IAU Minor Planet Center — MPCORB, Distant.txt, CometEls.txt',
  retrieved: new Date().toISOString().slice(0, 10),
};
fs.writeFileSync(path.join(outDir, 'sso_meta.json'), JSON.stringify(meta, null, 2));

console.log('\n--- solar system layer ---');
console.log('objects :', writer.count.toLocaleString());
console.log('skipped :', skipped.toLocaleString(), '(unbound / malformed)');
console.log('bytes   :', (out.length / 1e6).toFixed(1), 'MB');
console.table(byClass);
