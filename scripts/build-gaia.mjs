#!/usr/bin/env node
// Reduce the raw Gaia DR3 download into the packed point layout the renderer uses.
//
// Input is whatever scripts/fetch-gaia.mjs left in <rawDir>: ~120 CSV chunks,
// ~9.6 GB, ~98M stars with parallax_over_error > 10. That is far too much to hold
// as text, so this streams line by line and never materialises a chunk as a string.
//
// Distance is the naive parallax inversion, 1000/parallax mas. That inversion is
// only honest because the fetch already cut at S/N > 10: above that the fractional
// distance error is under ~10% and the posterior is dominated by the measurement,
// not the prior. Below it the inversion is actively misleading, which is why those
// stars were never downloaded.
//
// Output (same layout as every other layer):
//   gaia_pos.bin  Float32 x,y,z   (parsecs, equatorial ICRS cartesian, Sun at origin)
//   gaia_col.bin  Uint8 r,g,b,size
//   gaia.json     manifest + provenance
//
// Usage: node scripts/build-gaia.mjs <rawDir> <outDir> [--max N] [--cap-mag M]

import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';

const [, , rawDir, outDir] = process.argv;
if (!rawDir || !outDir) {
  console.error('usage: node scripts/build-gaia.mjs <rawDir> <outDir> [--max N] [--cap-mag M]');
  process.exit(1);
}
const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i >= 0 ? +process.argv[i + 1] : d; };
const MAX_STARS = arg('max', 120e6);      // hard ceiling on the preallocation
const CAP_MAG = arg('cap-mag', Infinity); // drop everything fainter than this G
// Available but deliberately unused by default. Gaia saturates at the bright end —
// Vega (G=0.03) and Altair (G=0.76) have no usable DR3 parallax and are simply
// absent — so it is tempting to cut Gaia bright-ward and let the older 2M-star
// layer own that range. Measurement says don't: that layer has Vega and Proxima
// but is MISSING Barnard's Star at V=9.5, which Gaia places to 0.001 pc. The two
// catalogues are complementary, not nested, so any bright cut deletes real stars
// nothing else carries. Both layers are drawn in full instead; the price is that
// the ~2M stars in both get their flux counted twice, which asinh tone mapping
// largely absorbs.
const MIN_MAG = arg('min-mag', -Infinity); // drop everything brighter than this G

const D = Math.PI / 180;
fs.mkdirSync(outDir, { recursive: true });

// ---------------------------------------------------------------- colour
// Gaia gives BP-RP, a colour index. Ballesteros' formula is written for B-V, and
// BP-RP is a wider baseline, so feeding one into the other is an approximation —
// but it is monotonic and lands O/B stars blue and M stars red, which is the whole
// job here. The result is a rendering hue, not a published temperature.
function teffFromBpRp(bpRp) {
  const bv = 0.85 * bpRp;   // rough BP-RP -> B-V rescale
  return 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62));
}

// Blackbody -> sRGB, Tanner Helland's piecewise fit. Normalised so the brightest
// channel is 255: the size byte carries brightness, this carries hue only.
function rgbFromTeff(T) {
  const t = Math.min(40000, Math.max(1000, T)) / 100;
  let r, g, b;
  if (t <= 66) { r = 255; } else { r = 329.698727446 * Math.pow(t - 60, -0.1332047592); }
  if (t <= 66) { g = 99.4708025861 * Math.log(t) - 161.1195681661; }
  else { g = 288.1221695283 * Math.pow(t - 60, -0.0755148492); }
  if (t >= 66) { b = 255; }
  else if (t <= 19) { b = 0; }
  else { b = 138.5177312231 * Math.log(t - 10) - 305.0447927307; }
  r = Math.max(0, Math.min(255, r));
  g = Math.max(0, Math.min(255, g));
  b = Math.max(0, Math.min(255, b));
  const m = Math.max(r, g, b) || 1;
  return [Math.round(255 * r / m), Math.round(255 * g / m), Math.round(255 * b / m)];
}

// Precompute a 512-entry colour ramp over the BP-RP range Gaia actually spans.
// Per-star pow/log across 98M stars is pure waste when the input is one number.
const RAMP_LO = -0.6, RAMP_HI = 5.0, RAMP_N = 512;
const RAMP = new Uint8Array(RAMP_N * 3);
for (let i = 0; i < RAMP_N; i++) {
  const bpRp = RAMP_LO + (RAMP_HI - RAMP_LO) * (i / (RAMP_N - 1));
  const [r, g, b] = rgbFromTeff(teffFromBpRp(bpRp));
  RAMP[i * 3] = r; RAMP[i * 3 + 1] = g; RAMP[i * 3 + 2] = b;
}
const rampIndex = (bpRp) => {
  const f = (bpRp - RAMP_LO) / (RAMP_HI - RAMP_LO);
  return Math.max(0, Math.min(RAMP_N - 1, Math.round(f * (RAMP_N - 1))));
};

// ---------------------------------------------------------------- brightness
// The size byte is the octree's importance key, so it must rank stars the way an
// observer would: by apparent brightness. Gaia's G runs about -1 (Sirius-class) to
// 21 (the survey floor); this sample, cut at parallax S/N > 10, is mostly G 8-19.
// Map bright -> high byte, and keep the faint end off zero so it still renders.
const SIZE_BRIGHT = 3.0, SIZE_FAINT = 20.0;
function sizeByte(G) {
  const f = (SIZE_FAINT - G) / (SIZE_FAINT - SIZE_BRIGHT);
  return Math.max(12, Math.min(255, Math.round(12 + 243 * Math.max(0, Math.min(1, f)))));
}

// ---------------------------------------------------------------- parse
const files = fs.readdirSync(rawDir).filter((f) => /^gaia_\d+\.csv$/.test(f)).sort();
if (!files.length) { console.error('no gaia_*.csv in ' + rawDir); process.exit(1); }
console.log(`chunks: ${files.length}`);

const pos = new Float32Array(MAX_STARS * 3);
const col = new Uint8Array(MAX_STARS * 4);

let n = 0, seen = 0;
let badParallax = 0, badNum = 0, tooFar = 0, magCut = 0, noPhot = 0, noColour = 0, brightCut = 0;
let dmin = Infinity, dmax = 0;
const magHist = new Int32Array(32);

for (const f of files) {
  const rl = readline.createInterface({
    input: fs.createReadStream(path.join(rawDir, f), { highWaterMark: 1 << 22 }),
    crlfDelay: Infinity,
  });
  let first = true;
  for await (const line of rl) {
    if (first) { first = false; continue; }     // header
    if (!line) continue;
    seen++;

    // source_id,ra,dec,parallax,parallax_over_error,phot_g_mean_mag,bp_rp
    // Hand-split rather than line.split(','): this runs ~98M times.
    let a = line.indexOf(',');                       // after source_id
    let b = line.indexOf(',', a + 1);                // after ra
    let c = line.indexOf(',', b + 1);                // after dec
    let d = line.indexOf(',', c + 1);                // after parallax
    let e = line.indexOf(',', d + 1);                // after parallax_over_error
    let g = line.indexOf(',', e + 1);                // after phot_g_mean_mag
    if (g < 0) { badNum++; continue; }

    // An empty CSV field coerces to 0, not NaN. That is not a rounding detail here:
    // a missing phot_g_mean_mag would become G=0, i.e. brighter than Sirius, and
    // since the octree samples brightest-first those sources would be the first
    // thing loaded at every zoom level. Empty must mean missing.
    const num = (s) => (s.length ? +s : NaN);
    const ra = num(line.slice(a + 1, b));
    const dec = num(line.slice(b + 1, c));
    const plx = num(line.slice(c + 1, d));
    const G = num(line.slice(e + 1, g));
    const bpRpRaw = num(line.slice(g + 1));

    if (!Number.isFinite(ra) || !Number.isFinite(dec)) { badNum++; continue; }
    // No G means no measured brightness, and brightness is what orders the octree.
    // There is nothing honest to substitute, so these are dropped.
    if (!Number.isFinite(G)) { noPhot++; continue; }
    if (!(plx > 0)) { badParallax++; continue; }
    if (G > CAP_MAG) { magCut++; continue; }
    if (G < MIN_MAG) { brightCut++; continue; }
    // Colour is hue only, so a missing BP-RP can safely fall back to solar.
    const bpRp = Number.isFinite(bpRpRaw) ? bpRpRaw : (noColour++, 0.82);

    const dist = 1000 / plx;                          // mas -> parsecs
    // Nothing in a parallax-selected sample belongs past the far side of the disc;
    // a handful of near-zero parallaxes survive S/N cuts and land absurdly far.
    if (!(dist > 0 && dist < 200000)) { tooFar++; continue; }

    if (n >= MAX_STARS) break;

    const cd = Math.cos(dec * D);
    const raR = ra * D;
    pos[n * 3] = dist * cd * Math.cos(raR);
    pos[n * 3 + 1] = dist * cd * Math.sin(raR);
    pos[n * 3 + 2] = dist * Math.sin(dec * D);

    const ri = rampIndex(bpRp) * 3;
    col[n * 4] = RAMP[ri];
    col[n * 4 + 1] = RAMP[ri + 1];
    col[n * 4 + 2] = RAMP[ri + 2];
    col[n * 4 + 3] = sizeByte(G);

    if (dist < dmin) dmin = dist;
    if (dist > dmax) dmax = dist;
    const mi = Math.max(0, Math.min(31, Math.floor(G)));
    magHist[mi]++;
    n++;
  }
  rl.close();
  process.stdout.write(`\r  ${f}  kept ${n.toLocaleString()} / seen ${seen.toLocaleString()}   `);
  if (n >= MAX_STARS) { console.log('\n  hit --max ceiling, stopping'); break; }
}
console.log('');

// ---------------------------------------------------------------- write
const posOut = Buffer.from(pos.buffer, 0, n * 3 * 4);
const colOut = Buffer.from(col.buffer, 0, n * 4);
fs.writeFileSync(path.join(outDir, 'gaia_pos.bin'), posOut);
fs.writeFileSync(path.join(outDir, 'gaia_col.bin'), colOut);

fs.writeFileSync(path.join(outDir, 'gaia.json'), JSON.stringify({
  count: n,
  bytes: posOut.length + colOut.length,
  distancePcRange: [+dmin.toFixed(2), Math.round(dmax)],
  magnitudeRange: [Number.isFinite(MIN_MAG) ? MIN_MAG : null, Number.isFinite(CAP_MAG) ? CAP_MAG : null],
  droppedNoPhotometry: noPhot,
  droppedTooBright: brightCut,
  keptWithoutColour: noColour,
  source: 'ESA Gaia DR3 gaia_source (parallax_over_error > 10)',
  distances: 'parallax inversion d = 1000/plx[mas]; valid because the S/N>10 cut keeps fractional error under ~10%',
  colour: 'BP-RP -> Teff (Ballesteros, approximate) -> blackbody sRGB; hue only, brightness carried by the size byte',
  frame: 'equatorial ICRS cartesian, parsecs, Sun at origin',
  retrieved: new Date().toISOString().slice(0, 10),
}, null, 2));

console.log(`\nstars kept   : ${n.toLocaleString()} of ${seen.toLocaleString()} rows`);
console.log(`rejected     : parallax<=0 ${badParallax.toLocaleString()}, unparsable ${badNum.toLocaleString()}, ` +
  `no G photometry ${noPhot.toLocaleString()}, out of range ${tooFar.toLocaleString()}, mag cut ${magCut.toLocaleString()}`);
console.log(`kept w/o colour: ${noColour.toLocaleString()} (BP-RP missing, drawn solar-coloured)`);
if (brightCut) console.log(`bright cut   : ${brightCut.toLocaleString()} at G < ${MIN_MAG} (left to the Tycho-era layer, where Gaia saturates)`);
console.log(`distance     : ${dmin.toFixed(1)} pc - ${(dmax / 1000).toFixed(1)} kpc`);
console.log(`bytes        : ${((posOut.length + colOut.length) / 1e6).toFixed(1)} MB`);
console.log('G magnitude histogram:');
for (let i = 0; i < 32; i++) {
  if (!magHist[i]) continue;
  const bar = '#'.repeat(Math.round(60 * magHist[i] / Math.max(...magHist)));
  console.log(`  G ${String(i).padStart(2)}  ${String(magHist[i]).padStart(10)}  ${bar}`);
}
