#!/usr/bin/env node
// Build the cosmic microwave background shell — the outermost thing that can be
// mapped at all.
//
// Source: WMAP 9-year Internal Linear Combination map (LAMBDA), HEALPix NESTED,
// Nside=512, thermodynamic mK. Downsampled to Nside=256 (786,432 pixels), which
// is still finer than the map's own 1-degree smoothing, so no real information
// is lost.
//
// This is not a catalogue of objects. It is a SURFACE: the shell at which the
// universe became transparent, ~380,000 years after the Big Bang. Every point is
// a direction on the sky with a measured temperature, placed at the comoving
// distance to last scattering. Nothing can ever be mapped beyond it with light.
//
// Usage: node scripts/build-cmb.mjs <rawDir> <outDir>

import fs from 'node:fs';
import path from 'node:path';

const rawDir = process.argv[2];
const outDir = process.argv[3];
if (!rawDir || !outDir) {
  console.error('usage: node scripts/build-cmb.mjs <rawDir> <outDir>');
  process.exit(1);
}

// ---------------------------------------------------------------- cosmology
const H0 = 67.66, OM = 0.3111, OL = 1 - OM, C_KMS = 299792.458;
const D_H = C_KMS / H0;
const Ez = (z) => Math.sqrt(OM * (1 + z) ** 3 + OL);
function comovingMpc(z) {
  const n = 4096, h = z / n;
  let s = 1 / Ez(0) + 1 / Ez(z);
  for (let i = 1; i < n; i++) s += (i % 2 ? 4 : 2) / Ez(i * h);
  return D_H * (h / 3) * s;
}
const Z_LSS = 1089.8;                       // Planck 2018 last-scattering redshift
const D_LSS = comovingMpc(Z_LSS);           // comoving distance, Mpc
console.log('last scattering: z =', Z_LSS, '->', D_LSS.toFixed(0), 'Mpc comoving',
  '(' + (D_LSS / 1000).toFixed(2) + ' Gpc)');

// ------------------------------------------------------------------ HEALPix
// pix2ang for the NESTED scheme. Standard algorithm; face lookup tables from
// the HEALPix reference implementation.
const JRLL = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
const JPLL = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7];
function pix2angNest(nside, ipix) {
  const npface = nside * nside, nl4 = 4 * nside;
  const face = Math.floor(ipix / npface);
  let v = ipix % npface, ix = 0, iy = 0, s = 0;
  while (v) { ix |= (v & 1) << s; v >>>= 1; iy |= (v & 1) << s; v >>>= 1; s++; }
  const jr = JRLL[face] * nside - ix - iy - 1;
  let nr, z, kshift;
  if (jr < nside) { nr = jr; z = 1 - (nr * nr) / (3 * nside * nside); kshift = 0; }
  else if (jr > 3 * nside) { nr = nl4 - jr; z = (nr * nr) / (3 * nside * nside) - 1; kshift = 0; }
  else { nr = nside; z = (2 * nside - jr) * 2 / (3 * nside); kshift = (jr - nside) & 1; }
  let jp = Math.floor((JPLL[face] * nr + ix - iy + 1 + kshift) / 2);
  if (jp > nl4) jp -= nl4;
  if (jp < 1) jp += nl4;
  return { theta: Math.acos(Math.max(-1, Math.min(1, z))), phi: (jp - (kshift + 1) * 0.5) * (Math.PI / 2 / nr) };
}

// ------------------------------------------------- galactic -> equatorial
const D = Math.PI / 180;
const unit = (ra, dec) => [Math.cos(dec) * Math.cos(ra), Math.cos(dec) * Math.sin(ra), Math.sin(dec)];
const EX = unit(266.40510 * D, -28.936175 * D);
const EZ = unit(192.85948 * D, 27.12825 * D);
const EY = [EZ[1] * EX[2] - EZ[2] * EX[1], EZ[2] * EX[0] - EZ[0] * EX[2], EZ[0] * EX[1] - EZ[1] * EX[0]];
const gal2eq = (x, y, z) => [
  x * EX[0] + y * EY[0] + z * EZ[0],
  x * EX[1] + y * EY[1] + z * EZ[1],
  x * EX[2] + y * EY[2] + z * EZ[2],
];

// ----------------------------------------------------------------- read FITS
const buf = fs.readFileSync(path.join(rawDir, 'wmap_ilc.fits'));
// primary header (1 block) + bintable header (1 block); rows are 8 bytes,
// big-endian float32 TEMPERATURE then N_OBS
const DATA_OFF = 5760, NROW = 3145728, ROWB = 8;
if (buf.length < DATA_OFF + NROW * ROWB) { console.error('unexpected FITS size'); process.exit(1); }

const NSIDE_IN = 512, NSIDE = 256;          // 4:1 downsample, NESTED makes this trivial
const NPIX = 12 * NSIDE * NSIDE;
const temp = new Float64Array(NPIX);
for (let i = 0; i < NPIX; i++) {
  let s = 0;
  for (let k = 0; k < 4; k++) s += buf.readFloatBE(DATA_OFF + (i * 4 + k) * ROWB);
  temp[i] = s / 4;                          // mK
}

// ------------------------------------------------------------------ validate
let mean = 0;
for (let i = 0; i < NPIX; i++) mean += temp[i];
mean /= NPIX;
let rms = 0, tmin = Infinity, tmax = -Infinity;
for (let i = 0; i < NPIX; i++) { const d = temp[i] - mean; rms += d * d; if (temp[i] < tmin) tmin = temp[i]; if (temp[i] > tmax) tmax = temp[i]; }
rms = Math.sqrt(rms / NPIX);
console.log('pixels        :', NPIX.toLocaleString(), '(Nside=' + NSIDE + ')');
console.log('mean          :', (mean * 1000).toFixed(2), 'uK   (monopole is removed, so ~0 expected)');
console.log('RMS anisotropy:', (rms * 1000).toFixed(1), 'uK   (CMB anisotropy is ~100 uK)');
console.log('range         :', (tmin * 1000).toFixed(0), 'to', (tmax * 1000).toFixed(0), 'uK');

// ------------------------------------------------------------------- build
// Divergent colour scale: cold spots blue, hot spots red, clipped at 3 sigma so
// the familiar CMB mottling is visible rather than washed out by rare extremes.
const CLIP = 3 * rms;
function tColor(dT) {
  const t = Math.max(-1, Math.min(1, dT / CLIP));
  if (t < 0) { const u = -t; return [0.16 + 0.50 * (1 - u), 0.30 + 0.55 * (1 - u), 0.75 + 0.25 * (1 - u)]; }
  return [0.66 + 0.34 * t, 0.85 - 0.55 * t, 0.99 - 0.80 * t];
}

const R_PC = D_LSS * 1e6;
const pos = new Float32Array(NPIX * 3);
const col = new Uint8Array(NPIX * 4);
// HEALPix pixel centres form a regular lattice, which beats against the screen
// grid and produces concentric moire rings that look like data but are not.
// Jitter each point within its own pixel to break the lattice; the pixel is
// ~13.7 arcmin at Nside=256, far below the map's 1-degree smoothing, so this
// changes nothing physical.
const PIXSCALE = Math.sqrt(4 * Math.PI / NPIX);
for (let i = 0; i < NPIX; i++) {
  const a = pix2angNest(NSIDE, i);
  let theta = a.theta + (Math.random() - 0.5) * PIXSCALE;
  theta = Math.max(1e-6, Math.min(Math.PI - 1e-6, theta));
  const phi = a.phi + (Math.random() - 0.5) * PIXSCALE / Math.sin(theta);
  // HEALPix theta is colatitude; WMAP maps are in galactic coordinates
  const b = Math.PI / 2 - theta, l = phi;
  const g = [Math.cos(b) * Math.cos(l), Math.cos(b) * Math.sin(l), Math.sin(b)];
  const e = gal2eq(g[0], g[1], g[2]);
  pos[i * 3] = e[0] * R_PC; pos[i * 3 + 1] = e[1] * R_PC; pos[i * 3 + 2] = e[2] * R_PC;
  const c = tColor(temp[i] - mean);
  col[i * 4] = Math.round(255 * c[0]);
  col[i * 4 + 1] = Math.round(255 * c[1]);
  col[i * 4 + 2] = Math.round(255 * c[2]);
  col[i * 4 + 3] = 150;                     // large, so the shell reads as a surface
}

fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, 'cmb_pos.bin'), Buffer.from(pos.buffer));
fs.writeFileSync(path.join(outDir, 'cmb_col.bin'), Buffer.from(col.buffer));
fs.writeFileSync(path.join(outDir, 'cmb.json'), JSON.stringify({
  count: NPIX,
  nside: NSIDE,
  redshift: Z_LSS,
  comovingMpc: Math.round(D_LSS),
  rmsMicroK: +(rms * 1000).toFixed(1),
  source: 'WMAP 9-year Internal Linear Combination map (NASA LAMBDA), HEALPix Nside=512 -> 256',
  note: 'a surface, not a catalogue: the shell where the universe became transparent',
  frame: 'equatorial ICRS cartesian, parsecs, Sun at origin',
  retrieved: new Date().toISOString().slice(0, 10),
}, null, 2));

console.log('radius        :', (R_PC / 1e6 / 1000).toFixed(2), 'Gpc');
console.log('on disk       :', ((pos.byteLength + col.byteLength) / 1e6).toFixed(1), 'MB');
