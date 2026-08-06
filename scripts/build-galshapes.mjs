#!/usr/bin/env node
// Render nearby galaxies as actual galaxies instead of identical spheres.
//
// Source: RC3 (de Vaucouleurs Third Reference Catalogue, VizieR VII/155), which
// carries for each galaxy:
//   D25  log apparent diameter        -> physical size, given a distance
//   R25  log major/minor axis ratio   -> inclination of the disc
//   PA   position angle               -> how it is oriented on the sky
//   T    de Vaucouleurs type          -> disc or spheroid, and stellar colour
//
// Each galaxy becomes a small oriented point cloud: an inclined exponential disc
// for spirals, a squashed spheroid for ellipticals, both aligned to the measured
// position angle. So M31 appears edge-on tilted the way it actually is, and an
// elliptical looks like an elliptical.
//
// Usage: node scripts/build-galshapes.mjs <rawDir> <outDir>

import fs from 'node:fs';
import path from 'node:path';

const rawDir = process.argv[2], outDir = process.argv[3];
if (!rawDir || !outDir) { console.error('usage: build-galshapes.mjs <rawDir> <outDir>'); process.exit(1); }

const D = Math.PI / 180, H0 = 70;           // km/s/Mpc for the Hubble-flow distances
const num = (s) => { const v = parseFloat(s); return Number.isFinite(v) ? v : null; };
const sex2deg = (s, isRA) => {
  const m = (s || '').trim().split(/\s+/).map(parseFloat);
  if (m.length < 2 || m.some((v) => !Number.isFinite(v))) return null;
  const sign = /^-/.test(s.trim()) ? -1 : 1;
  const v = Math.abs(m[0]) + (m[1] || 0) / 60 + (m[2] || 0) / 3600;
  return sign * v * (isRA ? 15 : 1);
};

// Galaxies close enough that redshift is a bad distance indicator; these are the
// ones the Local Group view actually shows, so they get literature distances.
const NEARBY = [
  ['M31', 10.68, 41.27, 0.78], ['M33', 23.46, 30.66, 0.85],
  ['LMC', 80.00, -69.76, 0.050], ['SMC', 13.19, -72.83, 0.062],
  ['NGC6822', 296.24, -14.80, 0.50], ['IC10', 5.10, 59.29, 0.66],
  ['NGC300', 13.72, -37.68, 2.0], ['NGC253', 11.89, -25.29, 3.5],
  ['CenA', 201.36, -43.02, 3.8], ['M81', 148.89, 69.07, 3.6],
  ['M82', 148.97, 69.68, 3.5], ['M83', 204.25, -29.87, 4.9],
  ['M94', 192.72, 41.12, 4.7], ['M101', 210.80, 54.35, 6.4],
  ['M51', 202.47, 47.20, 8.6], ['M104', 190.00, -11.62, 9.6],
  ['M87', 187.71, 12.39, 16.4], ['M49', 187.44, 8.00, 16.5],
  ['M86', 186.55, 12.95, 16.0], ['M60', 190.92, 11.55, 16.5],
];

// Galaxies too large or too nearby for RC3 to describe usefully. The Magellanic
// Clouds span degrees and sit at 50-62 kpc, where redshift is meaningless; they
// are also the most visually important objects at Local Group scale, so they get
// explicit parameters. [name, RA, Dec, d(Mpc), diameter(arcmin), axisRatio, PA, T]
const EXTRA = [
  ['LMC', 80.894, -69.756, 0.0496, 645, 0.90, 170, 9.5],
  ['SMC', 13.187, -72.829, 0.0620, 320, 0.55, 45, 9.5],
];

const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5;

const lines = fs.readFileSync(path.join(rawDir, 'rc3.tsv'), 'latin1').split('\n');
// Append the Magellanic Clouds as RC3-shaped rows. RC3 does not describe them
// usefully — they span degrees — and they are the most prominent galaxies in the
// Local Group view, so they are supplied explicitly and then shaped identically.
for (const [, ra, de, dMpc, diam, q, pa, t] of EXTRA) {
  const h = ra / 15, hm = (h % 1) * 60;
  const ad = Math.abs(de), am = (ad % 1) * 60;
  const raS = `${String(Math.floor(h)).padStart(2, '0')} ${String(Math.floor(hm)).padStart(2, '0')} ${((hm % 1) * 60).toFixed(1)}`;
  const deS = `${de < 0 ? '-' : '+'}${String(Math.floor(ad)).padStart(2, '0')} ${String(Math.floor(am)).padStart(2, '0')} ${Math.round((am % 1) * 60)}`;
  lines.push([raS, deS, 'MagellanicCloud', t, Math.log10(diam / 0.1), -Math.log10(q), pa, '', '', ''].join('\t'));
  NEARBY.push(['MC', ra, de, dMpc]);
}
const pos = [], col = [];
let nGal = 0, nPts = 0, nOverride = 0;
const claimed = new Set();
const byType = { spiral: 0, lenticular: 0, elliptical: 0, irregular: 0 };

for (const L of lines) {
  if (!L || L[0] === '#') continue;
  const f = L.split('\t');
  if (f.length < 9) continue;
  const ra = sex2deg(f[0], true), de = sex2deg(f[1], false);
  if (ra === null || de === null) continue;
  const T = num(f[3]), D25 = num(f[4]), R25 = num(f[5]), PA = num(f[6]);
  const V3K = num(f[7]), cz = num(f[8]);
  if (D25 === null) continue;

  // distance: literature for the nearby set, otherwise Hubble flow
  let dMpc = null;
  for (let k = 0; k < NEARBY.length; k++) {
    if (claimed.has(k)) continue;
    const [, nra, nde, nd] = NEARBY[k];
    if (Math.abs(nra - ra) < 0.25 && Math.abs(nde - de) < 0.25) { dMpc = nd; claimed.add(k); nOverride++; break; }
  }
  if (dMpc === null) {
    const v = V3K !== null ? V3K : cz;
    if (v === null || v < 600) continue;    // peculiar motion dominates below this
    dMpc = v / H0;
  }
  if (dMpc > 140) continue;                 // beyond this they are sub-pixel anyway

  // geometry
  const diamArcmin = 0.1 * Math.pow(10, D25);
  const rPc = (dMpc * 1e6) * (diamArcmin / 2 / 60) * D;   // semi-major axis, parsecs
  if (!(rPc > 0) || rPc > 2e5) continue;
  const q = R25 !== null ? Math.max(0.12, Math.min(1, Math.pow(10, -R25))) : 0.7;
  const pa = (PA !== null ? PA : Math.random() * 180) * D;
  const t = T !== null ? T : 3;

  // sky frame at this direction: line of sight, north, east
  const cd = Math.cos(de * D), sd = Math.sin(de * D), cr = Math.cos(ra * D), sr = Math.sin(ra * D);
  const R = [cd * cr, cd * sr, sd];                       // outward
  const N = [-sd * cr, -sd * sr, cd];                     // toward +Dec
  const E = [-sr, cr, 0];                                 // toward +RA
  // PA measured from north toward east
  const A = [0, 1, 2].map((k) => Math.cos(pa) * N[k] + Math.sin(pa) * E[k]);   // major axis
  const B = [0, 1, 2].map((k) => -Math.sin(pa) * N[k] + Math.cos(pa) * E[k]);  // minor axis

  // How many points this galaxy is worth. The cap is high because the camera can
  // now fly to a galaxy and orbit it, so the nearest few need real detail.
  // Strongly superlinear in apparent size: the handful of galaxies you can
  // actually fly to need enough points to read as continuous surface brightness
  // (~5-10 per pixel), while the thousands of small ones stay cheap. The layer is
  // streamed through the LOD octree, so the total never has to be resident.
  const nP = Math.max(30, Math.min(350000, Math.round(1.5 * Math.pow(diamArcmin, 2.4))));

  const isE = t <= -3.5, isS0 = t > -3.5 && t < 0.5, isIrr = t >= 9.5;
  const NARM = Math.random() < 0.68 ? 2 : 4;
  if (isE) byType.elliptical++; else if (isS0) byType.lenticular++;
  else if (isIrr) byType.irregular++; else byType.spiral++;

  // inclination that reproduces the observed axis ratio
  const inc = Math.acos(Math.max(0.05, Math.min(1, q)));
  const ci = Math.cos(inc), si = Math.sin(inc);

  for (let i = 0; i < nP; i++) {
    let u = 0, v = 0, w = 0;
    if (isE || isS0) {
      // spheroid, centrally concentrated
      const rr = Math.pow(Math.random(), isE ? 1.7 : 1.4);
      const th = Math.random() * Math.PI * 2, cph = 2 * Math.random() - 1;
      const sph = Math.sqrt(1 - cph * cph);
      u = rr * sph * Math.cos(th); v = rr * sph * Math.sin(th) * q; w = rr * cph * q;
    } else if (isIrr) {
      // irregulars have no ordered structure: clumpy star formation, puffy
      const rr = Math.min(1, -Math.log(1 - Math.random() * 0.98) / 2.6);
      const th = Math.random() * Math.PI * 2;
      u = rr * Math.cos(th) + gauss() * 0.30;
      v = rr * Math.sin(th) + gauss() * 0.30;
      w = gauss() * 0.22;
    } else {
      // Spiral. Hubble type sets the winding and how much of the light is bulge:
      // Sa is tightly wound with a big bulge, Sc is open with almost none.
      const wind = Math.max(1.05, 2.85 - 0.19 * t);
      const bulgeFrac = Math.max(0.03, Math.min(0.45, 0.42 - 0.045 * t));

      if (Math.random() < bulgeFrac) {
        // central bulge: old, concentrated, rounder than the disc
        const rr = Math.pow(Math.random(), 2.1) * 0.30;
        const th = Math.random() * Math.PI * 2, cph = 2 * Math.random() - 1;
        const sph = Math.sqrt(1 - cph * cph);
        u = rr * sph * Math.cos(th); v = rr * sph * Math.sin(th); w = rr * cph * 0.62;
      } else {
        const rr = Math.min(1, -Math.log(1 - Math.random() * 0.985) / 2.9);
        // snap most of the disc onto logarithmic arms; the rest is inter-arm field
        const onArm = Math.random() < 0.66;
        let th;
        if (onArm) {
          const arm = Math.floor(Math.random() * NARM) * (Math.PI * 2 / NARM);
          // arms broaden outward, as real ones do
          th = arm + Math.log(Math.max(rr, 0.02) / 0.02) * wind
             + gauss() * (0.10 + 0.20 * rr) * Math.PI;
        } else {
          th = Math.random() * Math.PI * 2;
        }
        // Dust lane: a narrow void on the inner edge of each arm. Additive blending
        // cannot draw absorption, so the lane is carved as an absence of stars —
        // which is what makes it read dark against the disc behind it.
        const armPhase = ((th - Math.log(Math.max(rr, 0.02) / 0.02) * wind) % (Math.PI * 2 / NARM) + Math.PI * 2) % (Math.PI * 2 / NARM);
        const laneAt = 0.20;
        if (rr > 0.14 && Math.abs(armPhase - laneAt) < 0.085 && Math.random() < 0.82) continue;

        u = rr * Math.cos(th); v = rr * Math.sin(th); w = gauss() * 0.045;
      }
    }
    // place in 3D: disc plane spans A and (ci*B + si*R)
    const x = u * A[0] + v * (ci * B[0] + si * R[0]) + w * (si * B[0] - ci * R[0]);
    const y = u * A[1] + v * (ci * B[1] + si * R[1]) + w * (si * B[1] - ci * R[1]);
    const z = u * A[2] + v * (ci * B[2] + si * R[2]) + w * (si * B[2] - ci * R[2]);
    const dPc = dMpc * 1e6;
    pos.push(R[0] * dPc + x * rPc, R[1] * dPc + y * rPc, R[2] * dPc + z * rPc);

    // stellar populations: ellipticals are old and red, spirals blue with HII
    let r, g, b;
    if (isE || isS0) { r = 255; g = 216 - Math.random() * 26; b = 165 - Math.random() * 34; }
    else {
      const rad = Math.hypot(u, v);
      if (rad < 0.22 && Math.random() < 0.7) { r = 255; g = 226; b = 176; }        // bulge
      else if (rad > 0.18 && Math.random() < 0.05 + 0.016 * Math.max(0, t)) { r = 255; g = 140; b = 190; }  // HII on the arms
      else { r = 176 + Math.random() * 40; g = 206 + Math.random() * 30; b = 255; } // disc
    }
    col.push(r, g, b, 40);
    nPts++;
  }
  nGal++;
}

fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, 'galshape_pos.bin'), Buffer.from(new Float32Array(pos).buffer));
fs.writeFileSync(path.join(outDir, 'galshape_col.bin'), Buffer.from(new Uint8Array(col).buffer));
fs.writeFileSync(path.join(outDir, 'galshapes.json'), JSON.stringify({
  galaxies: nGal, points: nPts,
  source: 'RC3 (de Vaucouleurs et al., VizieR VII/155) — D25 diameter, R25 axis ratio, PA, type T',
  distances: `literature for ${NEARBY.length} nearby galaxies, otherwise Hubble flow at H0=${H0}`,
  frame: 'equatorial ICRS cartesian, parsecs, Sun at origin',
  retrieved: new Date().toISOString().slice(0, 10),
}, null, 2));

console.log('galaxies shaped :', nGal.toLocaleString(), '| literature distances used:', nOverride);
console.log('points          :', nPts.toLocaleString(),
  '|', ((pos.length * 4 + col.length) / 1e6).toFixed(1), 'MB');
console.table(byType);
