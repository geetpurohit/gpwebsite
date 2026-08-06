#!/usr/bin/env node
// Stellar streams: the shredded remains of globular clusters and dwarf galaxies
// being pulled apart by the Milky Way. They are the most striking real structure
// in the halo and the clearest evidence that the halo was assembled from debris.
//
// Source: galstreams (Mateu 2023), a compilation of published stream tracks.
// Each track is a polyline of (ra, dec, distance) along the stream's path.
//
// Usage: node scripts/build-streams.mjs <tracksDir> <outDir>

import fs from 'node:fs';
import path from 'node:path';

const tracksDir = process.argv[2], outDir = process.argv[3];
if (!tracksDir || !outDir) { console.error('usage: build-streams.mjs <tracksDir> <outDir>'); process.exit(1); }

const D = Math.PI / 180;
const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5;

const files = fs.readdirSync(tracksDir).filter((f) => f.endsWith('.ecsv') && !f.includes('summary'));
// track.<kind>.<Name>.<ref>.ecsv — keep one track per stream name
const best = new Map();
for (const f of files) {
  const parts = f.split('.');
  const name = parts.length > 2 ? parts[2] : f;
  if (!best.has(name)) best.set(name, f);
}

const pos = [], col = [];
let nStream = 0, nPts = 0, dmin = Infinity, dmax = 0;
const labels = [];
const FAMOUS = /^(Sgr|GD-1|Pal5|Orphan|Jhelum|Helmi|Fimbulthul|Phoenix|Tucana|ATLAS)/i;

for (const [name, file] of best) {
  const txt = fs.readFileSync(path.join(tracksDir, file), 'latin1').split('\n');
  let hdr = null; const rows = [];
  for (const L of txt) {
    if (!L || L[0] === '#') continue;
    if (!hdr) { hdr = L.trim().split(','); continue; }
    const v = L.trim().split(',');
    if (v.length < hdr.length) continue;
    rows.push(v);
  }
  if (!hdr || rows.length < 2) continue;
  const iRa = hdr.indexOf('ra'), iDe = hdr.indexOf('dec'), iD = hdr.indexOf('distance');
  if (iRa < 0 || iDe < 0 || iD < 0) continue;

  // sample along the polyline, giving the stream a physical width
  const track = [];
  for (const v of rows) {
    const ra = +v[iRa], de = +v[iDe], dkpc = +v[iD];
    if (![ra, de, dkpc].every(Number.isFinite) || !(dkpc > 0.1)) continue;
    const dPc = dkpc * 1000;
    track.push([
      dPc * Math.cos(de * D) * Math.cos(ra * D),
      dPc * Math.cos(de * D) * Math.sin(ra * D),
      dPc * Math.sin(de * D),
    ]);
    if (dPc < dmin) dmin = dPc;
    if (dPc > dmax) dmax = dPc;
  }
  if (track.length < 2) continue;

  // Published tracks carry thousands of nodes — far finer than anything visible
  // here. Thin to ~110 nodes, then give the stream a physical width so it reads
  // as a ribbon rather than a wire.
  const MAX_NODES = 110;
  const step = Math.max(1, Math.ceil(track.length / MAX_NODES));
  const thin = track.filter((_, i) => i % step === 0);
  const PER_SEG = 13, WIDTH = 170;
  for (let i = 0; i < thin.length - 1; i++) {
    const a = thin[i], b = thin[i + 1];
    for (let k = 0; k < PER_SEG; k++) {
      const t = k / PER_SEG;
      const x = a[0] + (b[0] - a[0]) * t + gauss() * WIDTH;
      const y = a[1] + (b[1] - a[1]) * t + gauss() * WIDTH;
      const z = a[2] + (b[2] - a[2]) * t + gauss() * WIDTH;
      pos.push(x, y, z);
      // blue horizontal-branch stars are the classic stream tracer, and are
      // genuinely blue-white — which also separates them from the RR Lyrae halo
      col.push(168 + Math.round(Math.random() * 34), 216 + Math.round(Math.random() * 24), 255, 44);
      nPts++;
    }
  }
  if (FAMOUS.test(name)) {
    const m = track[Math.floor(track.length / 2)];
    labels.push({ name: name.replace(/_/g, ' '), x: m[0], y: m[1], z: m[2] });
  }
  nStream++;
}

fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, 'stream_pos.bin'), Buffer.from(new Float32Array(pos).buffer));
fs.writeFileSync(path.join(outDir, 'stream_col.bin'), Buffer.from(new Uint8Array(col).buffer));
fs.writeFileSync(path.join(outDir, 'streams.json'), JSON.stringify({
  streams: nStream, points: nPts, labels,
  source: 'galstreams (Mateu 2023) — compilation of published stellar stream tracks',
  frame: 'equatorial ICRS cartesian, parsecs, Sun at origin',
  retrieved: new Date().toISOString().slice(0, 10),
}, null, 2));

console.log('streams :', nStream, '| points:', nPts.toLocaleString(),
  '|', ((pos.length * 4 + col.length) / 1e6).toFixed(1), 'MB');
console.log('distance:', (dmin / 1000).toFixed(1), '-', (dmax / 1000).toFixed(0), 'kpc | labels:', labels.length);
