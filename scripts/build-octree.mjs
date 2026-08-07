#!/usr/bin/env node
// Build a streaming LOD octree from a packed point layer.
//
// This is the piece that decouples catalogue size from what the GPU holds. A
// browser can keep maybe 10-40M points resident; a catalogue can be billions.
// An octree bridges that: you never need more points than you have pixels, so
// each node stores a representative sample of everything beneath it and the
// renderer descends only where a node is big on screen.
//
// Scheme is ADDITIVE (as in Potree): a node holds the N most important points in
// its box, and its children hold the rest. Drawing a node together with all of
// its loaded ancestors reconstructs the full set with no duplication on disk.
//
// Importance = the packed size byte, which encodes brightness. For an
// astronomical map that is exactly the right sampling rule: when you cannot draw
// everything, draw the brightest, because that is what an observer would see.
// Uniform random subsampling would drop the few objects that dominate the view.
//
// Usage: node scripts/build-octree.mjs <posBin> <colBin> <outDir> [maxPerNode]

import fs from 'node:fs';
import path from 'node:path';

const [, , posPath, colPath, outDir, maxArg] = process.argv;
if (!posPath || !colPath || !outDir) {
  console.error('usage: node scripts/build-octree.mjs <posBin> <colBin> <outDir> [maxPerNode]');
  process.exit(1);
}
const MAX_PER_NODE = +(maxArg || 50000);
const MAX_DEPTH = 12;

const posBuf = fs.readFileSync(posPath);
const colBuf = fs.readFileSync(colPath);
const POS = new Float32Array(posBuf.buffer, posBuf.byteOffset, posBuf.length / 4);
const COL = new Uint8Array(colBuf.buffer, colBuf.byteOffset, colBuf.length);
const N = POS.length / 3;
if (COL.length / 4 !== N) {
  console.error(`point count mismatch: pos has ${N}, col has ${COL.length / 4}`);
  process.exit(1);
}
console.log('points:', N.toLocaleString());

// ------------------------------------------------------------------- bounds
let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
for (let i = 0; i < N; i++) {
  for (let k = 0; k < 3; k++) {
    const v = POS[i * 3 + k];
    if (v < lo[k]) lo[k] = v;
    if (v > hi[k]) hi[k] = v;
  }
}
// cube the root so subdivision stays isotropic
const centre = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
const half = Math.max(...[0, 1, 2].map((k) => (hi[k] - lo[k]) / 2)) * 1.001 || 1;
console.log('root centre:', centre.map((v) => v.toFixed(1)).join(', '), '| half-size:', half.toFixed(1));

fs.mkdirSync(outDir, { recursive: true });

const nodes = [];
let written = 0, totalPts = 0;

// Everything below stays in Int32Array index buffers and never builds a JS number
// array. At Gaia scale (~98M points) `Array.from` on the root alone would cost the
// best part of a gigabyte, and a comparator sort would be ~2.6 billion JS calls.
// Importance is a single byte, so a counting sort is both exact and O(n).
function sortByBrightnessDesc(idx) {
  const n = idx.length;
  const counts = new Int32Array(256);
  for (let i = 0; i < n; i++) counts[COL[idx[i] * 4 + 3]]++;
  // brightest (255) lands first
  const cur = new Int32Array(256);
  let acc = 0;
  for (let v = 255; v >= 0; v--) { cur[v] = acc; acc += counts[v]; }
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) { const s = idx[i]; out[cur[COL[s * 4 + 3]]++] = s; }
  return out;
}

// idx: Int32Array of point indices belonging to this node
function build(idx, name, c, h, depth) {
  if (idx.length === 0) return null;

  // brightest first — the size byte is the brightness proxy
  const sorted = sortByBrightnessDesc(idx);
  const keepCount = (depth >= MAX_DEPTH) ? sorted.length : Math.min(MAX_PER_NODE, sorted.length);

  // write this node's own points
  const p = new Float32Array(keepCount * 3);
  const q = new Uint8Array(keepCount * 4);
  for (let i = 0; i < keepCount; i++) {
    const s = sorted[i];
    p[i * 3] = POS[s * 3]; p[i * 3 + 1] = POS[s * 3 + 1]; p[i * 3 + 2] = POS[s * 3 + 2];
    q[i * 4] = COL[s * 4]; q[i * 4 + 1] = COL[s * 4 + 1]; q[i * 4 + 2] = COL[s * 4 + 2]; q[i * 4 + 3] = COL[s * 4 + 3];
  }
  // One file per node, not two: halves the request count, which matters both for
  // Cloudflare Pages file limits and for R2 Class B operation billing.
  // Layout: [uint32 count][Float32 xyz * count][Uint8 rgba * count]
  const head = Buffer.alloc(4);
  head.writeUInt32LE(keepCount, 0);
  fs.writeFileSync(path.join(outDir, name + '.bin'),
    Buffer.concat([head, Buffer.from(p.buffer), Buffer.from(q.buffer)]));
  written++; totalPts += keepCount;
  if (written % 250 === 0) {
    process.stdout.write(`\r  nodes ${written.toLocaleString()}  points ${totalPts.toLocaleString()}   `);
  }

  const node = { n: name, c: c.map((v) => +v.toFixed(2)), h: +h.toFixed(2), k: keepCount, d: depth, ch: [] };
  nodes.push(node);

  const restLen = sorted.length - keepCount;
  let buckets = null;
  if (restLen) {
    // partition the remainder into octants: count, allocate exactly, then fill
    const oct = new Uint8Array(restLen);
    const ocount = new Int32Array(8);
    for (let i = 0; i < restLen; i++) {
      const s = sorted[keepCount + i];
      const o = (POS[s * 3] > c[0] ? 1 : 0) | (POS[s * 3 + 1] > c[1] ? 2 : 0) | (POS[s * 3 + 2] > c[2] ? 4 : 0);
      oct[i] = o; ocount[o]++;
    }
    buckets = new Array(8);
    for (let o = 0; o < 8; o++) buckets[o] = ocount[o] ? new Int32Array(ocount[o]) : null;
    const fill = new Int32Array(8);
    for (let i = 0; i < restLen; i++) {
      const o = oct[i];
      buckets[o][fill[o]++] = sorted[keepCount + i];
    }
  }

  if (buckets) {
    const hh = h / 2;
    for (let o = 0; o < 8; o++) {
      if (!buckets[o]) continue;
      const cc = [
        c[0] + ((o & 1) ? hh : -hh),
        c[1] + ((o & 2) ? hh : -hh),
        c[2] + ((o & 4) ? hh : -hh),
      ];
      const child = build(buckets[o], name + o, cc, hh, depth + 1);
      // drop the reference as soon as the subtree is done so the deepest path,
      // not the whole level, bounds peak memory
      buckets[o] = null;
      if (child) node.ch.push(child);
    }
  }
  return node;
}

const all = new Int32Array(N);
for (let i = 0; i < N; i++) all[i] = i;
const root = build(all, 'r', centre, half, 0);

// The index is a flat tree; the loader fetches <name>.pos/.col on demand.
fs.writeFileSync(path.join(outDir, 'index.json'), JSON.stringify({
  root, nodes: written, points: totalPts, maxPerNode: MAX_PER_NODE,
  scheme: 'additive — a node plus its loaded ancestors reconstructs the full set',
  sampling: 'brightest-first by packed size byte',
}));

let maxDepth = 0;
for (const n of nodes) if (n.d > maxDepth) maxDepth = n.d;
console.log('nodes written :', written.toLocaleString());
console.log('points stored :', totalPts.toLocaleString(), totalPts === N ? '(lossless)' : `(MISMATCH, expected ${N})`);
console.log('max depth     :', maxDepth);
console.log('root holds    :', root.k.toLocaleString(), 'points =', (100 * root.k / N).toFixed(2) + '% of the catalogue');
