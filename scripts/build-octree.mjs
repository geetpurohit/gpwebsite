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

// idx: Int32Array of point indices belonging to this node
function build(idx, name, c, h, depth) {
  if (idx.length === 0) return null;

  // brightest first — the size byte is the brightness proxy
  const order = Array.from(idx);
  order.sort((a, b) => COL[b * 4 + 3] - COL[a * 4 + 3]);

  const keepCount = (depth >= MAX_DEPTH) ? order.length : Math.min(MAX_PER_NODE, order.length);
  const keep = order.slice(0, keepCount);
  const rest = order.slice(keepCount);

  // write this node's own points
  const p = new Float32Array(keep.length * 3);
  const q = new Uint8Array(keep.length * 4);
  for (let i = 0; i < keep.length; i++) {
    const s = keep[i];
    p[i * 3] = POS[s * 3]; p[i * 3 + 1] = POS[s * 3 + 1]; p[i * 3 + 2] = POS[s * 3 + 2];
    q[i * 4] = COL[s * 4]; q[i * 4 + 1] = COL[s * 4 + 1]; q[i * 4 + 2] = COL[s * 4 + 2]; q[i * 4 + 3] = COL[s * 4 + 3];
  }
  // One file per node, not two: halves the request count, which matters both for
  // Cloudflare Pages file limits and for R2 Class B operation billing.
  // Layout: [uint32 count][Float32 xyz * count][Uint8 rgba * count]
  const head = Buffer.alloc(4);
  head.writeUInt32LE(keep.length, 0);
  fs.writeFileSync(path.join(outDir, name + '.bin'),
    Buffer.concat([head, Buffer.from(p.buffer), Buffer.from(q.buffer)]));
  written++; totalPts += keep.length;

  const node = { n: name, c: c.map((v) => +v.toFixed(2)), h: +h.toFixed(2), k: keep.length, d: depth, ch: [] };
  nodes.push(node);

  if (rest.length) {
    // partition the remainder into octants
    const buckets = Array.from({ length: 8 }, () => []);
    for (const s of rest) {
      const o = (POS[s * 3] > c[0] ? 1 : 0) | (POS[s * 3 + 1] > c[1] ? 2 : 0) | (POS[s * 3 + 2] > c[2] ? 4 : 0);
      buckets[o].push(s);
    }
    const hh = h / 2;
    for (let o = 0; o < 8; o++) {
      if (!buckets[o].length) continue;
      const cc = [
        c[0] + ((o & 1) ? hh : -hh),
        c[1] + ((o & 2) ? hh : -hh),
        c[2] + ((o & 4) ? hh : -hh),
      ];
      const child = build(buckets[o], name + o, cc, hh, depth + 1);
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
