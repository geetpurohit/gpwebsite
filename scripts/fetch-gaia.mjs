#!/usr/bin/env node
// Pull Gaia DR3 stars with usable parallaxes, in resumable chunks.
//
// 1.47 billion Gaia sources have a parallax, but 87% of them have S/N < 5, where the
// inferred distance is dominated by the prior rather than the measurement. Those are
// sky positions, not distances, so they are deliberately excluded. The default cut
// (parallax_over_error > 10) leaves ~98M stars whose distances are real.
//
// ESA's TAP caps how much one job may return, so the sky is split by source_id range
// — which is a HEALPix ordering, so each chunk is a contiguous patch of sky. Chunks
// are written individually and skipped if present, so this can be killed and rerun.
//
// Usage: node scripts/fetch-gaia.mjs <outDir> [--chunks 120] [--snr 10]
//
// Expect hours. Leave it running; then build with scripts/build-gaia.mjs.

import fs from 'node:fs';
import path from 'node:path';

const outDir = process.argv[2];
if (!outDir) { console.error('usage: node scripts/fetch-gaia.mjs <outDir> [--chunks N] [--snr X]'); process.exit(1); }
const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i >= 0 ? +process.argv[i + 1] : d; };
const CHUNKS = arg('chunks', 120);
const SNR = arg('snr', 10);

const TAP = 'https://gea.esac.esa.int/tap-server/tap';
const MAX_SOURCE_ID = 6917529027641081856n;   // 2^63 - 2^59, above every real source_id
fs.mkdirSync(outDir, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function runChunk(i) {
  const dest = path.join(outDir, `gaia_${String(i).padStart(4, '0')}.csv`);
  if (fs.existsSync(dest) && fs.statSync(dest).size > 200) return 'skip';

  const lo = (MAX_SOURCE_ID * BigInt(i)) / BigInt(CHUNKS);
  const hi = (MAX_SOURCE_ID * BigInt(i + 1)) / BigInt(CHUNKS);
  const adql = `SELECT source_id, ra, dec, parallax, parallax_over_error, phot_g_mean_mag, bp_rp
    FROM gaiadr3.gaia_source
    WHERE parallax_over_error > ${SNR}
      AND source_id >= ${lo} AND source_id < ${hi}`;

  // submit
  const body = new URLSearchParams({
    REQUEST: 'doQuery', LANG: 'ADQL', FORMAT: 'csv', PHASE: 'RUN', QUERY: adql,
  });
  const sub = await fetch(TAP + '/async', { method: 'POST', body });
  if (!sub.ok) throw new Error('submit ' + sub.status);
  const jobUrl = sub.url && sub.url.includes('/async/') ? sub.url : sub.headers.get('location');
  if (!jobUrl) throw new Error('no job url');

  // poll
  for (let t = 0; t < 900; t++) {
    await sleep(2000);
    const ph = await (await fetch(jobUrl + '/phase')).text();
    if (ph.trim() === 'COMPLETED') break;
    if (['ERROR', 'ABORTED'].includes(ph.trim())) throw new Error('job ' + ph.trim());
  }

  const res = await fetch(jobUrl + '/results/result');
  if (!res.ok) throw new Error('fetch result ' + res.status);
  const txt = await res.text();
  fs.writeFileSync(dest, txt);
  // best-effort cleanup so the archive does not accumulate our jobs
  fetch(jobUrl, { method: 'DELETE' }).catch(() => {});
  return (txt.split('\n').length - 2);
}

console.log(`Gaia DR3, parallax S/N > ${SNR}, ${CHUNKS} chunks -> ${outDir}`);
let total = 0, done = 0, failed = 0;
for (let i = 0; i < CHUNKS; i++) {
  try {
    const r = await runChunk(i);
    if (r === 'skip') { done++; console.log(`[${i + 1}/${CHUNKS}] already present`); continue; }
    total += r; done++;
    console.log(`[${i + 1}/${CHUNKS}] ${r.toLocaleString()} rows   (running total ${total.toLocaleString()})`);
  } catch (e) {
    failed++;
    console.log(`[${i + 1}/${CHUNKS}] FAILED: ${String(e.message).slice(0, 80)} — rerun to retry`);
    await sleep(5000);
  }
}
console.log(`\nchunks ok ${done}, failed ${failed}, rows ${total.toLocaleString()}`);
