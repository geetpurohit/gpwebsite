#!/usr/bin/env node
// Audit every object the sync should have uploaded, and optionally re-upload the
// ones that are missing or truncated.
//
// WHY THIS EXISTS: sync-r2.mjs reports failures only to stdout. If that output is
// lost — piped through `tail`, scrollback truncated, terminal closed — you have no
// idea which of several thousand files did not make it, and a missing octree node
// is invisible until someone happens to zoom into that corner of the sky. This
// reconstructs the truth from the bucket itself rather than from a log.
//
// Compares local size against content-length, so a truncated upload is caught too,
// not just an absent one.
//
// Usage:
//   node scripts/audit-r2.mjs <baseUrl> [--fix <bucket>] [--concurrency 24] [--prefix v1]

import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const run = promisify(execFile);
const args = process.argv.slice(2);
const base = (args.find((a) => !a.startsWith('--')) || '').replace(/\/+$/, '');
const flag = (n, d) => { const i = args.indexOf('--' + n); return i >= 0 ? args[i + 1] : d; };
if (!base) {
  console.error('usage: node scripts/audit-r2.mjs <baseUrl> [--fix <bucket>] [--concurrency 24]');
  process.exit(1);
}
const FIX = args.includes('--fix') ? flag('fix') : null;
const CONC = +flag('concurrency', 24);
const PREFIX = flag('prefix', 'v1');

// The base is the BUCKET ROOT; the prefix is added per key. Passing the URL that
// DATA_BASE uses (which already ends in /v1) silently builds /v1/v1/... and every
// single object reports 404 — an alarming and completely wrong result. Catch it.
if (new RegExp(`/${PREFIX}/?$`).test(base)) {
  console.error(`base already ends in /${PREFIX}, and the prefix is added per key.`);
  console.error(`That would check /${PREFIX}/${PREFIX}/... and report everything missing.`);
  console.error(`Pass the bucket root instead: ${base.replace(new RegExp(`/${PREFIX}/?$`), '')}`);
  process.exit(2);
}

const ROOT = 'public/universe';
const SKIP = new Set(['gaia_pos.bin', 'gaia_col.bin']);
const IMMUTABLE = 'public, max-age=31536000, immutable';
const MUTABLE = 'public, max-age=300';
const ctFor = (f) => (f.endsWith('.json') ? 'application/json' : 'application/octet-stream');
const ccFor = (f) => (f.endsWith('.json') ? MUTABLE : IMMUTABLE);

function walk(dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const fp = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(fp));
    else if (!SKIP.has(e.name)) out.push(fp);
  }
  return out;
}

const files = walk(ROOT);
const keyOf = (f) => `${PREFIX}/${path.relative(ROOT, f).split(path.sep).join('/')}`;
console.log(`auditing ${files.length.toLocaleString()} objects against ${base}`);

const missing = [], truncated = [];
let checked = 0;

// Every probe carries a unique query string. Without it an audit run DURING an
// upload is actively harmful: the edge caches each 404 it serves with a 4-hour
// max-age, so keys probed a moment too early keep returning 404 long after the
// object lands, and a later audit reads back its own poison and reports files
// missing that are demonstrably present. The query string is not part of the R2
// object key, so this asks the bucket rather than the cache.
const bust = () => `?audit=${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;

async function check(f) {
  const key = keyOf(f);
  const localSize = fs.statSync(f).size;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const r = await fetch(`${base}/${key}${bust()}`, { method: 'HEAD' });
      if (r.status === 404) { missing.push(f); break; }
      if (!r.ok) { if (attempt === 3) missing.push(f); continue; }
      // A short object is worse than an absent one: it parses as valid and yields
      // silently wrong geometry rather than a clean failure. But an ABSENT
      // content-length is not evidence of that — the edge omits it whenever the
      // response is compressed or revalidated, which made every .json look
      // truncated on the first run when they were all byte-perfect. Only a
      // present, non-zero, mismatched length counts.
      const raw = r.headers.get('content-length');
      const len = raw == null ? null : +raw;
      if (len !== null && len > 0 && len !== localSize) truncated.push({ f, len, localSize });
      break;
    } catch {
      if (attempt === 3) missing.push(f);
      await new Promise((r2) => setTimeout(r2, 400 * attempt));
    }
  }
  checked++;
  if (checked % 250 === 0) process.stdout.write(`\r  checked ${checked}/${files.length}   `);
}

{
  const queue = [...files];
  await Promise.all(Array.from({ length: CONC }, async () => {
    while (queue.length) await check(queue.pop());
  }));
}
console.log(`\r  checked ${checked}/${files.length}   `);

const broken = [...missing, ...truncated.map((t) => t.f)];
console.log(`\nmissing   : ${missing.length}`);
console.log(`truncated : ${truncated.length}`);
for (const t of truncated.slice(0, 10)) {
  console.log(`   ${keyOf(t.f)}  remote ${t.len} vs local ${t.localSize}`);
}

if (!broken.length) { console.log('\nbucket matches local exactly.'); process.exit(0); }

if (!FIX) {
  console.log(`\n${broken.length} object(s) need re-uploading. Re-run with: --fix <bucket>`);
  fs.writeFileSync('r2-missing.txt', broken.join('\n'));
  console.log('list written to r2-missing.txt');
  process.exit(1);
}

// ---- re-upload only what is broken ----
const WRANGLER = 'node_modules/wrangler/bin/wrangler.js';
console.log(`\nre-uploading ${broken.length} object(s) to ${FIX}...`);
let done = 0, failed = 0;
const stillBad = [];
async function upload(f, attempt = 1) {
  const key = keyOf(f);
  try {
    await run(process.execPath, [
      WRANGLER, 'r2', 'object', 'put', `${FIX}/${key}`,
      '--file', f, '--remote', '-y',
      '--content-type', ctFor(f), '--cache-control', ccFor(f),
    ], { maxBuffer: 1 << 24 });
    done++;
  } catch (e) {
    if (attempt < 5) {
      await new Promise((r) => setTimeout(r, 1000 * attempt * attempt));
      return upload(f, attempt + 1);
    }
    failed++; stillBad.push(key);
    console.error(`\nFAILED ${key}: ${String(e.stderr || e.message).slice(0, 160)}`);
  }
  if ((done + failed) % 10 === 0) process.stdout.write(`\r  ${done + failed}/${broken.length}   `);
}
{
  // Deliberately gentler than the main sync: every wrangler call is a fresh Node
  // process, and the spawn storm is what pegs the CPU, not the bytes.
  const queue = [...broken];
  await Promise.all(Array.from({ length: Math.min(CONC, 6) }, async () => {
    while (queue.length) await upload(queue.pop());
  }));
}
console.log(`\n\nre-uploaded ${done}, still failing ${failed}`);
if (stillBad.length) {
  fs.writeFileSync('r2-missing.txt', stillBad.join('\n'));
  console.log('remaining failures written to r2-missing.txt');
}
process.exit(failed ? 1 : 0);
