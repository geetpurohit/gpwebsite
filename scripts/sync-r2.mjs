#!/usr/bin/env node
// Upload public/universe to a Cloudflare R2 bucket, then get it out of the repo.
//
// WHY: the data is ~212 MB and growing. That does not belong in git, and it will
// not fit a Pages deploy (20,000 file limit, and this is already thousands).
// R2 gives 10 GB free with ZERO egress fees, and every file here is immutable, so
// a one-year cache header means Cloudflare's edge serves nearly all requests
// without ever touching the bucket — which also keeps you under the free
// 10M/month Class B operation limit.
//
// PREREQUISITES
//   npm i -D wrangler
//   npx wrangler login
//   npx wrangler r2 bucket create universe-data
//
// Then bind a CUSTOM DOMAIN to the bucket in the Cloudflare dashboard
// (R2 > your bucket > Settings > Public access > Custom domain).
// Do NOT use the r2.dev URL: it is rate-limited and explicitly not for production.
//
// Finally set DATA_BASE in src/pages/projects/universe.astro to that domain, e.g.
//   const DATA_BASE = 'https://universe-data.geetpurohit.com';
//
// Usage:
//   node scripts/sync-r2.mjs <bucket> [--prefix v1] [--dry] [--concurrency 8]

import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const run = promisify(execFile);
const args = process.argv.slice(2);
const bucket = args.find((a) => !a.startsWith('--'));
const flag = (n, d) => { const i = args.indexOf('--' + n); return i >= 0 ? args[i + 1] : d; };
const DRY = args.includes('--dry');
const PREFIX = flag('prefix', 'v1');
const CONC = +flag('concurrency', 8);

if (!bucket) {
  console.error('usage: node scripts/sync-r2.mjs <bucket> [--prefix v1] [--dry] [--concurrency 8]');
  process.exit(1);
}

const WRANGLER = 'node_modules/wrangler/bin/wrangler.js';
if (!fs.existsSync(WRANGLER)) { console.error('wrangler not installed: npm i -D wrangler'); process.exit(1); }

const ROOT = 'public/universe';
if (!fs.existsSync(ROOT)) { console.error('missing ' + ROOT); process.exit(1); }

// Octree nodes and packed binaries never change once written, so they can be
// cached for a year. The small JSON manifests are the only things that get
// rewritten by a rebuild, so they get a short TTL.
const IMMUTABLE = 'public, max-age=31536000, immutable';
const MUTABLE = 'public, max-age=300';
const ctFor = (f) => (f.endsWith('.json') ? 'application/json' : 'application/octet-stream');
const ccFor = (f) => (f.endsWith('.json') ? MUTABLE : IMMUTABLE);

// Build intermediates that are fully superseded by an octree and have no runtime
// fallback path. gaia_pos/gaia_col alone are 1.45 GB — seven times the entire rest
// of the bucket — and nothing ever fetches them, so uploading them would be pure
// cost. Layers that DO have a monolithic fallback in universe.astro are not listed
// here; deleting those is a separate call.
const SKIP = new Set(['gaia_pos.bin', 'gaia_col.bin']);

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
const skipped = walk.skipped = [...SKIP].filter((f) => fs.existsSync(path.join(ROOT, f)));
if (skipped.length) {
  const mb = skipped.reduce((s, f) => s + fs.statSync(path.join(ROOT, f)).size, 0) / 1e6;
  console.log(`skipping ${skipped.length} build intermediate(s), ${mb.toFixed(0)} MB: ${skipped.join(', ')}`);
}
const total = files.reduce((s, f) => s + fs.statSync(f).size, 0);
console.log(`files: ${files.length.toLocaleString()}  |  total: ${(total / 1e6).toFixed(1)} MB`);
console.log(`bucket: ${bucket}  |  prefix: ${PREFIX}  |  ${DRY ? 'DRY RUN' : 'uploading'}`);

if (DRY) {
  const byExt = {};
  for (const f of files) {
    const e = path.extname(f) || '(none)';
    byExt[e] = byExt[e] || { n: 0, b: 0 };
    byExt[e].n++; byExt[e].b += fs.statSync(f).size;
  }
  console.table(Object.fromEntries(Object.entries(byExt)
    .map(([k, v]) => [k, { files: v.n, MB: +(v.b / 1e6).toFixed(1), cacheControl: ccFor('x' + k) }])));
  console.log('\nnothing uploaded (--dry). Drop --dry to run for real.');
  process.exit(0);
}

let done = 0, failed = 0;
const failures = [];
// Uploads occasionally fail on a transient 'Failed to fetch'. Retry with backoff
// rather than making the whole 258-file run a coin flip.
async function upload(f, attempt = 1) {
  const key = `${PREFIX}/${path.relative(ROOT, f).split(path.sep).join('/')}`;
  try {
    await run(process.execPath, [
      WRANGLER, 'r2', 'object', 'put', `${bucket}/${key}`,
      '--file', f, '--remote', '-y',
      '--content-type', ctFor(f),
      '--cache-control', ccFor(f),
    ], { maxBuffer: 1 << 24 });
  } catch (e) {
    if (attempt < 4) {
      await new Promise((r) => setTimeout(r, 800 * attempt * attempt));
      return upload(f, attempt + 1);
    }
    failed++;
    failures.push({ f, key });
    console.error('\nFAILED after 3 retries:', key, String(e.stderr || e.message).slice(0, 140));
    return;
  }
  done++;
  if (done % 25 === 0 || done === files.length) {
    process.stdout.write(`\r  ${done}/${files.length} uploaded`);
  }
}

// simple worker pool
const queue = [...files];
await Promise.all(Array.from({ length: CONC }, async () => {
  while (queue.length) await upload(queue.pop());
}));

console.log(`\ndone: ${done} uploaded, ${failed} failed`);
if (failures.length) {
  console.log(`\nRetry just these:`);
  for (const { f, key } of failures) {
    console.log(`  node ${WRANGLER} r2 object put ${bucket}/${key} --file ${f} --remote -y ` +
      `--content-type ${ctFor(f)} --cache-control "${ccFor(f)}"`);
  }
}
if (!failed) {
  console.log(`\nNext:`);
  console.log(`  1. bind a custom domain to the bucket (not the r2.dev URL)`);
  console.log(`  2. add a CORS rule allowing your site origin`);
  console.log(`  3. set DATA_BASE = 'https://<your-domain>/${PREFIX}' in universe.astro`);
  console.log(`  4. git rm -r --cached public/universe && add it to .gitignore`);
}
