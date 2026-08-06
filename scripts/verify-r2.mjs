#!/usr/bin/env node
// Check a public R2 base URL is actually serving the universe data correctly
// BEFORE you stop tracking public/universe in git. Getting this order wrong
// means the deployed page shows "Could not load the map data".
//
// Usage: node scripts/verify-r2.mjs https://universe-data.example.com/v1

const base = (process.argv[2] || '').replace(/\/+$/, '');
if (!base) { console.error('usage: node scripts/verify-r2.mjs <publicBaseUrl>'); process.exit(1); }

// one of each kind: manifest, packed binary, octree index, octree node
const CHECKS = [
  { path: '/meta.json', kind: 'manifest', json: true },
  { path: '/sso_meta.json', kind: 'manifest', json: true },
  { path: '/quasars.json', kind: 'manifest', json: true },
  { path: '/oct/stars/index.json', kind: 'octree index', json: true },
  { path: '/oct/gal/index.json', kind: 'octree index', json: true },
  { path: '/oct/stars/r.bin', kind: 'octree node' },
  { path: '/sso.bin', kind: 'binary' },
  { path: '/cmb_pos.bin', kind: 'binary' },
  { path: '/stream_pos.bin', kind: 'binary' },
  { path: '/globulars.json', kind: 'manifest', json: true },
];

let bad = 0;
console.log('base:', base, '\n');
for (const c of CHECKS) {
  const url = base + c.path;
  try {
    // Send an Origin header: R2 (like S3) only emits Access-Control-Allow-Origin
    // when the request is actually cross-origin. Without it the CORS check is a
    // false negative — the rule can be perfectly correct and still look missing.
    const ORIGIN = process.env.VERIFY_ORIGIN || 'https://geetpurohit.com';
    const hdrs = { origin: ORIGIN };
    // HEAD first so we do not pull 20 MB just to check headers
    let r = await fetch(url, { method: 'HEAD', headers: hdrs });
    if (!r.ok) r = await fetch(url, { headers: { ...hdrs, range: 'bytes=0-1023' } });
    const ok = r.ok || r.status === 206;
    const len = +(r.headers.get('content-length') || 0);
    const cc = r.headers.get('cache-control') || '(none)';
    const ct = r.headers.get('content-type') || '(none)';
    const cors = r.headers.get('access-control-allow-origin') || '(none)';
    if (!ok) { bad++; console.log(`FAIL ${r.status}  ${c.path}`); continue; }

    const warn = [];
    if (c.json && !/json/i.test(ct)) warn.push('content-type=' + ct);
    if (!c.json && /json/i.test(ct)) warn.push('content-type=' + ct);
    if (!c.json && !/immutable/.test(cc)) warn.push('cache-control=' + cc);
    if (cors === '(none)') warn.push('no CORS header');
    console.log(`ok   ${String(r.status).padEnd(3)} ${(len / 1e6).toFixed(2).padStart(7)} MB  ${c.path}` +
      (warn.length ? '   <-- ' + warn.join(', ') : ''));
    if (warn.length) bad++;
  } catch (e) {
    bad++;
    console.log(`FAIL      ${c.path}   ${String(e.message).slice(0, 70)}`);
  }
}

console.log();
if (bad) {
  console.log(`${bad} problem(s). Do NOT gitignore public/universe yet.`);
  console.log('Common causes: custom domain not bound, CORS rule missing, wrong --prefix.');
  process.exit(1);
}
console.log('All good. Safe to point DATA_BASE here and stop tracking public/universe.');
