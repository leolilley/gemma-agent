#!/usr/bin/env python3
"""Write one signed correction item for the GPU-lane records whose started_at is wrong.

Why: until 2026-10-19 the ingest tool read ZIP timestamps as local time (a 13-hour shift on this machine) and used the
results file's mtime, the moment the whole run finished, as every task's started_at. The records are named with their
digests in signed ledger entries, so they are not edited; this item states, per record, the recorded value and the
corrected one (the attempt's first model call from its trace, the same t0 scripts/curve.py uses), and scripts/verify.py
applies the correction to its chronology checks. Idempotent: records already corrected by an earlier item are skipped.

Usage: scripts/correct-started-at.py [--dry-run]      (needs the RyeOS daemon for the signing step)
"""
import argparse, json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KDIR = ROOT / '.ai/knowledge/gemma-agent'
sys.path.insert(0, str(ROOT / '.ai/tools/gemma-agent/lib'))
from ingest_run import trace_start  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
already = set()
for cp in KDIR.glob('correction-*.md'):
    for line in cp.read_text().splitlines():
        if line.startswith('| ') and not line.startswith('| attempt_id') and not line.startswith('| ---'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            if len(cells) >= 2 and cells[1] == 'started_at': already.add(cells[0])
rows = []
for p in sorted((ROOT / 'records').glob('*.json')):
    r = json.loads(p.read_text())
    if r.get('site') != 'kaggle-l4x4' or r['attempt_id'] in already or r.get('started_at_source'): continue
    fixed = trace_start(ROOT / r['results_dir'], r['task_id'])
    if fixed and fixed != r['started_at']: rows.append((r['attempt_id'], r['started_at'], fixed))
print(f'{len(rows)} GPU-lane records with a wrong started_at ({len(already)} already corrected)')
if not rows: sys.exit(0)
shifts = sorted((datetime.fromisoformat(f) - datetime.fromisoformat(o)).total_seconds() / 3600 for _, o, f in rows)
print(f'correction in hours (corrected minus recorded): min {shifts[0]:+.2f}, median {shifts[len(shifts) // 2]:+.2f}, max {shifts[-1]:+.2f}')
now = datetime.now(timezone.utc).replace(microsecond=0); name = f"correction-{now.strftime('%Y%m%dT%H%M%SZ')}-started-at"
body = [f'---\nkind: knowledge\nid: gemma-agent/{name}\nversion: "1.0.0"\n---\n', f'# Correction {now.isoformat()}: started_at of {len(rows)} GPU-lane records\n',
        'The ingest tool, until 2026-10-19 03:50 UTC, read ZIP entry times as local time (UTC+13 on the ingesting machine) and used the '
        'modification time of the run\'s results file, the moment the whole run finished, as every task\'s `started_at`. Both are wrong: '
        'the recorded values are about 13 hours early and are completion times. The corrected value is the attempt\'s first model call '
        '(the trace\'s first agent step timestamp minus its elapsed seconds, UTC), the same origin `scripts/curve.py` uses for solve times. '
        'The record files are not edited, because signed ledger entries name their digests; `scripts/verify.py` applies this item to its '
        'chronology checks. The trace timing behind every curve and prediction was never affected. Records written after the fix carry '
        '`started_at_source`.\n', '| attempt_id | field | recorded | corrected | reason |', '| --- | --- | --- | --- | --- | ']
body += [f'| {a_} | started_at | {o} | {f} | run-finished mtime read in local time; replaced by the trace\'s first model call |' for a_, o, f in rows]
out = KDIR / f'{name}.md'
if a.dry_run: print(f'dry run: would write {out.relative_to(ROOT)}'); print('\n'.join(body[-3:])); sys.exit(0)
out.write_text('\n'.join(body) + '\n')
r = subprocess.run(['ryeos', 'sign', f'knowledge:gemma-agent/{name}', '--project', str(ROOT)], capture_output=True, text=True)
try: res = json.loads(r.stdout)
except Exception: res = {}
if r.returncode != 0 or res.get('failed') or not res.get('signed'):
    out.unlink(); sys.exit(f'signing failed; item removed:\n{(r.stdout + r.stderr)[-600:]}')
print(f'wrote and signed {out.relative_to(ROOT)}: {out.read_text().splitlines()[0][:80]}...')
