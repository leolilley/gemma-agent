#!/usr/bin/env python3
"""Two-lane task soundness table. Joins the measurement notebook's Docker-lane audit (reference/mbyt-soundness/
task_soundness.csv) with our Kaggle-lane audit (results/kaggle/lane-audit-<n>/lane_audit.jsonl, the harness's own
verify_task with empty and gold controls under sandbox='subprocess'). Writes docs/results/lane-soundness.csv and prints
per-repo counts. A task enters the GPU grid only if it is sound on both lanes. Stdlib only, deterministic.

Usage: scripts/lane-soundness.py results/kaggle/lane-audit-2/lane_audit.jsonl"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
audit = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results/kaggle/lane-audit-1/lane_audit.jsonl'
docker = {r['task_id']: r for r in csv.DictReader(open(ROOT / 'reference/mbyt-soundness/task_soundness.csv'))}
kaggle = {}
for line in open(audit):
    if line.strip():
        r = json.loads(line); kaggle[r['task_id']] = r

def imp(r):
    f = r['import'].get('file', '')
    return 'workspace' if '/workspace/' in f else 'image' if 'dist-packages' in f else 'error'

rows = []
for tid in sorted(set(docker) | set(kaggle), key=lambda t: (t.split('_')[0], int(t.split('_')[1]))):
    d, k = docker.get(tid), kaggle.get(tid)
    rows.append({'task_id': tid, 'repo': (d or k)['repo'], 'docker_sound': d['sound'] if d else '', 'docker_status': d['status'] if d else '',
                 'kaggle_sound': int(k['sound']) if k else '', 'kaggle_import': imp(k) if k else '',
                 'kaggle_empty_resolved': k['empty']['resolved'] if k else '', 'kaggle_gold_resolved': k['gold']['resolved'] if k else '',
                 'kaggle_empty_error': (k['empty']['error'] or '') if k else '', 'kaggle_gold_error': (k['gold']['error'] or '') if k else '',
                 'both_sound': int(bool(d and k and d['sound'] == '1' and k['sound']))})
out = ROOT / 'docs/results/lane-soundness.csv'; out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
by = defaultdict(Counter)
for r in rows:
    c = by[r['repo']]; c['tasks'] += 1
    c['docker_sound'] += r['docker_sound'] == '1'; c['kaggle_audited'] += r['kaggle_sound'] != ''; c['kaggle_sound'] += r['kaggle_sound'] == 1; c['both'] += r['both_sound']
print(f'{"repo":<18} {"tasks":>5} {"docker":>7} {"kaggle audited":>14} {"kaggle sound":>12} {"both":>5}')
for repo, c in sorted(by.items()): print(f'{repo:<18} {c["tasks"]:>5} {c["docker_sound"]:>7} {c["kaggle_audited"]:>14} {c["kaggle_sound"]:>12} {c["both"]:>5}')
both = [r['task_id'] for r in rows if r['both_sound']]
print(f'\nsound on both lanes: {len(both)}'); print('wrote', out)
json.dump(both, open(ROOT / 'docs/results/gpu-lane-tasks.json', 'w'), indent=0)
