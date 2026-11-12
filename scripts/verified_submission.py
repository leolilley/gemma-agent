#!/usr/bin/env python3
"""Verified submission, as stated in DECISIONS (2026-11-02, bf18-e4b-verify) before the arm ran: an attempt's last
run_command after its last edit exits 0. Edits are scripts/episodes.py edits. An attempt with no edit, or no
run_command after its last edit, is not verified. Per bundle on one run's records, and the paired task-level difference
of two bundles with the campaign's cluster bootstrap (scripts/regenerate.py).

Usage: scripts/verified_submission.py <bundle_a> <run_id_a> <bundle_b> <run_id_b>"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import curve, episodes, regenerate  # noqa: E402


def verified(record) -> int:
    calls, _ = curve.trace_calls(record, ROOT)
    edits = [c['index'] for c in calls if episodes.is_edit(c)]
    if not edits: return 0
    runs = [c for c in calls if c['index'] > edits[-1] and c['tool'] == 'run_command']
    return int(bool(runs) and not runs[-1]['malformed'] and (runs[-1]['result'] or {}).get('exit_code') == 0)


def arm(bundle, run_id):
    out = {}
    for f in sorted((ROOT / 'records').glob(f'*-{bundle}-r*-{run_id}.json')):
        r = json.loads(f.read_text())
        if r['bundle'] == bundle and r.get('validity') == 'valid': out[r['task_id']] = verified(r)
    return out


if __name__ == '__main__':
    ba, ra, bb, rb = sys.argv[1:5]
    a, b = arm(ba, ra), arm(bb, rb); shared = sorted(set(a) & set(b))
    ci = regenerate.cluster_bootstrap({t: [a[t] - b[t]] for t in shared}, lambda s: sum(x[0] for x in s) / len(s)) if shared else None
    print(json.dumps({ba: f'{sum(a.values())}/{len(a)}', bb: f'{sum(b.values())}/{len(b)}', 'shared': len(shared),
                      'a_minus_b': ci and {k: round(ci[k], 4) for k in ('point', 'lo', 'hi')}}))
