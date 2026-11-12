#!/usr/bin/env python3
"""The prompt-lever comparison as pre-registered in DECISIONS (2026-11-05T21:00Z): bf18-window against bf18, first attempts
at the cell (8 min, 28 calls, Kaggle L4x4), from the records and their traces. Rule: resubmit only if the arm resolves at
least bf18 + 5 of the 68. Secondary: first-edit call index, calls per attempt, cap-outs, clock-outs, greps and read windows
per attempt. Stdlib only.

Usage: scripts/window_analysis.py [--arm bf18-window] [--control bf18] [--margin 5]"""
import argparse, collections, json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def attempts(bundle):
    out = {}
    for f in sorted((ROOT / 'records').glob(f'*-{bundle}-r0-*.json')):
        r = json.loads(f.read_text())
        if r['bundle'] != bundle or r.get('validity', 'valid') != 'valid' or r.get('infrastructure_failure') or r['site'] != 'kaggle-l4x4': continue
        if r['budget'].get('max_time_minutes') != 8 or r['budget'].get('max_tool_calls') != 28 or r.get('repeat_index', 0) != 0: continue
        out.setdefault(r['task_id'], r)
    return out


def calls(r):
    tr = json.loads((ROOT / r['results_dir'] / 'traces' / f"trace_{r['task_id']}.json").read_text())
    return [(c['function_name'], c.get('arguments') or {}) for s in tr['steps'] for c in (s.get('tool_calls') or [])]


def profile(r):
    cs = calls(r); n = len(cs)
    edit = next((i + 1 for i, (f, a) in enumerate(cs) if f in ('edit_file', 'write_file')), None)
    pre = cs[:(edit or n)]
    greps = sum(1 for f, a in pre if f == 'run_command' and 'grep' in a.get('command', ''))
    reads = [a for f, a in pre if f == 'read_file']
    windows = [(a.get('end_line') or a.get('start_line')) - a['start_line'] for a in reads if a.get('start_line') is not None]
    files = collections.Counter(a.get('filepath') for a in reads)
    return {'calls': n, 'first_edit': edit, 'pre_edit_greps': greps, 'pre_edit_reads': len(reads), 'unranged_reads': sum(1 for a in reads if a.get('start_line') is None),
            'window_lines_median': statistics.median(windows) if windows else None, 'same_file_extra_reads': sum(v - 1 for v in files.values()),
            'cap': n >= 28, 'clock': r['wall_seconds'] >= 470 and n < 28, 'resolved': bool(r['outcome'].get('resolved')), 'submit': any(f == 'submit_patch' for f, a in cs)}


def summarise(ps):
    med = lambda k: statistics.median([p[k] for p in ps if p[k] is not None]) if any(p[k] is not None for p in ps) else None
    return {'n': len(ps), 'resolved': sum(p['resolved'] for p in ps), 'cap_outs': sum(p['cap'] and not p['resolved'] for p in ps), 'clock_outs': sum(p['clock'] and not p['resolved'] for p in ps),
            'no_edit': sum(p['first_edit'] is None for p in ps), 'first_edit_median': med('first_edit'), 'calls_median': med('calls'), 'greps_pre_edit_median': med('pre_edit_greps'),
            'reads_pre_edit_median': med('pre_edit_reads'), 'unranged_reads_median': med('unranged_reads'), 'window_lines_median': med('window_lines_median'), 'same_file_extra_reads_median': med('same_file_extra_reads')}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='bf18-window'); ap.add_argument('--control', default='bf18'); ap.add_argument('--margin', type=int, default=5); a = ap.parse_args()
    arm, ctl = attempts(a.arm), attempts(a.control); common = sorted(set(arm) & set(ctl))
    pa = {t: profile(arm[t]) for t in common}; pc = {t: profile(ctl[t]) for t in common}
    flips = {'arm_only': [t for t in common if pa[t]['resolved'] and not pc[t]['resolved']], 'control_only': [t for t in common if pc[t]['resolved'] and not pa[t]['resolved']]}
    rep = {'tasks': len(common), 'arm': {a.arm: summarise(list(pa.values()))}, 'control': {a.control: summarise(list(pc.values()))}, 'flips': flips,
           'rule': {'margin': a.margin, 'arm_resolved': sum(p['resolved'] for p in pa.values()), 'control_resolved': sum(p['resolved'] for p in pc.values())}}
    rep['rule']['pass'] = rep['rule']['arm_resolved'] >= rep['rule']['control_resolved'] + a.margin
    for grp, sel in (('control_cap_outs', lambda t: pc[t]['cap'] and not pc[t]['resolved']), ('control_resolved', lambda t: pc[t]['resolved'])):
        ts = [t for t in common if sel(t)]; rep[grp] = {'tasks': len(ts), 'arm_resolved': sum(pa[t]['resolved'] for t in ts), 'arm_first_edit_median': statistics.median([pa[t]['first_edit'] for t in ts if pa[t]['first_edit']]) if ts else None}
    print(json.dumps(rep, indent=1)); return 0


if __name__ == '__main__':
    sys.exit(main())
