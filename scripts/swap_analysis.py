#!/usr/bin/env python3
"""The frontier swap's analysis as fixed in DECISIONS (2026-11-02T08:10Z), from the kernel's generations.jsonl and the
signed prefix selection. Stdlib only, deterministic.

Unit = prefix: its rate for an outcome is the share of its completions (one greedy, four sampled) with that outcome;
errors are dropped and counted. Intervals: task-level cluster bootstrap (scripts/regenerate.py: seed 20261013, 2,000
draws, 90%), a task's value being the mean over its prefixes of the paired difference.
  Primary 1, looping sources (prefixes of c1 and c2 attempts, both points): base minus the adapter that made the history.
  Primary 2, clean sources (E4B base and 31B prefixes, all points): each adapter minus base.
Secondary: the same for edit, final and length_cap; greedy alone; per source and point.

Usage: scripts/swap_analysis.py <swap output dir> [--prefixes results/swap/datasets/20261102/prefixes.jsonl]"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import regenerate  # noqa: E402

OUTCOMES = ('repeat_previous', 'edit', 'final', 'length_cap')
ADAPTER_OF = {'bf18-e4b-lora-c1': 'c1', 'bf18-e4b-lora-c2': 'c2'}


def rates(gens, samples=None):
    acc = defaultdict(lambda: defaultdict(list))   # (prefix, executor) -> outcome -> [0/1]
    errors = 0
    for g in gens:
        if samples and g['sample'] not in samples: continue
        if 'error' in g: errors += 1; continue
        for o in OUTCOMES: acc[(g['prefix_id'], g['executor'])][o].append(int(g[o]))
    return {k: {o: sum(v) / len(v) for o, v in d.items()} for k, d in acc.items()}, errors


def contrast(r, prefixes, outcome, a_of, b_of):
    """Per task: mean over its prefixes of rate(a) - rate(b); then the cluster bootstrap over tasks."""
    per_task = defaultdict(list)
    for p in prefixes:
        a, b = a_of(p), b_of(p)
        if (p['prefix_id'], a) in r and (p['prefix_id'], b) in r:
            per_task[p['task_id']].append(r[(p['prefix_id'], a)][outcome] - r[(p['prefix_id'], b)][outcome])
    if not per_task: return None
    pt = {t: [sum(v) / len(v)] for t, v in per_task.items()}
    ci = regenerate.cluster_bootstrap(pt, lambda s: sum(x[0] for x in s) / len(s))
    return {'diff': round(ci['point'], 4), 'lo': round(ci['lo'], 4), 'hi': round(ci['hi'], 4), 'tasks': len(pt), 'prefixes': sum(len(v) for v in per_task.values())}


def mean_rate(r, prefixes, ex, outcome):
    v = [r[(p['prefix_id'], ex)][outcome] for p in prefixes if (p['prefix_id'], ex) in r]
    return round(sum(v) / len(v), 4) if v else None


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--prefixes', default='results/swap/datasets/20261102/prefixes.jsonl'); a = ap.parse_args()
    out = Path(a.out); gens = [json.loads(l) for l in (out / 'generations.jsonl').read_text().splitlines() if l.strip()]
    ids = {g['prefix_id'] for g in gens}
    prefixes = [p for p in (json.loads(l) for l in (ROOT / a.prefixes).read_text().splitlines()) if p['prefix_id'] in ids]
    looping = [p for p in prefixes if p['source_kind'] == 'looping']; clean = [p for p in prefixes if p['source_kind'] == 'other']
    report = {'completions': len(gens), 'prefixes': len(prefixes), 'looping_prefixes': len(looping), 'clean_prefixes': len(clean)}
    for label, samples in (('all', None), ('greedy', {'greedy'})):
        r, errors = rates(gens, samples); sec = {'errors': errors}
        for o in OUTCOMES:
            sec[o] = {
                'primary1_base_minus_source_adapter': contrast(r, looping, o, lambda p: 'base', lambda p: ADAPTER_OF[p['source_bundle']]),
                'primary2_c1_minus_base': contrast(r, clean, o, lambda p: 'c1', lambda p: 'base'),
                'primary2_c2_minus_base': contrast(r, clean, o, lambda p: 'c2', lambda p: 'base'),
                'rates': {grp: {ex: mean_rate(r, ps, ex, o) for ex in ('base', 'c1', 'c2')} for grp, ps in (('looping', looping), ('clean', clean))}}
        groups = defaultdict(list)
        for p in prefixes: groups[f"{p['source_bundle']} {p['point']}"].append(p)
        sec['by_source_point_repeat'] = {k: {'n': len(v), **{ex: mean_rate(r, v, ex, 'repeat_previous') for ex in ('base', 'c1', 'c2')}} for k, v in sorted(groups.items())}
        report[label] = sec
    (out / 'analysis.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('completions', 'prefixes', 'looping_prefixes', 'clean_prefixes')}))
    for o in OUTCOMES:
        s = report['all'][o]
        print(o, 'rates', s['rates']); print('  P1 base-adapter', s['primary1_base_minus_source_adapter']); print('  P2 c1-base', s['primary2_c1_minus_base'], ' c2-base', s['primary2_c2_minus_base'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
