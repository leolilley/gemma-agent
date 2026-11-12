#!/usr/bin/env python3
"""tool:gemma-agent/measure — apply the signed measures (.ai/config/gemma-agent/measures.yaml) to the attempt records
and order the executors they cover.

A measure is one entry of that config plus the code that computes it (this file, scripts/episodes.py, scripts/curve.py,
scripts/regenerate.py, the validity rule). Its digest covers both: editing either makes a different measure.

per_attempt measures give each valid attempt a value; an executor's value on a task is the mean over its attempts there.
Executors (bundles) are compared only within one condition (site, clock, call cap, turn cap) and only on the tasks both
have valid attempts for: the per-task difference is resampled over tasks (the campaign's cluster bootstrap: seed
20261013, 2,000 draws, 90% interval, one stream per task set). A verdict names the better executor when the interval
excludes zero, 'unresolved' when it does not, and 'no_shared_evidence' below min_shared_tasks.

declared measures read one number per executor from a signed training record (held-out loss of a cycle against its
parent bundle on that cycle's validation split); they carry no interval and their verdict is the direction alone.

A pair's order combines the verdicts of its ordering measures (role: ordering) across conditions; diagnostic measures
are reported and do not vote (order_with_diagnostics gives the order had they voted): 'A dominates' when every verdict with evidence favours A,
'A ahead' when some favour A, none favour B and the rest are unresolved, 'conflict' when verdicts favour both, 'unresolved' when none is resolved, 'no_shared_evidence' when no measure
had evidence for both. Every measurement names the measure digest and the digest of the attempts it read (each record with its trace).

Writes results/measures/<run>/measurements.json, order.json and order.md."""
import argparse, hashlib, json, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def read_config(p: Path) -> dict:
    """The measures config: top-level scalars, one `measures:` mapping of name -> flat mapping of scalars. Stdlib only."""
    out, measures, cur = {}, {}, None
    for raw in p.read_text().splitlines():
        line = raw.split(' #', 1)[0].rstrip() if not raw.lstrip().startswith('#') else ''
        if not line.strip(): continue
        ind = len(line) - len(line.lstrip()); k, _, v = line.strip().partition(':'); v = v.strip().strip('"\'')
        if ind == 0:
            if k == 'measures': out['measures'] = measures
            else: out[k] = v
        elif ind == 2: cur = measures.setdefault(k, {})
        elif ind == 4 and cur is not None: cur[k] = v
    return out


def setup(root: Path):
    sys.path.insert(0, str(root / '.ai/tools/gemma-agent/lib')); sys.path.insert(0, str(root / 'scripts'))
    import regenerate, episodes, curve  # noqa: F401
    from validity import classify
    return regenerate, episodes, curve, classify


def code_files(root: Path) -> list:
    return [root / p for p in ('.ai/tools/gemma-agent/lib/measures.py', '.ai/tools/gemma-agent/lib/validity.py', '.ai/tools/gemma-agent/lib/lane.py',
                               'scripts/episodes.py', 'scripts/curve.py', 'scripts/regenerate.py')]


# per-attempt functions: (record, calls or None) -> float or None (None = not measurable on this attempt)
def f_resolved(r, calls, ep): return 1.0 if r['outcome'].get('resolved') else 0.0
def f_patch_produced(r, calls, ep): return 1.0 if (r['outcome'].get('agent_patch_size') or 0) > 0 else 0.0
def f_loop_share(r, calls, ep):
    if calls is None: return None
    return len(ep.analyse_calls(calls)['loop_calls']) / len(calls) if calls else 0.0
def f_repeat_previous_share(r, calls, ep):
    if calls is None: return None
    if len(calls) < 2: return 0.0
    keys = [ep.canonical(c) for c in calls]
    return sum(1 for a, b in zip(keys, keys[1:]) if a == b) / (len(calls) - 1)
def f_ended_by_limit(r, calls, ep, terminal=None): return 1.0 if terminal(r) in ('call cap', 'turn cap', 'time cap') else 0.0

PER_ATTEMPT = {'resolved': f_resolved, 'patch_produced': f_patch_produced, 'loop_share': f_loop_share,
               'repeat_previous_share': f_repeat_previous_share, 'ended_by_limit': f_ended_by_limit}
NEEDS_TRACE = {'loop_share', 'repeat_previous_share'}


def condition(r) -> str:
    b = r['budget']
    return f"{r['site']} ({b['max_time_minutes']:g} min / {b['max_tool_calls']} calls / {b['max_turns']} turns)"


def paired(regenerate, va: dict, vb: dict, better: str, min_tasks: int) -> dict:
    tasks = sorted(set(va) & set(vb))
    if len(tasks) < min_tasks: return {'verdict': 'no_shared_evidence', 'n_tasks': len(tasks)}
    sign = 1 if better == 'higher' else -1
    diff = {t: [sign * (sum(va[t]) / len(va[t]) - sum(vb[t]) / len(vb[t]))] for t in tasks}
    ci = regenerate.cluster_bootstrap(diff, lambda s: sum(x[0] for x in s) / len(s))
    v = 'A' if ci['lo'] > 0 else 'B' if ci['hi'] < 0 else 'unresolved'
    # the interval is reported in the measure's own direction (A minus B), not the oriented one
    lo, hi = (ci['lo'], ci['hi']) if sign == 1 else (-ci['hi'], -ci['lo'])
    return {'verdict': v, 'n_tasks': len(tasks), 'diff': round(sign * ci['point'], 4), 'lo': round(lo, 4), 'hi': round(hi, 4)}


def combine(a: str, b: str, verdicts: list) -> str:
    seen = [v['verdict'] for v in verdicts if v['verdict'] != 'no_shared_evidence']; res = set(seen) & {'A', 'B'}
    if not seen: return 'no_shared_evidence'
    if res == {'A', 'B'}: return 'conflict'
    if res and 'unresolved' in seen: return f"{a if res == {'A'} else b} ahead"
    if res: return f"{a if res == {'A'} else b} dominates"
    return 'unresolved'


def run(root: Path, p: dict) -> dict:
    regenerate, episodes, curve, classify = setup(root)
    cfg_path = root / '.ai/config/gemma-agent/measures.yaml'; cfg = read_config(cfg_path)
    min_tasks = int(cfg.get('min_shared_tasks', 4))
    code = {str(f.relative_to(root)): sha(f) for f in code_files(root)}
    mdig = {n: hashlib.sha256(canon({'name': n, 'entry': e, 'code': code}).encode()).hexdigest() for n, e in cfg['measures'].items()}

    rec_dir = (root / p.get('records', 'records')).resolve(); stamp = p.get('run') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if not rec_dir.is_relative_to(root.resolve()): return {'ok': False, 'error': 'records must be inside the project'}
    if not stamp.replace('-', '').replace('_', '').isalnum(): return {'ok': False, 'error': 'run must be letters, digits, - or _'}
    rec_files = sorted(rec_dir.glob('*.json'))
    recs = [json.loads(f.read_text()) for f in rec_files]
    # an attempt's digest covers its record and, when present, its trace (the trace measures read it)
    rec_sha = {}
    for f, r in zip(rec_files, recs):
        t = root / str(r.get('results_dir')) / 'traces' / f"trace_{r['task_id']}.json"
        rec_sha[f.name] = sha(f) + (':' + sha(t) if t.is_file() else '')
    valid = [r for r in recs if (r.get('validity') or classify(r)) == 'valid']
    calls_of = {}
    for r in valid:
        try: calls_of[r['attempt_id']] = curve.trace_calls(r, root)[0]
        except (OSError, KeyError, ValueError): calls_of[r['attempt_id']] = None
    terminal = regenerate.terminal_reason

    # per measure, condition, executor: task -> values, and the attempts that entered
    measurements, values = [], {}
    for name, e in cfg['measures'].items():
        if e.get('kind') != 'per_attempt': continue
        fn = PER_ATTEMPT[e['function']]
        cells = defaultdict(lambda: defaultdict(list)); used = defaultdict(list); skipped = defaultdict(int)
        for r in valid:
            v = fn(r, calls_of[r['attempt_id']], episodes, terminal) if e['function'] == 'ended_by_limit' else fn(r, calls_of[r['attempt_id']], episodes)
            key = (condition(r), r['bundle'])
            if v is None: skipped[key] += 1; continue
            cells[key][r['task_id']].append(v); used[key].append(r['attempt_id'])
        for key, per_task in sorted(cells.items()):
            files = sorted(f"{a}.json:{rec_sha[a + '.json']}" for a in used[key])
            measurements.append({'measure': name, 'measure_digest': mdig[name], 'condition': key[0], 'executor': key[1],
                                 'value': round(sum(sum(v) / len(v) for v in per_task.values()) / len(per_task), 4),
                                 'n_tasks': len(per_task), 'n_attempts': len(used[key]), 'skipped_no_trace': skipped[key],
                                 'records_digest': hashlib.sha256('\n'.join(files).encode()).hexdigest()})
        values[name] = cells

    declared = defaultdict(lambda: defaultdict(dict))   # measure -> condition -> executor -> value
    for name, e in cfg['measures'].items():
        if e.get('kind') != 'declared': continue
        for tr in sorted((root / 'results/lora/cycles').glob('*/train.json')):
            d = json.loads(tr.read_text()); t = d['training']; cond = f"validation split of cycle {d['cycle']} ({t['val_examples']} examples, {t['val_attempts']} attempts)"
            for ex, val in ((d['bundle']['parent'], t[e['base_field']]), (d['bundle']['name'], t[e['trained_field']])):
                declared[name][cond][ex] = val
                measurements.append({'measure': name, 'measure_digest': mdig[name], 'condition': cond, 'executor': ex, 'value': round(val, 4),
                                     'records_digest': hashlib.sha256(f"{tr.relative_to(root)}:{sha(tr)}".encode()).hexdigest()})

    executors = sorted({m['executor'] for m in measurements})
    roles = {n: e.get('role', 'ordering') for n, e in cfg['measures'].items()}
    pairs = []
    for i, a in enumerate(executors):
        for b in executors[i + 1:]:
            verdicts = []
            for name, cells in values.items():
                better = cfg['measures'][name]['better']
                for cond in sorted({c for c, _ in cells}):
                    if (cond, a) in cells and (cond, b) in cells:
                        verdicts.append({'measure': name, 'condition': cond, **paired(regenerate, cells[(cond, a)], cells[(cond, b)], better, min_tasks)})
            for name, conds in declared.items():
                better = cfg['measures'][name]['better']
                for cond, ex in conds.items():
                    if a in ex and b in ex:
                        d = ex[a] - ex[b]; s = d if better == 'higher' else -d
                        verdicts.append({'measure': name, 'condition': cond, 'verdict': 'A' if s > 0 else 'B' if s < 0 else 'unresolved', 'diff': round(d, 4), 'interval': None})
            pairs.append({'A': a, 'B': b, 'order': combine(a, b, [v for v in verdicts if roles[v['measure']] == 'ordering']),
                          'order_with_diagnostics': combine(a, b, verdicts), 'verdicts': verdicts})

    out = root / 'results/measures' / stamp; out.mkdir(parents=True, exist_ok=True)
    head = {'computed_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'config': str(cfg_path.relative_to(root)),
            'config_sha256': sha(cfg_path), 'config_version': cfg.get('version'), 'code': code, 'measure_digests': mdig,
            'records': len(recs), 'valid_records': len(valid), 'valid_without_trace': sum(1 for v in calls_of.values() if v is None),
            'records_set_digest': hashlib.sha256('\n'.join(f'{k}:{v}' for k, v in sorted(rec_sha.items())).encode()).hexdigest(),
            'min_shared_tasks': min_tasks}
    (out / 'measurements.json').write_text(json.dumps({**head, 'measurements': measurements}, indent=1) + '\n')
    (out / 'order.json').write_text(json.dumps({**head, 'pairs': pairs}, indent=1) + '\n')
    (out / 'order.md').write_text(render(head, measurements, pairs, cfg))
    counts = defaultdict(int)
    for x in pairs: counts[x['order'].rsplit(' ', 1)[-1]] += 1
    changed = sum(1 for x in pairs if x['order'] != x['order_with_diagnostics'])
    return {'ok': True, 'out': str(out.relative_to(root)), 'measurements': len(measurements), 'pairs': len(pairs), 'orders': dict(counts), 'changed_by_diagnostics': changed}


def render(head, measurements, pairs, cfg) -> str:
    L = [f"# Measurements and order ({head['computed_at']})", '',
         f"Config {head['config']} v{head['config_version']} (sha256 {head['config_sha256'][:16]}…); records {head['valid_records']} valid of {head['records']} "
         f"(set digest {head['records_set_digest'][:16]}…); {head['valid_without_trace']} valid records have no trace.", '',
         '| measure | role | better | digest |', '|---|---|---|---|']
    L += [f"| {n} | {e.get('role', 'ordering')} | {e['better']} | {head['measure_digests'][n][:12]} |" for n, e in cfg['measures'].items()]
    by_cond = defaultdict(lambda: defaultdict(dict)); n_of = defaultdict(dict)
    for m in measurements:
        by_cond[m['condition']][m['executor']][m['measure']] = m['value']; n_of[m['condition']][m['executor']] = m.get('n_tasks')
    names = list(cfg['measures'])
    for cond in sorted(by_cond):
        L += ['', f'## {cond}', '', '| executor | tasks | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
        for ex in sorted(by_cond[cond]):
            row = by_cond[cond][ex]
            L.append(f"| {ex} | {n_of[cond][ex] if n_of[cond][ex] is not None else ''} | " + ' | '.join('' if n not in row else f'{row[n]:g}' for n in names) + ' |')
    L += ['', '## Pairwise order', '', '| A | B | order | with diagnostics | resolved verdicts |', '|---|---|---|---|---|']
    for x in pairs:
        res = '; '.join(f"{v['measure']} → {x[v['verdict']]} ({v['condition'].split(' (')[0]}, n={v.get('n_tasks', '-')})" for v in x['verdicts'] if v['verdict'] in ('A', 'B'))
        L.append(f"| {x['A']} | {x['B']} | {x['order']} | {x['order_with_diagnostics']} | {res} |")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--project-path', required=True); a = ap.parse_args()
    params = json.loads(sys.stdin.read() or '{}')
    print(json.dumps(run(Path(a.project_path), params)))
