#!/usr/bin/env python3
"""tool:gemma-agent/dispatch — pick the next attempt from the cohort, or refuse. Stdlib only.
Truth is the records on disk plus the adapter log; nothing is marked, so the graph must run attempts
sequentially (budgets.api.max_concurrent_attempts is 1)."""
import argparse, json, os, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # lib/ is the import root even under python -I
from validity import classify

def flat_yaml_lists(text: str) -> dict:
    """Tiny reader for the two flat config files: top-level keys, one level of nesting, inline lists."""
    out, cur = {}, None
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].rstrip()
        if not line.strip(): continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(':'); val = val.strip()
        if val.startswith('[') and val.endswith(']'):
            val = [v.strip().strip('"\'') for v in val[1:-1].split(',') if v.strip()]
        elif re.fullmatch(r'-?\d+', val or 'x'): val = int(val)
        elif val.startswith('-'): continue
        if indent == 0:
            out[key] = val if val != '' else {}; cur = key if val == '' else None
        elif indent == 2 and cur is not None:
            if val == '': out[cur][key] = {}; out.setdefault('_sub', {})[(cur, key)] = True
            else: out[cur][key] = val
        elif indent == 4 and cur is not None:
            sub = [k for k in out[cur] if isinstance(out[cur][k], dict)][-1]
            out[cur][sub][key] = val if val != '' else {}
    out.pop('_sub', None); return out

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--project-path', required=True); a = ap.parse_args()
    root = Path(a.project_path).resolve(); params = json.loads(sys.stdin.read() or '{}')
    cfg = root / '.ai/config/gemma-agent'
    budgets = flat_yaml_lists((cfg / 'budgets.yaml').read_text()); cohorts = flat_yaml_lists((cfg / 'cohorts.yaml').read_text())
    name = params.get('cohort', 'noise_floor_requests'); co = cohorts['cohorts'][name]
    tasks, repeats, bundle, site = co['tasks'], int(co['repeats']), co['bundle'], co['site']
    now = time.time(); today = datetime.now(timezone.utc).strftime('%Y%m%d')
    # records already on disk for this cohort
    done = set(); started_today = 0
    for p in (root / 'records').glob('*.json'):
        try: r = json.loads(p.read_text())
        except Exception: continue
        if r.get('bundle') != bundle or r.get('site') != site: continue
        v = classify(r)   # docs/PLAN.md validity rule: only a valid attempt fills its slot; invalid ones are re-issued
        if v == 'valid' and r['task_id'] in tasks: done.add((r['task_id'], int(r.get('repeat_index', 0))))
        if v != 'infrastructure_invalid' and (r.get('started_at') or '').replace('-', '').startswith(today): started_today += 1
    # in-flight: a results dir without task_results.jsonl, touched in the last 90 minutes
    inflight = [d.name for d in (root / 'results').glob('*-r[0-9]*') if d.is_dir() and not (d / 'task_results.jsonl').exists() and now - d.stat().st_mtime < 90 * 60]
    cap = int(budgets['api']['attempts_per_day'])
    def out(obj): print(json.dumps(obj)); return 0
    if inflight: return out({'dispatch': 'refuse', 'reason': 'attempt in flight', 'inflight': inflight, 'done': len(done), 'planned': len(tasks) * repeats})
    if started_today >= cap: return out({'dispatch': 'refuse', 'reason': f'daily cap {cap} reached', 'started_today': started_today, 'done': len(done), 'planned': len(tasks) * repeats})
    for rep in range(repeats):          # breadth first: every task once before any repeat
        for t in tasks:
            if (t, rep) not in done:
                return out({'dispatch': 'launch', 'task_id': t, 'bundle': bundle, 'site': site, 'repeat_index': rep,
                            'max_time_minutes': int(budgets['api']['attempt_wall_clock_minutes']), 'done': len(done), 'planned': len(tasks) * repeats, 'started_today': started_today})
    return out({'dispatch': 'refuse', 'reason': 'cohort complete', 'done': len(done), 'planned': len(tasks) * repeats})

if __name__ == '__main__':
    sys.exit(main())
