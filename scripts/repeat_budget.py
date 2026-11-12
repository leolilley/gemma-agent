#!/usr/bin/env python3
"""Repeat counts split at the tool-call budget, from recorded traces only.

The harness refuses counted tool calls after the budget's last one ("Tool call budget exhausted") but does not end
the attempt: the model can keep asking until the turn limit, and uncounted tools (submit_patch, get_status) still
run. "Within budget" is every tool call before the attempt's first refusal; "past the cap" is every call from it on.

Writes docs/results/repeat-within-budget.json: per evaluation run of the five cycles (and the base arm) the calls,
repeats, tools repeated, edits and capped completions on each side; the repeat hazard within budget; and the cycle 3
expert labels split the same way. Post hoc (2026-11-12), after an outside review asked why calls exceed the budget.

Usage: scripts/repeat_budget.py [--out docs/results/repeat-within-budget.json]
"""
import argparse, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFUSED = 'Tool call budget exhausted'
EDITS = {'edit_file', 'write_file'}


def key(c: dict) -> tuple:
    f = c.get('function') or c
    fn = c.get('function_name') or f.get('name'); a = f.get('arguments')
    return fn, a if isinstance(a, str) else json.dumps(a, sort_keys=True)


def run_stats(run_id: str) -> dict:
    side = lambda: {'calls': 0, 'repeats': 0, 'edits': 0, 'completions_at_4096': 0, 'repeated_tools': Counter()}
    s = {'within': side(), 'past': side()}; hz = {k: [0, 0] for k in range(1, 6)}
    reach = three = 0; n_tasks = 0
    for f in sorted((ROOT / 'results/kaggle' / run_id / 'traces').glob('*.json')):
        n_tasks += 1; prev = None; run = 0; past = False; reps = 0
        for st in json.loads(f.read_text())['steps']:
            tc = st.get('tool_calls') or []
            if not tc: continue
            if not past and REFUSED in json.dumps(st.get('observation')): past = True
            d = s['past' if past else 'within']
            for c in tc:
                k = key(c); rep = k == prev; d['calls'] += 1; d['edits'] += k[0] in EDITS
                if prev is not None and not past: b = min(run, 5); hz[b][0] += 1; hz[b][1] += rep
                if rep:
                    run += 1; d['repeats'] += 1; d['repeated_tools'][k[0]] += 1; reps += not past
                else: run = 1
                prev = k
            d['completions_at_4096'] += ((st.get('metrics') or {}).get('completion_tokens') or 0) >= 4096
        reach += past; three += reps >= 3
    for d in s.values(): d['repeated_tools'] = dict(sorted(d['repeated_tools'].items()))
    return {'run_id': run_id, 'tasks': n_tasks, 'tasks_reaching_the_cap': reach, 'tasks_with_3_or_more_repeats_within_budget': three,
            **s, 'hazard_within_budget': {('5+' if k == 5 else str(k)): {'next_calls': v[0], 'repeats': v[1]} for k, v in hz.items()}}


def expert_labels() -> dict:
    d = ROOT / 'results/dagger'
    att = {a['attempt_id']: a for a in map(json.loads, (d / 'datasets/20261103/attempts.jsonl').read_text().splitlines())}
    usable = {(r['attempt_id'], r['call_index']) for r in map(json.loads, (d / 'labels/20261103/rendered.jsonl').read_text().splitlines()) if r.get('usable', True)}
    out = {s: {b: {'states': 0, 'expert_repeats': 0} for b in ('0 or 1', '2', '3 or 4', '5 or more')} for s in ('within', 'past')}
    tasks = {'within': set(), 'past': set()}
    for l in map(json.loads, (d / 'labels/20261103/labels.jsonl').read_text().splitlines()):
        if (l['attempt_id'], l['call_index']) not in usable: continue
        a = att[l['attempt_id']]; st = next(s for s in a['states'] if s['call_index'] == l['call_index'])
        msgs = a['messages'][:st['messages_before']]; side = 'past' if REFUSED in json.dumps(msgs) else 'within'
        cs = [key(c) for m in msgs if m.get('role') == 'assistant' for c in (m.get('tool_calls') or [])]; run = 0
        for k in reversed(cs):
            if k != cs[-1]: break
            run += 1
        b = '0 or 1' if run <= 1 else '2' if run == 2 else '3 or 4' if run <= 4 else '5 or more'
        out[side][b]['states'] += 1; out[side][b]['expert_repeats'] += bool(l['repeat_previous'])
        if run >= 3: tasks[side].add(l['task_id'])
    return {**out, 'tasks_with_3_or_more_identical': {k: sorted(v) for k, v in tasks.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(ROOT / 'docs/results/repeat-within-budget.json')); a = ap.parse_args()
    acc = {c: json.loads((ROOT / f'results/lora/cycles/{c}/accept.json').read_text()) for c in ('c1', 'c2', 'c3', 'c4', 'c5')}
    runs = {'base': run_stats(acc['c1']['base']['run_id']), **{c: run_stats(acc[c]['adapter']['run_id']) for c in acc}}
    pooled = {b: {'next_calls': sum(runs[c]['hazard_within_budget'][b]['next_calls'] for c in ('c1', 'c2', 'c3', 'c5')),
                  'repeats': sum(runs[c]['hazard_within_budget'][b]['repeats'] for c in ('c1', 'c2', 'c3', 'c5'))} for b in ('1', '2', '3', '4', '5+')}
    doc = {'definition': 'within = tool calls before the attempt\'s first refused call; past = from that call on', 'runs': runs,
           'hazard_within_budget_pooled_c1_c2_c3_c5': pooled, 'expert_labels_cycle_3': expert_labels()}
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    for c, r in runs.items():
        w, p = r['within'], r['past']
        print(f"{c:5} within {w['repeats']:4}/{w['calls']:4} {100 * w['repeats'] / w['calls']:3.0f}%  edits {w['edits']:3} ({100 * w['edits'] / w['calls']:2.0f}%)  capped {w['completions_at_4096']:3}"
              f" | past {p['repeats']:4}/{p['calls']:4} | cap tasks {r['tasks_reaching_the_cap']:2} | 3+ repeats {r['tasks_with_3_or_more_repeats_within_budget']:2} | {w['repeated_tools']}")
    print('pooled hazard', {b: f"{v['repeats']}/{v['next_calls']}" for b, v in pooled.items()})
    print('expert', json.dumps(doc['expert_labels_cycle_3']))
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
