#!/usr/bin/env python3
"""Draw the hand-check sample of terminal-reason labels (PLAN.md 'Release': a hand-checked sample of 30 taxonomy labels
with the agreement rate) and write one trace digest per attempt WITHOUT the automatic label, for a reader to label blind.

Sample: 30 valid, unresolved GPU-lane attempts on the 68-task set, drawn by random.Random(20261023).sample over the sorted
attempt ids. The digest per attempt: budget, wall seconds, calls executed by the harness, the agent's call sequence with the
tool, the file path or command, the result status / exit code and the elapsed time at each call, truncated outputs, and how the
attempt ended as the harness recorded it (error field). The automatic labels (regenerate.terminal_reason) are written to a
separate key file for the agreement count.

Usage: scripts/label-sample.py OUT_DIR"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / '.ai/tools/gemma-agent/lib'))
import curve, regenerate  # noqa: E402
from validity import classify  # noqa: E402

SEED, N = 20261023, 30
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
tasks = curve.grid_tasks(ROOT)
recs = [json.load(open(p)) for p in sorted((ROOT / 'records').glob('*.json'))]
for r in recs: r['validity'] = classify(r)
pool = sorted(r['attempt_id'] for r in recs if r['site'] == 'kaggle-l4x4' and r['validity'] == 'valid' and r['task_id'] in tasks and not r['outcome'].get('resolved'))
ids = sorted(random.Random(SEED).sample(pool, N)); by_id = {r['attempt_id']: r for r in recs}
key = {}
for n, aid in enumerate(ids, 1):
    r = by_id[aid]; calls, end = curve.trace_calls(r, ROOT)
    lines = [f"# attempt {n:02d}", f"budget: {r['budget']['max_time_minutes']:.0f} min, {r['budget']['max_tool_calls']} tool calls, {r['budget']['max_turns']} turns, command timeout {r['budget']['timeout_seconds']} s",
             f"wall seconds: {r['wall_seconds']}; harness tool_calls counted: {r['outcome'].get('tool_calls')}; model calls: {r['outcome'].get('total_llm_calls')}; harness error field: {r['outcome'].get('error')!r}; patch bytes: {r['outcome'].get('agent_patch_size')}; test exit code: {r['outcome'].get('test_exit_code')}",
             f"last trace timestamp (elapsed s): {None if end is None else round(end)}", '', 'calls (index, elapsed s at response, tool, target, result):']
    for c in calls:
        res = c.get('result') or {}; a = c.get('arguments') or {}
        tgt = a.get('command') or c.get('path') or ''
        status = 'MALFORMED (framework refused)' if c['malformed'] else (res.get('status', 'ok') + (f" exit {res['exit_code']}" if 'exit_code' in res else ''))
        extra = ''
        if c['tool'] == 'run_command': extra = ' | ' + ' '.join(str(res.get('stdout', ''))[-160:].split()) + (' | stderr: ' + ' '.join(str(res.get('stderr', ''))[-160:].split()) if res.get('stderr') else '')
        elif c['tool'] == 'edit_file' and res.get('status') == 'error': extra = ' | ' + ' '.join(str(res.get('error', res))[:160].split())
        lines.append(f"  {c['index']:2d}  {c['issued_at']:6.0f}s  {c['tool']:<14} {str(tgt)[:110]!s:<110} {status}{extra}")
    (out / f'attempt-{n:02d}.txt').write_text('\n'.join(lines) + '\n')
    key[f'{n:02d}'] = {'attempt_id': aid, 'task': r['task_id'], 'bundle': r['bundle'], 'automatic': regenerate.terminal_reason(r)}
(out / 'KEY.json').write_text(json.dumps({'seed': SEED, 'pool': len(pool), 'sample': key}, indent=1))
from collections import Counter
print(f'{N} of {len(pool)} unresolved valid GPU attempts; automatic labels: {dict(Counter(v["automatic"] for v in key.values()))}; digests in {out}')
