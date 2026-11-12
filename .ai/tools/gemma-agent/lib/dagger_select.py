#!/usr/bin/env python3
"""tool:gemma-agent/dagger_select: the learner's states for one DAgger round (DECISIONS 2026-11-03T05:50Z), by one fixed
rule, written as the dagger-label kernel's input. Params as JSON on stdin; one JSON object on stdout. Stdlib only.

Source: every valid attempt of one rollout run (`run_id`, the learner's bundle `bundle`) at the cell (8 min, 28 calls) on
one site (`site`: kaggle-l4x4, the default, or modal-<gpu>, the same kernel on Modal), trace sealed. Refused if any of the run's tasks is in the LoRA evaluation split (results/lora/cycles/c1/accept.json).
States: every model call k >= 1, the history before it; the previous call travels with it when it issued a tool call. The
kernel renders each history and keeps it only when the rendering has exactly the served prompt token count.

Output: results/dagger/datasets/<stamp>/attempts.jsonl (per attempt: the full message history and per state the number
of messages before it, the served count, the previous call; provenance with record, trace and seal digests) and
manifest.json."""
import argparse, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_select import calls, assistant, seal_table  # noqa: E402

sha = lambda b: hashlib.sha256(b).hexdigest()
NAME = re.compile(r'^[A-Za-z0-9_.-]+$')
SITE = re.compile(r'^(kaggle-l4x4|modal-[a-z0-9]+(-[a-z0-9]+)*)$')


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--project-path', required=True); a = ap.parse_args()
    root = Path(a.project_path).resolve(); p = json.loads(sys.stdin.read() or '{}')
    run_id, bundle, site = p.get('run_id', ''), p.get('bundle', ''), p.get('site', 'kaggle-l4x4')
    if not SITE.match(site): return print(json.dumps({'ok': False, 'error': 'site must be kaggle-l4x4 or modal-<gpu>'})) or 1
    if not (NAME.match(run_id) and NAME.match(bundle)): return print(json.dumps({'ok': False, 'error': 'run_id and bundle are required (letters, digits, . - _)'})) or 1
    if p.get('stamp') and not str(p['stamp']).replace('-', '').replace('_', '').isalnum(): return print(json.dumps({'ok': False, 'error': 'stamp must be letters, digits, - or _'})) or 1
    split_file = root / 'results/lora/cycles/c1/accept.json'; held_out = set(json.loads(split_file.read_text())['evaluation_tasks'])
    seal_name, sealed = seal_table(root)
    rows, counts = [], {'source_attempts': 0, 'unsealed': 0, 'states': 0, 'with_previous_call': 0}
    for rf in sorted((root / 'records').glob(f'*-{bundle}-r*-{run_id}.json')):
        raw = rf.read_bytes(); r = json.loads(raw)
        if not (r['bundle'] == bundle and r.get('run_id', run_id) == run_id and r.get('validity') == 'valid' and r['site'] == site
                and r['budget']['max_time_minutes'] == 8.0 and r['budget']['max_tool_calls'] == 28):
            continue
        if r['task_id'] in held_out: return print(json.dumps({'ok': False, 'error': f"{r['task_id']} is an evaluation task"})) or 1
        counts['source_attempts'] += 1
        tpath = f"{r['results_dir']}/traces/trace_{r['task_id']}.json"; tbytes = (root / tpath).read_bytes()
        if sealed.get(tpath) != sha(tbytes): counts['unsealed'] += 1; continue
        steps = json.loads(tbytes)['steps']; cs = calls(steps)
        system = next(s['message'] for s in steps if s.get('source') == 'system'); user = next(s['message'] for s in steps if s.get('source') == 'user')
        msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]; states = []
        for k, c in enumerate(cs):
            if k >= 1:
                prev = cs[k - 1]['tool_calls'][0] if cs[k - 1]['tool_calls'] else None
                states.append({'call_index': k, 'messages_before': len(msgs), 'served_prompt_tokens': c['metrics'].get('prompt_tokens'), 'previous_call': prev,
                               'learner_next': assistant(c)})
                counts['states'] += 1; counts['with_previous_call'] += prev is not None
            msgs = msgs + [assistant(c)] + [{'role': 'tool', **o} for o in c['observations']]
        rows.append({'attempt_id': r['attempt_id'], 'task_id': r['task_id'], 'messages': msgs, 'states': states,
                     'provenance': {'record': str(rf.relative_to(root)), 'record_sha256': sha(raw), 'trace': tpath, 'trace_sha256': sha(tbytes), 'seal': seal_name}})
    if not rows: return print(json.dumps({'ok': False, 'error': 'no sealed attempts for that run and bundle', 'counts': counts})) or 1
    stamp = p.get('stamp') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out = root / 'results/dagger/datasets' / stamp; out.mkdir(parents=True)
    with (out / 'attempts.jsonl').open('w') as fh:
        for row in rows: fh.write(json.dumps(row) + '\n')
    manifest = {'dataset': stamp, 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'rule': __doc__.split('Source:', 1)[1].split('Output:', 1)[0].strip(),
                'run_id': run_id, 'bundle': bundle, 'site': site, 'split': {'file': str(split_file.relative_to(root)), 'sha256': sha(split_file.read_bytes())}, 'seal': seal_name,
                'tasks': sorted(r['task_id'] for r in rows), 'counts': counts, 'attempts_sha256': sha((out / 'attempts.jsonl').read_bytes())}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({'ok': True, 'path': str(out.relative_to(root)), 'counts': counts, 'attempts_sha256': manifest['attempts_sha256']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
