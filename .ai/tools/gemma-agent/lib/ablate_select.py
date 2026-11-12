#!/usr/bin/env python3
"""tool:gemma-agent/ablate_select: the contrast set for the c1 loop-direction ablation (DECISIONS 2026-11-02T21:20Z and
the 21:50Z amendment), by one fixed rule, written as the ablate kernel's input. Params as JSON on stdin; one JSON object
on stdout. Stdlib only.

Source: the first attempt (repeat_index 0) of bf18-e4b-lora-c1 at the cell (kaggle-l4x4, 8 min, 28 calls) on each of the
16 tasks of the LoRA evaluation split (results/lora/cycles/c1/accept.json), trace sealed.
Positions: every model call k >= 1 whose previous call issued a tool call. Label: repeat when the call's first tool call
(name and canonical arguments) equals the previous call's, swap_select's call_key. The kernel renders the history before
each call and keeps the position only when the rendering has exactly the served prompt token count.

Output: results/ablate/datasets/<stamp>/attempts.jsonl (per attempt: the full message history, and per position the
number of messages before it, the served count and the label; provenance with record, trace and seal digests) and
manifest.json."""
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_select import calls, assistant, seal_table  # noqa: E402
from swap_select import call_key  # noqa: E402

sha = lambda b: hashlib.sha256(b).hexdigest()
SOURCE = 'bf18-e4b-lora-c1'


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--project-path', required=True); a = ap.parse_args()
    root = Path(a.project_path).resolve(); p = json.loads(sys.stdin.read() or '{}')
    split_file = (root / p.get('split', 'results/lora/cycles/c1/accept.json')).resolve()
    if not split_file.is_relative_to(root): return print(json.dumps({'ok': False, 'error': 'split must be inside the project'})) or 1
    if p.get('stamp') and not str(p['stamp']).replace('-', '').replace('_', '').isalnum(): return print(json.dumps({'ok': False, 'error': 'stamp must be letters, digits, - or _'})) or 1
    tasks = json.loads(split_file.read_text())['evaluation_tasks']
    seal_name, sealed = seal_table(root)
    stamp = p.get('stamp') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out = root / 'results/ablate/datasets' / stamp; out.mkdir(parents=True)
    counts = {'source_attempts': 0, 'unsealed': 0, 'positions': 0, 'repeat': 0, 'no_previous_call': 0}
    with (out / 'attempts.jsonl').open('w') as fh:
        for rf in sorted((root / 'records').glob('*.json')):
            raw = rf.read_bytes(); r = json.loads(raw)
            if not (r['bundle'] == SOURCE and r['task_id'] in tasks and r.get('repeat_index') == 0 and r.get('validity') == 'valid'
                    and r['site'] == 'kaggle-l4x4' and r['budget']['max_time_minutes'] == 8.0 and r['budget']['max_tool_calls'] == 28):
                continue
            counts['source_attempts'] += 1
            tpath = f"{r['results_dir']}/traces/trace_{r['task_id']}.json"; tbytes = (root / tpath).read_bytes()
            if sealed.get(tpath) != sha(tbytes): counts['unsealed'] += 1; continue
            steps = json.loads(tbytes)['steps']; cs = calls(steps); keys = [call_key(c) for c in cs]
            system = next(s['message'] for s in steps if s.get('source') == 'system'); user = next(s['message'] for s in steps if s.get('source') == 'user')
            msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]; positions = []
            for k, c in enumerate(cs):
                if k >= 1:
                    if keys[k - 1] is None: counts['no_previous_call'] += 1
                    else:
                        rep = keys[k] == keys[k - 1]
                        positions.append({'call_index': k, 'messages_before': len(msgs), 'served_prompt_tokens': c['metrics'].get('prompt_tokens'), 'repeat': rep})
                        counts['positions'] += 1; counts['repeat'] += int(rep)
                msgs = msgs + [assistant(c)] + [{'role': 'tool', **o} for o in c['observations']]
            fh.write(json.dumps({'attempt_id': r['attempt_id'], 'task_id': r['task_id'], 'messages': msgs, 'positions': positions,
                                 'provenance': {'record': str(rf.relative_to(root)), 'record_sha256': sha(raw), 'trace': tpath, 'trace_sha256': sha(tbytes), 'seal': seal_name}}) + '\n')
    manifest = {'dataset': stamp, 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'rule': __doc__.split('Source:', 1)[1].split('Output:', 1)[0].strip(),
                'split': {'file': str(split_file.relative_to(root)), 'sha256': sha(split_file.read_bytes()), 'tasks': tasks}, 'seal': seal_name,
                'counts': counts, 'attempts_sha256': sha((out / 'attempts.jsonl').read_bytes())}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({'ok': True, 'path': str(out.relative_to(root)), 'counts': counts, 'attempts_sha256': manifest['attempts_sha256']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
