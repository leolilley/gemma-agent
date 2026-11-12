#!/usr/bin/env python3
"""Cycle 3's training input as pre-registered (DECISIONS 2026-11-03T05:50Z): one rendered.jsonl holding the teacher rows
the lora-train kernel trained cycle 2 on (lora-render v2 output) and the kept DAgger rows (dagger-label output), plus
cycle 3's train.json (cycle 2's configuration, tiers qat and dagger, adapter lora_c3; one replica accumulating 8 for
the two-GPU Modal site). Stdlib only, deterministic.

--onset builds the post-hoc secondary arm c3-onset (DECISIONS 2026-11-03T20:45Z): only DAgger rows whose history does
not end in 3 or more identical tool calls (the expert follows entrenched loops), adapter lora_c3o.

Usage: scripts/dagger_rows.py <dagger-label output dir> [--onset] [--teacher .tmp/lora/render-out/v2/lora-render/rendered.jsonl] [--out datasets/lora-c3-data]"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEACHER_SHA = '242632951c2d16c26a5cc35fd0d9872ed6525aa775b9c27edb706a5d68a7fcb2'   # rendered.jsonl of lora-render v2, cycles 1 and 2
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
STATES = 'results/dagger/datasets/20261103/attempts.jsonl'


def trailing_run(messages: list) -> int:
    """How many identical tool calls (name and arguments) end the history."""
    calls = [(c['function']['name'], c['function']['arguments']) for m in messages if m['role'] == 'assistant' for c in (m.get('tool_calls') or [])]
    n = 0
    for c in reversed(calls):
        if c != calls[-1]: break
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('labels'); ap.add_argument('--teacher', default='.tmp/lora/render-out/v2/lora-render/rendered.jsonl')
    ap.add_argument('--out'); ap.add_argument('--onset', action='store_true'); a = ap.parse_args()
    a.out = a.out or ('datasets/lora-c3o-data' if a.onset else 'datasets/lora-c3-data')
    teacher, dagger = ROOT / a.teacher, Path(a.labels) / 'rendered.jsonl'
    if sha(teacher) != TEACHER_SHA: sys.exit(f'{teacher} is not the rendering cycles 1 and 2 trained on')
    rows = [json.loads(l) for l in dagger.read_text().splitlines() if l.strip()]
    if not rows or any(r['tier'] != 'dagger' or not r['usable'] for r in rows): sys.exit('the DAgger rows must all be usable rows of tier dagger')
    held_out = set(json.loads((ROOT / 'results/lora/cycles/c1/accept.json').read_text())['evaluation_tasks'])
    if any(r['task_id'] in held_out for r in rows): sys.exit('a DAgger row comes from an evaluation task')
    if a.onset:
        states = {}
        for l in (ROOT / STATES).read_text().splitlines():
            r = json.loads(l); states.update({(r['attempt_id'], s['call_index']): r['messages'][:s['messages_before']] for s in r['states']})
        rows = [r for r in rows if trailing_run(states[(r['attempt_id'], r['call_index'])]) < 3]
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    with (out / 'rendered.jsonl').open('w') as f:
        f.write(teacher.read_text())
        for r in rows: f.write(json.dumps(r) + '\n')
    c2 = json.loads((ROOT / 'results/lora/cycles/c2/train.json').read_text())
    cfg = c2.get('config') or c2
    train = {**{k: cfg[k] for k in ('max_total_tokens', 'max_completion_tokens', 'lora', 'optim', 'seed', 'replicas', 'val_attempt_fraction', 'val_max_examples', 'bundle')},
             'run_id': 'lora-c3o-e4b-r16a8-dagger1onset' if a.onset else 'lora-c3-e4b-r16a8-dagger1', 'tiers': ['qat', 'dagger'],
             'adapter_name': 'lora_c3o' if a.onset else 'lora_c3'}
    # on Modal 2x A100-80GB (DECISIONS 2026-11-03T09:05Z): one replica over both GPUs accumulating 8, the 8 examples per
    # update and the update count of c2's two replicas of 4
    assert cfg['replicas'] == 2 and cfg['optim']['accumulate'] == 4, cfg
    train['replicas'] = 1; train['optim'] = {**cfg['optim'], 'accumulate': 8}
    (out / 'train.json').write_text(json.dumps(train, indent=2) + '\n')
    (out / 'manifest.json').write_text(json.dumps({'teacher': {'file': a.teacher, 'sha256': TEACHER_SHA}, 'dagger': {'file': str(dagger), 'sha256': sha(dagger), 'rows': len(rows), 'onset_only': a.onset},
                                                    'rendered_sha256': sha(out / 'rendered.jsonl'), 'train_json_sha256': sha(out / 'train.json')}, indent=1) + '\n')
    print(json.dumps({'ok': True, 'dagger_rows': len(rows), 'tasks': sorted({r['task_id'] for r in rows}), 'train': train}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
