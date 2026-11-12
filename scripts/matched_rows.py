#!/usr/bin/env python3
"""Cycle 5's training input, the matched control pre-registered in DECISIONS 2026-11-11T03:15Z: the teacher's
exact-match rows cut to cycle 4's size. Stdlib only, deterministic.

From the teacher rendering (lora-render v2, the rendered.jsonl cycles 1 and 2 trained on) it keeps the rows the train
kernel would use for cycle 2 (usable, tier qat, within the caps), leaves the kernel's validation attempts whole (the
kernel picks them by sha256(seed:attempt) < fraction and trains on the rest), and of the remaining training rows keeps
the first N in sha256("matched-20261111:" + attempt_id + ":" + str(call_index)) order. The output rendered.jsonl holds
exactly those rows, so the unchanged kernel trains on N examples; train.json is cycle 4's with the run and adapter renamed.

Usage: scripts/matched_rows.py <teacher rendered.jsonl> [--n 204] [--out datasets/lora-data]"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEACHER_SHA = '242632951c2d16c26a5cc35fd0d9872ed6525aa775b9c27edb706a5d68a7fcb2'   # rendered.jsonl of lora-render v2, cycles 1 and 2
TAG = 'matched-20261111'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('teacher'); ap.add_argument('--n', type=int, default=204); ap.add_argument('--out', default='datasets/lora-data')
    a = ap.parse_args(); teacher = Path(a.teacher)
    if sha(teacher) != TEACHER_SHA: sys.exit(f'{teacher} is not the rendering cycles 1 and 2 trained on')
    c4 = json.loads((ROOT / 'results/lora/cycles/c4/train.json').read_text()); cfg = c4.get('config') or c4
    rows = [json.loads(l) for l in teacher.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r['usable'] and r['tier'] == 'qat' and r['prompt_tokens'] + r['completion_tokens'] <= cfg['max_total_tokens']
            and r['completion_tokens'] <= cfg['max_completion_tokens']]
    h = lambda s: int(hashlib.sha256(f'{cfg["seed"]}:{s}'.encode()).hexdigest()[:8], 16) / 0xffffffff          # the train kernel's rule
    val_att = {x for x in sorted({r['attempt_id'] for r in rows}) if h(x) < cfg['val_attempt_fraction']}
    val = [r for r in rows if r['attempt_id'] in val_att]; pool = [r for r in rows if r['attempt_id'] not in val_att]
    key = lambda r: hashlib.sha256(f'{TAG}:{r["attempt_id"]}:{r["call_index"]}'.encode()).hexdigest()
    keep = sorted(pool, key=key)[:a.n]
    if len(keep) != a.n: sys.exit(f'only {len(keep)} training rows available')
    chosen = {(r['attempt_id'], r['call_index']) for r in keep} | {(r['attempt_id'], r['call_index']) for r in val}
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    with (out / 'rendered.jsonl').open('w') as f:                      # the teacher file's own row order, so the kernel's seeded shuffle is reproducible
        for r in rows:
            if (r['attempt_id'], r['call_index']) in chosen: f.write(json.dumps(r) + '\n')
    train = {**{k: cfg[k] for k in ('max_total_tokens', 'max_completion_tokens', 'lora', 'optim', 'seed', 'replicas', 'val_attempt_fraction', 'val_max_examples', 'bundle')},
             'run_id': 'lora-c5-e4b-r16a8-teacher204-2ep', 'tiers': ['qat'], 'adapter_name': 'lora_c5'}
    manifest = {'rule': f'first {a.n} of the cycle 2 training rows in sha256("{TAG}:" + attempt_id + ":" + call_index) order, plus the kernel\'s validation attempts whole',
                'teacher': {'sha256': TEACHER_SHA, 'usable_qat_rows': len(rows)}, 'validation_rows': len(val), 'validation_attempts': sorted(val_att),
                'training_pool': len(pool), 'training_rows': len(keep), 'training_attempts': len({r['attempt_id'] for r in keep}), 'training_tasks': len({r['task_id'] for r in keep}),
                'rendered_sha256': sha(out / 'rendered.jsonl'), 'train': train}
    (out / 'matched-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'validation_attempts'}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
