#!/usr/bin/env python3
"""tool:gemma-agent/swap_select: choose the recorded prefixes for the frontier swap, by one rule fixed before any swap
output exists, and write them as the swap kernel's input. Params as JSON on stdin; one JSON object on stdout. Stdlib only.

The swap hands the history of a recorded attempt, cut just before one of its model calls, to another executor and asks
for that one next call only: no tool runs, the harness is not involved. The executors are the E4B QAT base and the two
cycle adapters, all served by one vLLM instance (adapters switched per request).

Sources: the first attempt (repeat_index 0) of four arms at the cell (kaggle-l4x4, 8 min, 28 calls) on the 16 tasks of the
LoRA evaluation split (results/lora/cycles/c1/accept.json): bf18 (31B 4-bit), bf18-e4b, bf18-e4b-lora-c1, bf18-e4b-lora-c2.
Cut points, by model call index k (0-based, as lora_select numbers calls; the prefix is the history before call k):
  looping sources (c1, c2): f, the first call whose tool call (name and arguments) equals the previous call's, and f + 4;
                            an attempt with no such call contributes nothing;
  other sources (31B, E4B base): k = 4, 8, 12.
A cut point is kept when the call exists, the call before it issued a tool call (the repeat comparison needs one), and
the trace is sealed. The swap kernel admits a prefix only when its rendering with the E4B template has exactly the
served prompt token count of that call (this also excludes ADK-compacted histories); the count travels with the prefix.

Output: results/swap/datasets/<stamp>/prefixes.jsonl (messages, the previous call, the source's own next call, served
counts, provenance with record, trace and seal digests) and manifest.json."""
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_select import calls, assistant, seal_table  # noqa: E402

sha = lambda b: hashlib.sha256(b).hexdigest()
SOURCES = {'bf18': 'other', 'bf18-e4b': 'other', 'bf18-e4b-lora-c1': 'looping', 'bf18-e4b-lora-c2': 'looping'}
FIXED_POINTS = (4, 8, 12); LOOP_OFFSET = 4


def call_key(c: dict):
    tc = c['tool_calls'][:1]
    if not tc: return None
    f = tc[0]['function']
    return json.dumps([f['name'], json.loads(f['arguments'] or '{}')], sort_keys=True, separators=(',', ':'))


def cut_points(cs: list, kind: str) -> list:
    if kind == 'other': return list(FIXED_POINTS)
    keys = [call_key(c) for c in cs]
    f = next((i for i in range(1, len(cs)) if keys[i] is not None and keys[i] == keys[i - 1]), None)
    return [] if f is None else [f, f + LOOP_OFFSET]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--project-path', required=True); a = ap.parse_args()
    root = Path(a.project_path).resolve(); p = json.loads(sys.stdin.read() or '{}')
    split_file = (root / p.get('split', 'results/lora/cycles/c1/accept.json')).resolve()
    if not split_file.is_relative_to(root): return print(json.dumps({'ok': False, 'error': 'split must be inside the project'})) or 1
    if p.get('stamp') and not str(p['stamp']).replace('-', '').replace('_', '').isalnum(): return print(json.dumps({'ok': False, 'error': 'stamp must be letters, digits, - or _'})) or 1
    tasks = json.loads(split_file.read_text())['evaluation_tasks']
    seal_name, sealed = seal_table(root)
    stamp = p.get('stamp') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out = root / 'results/swap/datasets' / stamp; out.mkdir(parents=True)
    counts = {'source_attempts': 0, 'no_loop': 0, 'points_missing': 0, 'no_previous_call': 0, 'unsealed': 0, 'prefixes': 0}
    sources = []
    with (out / 'prefixes.jsonl').open('w') as fh:
        for rf in sorted((root / 'records').glob('*.json')):
            raw = rf.read_bytes(); r = json.loads(raw)
            if not (r['bundle'] in SOURCES and r['task_id'] in tasks and r.get('repeat_index') == 0 and r.get('validity') == 'valid'
                    and r['site'] == 'kaggle-l4x4' and r['budget']['max_time_minutes'] == 8.0 and r['budget']['max_tool_calls'] == 28):
                continue
            counts['source_attempts'] += 1; kind = SOURCES[r['bundle']]
            tpath = f"{r['results_dir']}/traces/trace_{r['task_id']}.json"; tbytes = (root / tpath).read_bytes()
            if sealed.get(tpath) != sha(tbytes): counts['unsealed'] += 1; continue
            steps = json.loads(tbytes)['steps']; cs = calls(steps)
            system = next(s['message'] for s in steps if s.get('source') == 'system'); user = next(s['message'] for s in steps if s.get('source') == 'user')
            hist = [[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]]
            for c in cs: hist.append(hist[-1] + [assistant(c)] + [{'role': 'tool', **o} for o in c['observations']])
            pts = cut_points(cs, kind)
            if kind == 'looping' and not pts: counts['no_loop'] += 1
            sources.append({'attempt_id': r['attempt_id'], 'bundle': r['bundle'], 'calls': len(cs), 'points': pts})
            for k in pts:
                if k >= len(cs): counts['points_missing'] += 1; continue
                if not cs[k - 1]['tool_calls']: counts['no_previous_call'] += 1; continue
                ex = {'prefix_id': f"{r['attempt_id']}@{k}", 'attempt_id': r['attempt_id'], 'task_id': r['task_id'], 'source_bundle': r['bundle'],
                      'source_kind': kind, 'point': 'first_repeat' if kind == 'looping' and k == pts[0] else 'first_repeat+4' if kind == 'looping' else f'call_{k}',
                      'call_index': k, 'messages': hist[k], 'previous_call': cs[k - 1]['tool_calls'][0], 'source_next': assistant(cs[k]),
                      'served_prompt_tokens': cs[k]['metrics'].get('prompt_tokens'), 'served_completion_tokens': cs[k]['metrics'].get('completion_tokens'),
                      'provenance': {'record': str(rf.relative_to(root)), 'record_sha256': sha(raw), 'trace': tpath, 'trace_sha256': sha(tbytes), 'seal': seal_name}}
                fh.write(json.dumps(ex) + '\n'); counts['prefixes'] += 1
    by = {}
    for line in (out / 'prefixes.jsonl').read_text().splitlines():
        e = json.loads(line); by[f"{e['source_bundle']} {e['point']}"] = by.get(f"{e['source_bundle']} {e['point']}", 0) + 1
    manifest = {'dataset': stamp, 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'rule': __doc__.split('Sources:', 1)[1].split('Output:', 1)[0].strip(),
                'split': {'file': str(split_file.relative_to(root)), 'sha256': sha(split_file.read_bytes()), 'tasks': tasks}, 'seal': seal_name,
                'counts': counts, 'by_source_point': by, 'sources': sources, 'prefixes_sha256': sha((out / 'prefixes.jsonl').read_bytes())}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({'ok': True, 'path': str(out.relative_to(root)), 'counts': counts, 'by_source_point': by, 'prefixes_sha256': manifest['prefixes_sha256']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
