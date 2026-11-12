#!/usr/bin/env python3
"""tool:gemma-agent/self_select: build a self-distillation training set (cycle 4, DECISIONS 2026-11-04T06:32Z) from the campaign's own attempt records by one fixed rule, with
every example traceable to its record, its trace and the raw-file seal. Params as JSON on stdin; one JSON object on stdout.

Selection (lora_select's rule with the source swapped): an attempt is used when its record says resolved by the hidden
tests, validity valid, no infrastructure failure, the Gemma 4 E4B model under the collection bundle (param `bundle`,
default bf18-e4b-explore: the base bundle at temperature 0.7 on the 30-minute clock), and it ran on the Kaggle L4x4 lane
(the scorer's vLLM stack); its task is not in the split's evaluation set; and its trace file matches the seal. The
learner's own attempts, so the histories are ones it produces itself.

Each model call becomes one example: the history before it (system, task prompt, earlier assistant turns with reasoning and
tool calls, tool results) as OpenAI-style messages, and the call's own assistant message as the target. A call is the trace
step carrying token metrics plus any following metric-less step (the visible text of a final answer). The served prompt
token count travels with the example so the trainer can check its rendering against it."""
import argparse, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

GRID = ['fastapi_5077','fastapi_5624','fastapi_9425','fastapi_9555','fastapi_9753','fastapi_11194','fastapi_13537','fastapi_13920','fastapi_14258',
        'fastapi_14301','fastapi_14306','fastapi_14372','fastapi_14419','fastapi_14448','fastapi_14458','fastapi_14463','fastapi_14479','fastapi_14487',
        'fastapi_14492','fastapi_14616','fastapi_14786','fastapi_14794','fastapi_14873','fastapi_14986','fastapi_15280','fastapi_15588','fastapi_15589',
        'fastapi_15661','fastapi_15800','rich_2725','rich_2943','rich_3006','rich_3043','rich_3052','rich_3061','rich_3063','rich_3067','rich_3105',
        'rich_3180','rich_3278','rich_3454','rich_3468','rich_3470','rich_3471','rich_3472','rich_3506','rich_3518','rich_3521','rich_3535','rich_3675',
        'rich_3676','rich_3718','rich_3777','rich_3782','rich_3882','rich_3894','rich_3905','rich_3930','rich_3934','rich_3935','rich_3938','rich_3942',
        'rich_3944','rich_3953','rich_4075','rich_4076','rich_4077','rich_4079']
sha = lambda b: hashlib.sha256(b).hexdigest()


def split(seed: str, n_eval: int) -> list:
    """Evaluation tasks: the first n_eval of the 68 grid tasks ordered by sha256(seed:task), drawn per repository in
    proportion (fastapi 29, rich 39) so both repositories are represented."""
    by = {'fastapi': [t for t in GRID if t.startswith('fastapi')], 'rich': [t for t in GRID if t.startswith('rich')]}
    k_f = round(n_eval * len(by['fastapi']) / len(GRID))
    pick = lambda ts, k: sorted(ts, key=lambda t: sha(f'{seed}:{t}'.encode()))[:k]
    return sorted(pick(by['fastapi'], k_f) + pick(by['rich'], n_eval - k_f))


def seal_table(root: Path) -> tuple[str, dict]:
    seals = sorted((root / '.ai/knowledge/gemma-agent').glob('seal-*.md'))
    if not seals: return '', {}
    rows = {}
    for line in seals[-1].read_text().splitlines():
        m = re.match(r'^\| (\S+) \| ([0-9a-f]{64}) \|$', line)
        if m: rows[m.group(1)] = m.group(2)
    return seals[-1].name, rows


def calls(steps: list) -> list:
    """Group agent steps into model calls: a step with metrics opens a call; metric-less steps join the open call."""
    out = []
    for s in steps:
        if s.get('source') != 'agent': continue
        tc = s.get('tool_calls'); tc = json.loads(tc) if isinstance(tc, str) else (tc or [])
        ev = (s.get('extra') or {}).get('event_type')
        if s.get('metrics') or not out:
            out.append({'step_ids': [], 'reasoning': None, 'content': None, 'tool_calls': [], 'observations': [], 'metrics': s.get('metrics') or {}})
        c = out[-1]; c['step_ids'].append(s['step_id']); msg = s.get('message') or None
        if msg:
            if ev == 'thinking': c['reasoning'] = (c['reasoning'] + '\n' if c['reasoning'] else '') + msg
            else: c['content'] = (c['content'] + '\n' if c['content'] else '') + msg
        for t in tc:
            c['tool_calls'].append({'id': t.get('tool_call_id'), 'type': 'function',
                                    'function': {'name': t['function_name'], 'arguments': json.dumps(t.get('arguments') or {})}})
            obs = s.get('observation'); obs = json.loads(obs) if isinstance(obs, str) else obs
            c['observations'].append({'tool_call_id': t.get('tool_call_id'), 'content': (obs or {}).get('content', '') if isinstance(obs, dict) else str(obs)})
    return out


def assistant(c: dict) -> dict:
    m = {'role': 'assistant', 'content': c['content']}
    if c['reasoning']: m['reasoning_content'] = c['reasoning']
    if c['tool_calls']: m['tool_calls'] = c['tool_calls']
    return m


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--project-path', required=True); a = ap.parse_args()
    root = Path(a.project_path).resolve(); p = json.loads(sys.stdin.read() or '{}')
    seed, n_eval, bundle = p.get('split_seed', 'lora-split-20261029'), int(p.get('n_eval', 16)), p.get('bundle', 'bf18-e4b-explore')
    eval_tasks = split(seed, n_eval)
    seal_name, sealed = seal_table(root)
    tools_cfg = root / 'configs/lora/tools.json'; tools = json.loads(tools_cfg.read_text())['tools']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out = root / 'results/lora/datasets' / stamp; out.mkdir(parents=True)
    counts = {'records': 0, 'resolved_e4b_bundle_kaggle': 0, 'excluded_eval_task': 0, 'excluded_unsealed': 0, 'attempts_used': 0, 'examples': 0}
    used, n_ex = [], 0
    with (out / 'examples.jsonl').open('w') as fh:
        for rf in sorted((root / 'records').glob('*.json')):
            counts['records'] += 1; raw = rf.read_bytes(); r = json.loads(raw)
            if not ((r.get('outcome') or {}).get('resolved') is True and r.get('validity') == 'valid' and not r.get('infrastructure_failure')
                    and 'e4b' in (r.get('model_alias') or '') and r.get('bundle') == bundle and r.get('site') == 'kaggle-l4x4'):
                continue
            counts['resolved_e4b_bundle_kaggle'] += 1
            if r['task_id'] in eval_tasks: counts['excluded_eval_task'] += 1; continue
            tpath = f"{r['results_dir']}/traces/trace_{r['task_id']}.json"; tbytes = (root / tpath).read_bytes()
            if sealed.get(tpath) != sha(tbytes): counts['excluded_unsealed'] += 1; continue
            steps = json.loads(tbytes)['steps']
            system = next(s['message'] for s in steps if s.get('source') == 'system')
            user = next(s['message'] for s in steps if s.get('source') == 'user')
            history = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
            for i, c in enumerate(calls(steps)):
                ex = {'example_id': f"{r['attempt_id']}#{i}", 'attempt_id': r['attempt_id'], 'task_id': r['task_id'], 'repository': r['task_id'].split('_')[0],
                      'bundle': r['bundle'], 'model_served': r.get('model_served') or r.get('model_alias'), 'call_index': i, 'step_ids': c['step_ids'],
                      'messages': list(history), 'target': assistant(c), 'served_prompt_tokens': c['metrics'].get('prompt_tokens'),
                      'served_completion_tokens': c['metrics'].get('completion_tokens'),
                      'provenance': {'record': str(rf.relative_to(root)), 'record_sha256': sha(raw), 'trace': tpath, 'trace_sha256': sha(tbytes), 'seal': seal_name}}
                fh.write(json.dumps(ex) + '\n'); n_ex += 1
                history = history + [assistant(c)] + [{'role': 'tool', **o} for o in c['observations']]
            counts['attempts_used'] += 1; used.append(r['attempt_id'])
    counts['examples'] = n_ex
    manifest = {'dataset': stamp, 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'rule': (f'resolved by hidden tests, validity valid, no infrastructure failure, Gemma 4 E4B under bundle {bundle}, Kaggle L4x4 lane '
                         '(the scorer\'s vLLM stack), task not in the evaluation split, trace sealed; one example per model call, target = that call\'s assistant message'),
                'bundle': bundle,
                'split': {'seed': seed, 'n_eval': n_eval, 'evaluation_tasks': eval_tasks, 'rule': 'per-repository proportional, sha256(seed:task) order'},
                'tools': {'file': 'configs/lora/tools.json', 'sha256': sha(tools_cfg.read_bytes()), 'names': [t['function']['name'] for t in tools]},
                'seal': seal_name, 'counts': counts, 'attempts': used, 'examples_sha256': sha((out / 'examples.jsonl').read_bytes())}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({'ok': True, 'dataset': stamp, 'path': str(out.relative_to(root)), 'counts': counts, 'evaluation_tasks': eval_tasks,
                      'examples_sha256': manifest['examples_sha256']}))
    return 0

if __name__ == '__main__':
    sys.exit(main())
