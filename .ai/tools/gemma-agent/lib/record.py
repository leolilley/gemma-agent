"""One attempt record (docs/RECORDS.md, schema gemma-agent.attempt/v1) derived from the harness's outputs.
Shared by the attempt tool (API lane, one task per results dir) and the ingest tool (GPU lane, one results dir
per kernel run holding many tasks), so both lanes produce byte-for-byte the same record shape."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validity import classify


def flat_yaml(p: Path) -> dict:
    """Minimal reader for the flat `key: value` files in a bundle (eval_config.yaml, agent.yaml). Stdlib only,
    because the tools run under the node's interpreter, not the project venv."""
    out = {}
    for line in p.read_text().splitlines():
        line = line.split('#', 1)[0].rstrip()
        if ':' in line and not line.startswith((' ', '-')):
            k, v = line.split(':', 1); out[k.strip()] = v.strip().strip('"\'')
        elif ':' in line and line.startswith('  ') and not line.strip().startswith('-'):
            k, v = line.strip().split(':', 1); out[k.strip()] = v.strip().strip('"\'')
    return out


def sha256_dir(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob('*') if x.is_file() and x.name != 'PROVENANCE.md'):
        h.update(str(p.relative_to(root)).encode()); h.update(b'\0'); h.update(p.read_bytes()); h.update(b'\0')
    return h.hexdigest()


def sha256_file(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def task_outcome(results: Path, task: str) -> dict | None:
    """The harness's task_results.jsonl line for `task`, or None if the harness wrote none."""
    tr = results / 'task_results.jsonl'
    if not tr.exists(): return None
    for line in tr.read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            if o.get('instance_id') == task: return o
    return None


def derive(root: Path, *, task: str, bundle: str, results: Path, budget: dict, site: str, repeat: int, started_at: str,
           wall_seconds: float | None, model_served: str, harness: dict, sandbox: str, attempt_id: str,
           outcome: dict | None = None, lane: dict | None = None) -> dict:
    sub = root / 'submissions' / bundle
    if outcome is None: outcome = task_outcome(results, task) or {'error': 'no task result'}
    trace_p = results / 'traces' / f'trace_{task}.json'
    seq, first_edit, tokens = [], None, None
    if trace_p.exists():
        trace = json.load(open(trace_p))
        fm = trace.get('final_metrics') or {}
        tokens = {k: fm.get(f'total_{k}_tokens') for k in ('prompt', 'completion', 'cached')} if fm else None
        for s in trace.get('steps', []):
            for tc in s.get('tool_calls') or []:
                name = tc.get('function_name') or tc.get('name') or (tc.get('function') or {}).get('name') or '?'
                seq.append(name)
                if first_edit is None and name in ('edit_file', 'write_file'): first_edit = len(seq)
    rep = sum(seq[i] == seq[i - 1] for i in range(1, len(seq)))
    record = {
        'schema': 'gemma-agent.attempt/v1', 'attempt_id': attempt_id, 'task_id': task, 'bundle': bundle,
        'config_digest': sha256_dir(sub), 'model_alias': flat_yaml(sub / 'agent.yaml').get('model'),
        'model_served': model_served, 'harness': harness, 'sandbox': sandbox,
        'snapshot_digest': sha256_file(root / 'data/snapshots' / f'{task}.tgz'), 'budget': budget, 'site': site,
        'repeat_index': repeat, 'started_at': started_at, 'wall_seconds': wall_seconds,
        'outcome': {k: outcome.get(k) for k in ('resolved', 'test_exit_code', 'agent_patch_size', 'duration_seconds', 'error', 'tool_calls', 'total_llm_calls')},
        'tool_sequence': seq, 'first_edit_call_index': first_edit, 'repetition_runs': rep, 'tokens': tokens,
        'patch_digest': sha256_file(results / 'patches' / f'{task}.patch'), 'results_dir': str(results.relative_to(root)),
        'infrastructure_failure': bool((outcome.get('total_llm_calls') or 0) == 0 and outcome.get('error')),
    }
    if lane is not None: record['lane'] = lane   # API-lane cost and availability over the attempt window (lib/lane.py)
    record['validity'] = classify(record)
    return record
