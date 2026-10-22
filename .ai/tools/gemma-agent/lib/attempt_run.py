#!/usr/bin/env python3
"""tool:gemma-agent/attempt — one harness attempt, one record. Params arrive as JSON on stdin.
Prints one JSON object on stdout. Never prints secret values."""
import argparse, hashlib, json, os, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # lib/ is the import root even under python -I
from record import derive, flat_yaml
from lane import summarize as lane_summary

ADAPTER = 'http://127.0.0.1:8001/v1'
HARNESS = ('swegemma', 'adk_submission', 'adk_eval_core', 'google.adk')

def versions(py: Path) -> dict:
    code = ("import json,importlib.metadata as m; print(json.dumps({d: m.version(d) for d in "
            "['swegemma','adk-submission','adk-eval-core','google-adk','litellm']}))")
    out = subprocess.run([str(py), '-c', code], capture_output=True, text=True, timeout=60)
    return json.loads(out.stdout) if out.returncode == 0 else {'error': out.stderr[-200:]}

def adapter_reachable(key: str) -> bool:
    try:
        req = urllib.request.Request(f'{ADAPTER}/models', headers={'Authorization': f'Bearer {key}'})
        with urllib.request.urlopen(req, timeout=5) as r: return r.status == 200
    except Exception: return False

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--project-path', required=True)
    a = ap.parse_args(); root = Path(a.project_path).resolve()
    params = json.loads(sys.stdin.read() or '{}')
    key = os.environ.get('LITELLM_API_KEY', '')
    secret = {'name': 'LITELLM_API_KEY', 'present': bool(key), 'length': len(key)}
    if params.get('dry_run'):
        import grp, stat
        sock = Path('/var/run/docker.sock')
        try: dv = subprocess.run(['docker', 'version', '--format', '{{.Server.Version}}'], capture_output=True, text=True, timeout=20); docker_cli = (dv.returncode, (dv.stdout or dv.stderr).strip()[:160])
        except Exception as ex: docker_cli = (-1, type(ex).__name__)
        docker = {'socket_exists': sock.exists(), 'socket_is_socket': sock.exists() and stat.S_ISSOCK(sock.stat().st_mode),
                  'DOCKER_HOST': os.environ.get('DOCKER_HOST'), 'uid': os.getuid(), 'groups': sorted(grp.getgrgid(g).gr_name for g in os.getgroups() if g < 70000)[:12],
                  'var_run_listing': sorted(os.listdir('/var/run'))[:15] if Path('/var/run').exists() else None, 'root_listing': sorted(os.listdir('/'))[:25], 'docker_cli': docker_cli,
                  'mountinfo_lines': sum(1 for _ in open('/proc/self/mountinfo')) if Path('/proc/self/mountinfo').exists() else None}
        import socket as _s
        try:
            c = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM); c.settimeout(5); c.connect('/var/run/docker.sock')
            c.sendall(b'GET /version HTTP/1.0\r\nHost: docker\r\n\r\n'); docker['raw_connect'] = c.recv(120).decode('utf-8', 'replace').split('\r\n')[0]; c.close()
        except OSError as ex: docker['raw_connect'] = f'{type(ex).__name__} errno={ex.errno} {ex.strerror}'
        try: dv2 = subprocess.run(['docker', 'version'], capture_output=True, text=True, timeout=20); docker['docker_cli_full'] = (dv2.returncode, dv2.stdout[-200:], dv2.stderr[-300:])
        except Exception as ex: docker['docker_cli_full'] = type(ex).__name__
        docker['env'] = {k: os.environ.get(k) for k in ('HOME', 'PATH', 'USER', 'XDG_RUNTIME_DIR', 'TMPDIR')}
        docker['var_run_target'] = os.path.realpath('/var/run')
        try: docker['mounts'] = [l.split()[4] for l in open('/proc/self/mountinfo')][:24]
        except Exception as ex: docker['mounts'] = type(ex).__name__
        print(json.dumps({'ok': secret['present'], 'secret': secret, 'adapter_reachable': adapter_reachable(key) if key else False, 'docker': docker,
                          'harness_versions': versions(root / '.venv/bin/python'), 'cwd': os.getcwd(),
                          'env_names_sample': sorted(k for k in os.environ if k.startswith(('RYE', 'LITELLM')))[:12]}))
        return 0
    if not key:
        print(json.dumps({'ok': False, 'error': 'LITELLM_API_KEY not delivered to tool environment', 'secret': secret})); return 2
    task = params['task_id']; bundle = params.get('bundle', 'bf18'); repeat = int(params.get('repeat_index', 0))
    sub = root / 'submissions' / bundle
    e = flat_yaml(sub / 'eval_config.yaml')
    budget = {'max_time_minutes': float(params.get('max_time_minutes', 30)), 'max_tool_calls': int(e['max_tool_calls']),
              'max_turns': int(e['max_turns']), 'timeout_seconds': int(e['timeout_seconds'])}
    attempt_id = f"{task}-{bundle}-r{repeat}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    results = root / 'results' / attempt_id
    cmd = [str(root / '.venv/bin/swegemma'), 'eval', '--tasks', str(root / 'data/tasks.jsonl'), '--snapshots-dir', str(root / 'data/snapshots'),
           '--submission-dir', str(sub), '--results-dir', str(results), '--sandbox', 'docker',
           '--models-yaml', str(root / 'configs/models.gemini-adapter.yaml'),
           '--max-time-minutes', str(budget['max_time_minutes']), '--max-tool-calls', str(budget['max_tool_calls']),
           '--max-turns', str(budget['max_turns']), '--timeout-seconds', str(budget['timeout_seconds']),
           '--task-id', task, '--display', 'quiet']
    started = time.time()
    run = subprocess.run(cmd, cwd=root, capture_output=True, text=True, env={**os.environ, 'LITELLM_API_KEY': key, 'DOCKER_HOST': os.environ.get('DOCKER_HOST') or 'unix:///var/run/docker.sock'})
    (results / 'harness-stderr.log').parent.mkdir(parents=True, exist_ok=True)
    (results / 'harness-stderr.log').write_text(run.stderr[-20000:])
    tr = results / 'task_results.jsonl'
    fallback = None if tr.exists() and tr.read_text().strip() else {'error': 'no task result', 'exit': run.returncode}
    record = derive(root, task=task, bundle=bundle, results=results, budget=budget, site=params.get('site', 'local-docker-api'),
                    repeat=repeat, started_at=datetime.fromtimestamp(started, timezone.utc).isoformat(), wall_seconds=round(time.time() - started, 1),
                    model_served='gemini-api:gemma-4-31b-it via scripts/gemini-adapter.py (vLLM thinking fields dropped; provider-default thinking)',
                    harness=versions(root / '.venv/bin/python'), sandbox='docker:swebench-sandbox:latest', attempt_id=attempt_id, outcome=fallback,
                    lane=lane_summary(root / 'results/adapter-log.jsonl', started, time.time()))
    (root / 'records').mkdir(exist_ok=True)
    (root / 'records' / f'{attempt_id}.json').write_text(json.dumps(record, indent=2))
    # The graph runtime reads this object: `ok: false` or a non-null top-level `error` ends the pass (on_error: fail).
    # An unresolved, capped or provider-interrupted attempt is a measurement and the pass goes on, so the harness's own
    # error text travels as `harness_error`; only "the harness never reached the model" is raised as a tool failure.
    infra = record['validity'] == 'infrastructure_invalid'
    print(json.dumps({'ok': not infra, 'validity': record['validity'], 'attempt_id': attempt_id,
                      'resolved': record['outcome']['resolved'], 'tool_calls': record['outcome']['tool_calls'],
                      'harness_error': record['outcome']['error'], 'error': record['outcome']['error'] if infra else None,
                      'record': f'records/{attempt_id}.json'}))
    return 0

if __name__ == '__main__':
    sys.exit(main())
