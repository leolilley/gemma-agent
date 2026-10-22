#!/usr/bin/env python3
"""Push one planned GPU-lane run: dataset version (run.json + bundle) then the kernel. Replaces hand-typed command
lines after the 2026-10-14 05:38 push passed 29 task ids as one word. Task sets come from docs/results/gpu-lane-tasks.json
(the two-lane-sound tasks); shards are by repo. Verifies run.json before the kernel push. Standing go: DECISIONS 2026-10-14.

Usage: scripts/push-run.py <arm> <shard> <repeat> [--kernel eval-4bit|eval-bf16] [--concurrency N] [--label NAME] [--forecast FILE] [--dry-run]
  --forecast: a signed forecast file (PLAN.md A3 ii) to ship inside the dataset version; its sha256 goes into run.json and from
              there into every record of the run, which is how verify.py ties the outcomes to a forecast that predates them
  --label: run-id shard label for a task list (default 'subset'); needed when one arm is split into several runs
  arm:   4bit-cell | bf16-cell | 4bit-lifted | bf16-lifted | 4bit-calls | bf16-calls
         (bundle follows: bf18, bf18-bf16, bf18-lifted, bf18-bf16-lifted, bf18-calls, bf18-bf16-calls)
  shard: rich | fastapi | all | a comma-separated task list (for lifted reruns of specific failures)"""
import argparse, json, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = {'4bit-cell': 'bf18', 'bf16-cell': 'bf18-bf16', '4bit-lifted': 'bf18-lifted', 'bf16-lifted': 'bf18-bf16-lifted',
          '4bit-calls': 'bf18-calls', 'bf16-calls': 'bf18-bf16-calls', '12b-cell': 'bf18-12b'}
KERNEL = {'4bit': 'eval-4bit', 'bf16': 'eval-bf16', '12b': 'eval-4bit'}   # the 12B W4A16 checkpoint is attached to the 4-bit kernel

ap = argparse.ArgumentParser(); ap.add_argument('arm', choices=BUNDLE); ap.add_argument('shard'); ap.add_argument('repeat', type=int)
ap.add_argument('--kernel'); ap.add_argument('--concurrency', type=int, default=1); ap.add_argument('--label', default='subset'); ap.add_argument('--forecast', default=None); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
if a.forecast and not (ROOT / a.forecast).is_file(): sys.exit(f'forecast file not found: {a.forecast}')
sys.path.insert(0, str(ROOT / 'scripts')); from forecast_check import check_forecast, ForecastError
# The prospective test of PLAN.md A3 ii: this arm on these tasks may only run under a signed forecast that covers them.
FORECAST_RUNS = {'bf16-cell': 'fastapi_'}
PUBKEY = ROOT / 'docs/results/signing-public-key.pem'; SITE = 'kaggle-l4x4'
tasks = json.load(open(ROOT / 'docs/results/gpu-lane-tasks.json'))
if a.shard in ('rich', 'fastapi'): ids = [t for t in tasks if t.startswith(a.shard + '_')]
elif a.shard == 'all': ids = list(tasks)
else: ids = [t for t in a.shard.split(',') if t]
unknown = [t for t in ids if t not in tasks]
if unknown: sys.exit(f'not in the two-lane-sound set: {unknown}')
forecast_info = None
if a.arm in FORECAST_RUNS and any(t.startswith(FORECAST_RUNS[a.arm]) for t in ids) and not a.forecast:
    sys.exit(f'refused: {a.arm} on {FORECAST_RUNS[a.arm]}* tasks is the prospective forecast run (PLAN.md A3 ii) and needs --forecast <signed .ai/knowledge/gemma-agent/forecast-*.md covering these tasks>')
if a.forecast:
    try: forecast_info = check_forecast(ROOT / a.forecast, BUNDLE[a.arm], SITE, ids, PUBKEY, knowledge_dir=ROOT / '.ai/knowledge/gemma-agent')
    except ForecastError as e: sys.exit(f'refused: {e}')
    print(f"forecast {a.forecast}: signed {forecast_info['signed_at']} by {forecast_info['signer'][:12]}, covers {len(forecast_info['tasks'])} tasks, file sha256 {forecast_info['file_digest'][:12]}")
kernel = a.kernel or KERNEL[a.arm.split('-')[0]]
run_id = f"grid-{a.arm}-{a.shard if a.shard in ('rich', 'fastapi', 'all') else a.label}-r{a.repeat}-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
print(f'run {run_id}: bundle {BUNDLE[a.arm]}, {len(ids)} tasks, concurrency {a.concurrency}, kernel {kernel}')
cmd = ['scripts/push-bundles-dataset.sh', run_id, BUNDLE[a.arm], str(a.concurrency), *ids]
if a.dry_run:
    # stage run.json only (the script pushes too, so replicate its manifest step here)
    import yaml
    e = yaml.safe_load(open(ROOT / f'submissions/{BUNDLE[a.arm]}/eval_config.yaml'))['evaluation']
    run = {'run_id': run_id, 'bundle': BUNDLE[a.arm], 'task_ids': ids, 'concurrency': a.concurrency, 'repeat_index': a.repeat,
           'budgets': {k: e[k] for k in ('max_time_minutes', 'max_tool_calls', 'max_turns', 'timeout_seconds')}}
    if a.forecast: run['forecast_file'] = a.forecast; run['forecast_sha256'] = __import__('hashlib').sha256((ROOT / a.forecast).read_bytes()).hexdigest()
    print(json.dumps(run, indent=1)[:900]); print('dry run: nothing pushed'); sys.exit(0)
r = subprocess.run(cmd, cwd=ROOT, env={**__import__('os').environ, 'REPEAT': str(a.repeat), 'FORECAST': a.forecast or ''}, capture_output=True, text=True)
print('\n'.join(l for l in (r.stdout + r.stderr).splitlines() if '%' not in l)[-800:])
if r.returncode != 0: sys.exit('dataset push failed')
manifest = json.load(open(ROOT / 'datasets/bundles/run.json'))
assert manifest['run_id'] == run_id and len(manifest['task_ids']) == len(ids) and manifest['repeat_index'] == a.repeat, manifest
if a.forecast: assert manifest.get('forecast_sha256') == __import__('hashlib').sha256((ROOT / a.forecast).read_bytes()).hexdigest(), 'run.json forecast digest differs from the file'
print(f"run.json verified: {len(manifest['task_ids'])} tasks, repeat {manifest['repeat_index']}")
time.sleep(90)
r = subprocess.run([str(ROOT / '.venv/bin/kaggle'), 'kernels', 'push', '-p', str(ROOT / 'notebooks' / kernel)], capture_output=True, text=True)
push_output = (r.stdout + r.stderr).strip(); print(push_output[-300:])
kv = re.search(r'Kernel version (\d+)', push_output); kernel_version = int(kv.group(1)) if kv else None
if kernel_version is None:
    # The dataset version is up (with run.json and, for a forecast run, forecast.md) but no kernel runs it: say so in
    # DECISIONS as a refusal, not a push, and exit non-zero. The same command is run again when Kaggle accepts it.
    reason = next((l for l in push_output.splitlines() if 'error' in l.lower() or 'quota' in l.lower()), push_output[-200:])
    with open(ROOT / 'docs/DECISIONS.md', 'a') as f:
        f.write(f"- {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC — Push REFUSED by Kaggle (scripts/push-run.py): `{run_id}` = bundle `{BUNDLE[a.arm]}`, {len(ids)} tasks ({a.shard}), kernel `{kernel}`"
                + (f", forecast `{a.forecast}` sha256 {manifest['forecast_sha256'][:12]}" if a.forecast else '') + f"; dataset version with run.json created, no kernel version, no attempt ran: {reason.strip()}\n")
    sys.exit(f'kernel push refused: {reason.strip()}')
if a.forecast:
    # External evidence for the forecast's chronology (PAPER.md 3.2): the dataset version that carries the forecast, with
    # Kaggle's own timestamp, and the kernel version that read it. Kept under docs/results/forecast-runs/. The kaggle
    # package lives in the project venv, so the lookup runs there.
    ds = None
    try:
        q = subprocess.run([str(ROOT / '.venv/bin/python'), '-c', (
            "import json\nfrom kaggle.api.kaggle_api_extended import KaggleApi\napi = KaggleApi(); api.authenticate()\n"
            "for d in api.dataset_list(mine=True, search='gemma-agent-bundles'):\n"
            "    if d.ref == 'leolilley/gemma-agent-bundles': print(json.dumps({'ref': d.ref, 'version': d.current_version_number, 'last_updated_utc': str(d.last_updated), 'private': bool(d.is_private)}))")],
            capture_output=True, text=True, timeout=120)
        ds = json.loads(q.stdout.strip().splitlines()[-1]) if q.stdout.strip() else {'error': (q.stderr or 'no output')[-200:]}
    except Exception as e: ds = {'error': str(e)[:200]}
    ev = {'run_id': run_id, 'bundle': BUNDLE[a.arm], 'site': SITE, 'task_ids': ids, 'forecast_file': a.forecast, 'forecast_sha256': manifest['forecast_sha256'],
          'forecast_signed_at': forecast_info['signed_at'], 'dataset_version': ds, 'kernel': {'ref': f'leolilley/gemma-agent-{kernel}', 'version': kernel_version},
          'pushed_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
          'note': 'The kernel version reads the dataset at this version (its input pin is shown on the kernel version page); the kernel copies forecast.md into its output, so the returned archive carries the forecast too.'}
    (ROOT / 'docs/results/forecast-runs').mkdir(parents=True, exist_ok=True)
    (ROOT / 'docs/results/forecast-runs' / f'{run_id}.json').write_text(json.dumps(ev, indent=2)); print('forecast-run evidence:', json.dumps(ds))
with open(ROOT / 'docs/DECISIONS.md', 'a') as f:
    f.write(f"- {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC — Pushed (standing go, scripts/push-run.py): `{run_id}` = bundle `{BUNDLE[a.arm]}`, {len(ids)} tasks ({a.shard}), concurrency {a.concurrency}, repeat {a.repeat}, kernel `{kernel}`" + (f", forecast `{a.forecast}` sha256 {manifest['forecast_sha256'][:12]} shipped in dataset version {ev['dataset_version'] and ev['dataset_version'].get('version')}, kernel version {kernel_version}" if a.forecast else '') + ".\n")
