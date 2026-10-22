"""Lane audit, CPU only: is a task sound on Kaggle's subprocess sandbox? For each task the harness's own verify_task
runs twice on a clean snapshot: with the empty patch (must fail) and with the gold patch (must pass). Version 2 covers all
129 public tasks (version 1: the 52 grid tasks, 39 sound, 2026-10-13 23:28 UTC). Also records which
copy of the repository the sandbox imports (workspace or the image's site-packages). No GPU, no model, no agent.
Output: /kaggle/working/lane_audit.jsonl (one line per task, appended as it goes) and lane_audit.zip."""
import asyncio, glob, importlib, json, logging, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
os.environ.update({'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'LITELLM_TELEMETRY': 'False', 'LITELLM_LOG': 'WARNING', 'OTEL_SDK_DISABLED': 'true'})
logging.basicConfig(level=logging.WARNING)

# 1. wheelhouse (identical to the organizer notebook)
WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
for pth in glob.glob('/usr/local/lib/python*/*-packages/*cutlass*.pth'):
    try: os.unlink(pth)
    except OSError: pass
tmp_whl = Path('/tmp/wheelhouse'); tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower(): continue
    target = tmp_whl / (w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name)
    if not target.exists():
        try: os.symlink(w, target)
        except OSError: shutil.copyfile(w, target)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
                *sorted(str(w) for w in tmp_whl.glob('*.whl'))], check=True)
importlib.invalidate_caches()


DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
WORK = Path('/kaggle/working'); OUT = WORK / 'lane_audit.jsonl'; RESULTS = WORK / 'audit-results'; RESULTS.mkdir(exist_ok=True)
TASKS = ["fastapi_15661", "fastapi_15588", "fastapi_15589", "fastapi_15030", "fastapi_14962", "fastapi_15280", "fastapi_14964", "fastapi_15023", "fastapi_14986", "fastapi_14978", "requests_7505", "requests_7502", "requests_7433", "requests_7427", "requests_7315", "requests_7328", "requests_7309", "requests_7205", "rich_4070", "rich_4077", "rich_4079", "rich_4076", "rich_4075", "rich_4006", "fastapi_14258", "fastapi_14851", "fastapi_14873", "fastapi_13920", "fastapi_14794", "fastapi_14786", "fastapi_14492", "fastapi_14479", "fastapi_14463", "fastapi_12942", "fastapi_14099", "fastapi_14609", "fastapi_14372", "fastapi_14349", "fastapi_14512", "fastapi_14487", "fastapi_11355", "fastapi_14485", "fastapi_14482", "fastapi_14306", "fastapi_14458", "fastapi_14459", "fastapi_14448", "fastapi_14455", "fastapi_9753", "fastapi_14430", "fastapi_5077", "fastapi_9555", "fastapi_14361", "fastapi_13207", "fastapi_14360", "fastapi_14356", "fastapi_5624", "fastapi_14419", "fastapi_14301", "fastapi_14262", "fastapi_14266", "fastapi_14246", "fastapi_14186", "fastapi_9425", "fastapi_13713", "fastapi_14077", "rich_3718", "rich_3934", "rich_3953", "rich_3944", "rich_3942", "rich_3938", "rich_3882", "rich_3894", "rich_3935", "rich_3905", "rich_3930", "httpx_3672", "fastapi_14791", "fastapi_15800", "fastapi_15785", "fastapi_14371", "fastapi_14605", "fastapi_15763", "fastapi_14583", "fastapi_15745", "fastapi_14297", "fastapi_14303", "fastapi_13537", "fastapi_14953", "fastapi_13786", "fastapi_14616", "fastapi_11194", "requests_6757", "requests_6629", "requests_6644", "requests_6589", "requests_6592", "rich_3180", "rich_3486", "rich_3777", "rich_3468", "rich_3782", "rich_3772", "rich_3296", "rich_3675", "rich_3676", "rich_3518", "rich_3535", "rich_3521", "rich_3506", "rich_3480", "rich_3472", "rich_3454", "rich_3471", "rich_3470", "rich_3469", "rich_3052", "rich_3278", "rich_2943", "rich_2725", "rich_3105", "rich_3130", "rich_3043", "rich_3067", "rich_3006", "rich_3064", "rich_3063", "rich_3061"]
PKG = {'psf/requests': 'requests', 'encode/httpx': 'httpx', 'Textualize/rich': 'rich', 'fastapi/fastapi': 'fastapi'}

from adk_eval_core.tracing import SessionTrace
from swegemma.config import EvalConfig
from swegemma.evaluate import Evaluator
from swegemma.harness.container_setup import extract_snapshot
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks

cfg = EvalConfig(tasks_path=DATA_DIR / 'tasks.jsonl', snapshots_dir=DATA_DIR / 'snapshots', results_dir=RESULTS, submission_dir=DATA_DIR / 'sample_submission',
                 models=None, sandbox='subprocess', timeout_seconds=240, verification_timeout_seconds=300, wheels_dir=DATA_DIR / 'wheels',
                 graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'), task_ids=TASKS, display_mode='quiet')
ev = Evaluator(cfg); docker = ev.docker
tasks = {t.instance_id: t for t in load_tasks(cfg.tasks_path)}
print(f'lane audit: {len(TASKS)} tasks, sandbox=subprocess, python {sys.version.split()[0]}')

def import_probe(snapshot: Path, pkg: str) -> dict:
    sid = docker.start()
    try:
        extract_snapshot(docker, sid, snapshot)
        r = docker.exec(sid, f"python3 -c 'import {pkg}, sys; print({pkg}.__file__)'", timeout=60)
        r2 = docker.exec(sid, "python3 -c 'import sys; print(chr(10).join(sys.path))'", timeout=60)
        return {'exit': r.exit_code, 'file': (r.stdout or '').strip()[-200:], 'err': (r.stderr or '').strip()[-200:], 'sys_path': [p for p in (r2.stdout or '').splitlines() if p][:12]}
    finally:
        try: docker.stop(sid)
        except Exception: pass

async def control(task, snapshot: Path, patch: str) -> dict:
    t0 = time.perf_counter()
    try:
        res = await verify_task(docker, cfg, task, snapshot, agent_patch=patch, agent_error=None, trace=SessionTrace(), start_time=t0)
        return {'resolved': bool(res.resolved), 'exit_code': res.test_exit_code, 'error': (res.error or None), 'seconds': round(time.perf_counter() - t0, 1), 'test_tail': (res.test_output or '')[-400:]}
    except Exception as e:
        return {'resolved': None, 'exit_code': None, 'error': f'{type(e).__name__}: {e}'[:300], 'seconds': round(time.perf_counter() - t0, 1), 'test_tail': ''}

async def main():
    n_sound = 0
    for i, tid in enumerate(TASKS, 1):
        task = tasks[tid]; snapshot = DATA_DIR / 'snapshots' / f'{tid}.tgz'
        row = {'task_id': tid, 'repo': task.repo, 'import': import_probe(snapshot, PKG.get(task.repo, 'x'))}
        row['empty'] = await control(task, snapshot, '')
        row['gold'] = await control(task, snapshot, task.patch)
        row['sound'] = bool(row['empty']['resolved'] is False and row['gold']['resolved'] is True and not row['empty']['error'] and not row['gold']['error'])
        n_sound += row['sound']
        with open(OUT, 'a') as f: f.write(json.dumps(row) + '\n')
        print(f"[{i}/{len(TASKS)}] {tid:<15} import={row['import']['file'][-60:]:<60} empty={row['empty']['resolved']} gold={row['gold']['resolved']} sound={row['sound']} ({row['empty']['seconds']}s/{row['gold']['seconds']}s)", flush=True)
    print(f'sound on this lane: {n_sound}/{len(TASKS)}')

asyncio.run(main())
with zipfile.ZipFile(WORK / 'lane_audit.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(OUT, OUT.name)
    for p in RESULTS.rglob('*'):
        if p.is_file() and p.stat().st_size < 2_000_000: z.write(p, p.relative_to(WORK))
print('archived', WORK / 'lane_audit.zip')
