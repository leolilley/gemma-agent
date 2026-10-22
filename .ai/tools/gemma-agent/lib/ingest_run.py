#!/usr/bin/env python3
"""tool:gemma-agent/ingest — turn one L4x4 kernel run (results-<run_id>.zip from `kaggle kernels output`, or its
extracted directory) into attempt records, one per task, with the same derivation as the API lane. Stdlib only.
Idempotent: a record that already exists is left alone and counted as skipped."""
import hashlib
import argparse, json, re, shutil, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # lib/ is the import root even under python -I
from record import derive

WHEELS = {'swegemma': 'swegemma', 'adk_submission': 'adk-submission', 'adk_eval_core': 'adk-eval-core', 'google_adk': 'google-adk', 'litellm': 'litellm'}


def harness_versions(wheel_names: list) -> dict:
    out = {}
    for w in wheel_names:
        m = re.match(r'([A-Za-z0-9_]+)-(\d[^-]*)-', w)
        if m and m.group(1) in WHEELS: out[WHEELS[m.group(1)]] = m.group(2)
    return out


def unpack(root: Path, archive: Path) -> Path:
    """results-<run_id>.zip holds results/<run_id>/...; land it at results/kaggle/<run_id>/ with mtimes kept."""
    with zipfile.ZipFile(archive) as z:
        names = z.namelist(); tops = {Path(n).parts[1] for n in names if len(Path(n).parts) > 2 and Path(n).parts[0] == 'results'}
        assert len(tops) == 1, f'archive holds {tops}, expected one run'
        run_id = tops.pop(); dest = root / 'results/kaggle' / run_id
        if dest.exists(): return dest
        tmp = root / 'results/kaggle' / f'.unpack-{run_id}'; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        for info in z.infolist():
            if info.is_dir(): continue
            target = tmp / info.filename; target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, 'wb') as dst: shutil.copyfileobj(src, dst)
            # ZIP times are the Kaggle machine's clock, which runs UTC; naive datetime() would read them as local time
            ts = datetime(*info.date_time, tzinfo=timezone.utc).timestamp(); __import__('os').utime(target, (ts, ts))
        shutil.move(str(tmp / 'results' / run_id), str(dest)); shutil.rmtree(tmp, ignore_errors=True)
        return dest


def trace_start(run_dir: Path, task: str) -> str | None:
    """The attempt's own start: the first model call's timestamp minus its elapsed time, from the harness trace (the
    same t0 scripts/curve.py uses). None when the run left no trace for the task."""
    p = run_dir / 'traces' / f'trace_{task}.json'
    if not p.exists(): return None
    for s in json.loads(p.read_text()).get('steps', []):
        ex = s.get('extra') or {}
        if s.get('source') == 'agent' and 'timestamp' in ex:
            return datetime.fromtimestamp(ex['timestamp'] - float(ex.get('elapsed_s') or 0), timezone.utc).isoformat()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--project-path', required=True); a = ap.parse_args()
    root = Path(a.project_path).resolve(); params = json.loads(sys.stdin.read() or '{}')
    site = params.get('site', 'kaggle-l4x4')
    if params.get('archive'): run_dir = unpack(root, root / params['archive'])
    elif params.get('run_dir'): run_dir = root / params['run_dir']
    else: print(json.dumps({'ok': False, 'error': 'give archive (results-<run_id>.zip) or run_dir'})); return 2
    run = json.loads((run_dir / 'run.json').read_text()); ident = json.loads((run_dir / 'identity.json').read_text())
    bundle, repeat, run_id = run['bundle'], int(run.get('repeat_index', 0)), run['run_id']
    budget = {'max_time_minutes': float(run['budgets']['max_time_minutes']), **{k: int(run['budgets'][k]) for k in ('max_tool_calls', 'max_turns', 'timeout_seconds')}}
    weights = 'w4a16' if 'w4a16' in ident['model'] else 'bf16 weights'
    model_served = f"vllm:{ident['model_path']} dtype={ident.get('dtype', 'bfloat16')} tp={ident['tp']} ({weights}; bundle thinking_config as served)"
    tr = run_dir / 'task_results.jsonl'
    # run-level fallback only: the results file's mtime is when the whole run finished, not when any task started
    run_finished_at = datetime.fromtimestamp(tr.stat().st_mtime if tr.exists() else run_dir.stat().st_mtime, timezone.utc).isoformat()
    (root / 'records').mkdir(exist_ok=True)
    written, skipped, resolved, missing = [], [], 0, []
    for task in run['task_ids']:
        attempt_id = f'{task}-{bundle}-r{repeat}-{run_id}'; out = root / 'records' / f'{attempt_id}.json'
        if out.exists(): skipped.append(task); continue
        started_at = trace_start(run_dir, task)
        rec = derive(root, task=task, bundle=bundle, results=run_dir, budget=budget, site=site, repeat=repeat,
                     started_at=started_at or run_finished_at, wall_seconds=None, model_served=model_served,
                     harness=harness_versions(ident.get('wheelhouse', [])), sandbox='subprocess', attempt_id=attempt_id)
        rec['wall_seconds'] = rec['outcome'].get('duration_seconds')
        rec['started_at_source'] = 'trace: first model call minus elapsed' if started_at else 'run results mtime (no trace for the task; this is when the run finished)'
        # run.json may name a signed forecast shipped in the same dataset version (push-run.py --forecast); the digest
        # travels into the record so verify.py can tie this outcome to a forecast that existed before the run.
        if run.get('forecast_sha256'):
            rec['forecast_sha256'] = run['forecast_sha256']; rec['forecast_file'] = run.get('forecast_file')
            fc = run_dir / 'forecast.md'
            rec['forecast_copy_returned'] = fc.exists() and hashlib.sha256(fc.read_bytes()).hexdigest() == run['forecast_sha256']
        if rec['outcome'].get('error') == 'no task result': missing.append(task)
        out.write_text(json.dumps(rec, indent=2)); written.append(task); resolved += bool(rec['outcome'].get('resolved'))
    print(json.dumps({'ok': True, 'run_id': run_id, 'bundle': bundle, 'site': site, 'repeat_index': repeat, 'run_dir': str(run_dir.relative_to(root)),
                      'records_written': len(written), 'skipped_existing': len(skipped), 'resolved': resolved, 'no_task_result': missing}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
