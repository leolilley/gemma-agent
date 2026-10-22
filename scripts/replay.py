#!/usr/bin/env python3
"""Replay a clock-lifted attempt up to a chosen second and grade the repository state it had then.

The question it answers: had the clock stopped this attempt at T seconds, would the tests have passed? The lifted run's
trace records every tool call with its arguments and the elapsed time at which its observation came back. This script
takes the successful edits completed by T, applies them in order to a fresh copy of the task snapshot with the
harness's own edit function (swegemma.edit.apply_replacement, the one edit_file calls) and path rules, takes the git
diff against the snapshot baseline exactly as the harness's agent runner does, and hands that patch to the harness's
own verification (swegemma.harness.verification.verify_task) in a clean container, which is how every attempt in the
record set was graded. Nothing of the harness is reimplemented: the edits, the diff and the grading are its code.

An attempt whose edits all came after T has the baseline as its state at T; that is the empty-patch control of the
lane-soundness audit, which fails on every sound task, so the script reports it and does not start a container.

Usage (run with the project venv, which holds the harness):
  .venv/bin/python scripts/replay.py --task rich_3278 --run results/kaggle/grid-bf16-calls-rich-c3-r0-20261018 --at 480
  --dry-run        build the patch, grade nothing
  --out DIR        default results/replay; writes <task>-at<T>.json, .patch and .test_output.txt
Work directories go under <out>/replay-work and are removed afterwards.
"""
import argparse, asyncio, hashlib, json, shutil, subprocess, sys, tarfile, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import curve  # noqa: E402

EDIT_TOOLS = ('edit_file', 'write_file')


def git(ws: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(['git', '-c', 'user.name=replay', '-c', 'user.email=replay@localhost', *args], cwd=ws, capture_output=True, text=True)


def apply_edit(ws: Path, call: dict) -> None:
    """The harness's edit semantics on a local workspace: its path rules, its replacement function, its file encodings."""
    from swegemma.edit import apply_replacement
    from swegemma.tools.workspace import _resolve_workspace_path
    args = call['arguments']; dest = _resolve_workspace_path(args['filepath'], workspace_root=ws, allow_root=False)
    if call['tool'] == 'write_file':
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(args['content'], encoding='utf-8'); return
    if not dest.exists(): raise FileNotFoundError(f"File '{args['filepath']}' not found in workspace.")
    with open(dest, encoding='utf-8', errors='replace', newline='') as f: content = f.read()
    if len(content) == 0: raise ValueError(f"File '{args['filepath']}' is empty.")
    am = args.get('allow_multiple', False); am = am.strip().lower() in ('true', '1', 'yes') if isinstance(am, str) else bool(am)
    res = apply_replacement(content, args['old_string'], args['new_string'], filepath=args['filepath'], allow_multiple=am)
    with open(dest, 'w', encoding='utf-8', newline='') as f: f.write(res.new_content)


def state_patch(task: str, run: Path, at: float, work: Path) -> tuple[str, list, list]:
    calls, end = curve.trace_calls({'results_dir': str(run), 'task_id': task}, Path('/'))
    edits = [c for c in calls if c['tool'] in EDIT_TOOLS and c['ok']]
    within = [c for c in edits if c['done_at'] <= at]; after = [c for c in edits if c['done_at'] > at]
    if work.exists(): shutil.rmtree(work)
    ws = work / 'workspace'; ws.mkdir(parents=True)
    with tarfile.open(ROOT / 'data/snapshots' / f'{task}.tgz') as t: t.extractall(ws, filter='data')
    r = git(ws, 'add', '-A'); r = git(ws, 'commit', '-q', '--allow-empty', '-m', 'eval_baseline'); git(ws, 'tag', '-f', '_swegemma_baseline')
    if r.returncode != 0: raise RuntimeError(f'baseline commit failed: {r.stderr[-400:]}')
    applied = []
    for c in within:
        try: apply_edit(ws, c); applied.append({'index': c['index'], 'tool': c['tool'], 'path': c['path'], 'done_at_s': round(c['done_at'], 1), 'applied': True})
        except Exception as e:   # a replayed edit that the harness accepted should apply; if it does not, the replay is void and says so
            applied.append({'index': c['index'], 'tool': c['tool'], 'path': c['path'], 'done_at_s': round(c['done_at'], 1), 'applied': False, 'error': str(e)[:300]}); break
    git(ws, 'add', '-N', '.')
    r = git(ws, 'diff', '--binary', '_swegemma_baseline')
    if r.returncode != 0: raise RuntimeError(f'git diff failed: {r.stderr[-400:]}')
    return r.stdout, applied, [{'index': c['index'], 'tool': c['tool'], 'path': c['path'], 'done_at_s': round(c['done_at'], 1)} for c in after]


def grade(task_id: str, patch: str, bundle: str, results_dir: Path) -> dict:
    from swegemma.cli import _build_cli_model_registry
    from swegemma.config import EvalConfig
    from swegemma.harness.verification import verify_task
    from swegemma.models import load_tasks
    from swegemma.sandbox import ContainerConfig, ContainerManager
    task = next(t for t in load_tasks(ROOT / 'data/tasks.jsonl') if t.instance_id == task_id)
    config = EvalConfig(tasks_path=ROOT / 'data/tasks.jsonl', snapshots_dir=ROOT / 'data/snapshots', results_dir=results_dir, submission_dir=ROOT / 'submissions' / bundle,
                        models=_build_cli_model_registry(submission_dir=ROOT / 'submissions' / bundle), image='swebench-sandbox:latest', sandbox='docker', display_mode='quiet')
    docker = ContainerManager(ContainerConfig(image=config.image, timeout_seconds=300))
    try:
        res = asyncio.run(verify_task(docker, config, task, ROOT / 'data/snapshots' / f'{task_id}.tgz', agent_patch=patch, start_time=time.perf_counter()))
    finally:
        try: docker.cleanup_all()
        except Exception: pass
    return {'resolved': bool(res.resolved), 'test_exit_code': res.test_exit_code, 'error': res.error, 'duration_seconds': round(res.duration_seconds, 1), 'test_output': res.test_output or ''}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--task', required=True); ap.add_argument('--run', required=True); ap.add_argument('--at', type=float, default=480)
    ap.add_argument('--bundle', default='bf18-bf16-calls'); ap.add_argument('--out', default=str(ROOT / 'results/replay')); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); run = Path(a.run).resolve(); tag = f'{a.task}-at{int(a.at)}'
    lifted = next((json.loads(l) for l in (run / 'task_results.jsonl').read_text().splitlines() if l.strip() and json.loads(l).get('instance_id') == a.task), {})
    work = out / 'replay-work' / tag
    patch, applied, after = state_patch(a.task, run, a.at, work)
    report = {'task': a.task, 'run': run.name, 'at_s': a.at, 'lifted_resolved': bool(lifted.get('resolved')), 'lifted_duration_s': lifted.get('duration_seconds'),
              'edits_by_T': applied, 'edits_after_T': after, 'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(), 'patch_bytes': len(patch.encode()),
              'started_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'image': 'swebench-sandbox:latest', 'grading': 'swegemma.harness.verification.verify_task on the patch'}
    (out / f'{tag}.patch').write_text(patch)
    void = any(not e['applied'] for e in applied)
    if void: report['replay'] = {'void': True, 'reason': 'a replayed edit did not apply'}
    elif not patch.strip(): report['replay'] = {'resolved': False, 'reason': 'no edit by T: the state at T is the snapshot baseline, the empty-patch control, which fails on every sound task'}
    elif a.dry_run: report['replay'] = {'dry_run': True}
    else:
        g = grade(a.task, patch, a.bundle, work / 'harness'); (out / f'{tag}.test_output.txt').write_text(g.pop('test_output')); report['replay'] = g
    (out / f'{tag}.json').write_text(json.dumps(report, indent=2) + '\n')
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({k: v for k, v in report.items() if k not in ('edits_by_T', 'edits_after_T')}, indent=1))
    print('edits by T:', [(e['index'], e['path'], e['done_at_s'], e['applied']) for e in applied]); print('edits after T:', [(e['index'], e['path'], e['done_at_s']) for e in after])
    return 0


if __name__ == '__main__':
    sys.exit(main())
