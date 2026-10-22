#!/usr/bin/env python3
"""Calls-to-solve curve, pass rate versus clock, and the clock prediction, from attempt records and their traces.

Definitions (fixed here; PLAN.md "Analysis plan" cites them):
  call            a model response carrying a tool call, whether or not the harness could execute it (a malformed
                  invocation costs the same decode); calls_executed counts the ones that ran without a tool error, which
                  is what the harness's own tool_calls counts and what the 28-call cap is read against.
  solve call      the index (1-based, over calls) of the LAST successfully executed edit_file or write_file call of an
                  attempt: the call after which the graded patch was on disk. None when the attempt never edited.
  solve time      wall seconds from the agent's first model call to the completion of that edit (tool observation
                  timestamp), so it does not include the decode of the following call.
  S(k)            over a set of attempts, one per task, taken as uncensored by the clock: the fraction of TASKS that
                  were resolved with solve call <= k. S(cap) is the unbudgeted pass rate.
  P(C)            same, with solve time <= C seconds. P(C) read at the bundle's clock is the predicted pass rate of
                  the same configuration under that clock; comparing it with the observed cell pass rate is the test.
  prediction      per task: an attempt with the clock lifted predicts a pass under the clock when it resolved and its
                  solve time <= clock. The observed cell attempt on the same task is the outcome.
  source rule     solve_call_source / solve_time_source: the same, over edits to files under the repository's source
                  directory only (REPO_SOURCE), so a scratch script or a config file written after the fix does not move
                  the solve time. Declared 2026-10-20 after the rich retrodiction (rich_3043's miss was an edit to
                  .pre-commit-config.yaml 900 s after the fix) and before any
                  fastapi cell data; carried in the fastapi forecast beside the preregistered any-file rule, which stays
                  primary. fastapi's directories were amended the same day (docs_src/ added) after the lifted fastapi
                  outcomes were inspected and before any fastapi cell outcome; REPO_SOURCE_DECLARED keeps the original.

Usage: scripts/curve.py [--root DIR]      prints the summary from this project's records; regenerate.py imports the functions.
       scripts/curve.py --lifted-run DIR --cell-run DIR [--clock 480] [--lifted-clock S] [--cell-clock S]
                        [--repo-source PREFIX ...] [--json]
                        standalone, no records and no RyeOS: DIR is a swegemma results directory as the harness writes it
                        (task_results.jsonl and traces/trace_<task>.json; our kernel runs also carry run.json with the
                        budgets, read when present). Prints each run's S(k), P(C) and solve statistics, then the per-task
                        prediction of the cell run from the lifted run under both rules, with the two baselines and the
                        disagreements. --repo-source names the repository's source directories for the source rule when
                        the task ids are not ours (default: REPO_SOURCE by task-id prefix).
"""
import argparse, json, statistics, sys
from pathlib import Path

EDIT_TOOLS = {'edit_file', 'write_file'}
# Repository-source directories by task-id prefix: an edit under one is an edit to the repository's code, anything else
# (repro.py at the root, a scratch test) is not. Data, not a rule in code: extend the table for another repository.
# REPO_SOURCE_DECLARED is the mapping as declared on 2026-10-20 (kept so the forecast can show what it would have predicted);
# REPO_SOURCE is the mapping in force. fastapi gained docs_src/ on 2026-10-20 after the 20 lifted fastapi outcomes then
# available were inspected and before any fastapi cell outcome existed: fastapi's test suite imports the documentation
# examples under docs_src/, and fastapi_14492 was fixed by editing one of them (the maintainers' patch changes the same file).
REPO_SOURCE_DECLARED = {'rich': ('rich/',), 'fastapi': ('fastapi/',), 'requests': ('src/requests/', 'requests/'), 'httpx': ('httpx/',)}
REPO_SOURCE = {'rich': ('rich/',), 'fastapi': ('fastapi/', 'docs_src/'), 'requests': ('src/requests/', 'requests/'), 'httpx': ('httpx/',)}


SOURCE_OVERRIDE: tuple | None = None   # --repo-source: the same prefixes for every task


def is_source_path(task_id: str, path) -> bool:
    prefixes = SOURCE_OVERRIDE if SOURCE_OVERRIDE is not None else REPO_SOURCE.get(task_id.split('_')[0], ())
    return bool(path) and str(path).lstrip('./').startswith(tuple(prefixes))
CLOCKS = [60, 120, 180, 240, 300, 360, 420, 480, 600, 720, 900, 1200, 1500, 1800]
CALLS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]


def trace_calls(record: dict, root: Path) -> list:
    """One entry per tool call: index, tool, issued_at (elapsed s when the model's response arrived), done_at (elapsed s
    when the tool observation was recorded), prompt and completion tokens of the call that issued it."""
    p = root / record['results_dir'] / 'traces' / f"trace_{record['task_id']}.json"
    steps = json.loads(p.read_text())['steps']
    t0 = None; calls = []
    for s in steps:
        ex = s.get('extra') or {}
        if s.get('source') != 'agent' or 'timestamp' not in ex: continue
        if t0 is None: t0 = ex['timestamp'] - float(ex.get('elapsed_s') or 0)
        tcs = s.get('tool_calls') or []
        if not tcs: continue
        obs = s.get('observation') or {}; obs_ts = (obs.get('extra') or {}).get('timestamp')
        content = obs.get('content') if isinstance(obs.get('content'), str) else json.dumps(obs.get('content'))
        try: parsed = json.loads(content) if content else {}
        except json.JSONDecodeError: parsed = {}
        # malformed = the framework refused the invocation (top-level "error", no "status": missing argument, unknown
        # tool); it costs a model response, is not counted by the harness's tool_calls and runs nothing. ok = executed
        # and the tool reported no error; a run_command with a non-zero exit is executed-but-not-ok, an edit_file whose
        # old_string was not found likewise, and neither counts as an edit.
        malformed = isinstance(parsed, dict) and 'error' in parsed and 'status' not in parsed
        ok = isinstance(parsed, dict) and not malformed and parsed.get('status', 'ok') != 'error'
        m = s.get('metrics') or {}
        args = tcs[0].get('arguments') if isinstance(tcs[0].get('arguments'), dict) else {}
        calls.append({'index': len(calls) + 1, 'tool': tcs[0].get('function_name'), 'ok': ok, 'malformed': malformed, 'issued_at': ex['timestamp'] - t0,
                      'path': args.get('filepath') or args.get('path'), 'arguments': args,
                      'done_at': (obs_ts - t0) if obs_ts else ex['timestamp'] - t0,
                      'prompt': m.get('prompt_tokens'), 'completion': m.get('completion_tokens')})
    end = None
    for s in reversed(steps):
        ex = s.get('extra') or {}
        if 'timestamp' in ex and t0 is not None: end = ex['timestamp'] - t0; break
    return calls, end


def solve_point(record: dict, root: Path) -> dict:
    calls, end = trace_calls(record, root)
    edits = [c for c in calls if c['tool'] in EDIT_TOOLS and c['ok']]
    submits = [c for c in calls if c['tool'] == 'submit_patch']
    last_edit = edits[-1] if edits else None
    source_edits = [c for c in edits if is_source_path(record['task_id'], c.get('path'))]
    last_source = source_edits[-1] if source_edits else None
    return {'task': record['task_id'], 'bundle': record['bundle'], 'resolved': bool(record['outcome'].get('resolved')),
            'calls_used': len(calls), 'calls_executed': sum(1 for c in calls if not c['malformed']),
            'failed_invocations': sum(1 for c in calls if c['malformed']),
            'solve_call': last_edit['index'] if last_edit else None,
            'solve_time': last_edit['done_at'] if last_edit else None,
            'solve_call_source': last_source['index'] if last_source else None,
            'solve_time_source': last_source['done_at'] if last_source else None,
            'submit_time': submits[-1]['done_at'] if submits else None, 'end_time': end,
            'clock_s': float(record['budget']['max_time_minutes']) * 60, 'wall_s': record.get('wall_seconds'),
            'median_call_s': statistics.median(b['issued_at'] - a['issued_at'] for a, b in zip(calls, calls[1:])) if len(calls) > 1 else None}


def one_per_task(points: list) -> dict:
    """First attempt per task (repeat 0 semantics: records are sorted by file name, so the earliest run wins)."""
    d = {}
    for p in points: d.setdefault(p['task'], p)
    return d


def s_of_k(points: dict, ks=CALLS) -> dict:
    n = len(points)
    return {k: sum(1 for p in points.values() if p['resolved'] and p['solve_call'] is not None and p['solve_call'] <= k) / n for k in ks} if n else {}


def p_of_c(points: dict, clocks=CLOCKS) -> dict:
    n = len(points)
    return {c: sum(1 for p in points.values() if p['resolved'] and p['solve_time'] is not None and p['solve_time'] <= c) / n for c in clocks} if n else {}


def predict(lifted: dict, observed: dict, clock_s: float, time_key='solve_time') -> dict:
    """Per-task prediction of the cell outcome from the lifted-clock attempt; observed = cell attempts by task."""
    tasks = sorted(set(lifted) & set(observed)); rows = []
    for t in tasks:
        L, O = lifted[t], observed[t]
        pred = bool(L['resolved'] and L[time_key] is not None and L[time_key] <= clock_s)
        rows.append({'task': t, 'predicted': pred, 'observed': O['resolved'], 'lifted_resolved': L['resolved'],
                     'lifted_solve_call': L['solve_call'], 'lifted_solve_time': None if L[time_key] is None else round(L[time_key]),
                     'lifted_solve_time_source': None if L.get('solve_time_source') is None else round(L['solve_time_source']),
                     'cell_calls': O['calls_used'],
                     'cell_clock_bound': (bool(O['wall_s'] and O['wall_s'] >= O['clock_s'] - 15) if O.get('clock_s') else None)})   # None: clock unknown
    both = sum(1 for r in rows if r['predicted'] and r['observed']); po = sum(1 for r in rows if r['predicted'] and not r['observed'])
    op = sum(1 for r in rows if r['observed'] and not r['predicted']); neither = len(rows) - both - po - op
    # Baselines the clock rule must beat (PLAN.md A3 acceptance c): predict every task fails; predict the lifted outcome
    # with no clock rule. The second is the test that the clock, not the lifted run alone, carries the prediction.
    base_all_fail = sum(1 for r in rows if not r['observed'])
    base_lifted_only = sum(1 for r in rows if r['lifted_resolved'] == r['observed'])
    return {'n_tasks': len(rows), 'predicted_passes': both + po, 'observed_passes': both + op, 'agree': both + neither,
            'both': both, 'predicted_only': po, 'observed_only': op, 'neither': neither,
            'baseline_all_fail_agree': base_all_fail, 'baseline_lifted_only_agree': base_lifted_only, 'rows': rows}


def grid_tasks(root: Path) -> set:
    """The 68 two-lane-sound tasks (docs/results/gpu-lane-tasks.json); the smoke run's 8 noise-floor tasks are outside."""
    return set(json.loads((root / 'docs/results/gpu-lane-tasks.json').read_text()))


def points_by_bundle(records: list, root: Path, site='kaggle-l4x4', tasks: set | None = None) -> dict:
    out = {}; tasks = grid_tasks(root) if tasks is None else tasks
    for r in sorted(records, key=lambda r: r['attempt_id']):
        if r['site'] != site or r.get('validity') != 'valid' or r['task_id'] not in tasks: continue
        try: out.setdefault(r['bundle'], []).append(solve_point(r, root))
        except (FileNotFoundError, KeyError, json.JSONDecodeError): continue
    return out


def run_points(run_dir: Path, clock_s: float | None = None) -> list:
    """Points for one harness results directory (task_results.jsonl + traces/), no records needed. The clock is read from
    run.json (our kernel runs) unless given; with neither, 0 and the clock-bound flag is never set."""
    run_dir = Path(run_dir).resolve()
    if clock_s is None:
        rj = run_dir / 'run.json'
        try: clock_s = float(json.loads(rj.read_text())['budgets']['max_time_minutes']) * 60
        except (OSError, KeyError, ValueError, json.JSONDecodeError): clock_s = 0.0
    out = []
    for line in (run_dir / 'task_results.jsonl').read_text().splitlines():
        if not line.strip(): continue
        tr = json.loads(line); task = tr['instance_id']
        rec = {'task_id': task, 'bundle': run_dir.name, 'results_dir': str(run_dir), 'outcome': {'resolved': bool(tr.get('resolved'))},
               'budget': {'max_time_minutes': clock_s / 60}, 'wall_seconds': tr.get('duration_seconds')}
        try: out.append(solve_point(rec, Path('/')))
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e: print(f'skipping {task}: {type(e).__name__}', file=sys.stderr)
    return out


def summary_lines(name: str, one: dict) -> list:
    solved = [p for p in one.values() if p['resolved']]
    lines = [f"{name}: {len(one)} tasks, {len(solved)} resolved; median calls used {statistics.median(p['calls_used'] for p in one.values()):.0f}; "
             f"median seconds per call {statistics.median(p['median_call_s'] for p in one.values() if p['median_call_s']):.1f}"]
    if solved:
        lines.append(f"   solve call among passes: median {statistics.median(p['solve_call'] for p in solved):.0f}, max {max(p['solve_call'] for p in solved)}; "
                     f"solve time: median {statistics.median(p['solve_time'] for p in solved) / 60:.1f} min, max {max(p['solve_time'] for p in solved) / 60:.1f} min")
    lines.append('   S(k): ' + ' '.join(f'{k}:{v:.2f}' for k, v in s_of_k(one).items()))
    lines.append('   P(C): ' + ' '.join(f'{c // 60}m:{v:.2f}' for c, v in p_of_c(one).items()))
    return lines


def standalone(a) -> int:
    global SOURCE_OVERRIDE
    if a.repo_source: SOURCE_OVERRIDE = tuple(a.repo_source)
    lifted = one_per_task(run_points(Path(a.lifted_run), a.lifted_clock)); cell = one_per_task(run_points(Path(a.cell_run), a.cell_clock))
    pr = predict(lifted, cell, a.clock); ps = predict(lifted, cell, a.clock, time_key='solve_time_source')
    late = lambda r: [x['task'] for x in r['rows'] if x['lifted_resolved'] and not x['predicted']]
    if a.json:
        print(json.dumps({'lifted_run': str(a.lifted_run), 'cell_run': str(a.cell_run), 'clock_s': a.clock,
                          'lifted': {'n': len(lifted), 'S_k': s_of_k(lifted), 'P_C': p_of_c(lifted)}, 'cell': {'n': len(cell), 'S_k': s_of_k(cell), 'P_C': p_of_c(cell)},
                          'prediction_any_file': pr, 'prediction_source_rule': ps}, indent=1)); return 0
    for name, one in (('lifted run', lifted), ('cell run', cell)): print('\n'.join(summary_lines(name, one)))
    for label, r in (('any-file rule (primary)', pr), ('source rule', ps)):
        print(f"prediction of the cell run from the lifted run at {a.clock} s, {label}: {r['n_tasks']} tasks; predicted {r['predicted_passes']}, observed {r['observed_passes']}; "
              f"agree {r['agree']}/{r['n_tasks']}; baselines all-fail {r['baseline_all_fail_agree']}, lifted-only {r['baseline_lifted_only_agree']}; late solvers {late(r)}; "
              f"disagreements {[x['task'] for x in r['rows'] if x['predicted'] != x['observed']]}")
    print('| task | lifted resolved | solve call | solve time (s) | predicted | solve time, source (s) | predicted, source | observed | cell calls | cell at clock |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for r, s in zip(pr['rows'], ps['rows']):
        print(f"| {r['task']} | {r['lifted_resolved']} | {r['lifted_solve_call'] or ''} | {r['lifted_solve_time'] if r['lifted_solve_time'] is not None else ''} | {r['predicted']} | "
              f"{r['lifted_solve_time_source'] if r['lifted_solve_time_source'] is not None else ''} | {s['predicted']} | {r['observed']} | {r['cell_calls']} | {r['cell_clock_bound']} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument('--lifted-run', default=None); ap.add_argument('--cell-run', default=None); ap.add_argument('--clock', type=float, default=480)
    ap.add_argument('--lifted-clock', type=float, default=None); ap.add_argument('--cell-clock', type=float, default=None)
    ap.add_argument('--repo-source', nargs='*', default=None); ap.add_argument('--json', action='store_true'); a = ap.parse_args()
    if a.lifted_run or a.cell_run:
        if not (a.lifted_run and a.cell_run): sys.exit('standalone mode needs both --lifted-run and --cell-run')
        return standalone(a)
    root = Path(a.root); records = [json.loads(p.read_text()) for p in sorted((root / 'records').glob('*.json'))]
    by = points_by_bundle(records, root)
    for b, pts in by.items():
        one = one_per_task(pts); solved = [p for p in one.values() if p['resolved']]
        print(f"{b}: {len(one)} tasks, {len(solved)} resolved; median calls used {statistics.median(p['calls_used'] for p in one.values()):.0f}; "
              f"median seconds per call {statistics.median(p['median_call_s'] for p in one.values() if p['median_call_s']):.1f}")
        if solved:
            print(f"   solve call among passes: median {statistics.median(p['solve_call'] for p in solved):.0f}, max {max(p['solve_call'] for p in solved)}; "
                  f"solve time: median {statistics.median(p['solve_time'] for p in solved) / 60:.1f} min, max {max(p['solve_time'] for p in solved) / 60:.1f} min")
        print('   S(k):', ' '.join(f'{k}:{v:.2f}' for k, v in s_of_k(one).items()))
        print('   P(C):', ' '.join(f'{c // 60}m:{v:.2f}' for c, v in p_of_c(one).items()))
    if 'bf18-bf16-calls' in by and 'bf18-bf16' in by:
        pr = predict(one_per_task(by['bf18-bf16-calls']), one_per_task(by['bf18-bf16']), 480)
        print(f"bf16 cell predicted from bf16 calls-only at 480 s: {pr['n_tasks']} tasks; predicted {pr['predicted_passes']}, observed {pr['observed_passes']}; "
              f"agree {pr['agree']}/{pr['n_tasks']} (both {pr['both']}, predicted-only {pr['predicted_only']}, observed-only {pr['observed_only']}, neither {pr['neither']})")
        for r in pr['rows']: print('   ', r)
        ps = predict(one_per_task(by['bf18-bf16-calls']), one_per_task(by['bf18-bf16']), 480, time_key='solve_time_source')
        print(f"source rule (secondary; last edit under the repository's source directory): predicted {ps['predicted_passes']}, observed {ps['observed_passes']}; "
              f"agree {ps['agree']}/{ps['n_tasks']}; baselines all-fail {ps['baseline_all_fail_agree']}, lifted-only {ps['baseline_lifted_only_agree']}; "
              f"late solvers {[r['task'] for r in ps['rows'] if r['lifted_resolved'] and not r['predicted']]}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
