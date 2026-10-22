#!/usr/bin/env python3
"""Write, and later evaluate, the signed forecast of PLAN.md A3 (ii): per-task predictions of the bf16 cell outcome on
tasks whose bf16 cell attempt has not run, from the bf16 calls-only (clock lifted) attempt on the same task.

  write      scripts/forecast.py write --tasks fastapi [--clock 480] [--out PATH] [--sign] [--retrospective]
             Refuses unless every target task has a valid lifted attempt, and refuses if ANY target task already has a
             cell outcome (a record, or an un-ingested run directory) of the predicted bundle: a forecast is only a
             forecast while the outcomes are unseen. --retrospective is the test mode for data that already exist: the
             document is labelled a retrospective test, carries `Retrodicts:` instead of `Predicts:`, cannot be signed,
             cannot be written under .ai/knowledge/, and is refused by the launch guard. Writes the knowledge item
             .ai/knowledge/gemma-agent/forecast-bf16-cell-fastapi.md (or --out, for tests) carrying:
               Predicts: bundle=bf18-bf16 site=kaggle-l4x4 tasks=<all target tasks>
             the per-task table, the predicted count, the 90% acceptance interval, the three criteria with the
             thresholds fixed from the rich retrodiction at writing time, and the late solvers. --sign runs
             `ryeos sign knowledge:gemma-agent/<name>`; the file is never edited after that.
  evaluate   scripts/forecast.py evaluate <forecast file> [--json]
             Reads the forecast and the observed cell records of its bundle, site and tasks, and decides the criteria:
             (a) count inside the interval, (b) task-level agreement at least the threshold, (c1) agreement above the
             all-fail baseline, (c2) agreement above the lifted-only baseline. (c2) is untestable when the target set
             has no late solver (the two predictors coincide): then branch 2b, the lifted run predicted the clocked
             outcomes and the clock rule's increment is untested on this set. A failed applicable criterion is branch 2.
             Prints the branch and, with --json, the numbers for the paper.

Two rules are carried (adopted 2026-10-20, Leo's go on the fifth review): the PRIMARY rule is the preregistered one, solve
time = completion of the last successful edit to any file; the SOURCE rule counts edits under the repository's source
directories only (curve.REPO_SOURCE), declared after the rich retrodiction showed rich_3043's miss was an edit to a repository
config file (.pre-commit-config.yaml) 900 s after a 116-s fix, and before any fastapi cell data. Each rule has its own predicted count, interval, criteria and late
solvers in the file; the branch is decided on the primary rule alone, the source rule is reported beside it. fastapi's
source directories were amended on 2026-10-20 (docs_src/ added) after the lifted fastapi outcomes were inspected and before
any fastapi cell outcome existed; the file prints the directories in force, and every task whose source-rule prediction
differs under the mapping as first declared (curve.REPO_SOURCE_DECLARED), so the amendment's effect is on record.

Definitions come from scripts/curve.py (call, solve call, solve time, S(k), P(C), predict). The acceptance interval:
each task's prediction is wrong independently with probability q, the rich retrodiction's disagreement rate at writing
time; with N_p predicted passes among n tasks, the observed count is N_p - X + Y, X ~ Bin(N_p, q), Y ~ Bin(n - N_p, q);
[lo, hi] is the shortest central interval with P(count < lo) <= 0.05 and P(count > hi) <= 0.05.
"""
import argparse, json, math, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import curve  # noqa: E402

LIFTED, TARGET, SITE = 'bf18-bf16-calls', 'bf18-bf16', 'kaggle-l4x4'
NAME = 'forecast-bf16-cell-fastapi'


def load_points():
    records = [json.loads(p.read_text()) for p in sorted((ROOT / 'records').glob('*.json'))]
    return curve.points_by_bundle(records, ROOT, site=SITE)


def count_interval(n: int, n_pred: int, q: float, level=0.90):
    def binom(k, m, p): return math.comb(m, k) * p ** k * (1 - p) ** (m - k)
    pmf = [0.0] * (n + 1)
    for x in range(n_pred + 1):
        for y in range(n - n_pred + 1):
            pmf[n_pred - x + y] += binom(x, n_pred, q) * binom(y, n - n_pred, q)
    tail = (1 - level) / 2
    cdf = 0.0; lo = 0
    for k in range(n + 1):
        cdf += pmf[k]
        if cdf > tail: lo = k; break
    cdf = 0.0; hi = n
    for k in range(n, -1, -1):
        cdf += pmf[k]
        if cdf > tail: hi = k; break
    return lo, hi, pmf


def existing_outcomes(tasks) -> set:
    """Target tasks that already have a TARGET outcome anywhere: a record (any validity) or an un-ingested run directory."""
    seen = set(tasks)
    found = set()
    for p in (ROOT / 'records').glob('*.json'):
        r = json.loads(p.read_text())
        if r.get('bundle') == TARGET and r.get('site') == SITE and r.get('task_id') in seen: found.add(r['task_id'])
    for rj in (ROOT / 'results').rglob('run.json'):
        try: run = json.loads(rj.read_text())
        except Exception: continue
        if run.get('bundle') == TARGET: found |= seen & set(run.get('task_ids', []))
    return found


def write(a) -> int:
    by = load_points()
    tasks = sorted(t for t in curve.grid_tasks(ROOT) if t.startswith(a.tasks + '_')) if a.tasks in ('rich', 'fastapi') else sorted(a.tasks.split(','))
    lifted_all = curve.one_per_task(by.get(LIFTED, []))
    lifted = {t: lifted_all[t] for t in tasks if t in lifted_all}
    # the same points under the source mapping as first declared, to print where the amended mapping changes a prediction
    saved = curve.REPO_SOURCE; curve.REPO_SOURCE = curve.REPO_SOURCE_DECLARED
    try: lifted_declared = {t: p for t, p in curve.one_per_task(load_points().get(LIFTED, [])).items() if t in lifted}
    finally: curve.REPO_SOURCE = saved
    repos = sorted({t.split('_')[0] for t in tasks})
    dirs_in_force = '; '.join(f"{r}: {', '.join(curve.REPO_SOURCE.get(r, ()))}" for r in repos)
    dirs_declared = '; '.join(f"{r}: {', '.join(curve.REPO_SOURCE_DECLARED.get(r, ()))}" for r in repos)
    missing = [t for t in tasks if t not in lifted]
    if missing: sys.exit(f'refused: {len(missing)} of {len(tasks)} target tasks have no valid {LIFTED} attempt yet: {", ".join(missing[:8])}')
    seen = existing_outcomes(tasks)
    if seen and not a.retrospective:
        sys.exit(f'refused: {len(seen)} of {len(tasks)} target tasks already have a {TARGET} outcome ({", ".join(sorted(seen)[:6])}); '
                 'a forecast is only a forecast while the outcomes are unseen. For a test on known data use --retrospective.')
    if a.retrospective:
        if a.sign or not a.out or Path(a.out).resolve().is_relative_to((ROOT / '.ai').resolve()) or Path(a.out).name.startswith('forecast-'):
            sys.exit('refused: a retrospective test is written with --out outside .ai/, never named forecast-*, never signed')
        if not seen: print('note: --retrospective on tasks with no existing outcome; the document is still labelled a test')
    # thresholds from the rich retrodiction as it stands now (only tasks outside the target set, so a rich test is honest about it)
    rich_tasks = {t for t in curve.grid_tasks(ROOT) if t.startswith('rich_')} - set(tasks)
    retro = curve.predict({t: p for t, p in lifted_all.items() if t in rich_tasks}, {t: p for t, p in curve.one_per_task(by.get(TARGET, [])).items() if t in rich_tasks}, a.clock)
    if retro['n_tasks'] == 0: sys.exit('refused: no rich retrodiction to take the disagreement rate from')
    q = 1 - retro['agree'] / retro['n_tasks']; agree_rate = retro['agree'] / retro['n_tasks']
    retro_s = curve.predict({t: p for t, p in lifted_all.items() if t in rich_tasks}, {t: p for t, p in curve.one_per_task(by.get(TARGET, [])).items() if t in rich_tasks}, a.clock, time_key='solve_time_source')
    q_s = 1 - retro_s['agree'] / retro_s['n_tasks']; agree_rate_s = retro_s['agree'] / retro_s['n_tasks']
    rows = [{'task': t, 'lifted_resolved': p['resolved'], 'lifted_solve_call': p['solve_call'], 'lifted_solve_time_s': None if p['solve_time'] is None else round(p['solve_time']),
             'predicted': bool(p['resolved'] and p['solve_time'] is not None and p['solve_time'] <= a.clock),
             'lifted_solve_time_source_s': None if p.get('solve_time_source') is None else round(p['solve_time_source']),
             'predicted_source': bool(p['resolved'] and p.get('solve_time_source') is not None and p['solve_time_source'] <= a.clock)} for t, p in sorted(lifted.items())]
    n, n_pred = len(rows), sum(r['predicted'] for r in rows)
    lo, hi, _ = count_interval(n, n_pred, q)
    late = [r for r in rows if r['lifted_resolved'] and not r['predicted']]
    min_agree = math.ceil((agree_rate - 0.10) * n)
    n_pred_s = sum(r['predicted_source'] for r in rows); lo_s, hi_s, _ = count_interval(n, n_pred_s, q_s)
    late_s = [r for r in rows if r['lifted_resolved'] and not r['predicted_source']]; min_agree_s = math.ceil((agree_rate_s - 0.10) * n)
    pred_declared = {t: bool(p['resolved'] and p.get('solve_time_source') is not None and p['solve_time_source'] <= a.clock) for t, p in lifted_declared.items()}
    changed = [f"{r['task']} ({'pass' if pred_declared[r['task']] else 'fail'} as declared, {'pass' if r['predicted_source'] else 'fail'} amended"
               + (f" at {r['lifted_solve_time_source_s']} s" if r['predicted_source'] else '') + ')' for r in rows if pred_declared[r['task']] != r['predicted_source']]
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ledgers = sorted((ROOT / '.ai/knowledge/gemma-agent').glob('ledger-*.md'))
    kind = 'RETROSPECTIVE TEST (outcomes already known; not a forecast, cannot be signed)' if a.retrospective else 'Forecast'
    verb = 'Retrodicts' if a.retrospective else 'Predicts'
    body = ([] if a.retrospective else [f'---\nkind: knowledge\nid: gemma-agent/{a.name}\nversion: "1.0.0"\n---\n']) + [
            f'# {kind} {now.isoformat()}: bf16 cell outcomes on {n} {a.tasks if a.tasks in ("rich", "fastapi") else "listed"} tasks, from the bf16 calls-only attempts\n',
            f'{verb}: bundle={TARGET} site={SITE} tasks={",".join(r["task"] for r in rows)}\n',
            f'Rule (PLAN.md A3, PRIMARY, preregistered): a task is predicted to pass under the {a.clock // 60}-minute clock if and only if its clock-lifted attempt '
            f'(bundle `{LIFTED}`, first valid attempt per task) resolved with solve time <= {a.clock} s, solve time being the completion of the '
            f'last successfully executed edit to any file, from the harness trace (scripts/curve.py). SOURCE rule (secondary, declared 2026-10-20 after the '
            f'rich retrodiction and before any `{TARGET}` fastapi data; decides nothing): the same with solve time taken over edits under the repository\'s '
            f'source directory only (curve.REPO_SOURCE). Written from {len(list((ROOT / "records").glob("*.json")))} records; '
            f'latest ledger entry {ledgers[-1].name if ledgers else "none"}. ' + (
                f'{len(seen)} of {n} tasks named above ALREADY HAVE a `{TARGET}` outcome: this document tests the writer on known data and predicts nothing.\n'
                if a.retrospective else
                f'No `{TARGET}` attempt on any task named above exists at writing time (checked against the records and the run directories); '
                f'the cell run that produces them is pushed only after this file is signed, with this file shipped in its dataset version.\n'),
            f'Predicted passes: {n_pred} of {n}\n',
            f'Acceptance interval for the observed count: [{lo}, {hi}] (90%, under per-task disagreement rate q = {q:.3f} from the rich retrodiction '
            f'at writing time: agreement {retro["agree"]} of {retro["n_tasks"]} on rich tasks outside this set; count = N_p - X + Y, X ~ Bin({n_pred}, q), Y ~ Bin({n - n_pred}, q)).\n',
            f'Source rule, predicted passes: {n_pred_s} of {n}. Source rule, acceptance interval for the observed count: [{lo_s}, {hi_s}] (90%, q = {q_s:.3f} from the '
            f'rich retrodiction under the source rule: agreement {retro_s["agree"]} of {retro_s["n_tasks"]}). Source rule, task level: agreement at least {min_agree_s} of {n}. '
            f'Source rule, late solvers: {len(late_s)} ({", ".join(r["task"] for r in late_s) or "none"}). The source rule is evaluated on the same criteria and reported '
            f'beside the primary; it does not decide the branch.\n',
            f'Source directories in force (curve.REPO_SOURCE, applied uniformly to every task here): {dirs_in_force}. As first declared on 2026-10-20: {dirs_declared}. ' + (
            f'fastapi gained docs_src/ on 2026-10-20 after the lifted fastapi outcomes then available were inspected (fastapi_14492 was fixed by editing only a '
            f'documentation example that the test suite imports and the maintainers\' patch changes) and before any `{TARGET}` fastapi outcome existed; the amendment is '
            f'prospective with respect to the cell outcomes but was not chosen before seeing fastapi data. ' if 'fastapi' in repos else '') +
            f'Source-rule predictions that differ under the mapping as declared: {"; ".join(changed) or "none"}. No primary-rule prediction depends on the mapping.\n',
            'Criteria (PLAN.md A3, PAPER.md decision rule); every applicable criterion must pass:',
            f'- (a) count: the observed pass count lies in [{lo}, {hi}].',
            f'- (b) task level: per-task agreement is at least {min_agree} of {n} (rich agreement {agree_rate:.3f} minus 0.10).',
            f'- (c1) all-fail baseline: agreement exceeds the all-fail predictor (agreement = number of observed fails).',
            f'- (c2) lifted-only baseline: agreement exceeds the lifted-only predictor (the lifted outcome with no clock rule; agreement = tasks where '
            f'the lifted outcome equals the observed one). The rule and that predictor differ only on the late solvers, tasks the lifted attempt '
            f'resolved after {a.clock} s: {len(late)} here ({", ".join(r["task"] for r in late) or "none"}); the rule predicts each of them to fail. '
            + ('With no late solver the two predictors coincide, so (c2) is UNTESTABLE on this set: if (a), (b) and (c1) pass, the finding is that the '
               'lifted run predicted the clocked outcomes while the clock rule\'s increment stays untested here (branch 2b).' if not late else
               '(c2) is decided on that set.'),
            'A miss on any applicable criterion is a failed forecast; what failed it is a question for the GPU repeats and the trace reasons, never asserted from the miss.\n',
            '| task | lifted resolved | lifted solve call | lifted solve time (s) | predicted pass | solve time, source (s) | predicted pass, source |',
            '| --- | --- | --- | --- | --- | --- | --- |']
    body += [f'| {r["task"]} | {r["lifted_resolved"]} | {r["lifted_solve_call"] or ""} | {r["lifted_solve_time_s"] if r["lifted_solve_time_s"] is not None else ""} | {r["predicted"]} | '
             f'{r["lifted_solve_time_source_s"] if r["lifted_solve_time_source_s"] is not None else ""} | {r["predicted_source"]} |' for r in rows]
    body += ['', 'This file is not edited after signing. If the cell run is cut by quota, the forecast is published as unmet-by-quota, not withdrawn.']
    out = Path(a.out) if a.out else ROOT / '.ai/knowledge/gemma-agent' / f'{a.name}.md'
    if out.exists(): sys.exit(f'refused: {out} exists; a forecast is never rewritten')
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text('\n'.join(body) + '\n')
    print(f'wrote {out}: {n} tasks, predicted {n_pred}, interval [{lo}, {hi}], q {q:.3f}, min agreement {min_agree}, late solvers {len(late)}; '
          f'source rule: predicted {n_pred_s}, interval [{lo_s}, {hi_s}], q {q_s:.3f}, min agreement {min_agree_s}, late solvers {len(late_s)}')
    if a.sign:
        if a.out: sys.exit('refused: --sign only for the knowledge item (no --out)')
        r = subprocess.run(['ryeos', 'sign', f'knowledge:gemma-agent/{a.name}', '--project', str(ROOT)], capture_output=True, text=True)
        try: res = json.loads(r.stdout)
        except Exception: res = {}
        if r.returncode != 0 or res.get('failed') or not res.get('signed'): out.unlink(); sys.exit(f'signing failed; file removed:\n{(r.stdout + r.stderr)[-600:]}')
        print('signed:', out.read_text().splitlines()[0][:90])
    return 0


def evaluate(a) -> int:
    text = Path(a.forecast).read_text()
    pm = re.search(r'^(?:Predicts|Retrodicts):\s*bundle=(\S+)\s+site=(\S+)\s+tasks=(\S+)', text, re.M)
    bundle, site, tasks = pm.group(1), pm.group(2), pm.group(3).split(',')
    lo, hi = map(int, re.search(r'^Acceptance interval for the observed count: \[(\d+), (\d+)\]', text, re.M).groups())
    min_agree = int(re.search(r'agreement is at least (\d+) of', text).group(1))
    ms = re.search(r'Source rule, acceptance interval for the observed count: \[(\d+), (\d+)\].*?Source rule, task level: agreement at least (\d+) of', text, re.S)
    source = {'lo': int(ms.group(1)), 'hi': int(ms.group(2)), 'min_agree': int(ms.group(3))} if ms else None
    pred = {}; cols = None
    for line in text.splitlines():
        if line.startswith('| task'): cols = [x.strip() for x in line.strip('|').split('|')]; continue
        if line.startswith('| ') and not line.startswith('| ---') and cols:
            c = dict(zip(cols, [x.strip() for x in line.strip('|').split('|')]))
            pred[c['task']] = {'lifted_resolved': c['lifted resolved'] == 'True', 'predicted': c['predicted pass'] == 'True',
                               'predicted_source': c.get('predicted pass, source') == 'True' if 'predicted pass, source' in c else None}
    records = [json.loads(p.read_text()) for p in sorted((ROOT / 'records').glob('*.json'))]
    obs = curve.one_per_task(curve.points_by_bundle(records, ROOT, site=site, tasks=set(tasks)).get(bundle, []))
    have = [t for t in tasks if t in obs]
    if not have: print('no observed cell attempts yet for this forecast (pending)'); return 0
    rows = [{'task': t, **pred[t], 'observed': obs[t]['resolved']} for t in have]
    n = len(rows); n_obs = sum(r['observed'] for r in rows); n_pred = sum(r['predicted'] for r in rows)
    agree = sum(r['predicted'] == r['observed'] for r in rows)
    all_fail = sum(not r['observed'] for r in rows); lifted_only = sum(r['lifted_resolved'] == r['observed'] for r in rows)
    late = [r for r in rows if r['lifted_resolved'] and not r['predicted']]
    crit = {'a_count': lo <= n_obs <= hi, 'b_task_level': agree >= min_agree, 'c1_all_fail': agree > all_fail,
            'c2_lifted_only': (agree > lifted_only) if late else None}   # None: untestable, no late solver
    applicable = [v for v in crit.values() if v is not None]
    branch = 2 if not all(applicable) else ('2b' if crit['c2_lifted_only'] is None else 1)
    res = {'n_tasks': n, 'of': len(tasks), 'predicted': n_pred, 'observed': n_obs, 'interval': [lo, hi], 'agree': agree, 'min_agree': min_agree,
           'baseline_all_fail': all_fail, 'baseline_lifted_only': lifted_only, 'late_solvers': [(r['task'], 'right' if not r['observed'] else 'wrong') for r in late],
           'disagreements': [(r['task'], 'predicted pass, observed fail' if r['predicted'] else 'predicted fail, observed pass') for r in rows if r['predicted'] != r['observed']],
           'criteria': crit, 'branch': branch, 'retrospective': text.lstrip().startswith('# RETROSPECTIVE') or 'Retrodicts:' in text, 'complete': len(have) == len(tasks)}
    if source and all(r['predicted_source'] is not None for r in rows):   # the secondary rule, same criteria, decides nothing
        agree_s = sum(r['predicted_source'] == r['observed'] for r in rows); n_pred_s = sum(r['predicted_source'] for r in rows)
        late_s = [r for r in rows if r['lifted_resolved'] and not r['predicted_source']]
        res['source_rule'] = {'predicted': n_pred_s, 'interval': [source['lo'], source['hi']], 'agree': agree_s, 'min_agree': source['min_agree'],
                              'criteria': {'a_count': source['lo'] <= n_obs <= source['hi'], 'b_task_level': agree_s >= source['min_agree'], 'c1_all_fail': agree_s > all_fail,
                                           'c2_lifted_only': (agree_s > lifted_only) if late_s else None},
                              'late_solvers': [(r['task'], 'right' if not r['observed'] else 'wrong') for r in late_s],
                              'disagreements': [r['task'] for r in rows if r['predicted_source'] != r['observed']]}
    if a.json: print(json.dumps(res, indent=1)); return 0
    print(f"{n} of {len(tasks)} tasks observed{'' if res['complete'] else ' (INCOMPLETE: criteria are provisional)'}: predicted {n_pred}, observed {n_obs}, interval [{lo}, {hi}] -> (a) {crit['a_count']}")
    print(f"agreement {agree}/{n}, needs >= {min_agree} -> (b) {crit['b_task_level']}; all-fail baseline {all_fail} -> (c1) {crit['c1_all_fail']}; "
          f"lifted-only baseline {lifted_only} -> (c2) {'UNTESTABLE (no late solver)' if crit['c2_lifted_only'] is None else crit['c2_lifted_only']}")
    print(f"late solvers: {res['late_solvers']}; disagreements: {res['disagreements']}; branch {res['branch']}" + (' [retrospective test, not a forecast]' if res['retrospective'] else ''))
    if res.get('source_rule'):
        s = res['source_rule']
        print(f"source rule (secondary, decides nothing): predicted {s['predicted']}, interval {s['interval']}, agreement {s['agree']}/{n} (needs >= {s['min_agree']}), "
              f"criteria {s['criteria']}, late solvers {s['late_solvers']}, disagreements {s['disagreements']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('write'); w.add_argument('--tasks', default='fastapi'); w.add_argument('--clock', type=int, default=480); w.add_argument('--out', default=None)
    w.add_argument('--name', default=NAME); w.add_argument('--sign', action='store_true'); w.add_argument('--retrospective', action='store_true'); w.set_defaults(fn=write)
    e = sub.add_parser('evaluate'); e.add_argument('forecast'); e.add_argument('--json', action='store_true'); e.set_defaults(fn=evaluate)
    a = ap.parse_args(); return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
