#!/usr/bin/env python3
"""Regenerate every number in the paper from records/*.json. Stdlib only, deterministic, no network.

Writes docs/results/tables.md (what the paper quotes) and docs/results/numbers.json (what the tables are built from).
Rules it applies are the ones in docs/PLAN.md "Analysis plan": validity by rule (lib/validity.py), terminal reasons
kept distinct, intent-to-run and valid-run summaries side by side, repeats pooled within task, cluster bootstrap
over tasks for intervals (90%, seed 20261013; every cohort, a set of tasks, draws from its own stream seeded by the seed
and the sorted task ids, so an interval depends only on the records that enter it, statistics on the same cohort share
their resamples, and an identical estimator on an identical cohort gives an identical interval), no p-values.

Usage: scripts/regenerate.py [--records records] [--out docs/results]"""
import argparse, hashlib, json, random, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.ai/tools/gemma-agent/lib'))
from validity import classify  # noqa: E402
from lane import summarize as lane_summary  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts'))
import curve  # noqa: E402  (calls-to-solve curve, pass rate versus clock, clock prediction; definitions in that file)
import episodes  # noqa: E402  (loop calls and recovery episodes by the PLAN.md rules of 2026-10-15; definitions in that file)

# The 4-bit lifted subset as declared (DECISIONS 2026-10-15 01:30 UTC): random.Random(20261015).sample over the sorted ids
# of the 41 first-attempt 4-bit cell failures on the 68-task set. Kept as data so the reading does not depend on the
# failure set staying what it was; the derivation is re-run and any difference is written into numbers.json.
LIFTED_SUBSET = sorted('fastapi_13920 fastapi_14258 fastapi_14306 fastapi_14616 fastapi_15661 fastapi_15800 rich_3043 rich_3105 rich_3278 '
                       'rich_3470 rich_3506 rich_3521 rich_3535 rich_3675 rich_3777 rich_3782 rich_3930 rich_3934 rich_3935 rich_4076'.split())
LIFTED_SEED, LIFTED_CLOCK_S, LIFTED_CALLS, CELL_CLOCK_S, CELL_CALLS = 20261015, 16 * 60, 40, 480, 28

SEED, B, CI = 20261013, 2000, 0.90
TERMINAL_ORDER = ['resolved', 'tests failed', 'no edit', 'call cap', 'turn cap', 'time cap', 'provider', 'sandbox', 'harness']


def terminal_reason(r: dict) -> str:
    """One of TERMINAL_ORDER, decided from record fields only."""
    o, b, v = r['outcome'], r['budget'], r.get('validity') or classify(r)
    if v == 'infrastructure_invalid': return 'sandbox' if 'docker' in str(o.get('error', '')).lower() or 'sandbox' in str(o.get('error', '')).lower() else 'harness'
    if v == 'environment_invalid': return 'sandbox'
    if v == 'provider_interrupted': return 'provider'
    if o.get('resolved'): return 'resolved'
    err = str(o.get('error') or '').lower()
    if 'timeout' in err or 'time' in err and 'exceed' in err: return 'time cap'
    if (o.get('tool_calls') or 0) >= int(b['max_tool_calls']): return 'call cap'
    if 'turn' in err: return 'turn cap'
    if o.get('duration_seconds') and o['duration_seconds'] >= float(b['max_time_minutes']) * 60 - 5: return 'time cap'
    if r.get('first_edit_call_index') is None: return 'no edit'
    return 'tests failed'


def ceiling_bound(r: dict) -> bool:
    """An API-lane attempt the lane's own time ceiling ended before the call cap; reported with and without."""
    return r['site'].startswith('local-docker') and terminal_reason(r) == 'time cap' and (r['outcome'].get('tool_calls') or 0) < int(r['budget']['max_tool_calls'])


def arm_key(r: dict) -> str:
    return f"{r['bundle']} @ {r['site']} ({int(r['budget']['max_time_minutes'])} min / {r['budget']['max_tool_calls']} calls)"


def stream(tasks) -> random.Random:
    """The resampling stream for one cohort: the shared seed and the sorted task ids, nothing else. Every statistic on the
    same cohort draws the same resamples, so D, matched calls and as scored are read on the same draws, and the same
    estimator on the same cohort gives the same interval whatever else was computed before it."""
    return random.Random(f'{SEED}:' + hashlib.sha256('\n'.join(sorted(tasks)).encode()).hexdigest())


def cluster_bootstrap(per_task: dict, stat):
    """per_task: task -> list of 0/1 outcomes. Resample tasks with replacement, pool repeats within task."""
    tasks = sorted(per_task); rng = stream(tasks)
    if not tasks: return None
    vals = []
    for _ in range(B):
        sample = [per_task[rng.choice(tasks)] for _ in tasks]
        vals.append(stat(sample))
    vals.sort(); lo = vals[int((1 - CI) / 2 * B)]; hi = vals[int((1 + CI) / 2 * B) - 1]
    return {'point': stat([per_task[t] for t in tasks]), 'lo': lo, 'hi': hi, 'n_tasks': len(tasks), 'n_attempts': sum(len(v) for v in per_task.values())}


def cohort_tasks(name: str) -> list:
    """The task list of one cohort in the signed cohort config (.ai/config/gemma-agent/cohorts.yaml), read without a YAML
    library: the `tasks: [...]` line inside the named block."""
    text = (ROOT / '.ai/config/gemma-agent/cohorts.yaml').read_text()
    import re
    m = re.search(rf'^  {re.escape(name)}:\n((?:    .*\n)+)', text, re.M)
    if not m: raise SystemExit(f'cohort {name} not found in cohorts.yaml')
    t = re.search(r'^    tasks: \[(.*?)\]', m.group(1), re.M | re.S)
    return [x.strip() for x in t.group(1).split(',') if x.strip()] if t else []


def mean_pass(samples):  # per-task pass rate averaged over repeats, then over tasks
    return statistics.mean(statistics.mean(s) for s in samples) if samples else 0.0


def md_table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--records', default=str(ROOT / 'records')); ap.add_argument('--out', default=str(ROOT / 'docs/results'))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    records = [json.load(open(p)) for p in sorted(Path(a.records).glob('*.json'))]
    excluded = {k: v for k, v in json.loads((ROOT / 'docs/results/excluded-runs.json').read_text()).items() if not k.startswith('_')} if (ROOT / 'docs/results/excluded-runs.json').exists() else {}
    for r in records:
        r['validity'] = classify(r)
        if any(r.get('results_dir', '').startswith(k) for k in excluded): r['validity'] = 'environment_invalid'   # docs/results/excluded-runs.json, by run
        r['_terminal'] = terminal_reason(r); r['_ceiling'] = ceiling_bound(r)
    numbers = {'seed': SEED, 'bootstrap_samples': B, 'interval': CI, 'records': len(records)}
    from datetime import datetime
    adapter_log = ROOT / 'results/adapter-log.jsonl'
    for r in records:   # token totals: carried by newer records, derived from the trace for older ones
        if 'tokens' not in r:
            tp = ROOT / r.get('results_dir', '') / 'traces' / f"trace_{r['task_id']}.json"
            if tp.exists():
                fm = (json.load(open(tp)).get('final_metrics') or {})
                r['tokens'] = {k: fm.get(f'total_{k}_tokens') for k in ('prompt', 'completion', 'cached')} if fm else None
    for r in records:   # API-lane availability per attempt: carried by newer records, derived from the adapter log for older ones
        if r['site'].startswith('local-docker') and 'lane' not in r and r.get('wall_seconds'):
            t0 = datetime.fromisoformat(r['started_at']).timestamp(); r['lane'] = lane_summary(adapter_log, t0, t0 + r['wall_seconds'])

    # T1: intent-to-run and valid runs per arm, with terminal reasons
    arms = defaultdict(list)
    for r in records: arms[arm_key(r)].append(r)
    t1 = []
    for arm, rs in sorted(arms.items()):
        valid = [r for r in rs if r['validity'] == 'valid']
        reasons = Counter(r['_terminal'] for r in rs)
        t1.append([arm, len(rs), len(valid), sum(1 for r in valid if r['outcome'].get('resolved')), sum(1 for r in rs if r['_ceiling']),
                   ', '.join(f'{k} {reasons[k]}' for k in TERMINAL_ORDER if reasons[k])])
    numbers['arms'] = {row[0]: {'intent': row[1], 'valid': row[2], 'resolved': row[3], 'ceiling_bound': row[4]} for row in t1}

    # T2: noise floor per task, with and without ceiling-bound attempts. The noise-floor cohort is the one PLAN.md names
    # (cohort 1: bf18 on the API lane, the 8 sound requests and httpx tasks, 5 repeats; read from the signed cohort config).
    # Until 2026-10-23 this pooled every valid bf18 API-lane attempt, so the proxy-fidelity rich attempts (cohort
    # proxy_fidelity_api, one attempt each) diluted it as they landed; those now have their own reading (C5).
    nf_tasks = set(cohort_tasks('noise_floor_requests'))
    nf = [r for r in records if r['validity'] == 'valid' and r['site'].startswith('local-docker') and r['bundle'] == 'bf18' and r['task_id'] in nf_tasks]
    per_task_all, per_task_clean = defaultdict(list), defaultdict(list)
    for r in nf:
        per_task_all[r['task_id']].append(int(bool(r['outcome'].get('resolved'))))
        if not r['_ceiling']: per_task_clean[r['task_id']].append(int(bool(r['outcome'].get('resolved'))))
    t2 = [[t, len(per_task_all[t]), sum(per_task_all[t]), len(per_task_clean.get(t, [])), sum(per_task_clean.get(t, []))] for t in sorted(per_task_all)]
    numbers['noise_floor'] = {'all': cluster_bootstrap(per_task_all, mean_pass), 'without_ceiling_bound': cluster_bootstrap(per_task_clean, mean_pass)}
    # within-task variance only exists once a task has 2+ valid attempts
    multi = {t: v for t, v in per_task_all.items() if len(v) >= 2}
    numbers['noise_floor']['tasks_with_repeats'] = len(multi)
    numbers['noise_floor']['tasks_with_mixed_outcomes'] = sum(1 for v in multi.values() if 0 < sum(v) < len(v))

    # T3: cost vector per arm over valid attempts (wall clock, calls); tokens and retries arrive with the adapter summary
    t3 = []
    for arm, rs in sorted(arms.items()):
        valid = [r for r in rs if r['validity'] == 'valid']
        if not valid: continue
        wall = [r['wall_seconds'] or r['outcome'].get('duration_seconds') or 0 for r in valid]
        calls = [r['outcome'].get('total_llm_calls') or 0 for r in valid]
        lanes = [r['lane'] for r in valid if r.get('lane')]
        retry = f"{statistics.median([l['retry_fraction'] for l in lanes]):.2f}" if lanes else 'n/a'
        sleep = f"{statistics.median([l['retry_sleep_seconds'] for l in lanes]) / 60:.0f}" if lanes else 'n/a'
        toks = f"{statistics.median([l['prompt_tokens'] for l in lanes]) / 1000:.0f}k" if lanes else 'n/a'
        t3.append([arm, len(valid), f'{statistics.median(wall) / 60:.0f}', f'{min(wall) / 60:.0f} to {max(wall) / 60:.0f}', f'{statistics.median(calls):.0f}', f'{min(calls)} to {max(calls)}', retry, sleep, toks])

    # T4: terminal reasons per arm, valid attempts only (the taxonomy's deterministic layer)
    t4_arms = sorted(a for a in arms if any(r['validity'] == 'valid' for r in arms[a]))
    t4 = []
    for reason in TERMINAL_ORDER:
        counts = [sum(1 for r in arms[a] if r['validity'] == 'valid' and r['_terminal'] == reason) for a in t4_arms]
        if any(counts): t4.append([reason, *counts])
    t4.append(['valid attempts', *[sum(1 for r in arms[a] if r['validity'] == 'valid') for a in t4_arms]])
    # tokens per valid attempt per arm
    t5 = []
    for a in t4_arms:
        toks = [r['tokens'] for r in arms[a] if r['validity'] == 'valid' and r.get('tokens') and r['tokens'].get('prompt') is not None]
        if toks: t5.append([a, len(toks), f"{statistics.median(t['prompt'] for t in toks) / 1000:.0f}k", f"{statistics.median(t['completion'] for t in toks) / 1000:.1f}k", f"{statistics.median((t.get('cached') or 0) for t in toks) / 1000:.0f}k"])

    # T6: loop calls and recovery episodes per arm (PLAN.md rules of 2026-10-15; scripts/episodes.py), valid attempts
    # with a trace; after call 12, the share of calls that are loop calls and the share inside a recovery episode.
    t6 = []; numbers['episodes_per_arm'] = {}
    for a in t4_arms:
        ss = []
        for r in arms[a]:
            if r['validity'] != 'valid': continue
            try: ss.append(episodes.for_record(r, ROOT, 12))
            except (FileNotFoundError, KeyError, json.JSONDecodeError): continue
        if not ss: continue
        after = sum(x['calls_after_k'] for x in ss)
        row = {'attempts_with_trace': len(ss), 'loop_calls': sum(x['loop_calls'] for x in ss), 'repetition_loop_attempts': sum(x['repetition_loop'] for x in ss),
               'recovery_episodes': sum(x['recovery_episodes'] for x in ss), 'attempts_with_recovery': sum(1 for x in ss if x['recovery_episodes']),
               'calls_after_12': after, 'loop_share_after_12': (sum(x['loop_after_k'] for x in ss) / after) if after else None,
               'episode_share_after_12': (sum(x['in_episode_after_k'] for x in ss) / after) if after else None}
        numbers['episodes_per_arm'][a] = row
        t6.append([a, row['attempts_with_trace'], row['loop_calls'], row['repetition_loop_attempts'], row['recovery_episodes'], row['attempts_with_recovery'], row['calls_after_12'],
                   '-' if row['loop_share_after_12'] is None else f"{row['loop_share_after_12']:.2f}", '-' if row['episode_share_after_12'] is None else f"{row['episode_share_after_12']:.2f}"])

    # primary endpoint D (PLAN.md, revised 2026-10-15): what the clock takes from the precision contrast.
    #   D = (P_bf16,calls - P_4bit,calls) - (P_bf16,cell - P_4bit,cell), per task, averaged over the eligible tasks.
    # P_4bit,calls for a task is its calls-only rerun when one exists, else its 4-bit cell attempts that did not end at the
    # clock (a 30-min clock leaves those unchanged). A task is eligible when all four quantities exist. Interim while the
    # bf16 calls-only arm is landing: n is stated, and the two brackets are reported on their own as well.
    gpu = [r for r in records if r['site'] == 'kaggle-l4x4' and r['validity'] == 'valid']
    def outcomes(bundle, pred=lambda r: True):
        d = {}
        for r in gpu:
            if r['bundle'] == bundle and pred(r): d.setdefault(r['task_id'], []).append(int(bool(r['outcome'].get('resolved'))))
        return d
    cell4, cell16 = outcomes('bf18'), outcomes('bf18-bf16')
    calls16, calls4_rerun = outcomes('bf18-bf16-calls'), outcomes('bf18-calls')
    cell4_not_clock = outcomes('bf18', lambda r: terminal_reason(r) != 'time cap')
    calls4 = {t: calls4_rerun.get(t) or cell4_not_clock.get(t) for t in cell4}
    calls4 = {t: v for t, v in calls4.items() if v}
    eligible = sorted(set(cell4) & set(cell16) & set(calls16) & set(calls4))
    numbers['grid_cells_with_data'] = {'4bit_cell': len(cell4), 'bf16_cell': len(cell16), 'bf16_calls': len(calls16), '4bit_calls_rerun': len(calls4_rerun), 'eligible_for_D': len(eligible)}
    numbers['bf16_calls_clock_bound'] = sum(1 for r in gpu if r['bundle'] == 'bf18-bf16-calls' and terminal_reason(r) == 'time cap')
    numbers['primary_endpoint_D'] = None
    if len(eligible) >= 5:
        m = statistics.mean
        def numerics(t): return m(calls16[t]) - m(calls4[t])
        def as_scored(t): return m(cell16[t]) - m(cell4[t])
        def boot(stat, tasks):
            tasks = sorted(tasks); rng = stream(tasks); vals = sorted(m(stat(t) for t in [rng.choice(tasks) for _ in tasks]) for _ in range(B))
            return {'point': m(stat(t) for t in tasks), 'lo': vals[int((1 - CI) / 2 * B)], 'hi': vals[int((1 + CI) / 2 * B) - 1], 'n_tasks': len(tasks)}
        numbers['primary_endpoint_D'] = boot(lambda t: numerics(t) - as_scored(t), eligible)
        numbers['contrast_matched_calls'] = boot(numerics, eligible)
        numbers['contrast_as_scored'] = boot(as_scored, eligible)
        rich = [t for t in eligible if t.startswith('rich_')]
        numbers['primary_endpoint_D_rich'] = boot(lambda t: numerics(t) - as_scored(t), rich) if len(rich) >= 5 else None
        # sensitivity, not the endpoint: the first attempt per task in each quantity instead of repeats pooled within task
        def first(t): return (calls16[t][0] - calls4[t][0]) - (cell16[t][0] - cell4[t][0])
        numbers['primary_endpoint_D_first_attempts'] = boot(first, eligible)

    # calls-to-solve curve, pass rate versus clock, clock prediction (PLAN.md analysis plan, primary analysis A2 and A3,
    # added 2026-10-19; definitions in scripts/curve.py). One point per task per bundle (first attempt), 68-task set.
    pts = curve.points_by_bundle(records, ROOT)
    numbers['curve'] = {}; curve_lines = ['## C1. Calls to solve and pass rate versus clock, per arm (first attempt per task, 68-task set)', '',
                                          'S(k): share of tasks resolved with the last edit at call <= k. P(C): share resolved with the last edit done by C seconds. '
                                          'Solve call and solve time are medians over resolved attempts; s/call is the median gap between consecutive model responses.', '']
    c1 = []
    for b, plist in pts.items():
        one = curve.one_per_task(plist); solved = [p for p in one.values() if p['resolved']]
        S, P = curve.s_of_k(one), curve.p_of_c(one)
        numbers['curve'][b] = {'tasks': len(one), 'resolved': len(solved), 'median_calls_used': statistics.median(p['calls_used'] for p in one.values()),
                               'median_s_per_call': statistics.median(p['median_call_s'] for p in one.values() if p['median_call_s']),
                               'solve_call_median': statistics.median(p['solve_call'] for p in solved) if solved else None,
                               'solve_time_median_s': statistics.median(p['solve_time'] for p in solved) if solved else None,
                               'solve_time_max_s': max(p['solve_time'] for p in solved) if solved else None,
                               'S': {str(k): round(v, 4) for k, v in S.items()}, 'P': {str(c): round(v, 4) for c, v in P.items()}}
        c1.append([b, len(one), len(solved), f"{numbers['curve'][b]['median_calls_used']:.0f}", f"{numbers['curve'][b]['median_s_per_call']:.1f}",
                   f"{numbers['curve'][b]['solve_call_median']:.0f}" if solved else '-', f"{numbers['curve'][b]['solve_time_median_s'] / 60:.1f}" if solved else '-',
                   ' '.join(f"{S[k]:.2f}" for k in (8, 12, 16, 20, 24, 28)), ' '.join(f"{P[c]:.2f}" for c in (180, 300, 480, 720, 1200, 1800))])
    curve_lines += [md_table(['arm (bundle)', 'tasks', 'resolved', 'median calls', 's/call', 'solve call', 'solve min', 'S(k) at k = 8 12 16 20 24 28', 'P(C) at C = 3 5 8 12 20 30 min'], c1), '']
    (out / 'curve.csv').write_text('bundle,task,resolved,calls_used,calls_executed,failed_invocations,solve_call,solve_time_s,submit_time_s,end_time_s,clock_s,wall_s,median_call_s\n' + ''.join(
        f"{p['bundle']},{p['task']},{int(p['resolved'])},{p['calls_used']},{p['calls_executed']},{p['failed_invocations']},{p['solve_call'] if p['solve_call'] is not None else ''},"
        f"{'' if p['solve_time'] is None else round(p['solve_time'], 1)},{'' if p['submit_time'] is None else round(p['submit_time'], 1)},"
        f"{'' if p['end_time'] is None else round(p['end_time'], 1)},{p['clock_s']:.0f},{p['wall_s']},{'' if p['median_call_s'] is None else round(p['median_call_s'], 2)}\n"
        for plist in pts.values() for p in curve.one_per_task(plist).values()))
    numbers['clock_prediction'] = None
    if 'bf18-bf16-calls' in pts and 'bf18-bf16' in pts:
        pr = curve.predict(curve.one_per_task(pts['bf18-bf16-calls']), curve.one_per_task(pts['bf18-bf16']), 480)
        numbers['clock_prediction'] = pr
        ps = curve.predict(curve.one_per_task(pts['bf18-bf16-calls']), curve.one_per_task(pts['bf18-bf16']), 480, time_key='solve_time_source')
        numbers['clock_prediction_source_rule'] = ps
        curve_lines += ['## C2. Clock prediction: bf16 cell outcome predicted per task from the bf16 calls-only attempt (pass iff resolved and solve time <= 480 s)', '',
                        f"{pr['n_tasks']} tasks with both attempts: predicted {pr['predicted_passes']} passes, observed {pr['observed_passes']}; per-task agreement {pr['agree']}/{pr['n_tasks']} "
                        f"(both {pr['both']}, predicted only {pr['predicted_only']}, observed only {pr['observed_only']}, neither {pr['neither']}). "
                        f"Baselines: all-fail predictor agrees on {pr['baseline_all_fail_agree']}/{pr['n_tasks']}; lifted-outcome-only predictor (no clock rule) on {pr['baseline_lifted_only_agree']}/{pr['n_tasks']}.", '',
                        f"Source rule (secondary, declared 2026-10-20 after this retrodiction and before any fastapi cell data; solve time over edits under the repository's "
                        f"source directories only, curve.REPO_SOURCE; fastapi's gained docs_src/ on 2026-10-20 after inspection of lifted fastapi outcomes, see the forecast): predicted {ps['predicted_passes']}, observed {ps['observed_passes']}; agreement {ps['agree']}/{ps['n_tasks']}; late solvers "
                        f"{', '.join(r['task'] for r in ps['rows'] if r['lifted_resolved'] and not r['predicted']) or 'none'}. On rich this is a post-hoc reading; the fastapi forecast carries both rules.", '',
                        md_table(['task', 'lifted resolved', 'lifted solve call', 'lifted solve time (s)', 'predicted', 'solve time, source (s)', 'predicted, source', 'observed', 'cell calls', 'cell at clock'],
                                 [[r['task'], r['lifted_resolved'], r['lifted_solve_call'] or '-', r['lifted_solve_time'] or '-', r['predicted'], r['lifted_solve_time_source'] or '-', s['predicted'], r['observed'], r['cell_calls'], r['cell_clock_bound']] for r, s in zip(pr['rows'], ps['rows'])]), '']

    # C3. Size (PLAN.md secondary endpoints; cohort 3): 12B 4-bit at the cell beside 31B 4-bit at the cell on the same rich
    # tasks, first attempt per task. Seconds per call, calls reached, pass rate, terminal reasons, per-task agreement, and
    # the paired pass-rate difference with its interval (descriptive: one point, one repeat; the paper makes no capability
    # claim from it). Written 2026-10-23 before any 12B record existed.
    def first_by_task(bundle, prefix=''):
        d = {}
        for r in sorted(gpu, key=lambda r: r['attempt_id']):
            if r['bundle'] == bundle and r['task_id'].startswith(prefix) and r['task_id'] in curve.grid_tasks(ROOT): d.setdefault(r['task_id'], r)
        return d
    def arm_summary(recs_by_task, pts_by_task):
        res = [r for r in recs_by_task.values() if r['outcome'].get('resolved')]
        reasons = Counter(r['_terminal'] for r in recs_by_task.values())
        spc = [p['median_call_s'] for p in pts_by_task.values() if p.get('median_call_s')]
        return {'tasks': len(recs_by_task), 'resolved': len(res), 'pass_rate': len(res) / len(recs_by_task) if recs_by_task else None,
                'median_s_per_call': statistics.median(spc) if spc else None, 'median_calls_used': statistics.median(p['calls_used'] for p in pts_by_task.values()) if pts_by_task else None,
                'median_wall_s': statistics.median(r['wall_seconds'] or 0 for r in recs_by_task.values()) if recs_by_task else None,
                'terminal_reasons': {k: reasons[k] for k in TERMINAL_ORDER if reasons[k]}}
    size_lines = ['## C3. Size: 12B 4-bit at the cell beside 31B 4-bit at the cell, rich tasks, first attempt per task', '']
    r12, r31 = first_by_task('bf18-12b', 'rich_'), first_by_task('bf18', 'rich_')
    numbers['size'] = None
    if r12:
        common = sorted(set(r12) & set(r31)); r31c = {t: r31[t] for t in common}; r12c = {t: r12[t] for t in common}
        p12 = {t: p for t, p in curve.one_per_task(pts.get('bf18-12b', [])).items() if t in r12c}; p31 = {t: p for t, p in curve.one_per_task(pts.get('bf18', [])).items() if t in r31c}
        s12, s31 = arm_summary(r12c, p12), arm_summary(r31c, p31)
        o12 = {t: [int(bool(r12c[t]['outcome'].get('resolved')))] for t in common}; o31 = {t: [int(bool(r31c[t]['outcome'].get('resolved')))] for t in common}
        both = sum(1 for t in common if o12[t][0] and o31[t][0]); only12 = sum(1 for t in common if o12[t][0] and not o31[t][0]); only31 = sum(1 for t in common if o31[t][0] and not o12[t][0])
        diff = None
        if len(common) >= 5:
            rng = stream(common); m = statistics.mean
            vals = sorted(m(o12[t][0] - o31[t][0] for t in [rng.choice(common) for _ in common]) for _ in range(B))
            diff = {'point': m(o12[t][0] - o31[t][0] for t in common), 'lo': vals[int((1 - CI) / 2 * B)], 'hi': vals[int((1 + CI) / 2 * B) - 1], 'n_tasks': len(common)}
        numbers['size'] = {'tasks': common, '12b': s12, '31b': s31, 'agreement': {'both_pass': both, '12b_only': only12, '31b_only': only31, 'neither': len(common) - both - only12 - only31},
                           'pass_rate_difference_12b_minus_31b': diff}
        size_lines += [md_table(['arm', 'tasks', 'resolved', 'pass rate', 'median s/call', 'median calls', 'median wall (min)', 'terminal reasons'],
                                [[lbl, x['tasks'], x['resolved'], f"{x['pass_rate']:.2f}", f"{x['median_s_per_call']:.1f}" if x['median_s_per_call'] else '-', f"{x['median_calls_used']:.0f}" if x['median_calls_used'] is not None else '-',
                                  f"{x['median_wall_s'] / 60:.1f}" if x['median_wall_s'] else '-', ', '.join(f'{k} {v}' for k, v in x['terminal_reasons'].items())] for lbl, x in (('12B 4-bit (bf18-12b)', s12), ('31B 4-bit (bf18)', s31))]), '',
                       f"Per-task agreement on {len(common)} tasks: both pass {both}, 12B only {only12}, 31B only {only31}, neither {len(common) - both - only12 - only31}."
                       + (f" Pass-rate difference 12B minus 31B = {diff['point']:+.3f} [{diff['lo']:+.3f}, {diff['hi']:+.3f}] (one attempt per task per arm; descriptive)." if diff else ''), '']
    else: size_lines += ['No 12B records yet (cohort size_12b_cell, bundle bf18-12b).', '']

    # C4. The cap (PLAN.md secondary endpoints; G4 recovery): 31B 4-bit at the lifted cell (16 min, 40 calls) on the declared
    # subset of 20 first-attempt 4-bit cell failures. Passes; passes whose solve call is beyond 28 or whose solve time is beyond
    # 480 s (what the cap and the clock would have cut); terminal reasons; extra calls (index > 28) split into loop calls and
    # recovery episodes; S(k) for k = 28..40 over the 20. Recovery net of reruns: the lifted pass rate among the 20 minus the
    # cell rerun pass rate among first-attempt cell failures, read from the GPU-lane repeats (the 4-bit cell's second attempt on
    # the 10-task subset) as PLAN.md says. Written 2026-10-23 before any lifted record existed.
    lifted_lines = ['## C4. The cap: 31B 4-bit at the lifted cell (16 min / 40 calls) on the declared subset of 20 cell failures', '']
    cell4_first = first_by_task('bf18')
    fail_now = sorted(t for t, r in cell4_first.items() if not r['outcome'].get('resolved'))
    rederived = sorted(random.Random(LIFTED_SEED).sample(fail_now, 20)) if len(fail_now) >= 20 else None
    numbers['lifted_subset'] = {'declared': LIFTED_SUBSET, 'first_attempt_cell_failures_now': len(fail_now), 'rederived_equals_declared': rederived == LIFTED_SUBSET}
    lifted_lines.append(f"Declared subset (seed {LIFTED_SEED} over the sorted first-attempt 4-bit cell failures, DECISIONS 2026-10-15): {', '.join(LIFTED_SUBSET)}. "
                        f"Re-derived from the records now ({len(fail_now)} first-attempt failures): {'identical' if rederived == LIFTED_SUBSET else 'DIFFERS: ' + ', '.join(sorted(set(rederived or []) ^ set(LIFTED_SUBSET)))}.")
    # cell rerun baseline among cell failures, from the GPU-lane repeats
    rr = defaultdict(list)
    for r in sorted(gpu, key=lambda r: r['attempt_id']):
        if r['bundle'] == 'bf18' and r['task_id'] in cell4_first and r['attempt_id'] != cell4_first[r['task_id']]['attempt_id']: rr[r['task_id']].append(int(bool(r['outcome'].get('resolved'))))
    rerun_fail = {t: v for t, v in rr.items() if not cell4_first[t]['outcome'].get('resolved')}
    rerun_rate = (sum(statistics.mean(v) for v in rerun_fail.values()) / len(rerun_fail)) if rerun_fail else None
    numbers['lifted_subset']['cell_rerun_among_failures'] = {'tasks': sorted(rerun_fail), 'passes': sum(int(any(v)) for v in rerun_fail.values()), 'rate': rerun_rate}
    lifted_lines.append(f"Cell rerun baseline (4-bit cell second attempts on first-attempt failures, GPU-lane repeats): {sum(int(any(v)) for v in rerun_fail.values())} of {len(rerun_fail)} passed"
                        + (f" ({rerun_rate:.2f})." if rerun_rate is not None else '.'))
    lifted_lines.append('')
    lf = {t: r for t, r in first_by_task('bf18-lifted').items() if t in LIFTED_SUBSET}
    if lf:
        lp = {t: p for t, p in curve.one_per_task(pts.get('bf18-lifted', [])).items() if t in lf}
        summ = arm_summary(lf, lp)
        solved = [p for p in lp.values() if p['resolved']]
        beyond_calls = [p['task'] for p in solved if p['solve_call'] is not None and p['solve_call'] > CELL_CALLS]
        executed_index = {p['task']: p['solve_call_executed'] for p in solved}
        beyond_clock = [p['task'] for p in solved if p['solve_time'] is not None and p['solve_time'] > CELL_CLOCK_S]
        within_both = [p['task'] for p in solved if p['solve_call'] is not None and p['solve_call'] <= CELL_CALLS and p['solve_time'] is not None and p['solve_time'] <= CELL_CLOCK_S]
        ep = {}
        for t, r in lf.items():
            try: ep[t] = episodes.for_record(r, ROOT, CELL_CALLS)
            except (FileNotFoundError, KeyError, json.JSONDecodeError): continue
        extra = sum(x['calls_after_k'] for x in ep.values()); extra_loop = sum(x['loop_after_k'] for x in ep.values()); extra_ep = sum(x['in_episode_after_k'] for x in ep.values())
        Sk = {k: sum(1 for p in lp.values() if p['resolved'] and p['solve_call'] is not None and p['solve_call'] <= k) for k in range(CELL_CALLS, LIFTED_CALLS + 1, 2)}
        lifted_rate = summ['pass_rate']
        numbers['lifted_subset'].update({'tasks_run': sorted(lf), 'summary': summ, 'passes': summ['resolved'], 'passes_solve_call_beyond_28': beyond_calls, 'passes_solve_call_executed': executed_index, 'passes_solve_time_beyond_480': beyond_clock,
                                         'passes_within_cell_budget': within_both, 'extra_calls': {'n': extra, 'loop_calls': extra_loop, 'in_recovery_episode': extra_ep},
                                         'loop_calls_total': sum(x['loop_calls'] for x in ep.values()), 'recovery_episodes_total': sum(x['recovery_episodes'] for x in ep.values()),
                                         'repetition_loop_attempts': sum(x['repetition_loop'] for x in ep.values()),
                                         'S_count_by_k': {str(k): v for k, v in Sk.items()}, 'recovery_net_of_rerun': (lifted_rate - rerun_rate) if (lifted_rate is not None and rerun_rate is not None) else None})
        lifted_lines += [md_table(['arm', 'tasks', 'resolved', 'pass rate', 'median s/call', 'median calls', 'median wall (min)', 'terminal reasons'],
                                  [['31B 4-bit lifted (bf18-lifted)', summ['tasks'], summ['resolved'], f"{summ['pass_rate']:.2f}", f"{summ['median_s_per_call']:.1f}" if summ['median_s_per_call'] else '-',
                                    f"{summ['median_calls_used']:.0f}" if summ['median_calls_used'] is not None else '-', f"{summ['median_wall_s'] / 60:.1f}" if summ['median_wall_s'] else '-', ', '.join(f'{k} {v}' for k, v in summ['terminal_reasons'].items())]]), '',
                         f"Passes {summ['resolved']} of {summ['tasks']}: solve call beyond 28 on {len(beyond_calls)} ({', '.join(beyond_calls) or 'none'}); solve time beyond 480 s on {len(beyond_clock)} ({', '.join(beyond_clock) or 'none'}); "
                         f"inside both cell limits on {len(within_both)} ({', '.join(within_both) or 'none'}: passes a plain cell rerun could have produced).",
                         f"Extra calls (index > 28) over the {len(ep)} attempts with a trace: {extra}; loop calls {extra_loop}; inside a recovery episode {extra_ep}. "
                         f"Loop calls in all {sum(x['loop_calls'] for x in ep.values())}, repetition-loop attempts {sum(x['repetition_loop'] for x in ep.values())}, recovery episodes {sum(x['recovery_episodes'] for x in ep.values())}.",
                         'Tasks solved with solve call <= k, k = ' + ' '.join(f'{k}:{v}' for k, v in Sk.items()) + '.',
                         (f"Recovery net of reruns: lifted pass rate {lifted_rate:.2f} minus cell rerun rate among failures {rerun_rate:.2f} = {lifted_rate - rerun_rate:+.2f}." if rerun_rate is not None else 'Recovery net of reruns: no cell rerun baseline yet.'), '']
    else: lifted_lines += ['No lifted-subset records yet (bundle bf18-lifted on the declared 20).', '']

    # C5. Proxy fidelity (PLAN.md secondary endpoints; cohort proxy_fidelity_api): the 4-bit bundle on the API lane, one attempt
    # per task, against its first GPU-lane cell attempt on the same task. The lane differs in sandbox, serving and clock
    # (120-min ceiling; the clock does not bind), so this reads how far the cheap lane predicts the cell, per task.
    proxy_tasks = set(cohort_tasks('proxy_fidelity_api'))
    api_first = {}
    for r in sorted(records, key=lambda r: r['attempt_id']):
        if r['validity'] == 'valid' and r['site'].startswith('local-docker') and r['bundle'] == 'bf18' and r['task_id'] in proxy_tasks: api_first.setdefault(r['task_id'], r)
    proxy_lines = ['## C5. Proxy fidelity: the 4-bit bundle on the API lane (one attempt per task) against its first 4-bit cell attempt on the GPU lane', '']
    numbers['proxy_fidelity'] = None
    if api_first:
        cell_first = first_by_task('bf18')
        common = sorted(set(api_first) & set(cell_first))
        oa = {t: int(bool(api_first[t]['outcome'].get('resolved'))) for t in common}; oc = {t: int(bool(cell_first[t]['outcome'].get('resolved'))) for t in common}
        both = sum(1 for t in common if oa[t] and oc[t]); api_only = sum(1 for t in common if oa[t] and not oc[t]); cell_only = sum(1 for t in common if oc[t] and not oa[t])
        ceiling = sum(1 for t in common if api_first[t]['_ceiling'])
        numbers['proxy_fidelity'] = {'cohort_tasks': len(proxy_tasks), 'attempted': len(api_first), 'api_resolved': sum(1 for r in api_first.values() if r['outcome'].get('resolved')),
                                     'compared': len(common), 'both_pass': both, 'api_only': api_only, 'cell_only': cell_only, 'neither': len(common) - both - api_only - cell_only,
                                     'agreement': (both + len(common) - both - api_only - cell_only) / len(common) if common else None, 'api_ceiling_bound_among_compared': ceiling,
                                     'api_terminal_reasons': dict(Counter(r['_terminal'] for r in api_first.values()))}
        pf = numbers['proxy_fidelity']
        proxy_lines += [f"Cohort of {pf['cohort_tasks']} tasks, {pf['attempted']} attempted, {pf['api_resolved']} resolved on the API lane; terminal reasons {', '.join(f'{k} {v}' for k, v in pf['api_terminal_reasons'].items())}.",
                        f"Per-task against the first 4-bit cell attempt on {pf['compared']} tasks: both pass {pf['both_pass']}, API only {pf['api_only']}, cell only {pf['cell_only']}, neither {pf['neither']}; agreement "
                        + (f"{pf['agreement']:.2f}" if pf['agreement'] is not None else '-') + f"; API attempts ended by the lane's 120-min ceiling among them: {pf['api_ceiling_bound_among_compared']}.", '']
    else: proxy_lines += ['No proxy-fidelity records yet.', '']

    # write
    lines = ['# Regenerated tables', '', f'Generated by `scripts/regenerate.py` from {len(records)} records. Seed {SEED}, {B} bootstrap resamples, {int(CI * 100)}% intervals, tasks as clusters, one resampling stream per cohort. No p-values.', '',
             '## T1. Intent to run and valid runs, per arm', '', md_table(['arm', 'intent', 'valid', 'resolved', 'ceiling-bound', 'terminal reasons (all attempts)'], t1), '',
             '## T2. Noise floor per task (cohort noise_floor_requests: bf18, API lane, 8 tasks): attempts and passes, all valid and without ceiling-bound', '', md_table(['task', 'valid attempts', 'passes', 'without ceiling-bound', 'passes'], t2), '']
    for k, lbl in (('all', 'all valid attempts'), ('without_ceiling_bound', 'without ceiling-bound attempts')):
        s = numbers['noise_floor'][k]
        if s: lines.append(f"Pass rate, {lbl}: {s['point']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}] over {s['n_tasks']} tasks, {s['n_attempts']} attempts.")
    lines += [f"Tasks with 2+ valid attempts: {numbers['noise_floor']['tasks_with_repeats']}; with mixed outcomes: {numbers['noise_floor']['tasks_with_mixed_outcomes']}.", '',
              '## T3. Cost vector per arm (valid attempts; retry fraction = upstream retries / (completed calls + retries), API lane only)', '', md_table(['arm', 'n', 'median wall (min)', 'wall range', 'median calls', 'calls range', 'median retry fraction', 'median retry sleep (min)', 'median prompt tokens'], t3), '',
              '## T4. Terminal reasons per arm (valid attempts)', '', md_table(['terminal reason', *t4_arms], t4), '',
              '## T5. Tokens per valid attempt (median; from the trace)', '', md_table(['arm', 'n', 'prompt', 'completion', 'cached'], t5), '',
              '## T6. Loop calls and recovery episodes per arm (valid attempts with a trace; rules in scripts/episodes.py from PLAN.md 2026-10-15)', '',
              md_table(['arm', 'attempts', 'loop calls', 'repetition-loop attempts', 'recovery episodes', 'attempts with recovery', 'calls after 12', 'loop share after 12', 'in-episode share after 12'], t6), '',
              '## Primary endpoint D', '', *(['Not computable yet (fewer than 5 eligible tasks): ' + json.dumps(numbers['grid_cells_with_data'])] if not numbers['primary_endpoint_D'] else
              [f"Interim, {numbers['primary_endpoint_D']['n_tasks']} eligible tasks (all four quantities present): " + json.dumps(numbers['grid_cells_with_data']) + f"; bf16 calls-only attempts that reached the 30-min clock: {numbers['bf16_calls_clock_bound']}.", '',
               f"D = {numbers['primary_endpoint_D']['point']:+.3f} [{numbers['primary_endpoint_D']['lo']:+.3f}, {numbers['primary_endpoint_D']['hi']:+.3f}]",
               f"matched calls (bf16 - 4bit) = {numbers['contrast_matched_calls']['point']:+.3f} [{numbers['contrast_matched_calls']['lo']:+.3f}, {numbers['contrast_matched_calls']['hi']:+.3f}]",
               f"as scored (bf16 - 4bit) = {numbers['contrast_as_scored']['point']:+.3f} [{numbers['contrast_as_scored']['lo']:+.3f}, {numbers['contrast_as_scored']['hi']:+.3f}]"] +
              ([f"rich only: D = {numbers['primary_endpoint_D_rich']['point']:+.3f} [{numbers['primary_endpoint_D_rich']['lo']:+.3f}, {numbers['primary_endpoint_D_rich']['hi']:+.3f}] over {numbers['primary_endpoint_D_rich']['n_tasks']} tasks"] if numbers.get('primary_endpoint_D_rich') else [])), '']
    lines += ['', *curve_lines, *size_lines, *lifted_lines, *proxy_lines]
    (out / 'tables.md').write_text('\n'.join(lines)); (out / 'numbers.json').write_text(json.dumps(numbers, indent=2, sort_keys=True))
    print(f'wrote {out / "tables.md"} and numbers.json from {len(records)} records'); return 0


if __name__ == '__main__':
    sys.exit(main())
