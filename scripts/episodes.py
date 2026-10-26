#!/usr/bin/env python3
"""Loop calls and recovery episodes per attempt, from the trace JSON, by the rules fixed in PLAN.md on 2026-10-15
(before any calls-only or lifted data existed). Imported by regenerate.py; runnable alone for one record.

  edit               an edit_file call, or a write_file call to a path inside the repository (not under /tmp and not
                     absolute); the harness's result must not be an error (an edit_file whose old_string was not found
                     changed nothing).
  loop call          a tool call whose function name and arguments (canonical JSON: sorted keys, no whitespace) equal
                     those of an earlier call in the same attempt with no edit between the two. Malformed invocations
                     (the framework refused them) are compared like any other call: a repeated refused call is a loop.
  repetition-loop    an attempt with three or more loop calls (taxonomy class 3).
  recovery episode   a run_command whose result has a non-zero exit_code or an error status, followed by at least one
                     edit, followed by the same command string returning exit_code 0. The episode spans the calls from
                     the failing command to the passing one, inclusive. One failing command opens at most one episode
                     (the first passing repeat closes it); a later failure of the same command may open another.
  after call k       the calls with index > k; the tables report, among them, the share that are loop calls and the
                     share inside a recovery episode (PLAN.md: k = 12 per arm; k = 28 for the lifted subset, whose
                     extra calls are the ones the 28-call cap would have cut).

Usage: scripts/episodes.py records/<attempt>.json [--k 12]"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import curve  # noqa: E402

EDIT_TOOLS = {'edit_file', 'write_file'}


def is_edit(call: dict) -> bool:
    if call['tool'] not in EDIT_TOOLS or not call['ok']: return False
    if call['tool'] == 'write_file':
        p = str(call.get('path') or '')
        if not p or p.startswith('/') or p.startswith('tmp/') or '/tmp/' in p: return False
    return True


def canonical(call: dict) -> str:
    return json.dumps([call['tool'], call.get('arguments') or {}], sort_keys=True, separators=(',', ':'))


def analyse_calls(calls: list) -> dict:
    """calls: curve.trace_calls output (index, tool, ok, malformed, arguments, result)."""
    loop_idx, seen = [], {}
    for c in calls:
        key = canonical(c)
        if key in seen: loop_idx.append(c['index'])
        else: seen[key] = c['index']
        if is_edit(c): seen = {}   # an edit resets the window: a repeat after an edit is a retest, not a loop
    episodes, open_fail = [], {}   # command string -> index of the failing call that opened a candidate
    for c in calls:
        if c['tool'] != 'run_command': 
            if is_edit(c):
                for k in list(open_fail): open_fail[k] = (open_fail[k][0], True)
            continue
        cmd = str((c.get('arguments') or {}).get('command', ''))
        res = c.get('result') or {}
        failed = c['malformed'] or res.get('status') == 'error' or (res.get('exit_code') not in (None, 0))
        if failed:
            if cmd not in open_fail: open_fail[cmd] = (c['index'], False)
        elif cmd in open_fail:
            start, edited = open_fail.pop(cmd)
            if edited: episodes.append((start, c['index']))
    in_episode = set()
    for s, e in episodes: in_episode.update(range(s, e + 1))
    return {'loop_calls': loop_idx, 'recovery_episodes': episodes, 'in_episode': sorted(in_episode), 'n_calls': len(calls)}


def summarize(an: dict, k: int) -> dict:
    after = [i for i in range(k + 1, an['n_calls'] + 1)]
    return {'loop_calls': len(an['loop_calls']), 'repetition_loop': len(an['loop_calls']) >= 3, 'recovery_episodes': len(an['recovery_episodes']),
            'calls_after_k': len(after), 'loop_after_k': sum(1 for i in after if i in set(an['loop_calls'])),
            'in_episode_after_k': sum(1 for i in after if i in set(an['in_episode']))}


def for_record(record: dict, root: Path, k: int) -> dict:
    calls, _ = curve.trace_calls(record, root)
    return summarize(analyse_calls(calls), k)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('record'); ap.add_argument('--k', type=int, default=12); a = ap.parse_args()
    rec = json.loads(Path(a.record).read_text()); calls, _ = curve.trace_calls(rec, ROOT); an = analyse_calls(calls)
    print(json.dumps({'task': rec['task_id'], 'bundle': rec['bundle'], **an, 'summary': summarize(an, a.k)}, indent=1))
