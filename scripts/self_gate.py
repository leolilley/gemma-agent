#!/usr/bin/env python3
"""The cycle 4 (self-distillation) gate as fixed in DECISIONS (2026-11-04T06:32Z), from the swap kernel's generations.jsonl
and the swap v1 prefix selection. Stdlib only, deterministic; rates and intervals as scripts/swap_analysis.py.
  G1: repeat_previous, c4 minus base, on the clean prefixes (E4B base and 31B sources): 90% interval upper bound <= +0.05.
  G2: edit, c4 minus base, on the clean prefixes: point estimate >= -0.15.
Both must hold to spend the evaluation arm. Secondary, reported: repeat_previous c4 minus c2 on the looping prefixes.
Rates of every executor in the run are reported beside them.

Usage: scripts/self_gate.py <swap output dir> [--arm c4] [--prefixes results/swap/datasets/20261102/prefixes.jsonl]"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from swap_analysis import OUTCOMES, contrast, mean_rate, rates  # noqa: E402

G1_CEILING = 0.05
G2_FLOOR = -0.15


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--arm', default='c4'); ap.add_argument('--prefixes', default='results/swap/datasets/20261102/prefixes.jsonl'); a = ap.parse_args()
    out = Path(a.out); gens = [json.loads(l) for l in (out / 'generations.jsonl').read_text().splitlines() if l.strip()]
    ids = {g['prefix_id'] for g in gens}; executors = sorted({g['executor'] for g in gens})
    prefixes = [p for p in (json.loads(l) for l in (ROOT / a.prefixes).read_text().splitlines()) if p['prefix_id'] in ids]
    looping = [p for p in prefixes if p['source_kind'] == 'looping']; clean = [p for p in prefixes if p['source_kind'] == 'other']
    r, errors = rates(gens)
    arm = a.arm
    g1 = contrast(r, clean, 'repeat_previous', lambda p: arm, lambda p: 'base')
    g2 = contrast(r, clean, 'edit', lambda p: arm, lambda p: 'base')
    report = {'arm': arm, 'completions': len(gens), 'errors': errors, 'executors': executors, 'looping_prefixes': len(looping), 'clean_prefixes': len(clean),
              'G1': {**(g1 or {}), 'ceiling': G1_CEILING, 'pass': bool(g1 and g1['hi'] <= G1_CEILING)},
              'G2': {**(g2 or {}), 'floor': G2_FLOOR, 'pass': bool(g2 and g2['diff'] >= G2_FLOOR)}}
    report['gate'] = report['G1']['pass'] and report['G2']['pass']
    groups = {'looping': looping, 'c2_looping': [p for p in looping if p['source_bundle'] == 'bf18-e4b-lora-c2'], 'clean': clean,
              'clean_e4b': [p for p in clean if 'e4b' in p['source_bundle']], 'clean_31b': [p for p in clean if 'e4b' not in p['source_bundle']]}
    report['rates'] = {o: {g: {ex: mean_rate(r, ps, ex, o) for ex in executors} for g, ps in groups.items()} for o in OUTCOMES}
    report['secondary'] = {'repeat_arm_minus_c2_looping': contrast(r, looping, 'repeat_previous', lambda p: arm, lambda p: 'c2'),
                           **{o: {'arm_minus_base_clean': contrast(r, clean, o, lambda p: arm, lambda p: 'base'),
                                  'arm_minus_c2_clean': contrast(r, clean, o, lambda p: arm, lambda p: 'c2'),
                                  'arm_minus_base_looping': contrast(r, looping, o, lambda p: arm, lambda p: 'base'),
                                  'arm_minus_c2_looping': contrast(r, looping, o, lambda p: arm, lambda p: 'c2')} for o in OUTCOMES}}
    (out / f'gate-{arm}.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('arm', 'completions', 'errors', 'G1', 'G2', 'gate', 'rates')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
