#!/usr/bin/env python3
"""The ablation gate as fixed in DECISIONS (2026-11-02T21:20Z; the full arm 2026-11-03T00:45Z), from the ablate kernel's generations.jsonl and the swap v1
prefix selection. Stdlib only, deterministic; rates and intervals as scripts/swap_analysis.py.
  G1: repeat_previous, arm minus c1, on c1's own looping prefixes: the 90% interval entirely below 0.
  G2: edit, arm minus base, on the clean prefixes (E4B base and 31B sources): point estimate >= -0.15.
Both must hold to spend the evaluation arm. The other swap v1 contrasts are reported beside them, and for the full arm
the base-only ablation (basefull) against base.

Usage: scripts/ablate_gate.py <ablate output dir> [--arm c1abl|c1full] [--prefixes results/swap/datasets/20261102/prefixes.jsonl]"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from swap_analysis import OUTCOMES, contrast, mean_rate, rates  # noqa: E402

G2_FLOOR = -0.15


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--arm', default='c1abl', choices=('c1abl', 'c1full')); ap.add_argument('--prefixes', default='results/swap/datasets/20261102/prefixes.jsonl'); a = ap.parse_args()
    out = Path(a.out); gens = [json.loads(l) for l in (out / 'generations.jsonl').read_text().splitlines() if l.strip()]
    ids = {g['prefix_id'] for g in gens}
    prefixes = [p for p in (json.loads(l) for l in (ROOT / a.prefixes).read_text().splitlines()) if p['prefix_id'] in ids]
    c1_loop = [p for p in prefixes if p['source_bundle'] == 'bf18-e4b-lora-c1']; clean = [p for p in prefixes if p['source_kind'] == 'other']
    r, errors = rates(gens)
    arm = a.arm
    g1 = contrast(r, c1_loop, 'repeat_previous', lambda p: arm, lambda p: 'c1')
    g2 = contrast(r, clean, 'edit', lambda p: arm, lambda p: 'base')
    report = {'arm': arm, 'completions': len(gens), 'errors': errors, 'c1_looping_prefixes': len(c1_loop), 'clean_prefixes': len(clean),
              'G1': {**(g1 or {}), 'pass': bool(g1 and g1['hi'] < 0)}, 'G2': {**(g2 or {}), 'floor': G2_FLOOR, 'pass': bool(g2 and g2['diff'] >= G2_FLOOR)}}
    report['gate'] = report['G1']['pass'] and report['G2']['pass']
    groups = {'c1_looping': c1_loop, 'all_looping': [p for p in prefixes if p['source_kind'] == 'looping'], 'clean': clean}
    report['rates'] = {o: {g: {ex: mean_rate(r, ps, ex, o) for ex in ('base', 'c1', arm, *(('basefull',) if arm == 'c1full' else ()))} for g, ps in groups.items()} for o in OUTCOMES}
    report['secondary'] = {o: {'arm_minus_c1_clean': contrast(r, clean, o, lambda p: arm, lambda p: 'c1'),
                               'arm_minus_base_c1_looping': contrast(r, c1_loop, o, lambda p: arm, lambda p: 'base'),
                               'base_minus_c1_c1_looping': contrast(r, c1_loop, o, lambda p: 'base', lambda p: 'c1'),
                               **({'basefull_minus_base_clean': contrast(r, clean, o, lambda p: 'basefull', lambda p: 'base'),
                                   'basefull_minus_base_all_looping': contrast(r, groups['all_looping'], o, lambda p: 'basefull', lambda p: 'base')}
                                  if arm == 'c1full' else {})} for o in OUTCOMES}
    (out / f'gate-{arm}.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('arm', 'completions', 'errors', 'G1', 'G2', 'gate')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
