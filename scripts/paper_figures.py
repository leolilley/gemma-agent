#!/usr/bin/env python3
"""Draw the loop paper's three figures as plain SVG, from recorded files only (no plotting library, like figures.py).

  P1  share of tool calls that repeat the previous call, per bundle, over its 16-task evaluation run at the cell:
      counted from the evaluation traces named by each cycle's accept.json (the base run is the base arm of cycle 1).
      Calls are counted within the tool-call budget: up to the first call the harness refuses (P3 likewise).
  P2  the frontier swap: who wrote the history against who reads it, share of next calls repeating the history's last
      call, from the generations and prefixes of results/swap/runs/v1-20261102 (the interval from its analysis.json).
  P3  the chance that the next call repeats the previous one, by how many identical calls already end the history,
      in the evaluation rollouts of the adapters trained on the teacher's histories (cycles 1, 2, 3, 5) and of the base.

Usage: scripts/paper_figures.py [--out docs/results/figures]     (PNG: rsvg-convert -z 2 <svg> -o <png>)
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONT = 'font-family="Helvetica, Arial, sans-serif"'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
# sequential blue, light to dark, for the swap cells
RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab']


REFUSED = 'Tool call budget exhausted'


def calls(trace: dict, within_budget: bool = True) -> list:
    """The attempt's tool calls in order. The harness refuses counted tool calls after the budget's last one but lets
    the model keep asking until the turn limit; by default only the calls before the first refusal are returned."""
    out = []
    for s in trace['steps']:
        if within_budget and s.get('tool_calls') and REFUSED in json.dumps(s.get('observation')): break
        for c in (s.get('tool_calls') or []):
            fn = c.get('function_name') or c.get('name') or (c.get('function') or {}).get('name')
            args = c['arguments'] if 'arguments' in c else (c.get('function') or {}).get('arguments')
            out.append((fn, args if isinstance(args, str) else json.dumps(args, sort_keys=True)))
    return out


def repeat_share(run_id: str) -> tuple:
    n = r = 0
    for f in sorted((ROOT / 'results/kaggle' / run_id / 'traces').glob('*.json')):
        cs = calls(json.loads(f.read_text())); n += len(cs); r += sum(cs[i] == cs[i - 1] for i in range(1, len(cs)))
    return r, n


def text(x, y, s, size=13, fill=INK, anchor='start', weight='normal'):
    return f'<text x="{x}" y="{y}" {FONT} font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'


def p1() -> str:
    acc = {c: json.loads((ROOT / f'results/lora/cycles/{c}/accept.json').read_text()) for c in ('c1', 'c2', 'c3', 'c4', 'c5')}
    rows = [('Base E4B, no adapter', acc['c1']['base']['run_id'], '2, 2, 2', BLUE)]
    names = {'c1': 'Cycle 1: the 31B\'s histories', 'c2': 'Cycle 2: the 31B\'s, exact-match only', 'c3': 'Cycle 3: plus 31B labels on the learner\'s states',
             'c4': 'Cycle 4: the learner\'s own histories, 204 examples', 'c5': 'Cycle 5: the 31B\'s histories, 204 examples'}
    for c in ('c1', 'c2', 'c3', 'c4', 'c5'):
        rows.append((names[c], acc[c]['adapter']['run_id'], str(acc[c]['adapter']['resolved']), BLUE if c == 'c4' else ORANGE))
    W, H, X0, X1, Y0, STEP, BAR = 860, 370, 340, 670, 96, 40, 16
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         text(24, 32, 'Tool calls that repeat the previous call', 17, weight='bold'),
         text(24, 52, 'Share of tool calls within the 28-call budget, each bundle\'s run on the 16 held-out tasks', 12.5, INK2)]
    for v in (0, 20, 40, 60):
        x = X0 + (X1 - X0) * v / 60
        o += [f'<line x1="{x}" y1="{Y0 - 14}" x2="{x}" y2="{Y0 + STEP * len(rows) - 12}" stroke="{GRID}" stroke-width="1"/>', text(x, Y0 - 20, f'{v}%', 11.5, INK2, 'middle')]
    o.append(text(W - 24, Y0 - 20, 'resolved of 16', 11.5, INK2, 'end'))
    for i, (name, run, resolved, color) in enumerate(rows):
        r, n = repeat_share(run); share = r / n; y = Y0 + STEP * i
        w = max(2.0, (X1 - X0) * share * 100 / 60)
        o += [text(X0 - 12, y + 12, name, 13, INK, 'end'),
              f'<path d="M{X0},{y} h{max(0, w - 4):.1f} a4,4 0 0 1 4,4 v{BAR - 8} a4,4 0 0 1 -4,4 h-{max(0, w - 4):.1f} z" fill="{color}"/>',
              text(X0 + w + 8, y + 12, f'{share * 100:.0f}%', 13, INK, weight='bold'), text(X0 + w + 44, y + 12, f'{r} of {n}', 11.5, INK2),
              text(W - 24, y + 12, resolved, 13, INK, 'end')]
    ly = Y0 + STEP * len(rows) + 14
    o += [f'<rect x="24" y="{ly - 10}" width="12" height="12" rx="2" fill="{ORANGE}"/>', text(42, ly, 'adapter trained on the larger model\'s histories', 12, INK2),
          f'<rect x="330" y="{ly - 10}" width="12" height="12" rx="2" fill="{BLUE}"/>', text(348, ly, 'no adapter, or adapter trained on the learner\'s own histories', 12, INK2), '</svg>']
    return '\n'.join(o) + '\n'


def swap_rates() -> dict:
    """Next-call repeat rate per reader on the swap's cuts, each cut weighted equally: looping cuts by whether the
    reading adapter wrote the history, clean cuts for the base and for the two adapters pooled."""
    run = ROOT / 'results/swap/runs/v1-20261102'
    pre = {r['prefix_id']: r for r in map(json.loads, (ROOT / 'results/swap/datasets/20261102/prefixes.jsonl').read_text().splitlines())}
    writer = {'bf18-e4b-lora-c1': 'c1', 'bf18-e4b-lora-c2': 'c2'}
    cells = {}
    for g in map(json.loads, (run / 'generations.jsonl').read_text().splitlines()):
        p = pre[g['prefix_id']]
        if p['source_kind'] == 'looping': reader = 'base' if g['executor'] == 'base' else ('writer' if g['executor'] == writer[p['source_bundle']] else 'other')
        else: reader = 'base' if g['executor'] == 'base' else 'adapter'
        cells.setdefault(('looping' if p['source_kind'] == 'looping' else 'clean', reader), {}).setdefault(g['prefix_id'], []).append(g['repeat_previous'])
    return {k: (sum(sum(v) / len(v) for v in d.values()) / len(d), len(d)) for k, d in cells.items()}


def p2() -> str:
    a = json.loads((ROOT / 'results/swap/runs/v1-20261102/analysis.json').read_text())['all']['repeat_previous']
    d = a['primary1_base_minus_source_adapter']; r = swap_rates()
    W, H, X0, Y0, CW, CH, GAP = 860, 400, 300, 176, 170, 74, 4
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         text(24, 32, 'The frontier swap: who wrote the history, who reads it', 17, weight='bold'),
         text(24, 52, 'Share of next calls that repeat the history\'s last call. One greedy and four sampled completions per cut; no tool runs.', 12.5, INK2)]
    steps = ['1. Take a recorded attempt', '2. Cut its history before call k', '3. Hand it to another model for call k only']
    sx = 24
    for i, s in enumerate(steps):
        w = 22 + 6.0 * len(s)
        o += [f'<rect x="{sx}" y="74" width="{w:.0f}" height="30" rx="6" fill="none" stroke="{INK2}" stroke-width="1"/>', text(sx + w / 2, 94, s, 12.5, INK, 'middle')]
        if i < 2: o.append(f'<path d="M{sx + w + 6:.0f},89 h18 m-6,-5 l6,5 l-6,5" fill="none" stroke="{INK2}" stroke-width="1.5"/>')
        sx += w + 32
    o.append(text(X0 + 1.5 * CW + GAP, Y0 - 34, 'Model that reads the history and makes the next call', 12.5, INK2, 'middle'))
    for j, label in enumerate(('Base E4B', 'The other adapter', 'The adapter that wrote it')): o.append(text(X0 + j * (CW + GAP) + CW / 2, Y0 - 12, label, 13, INK, 'middle', 'bold'))

    def cell(x, y, w, v, sub=None):
        step = min(len(RAMP) - 1, int(v * 100 / 50 * (len(RAMP) - 1) + 0.5)); ink = '#ffffff' if step >= 6 else INK
        out = [f'<rect x="{x}" y="{y}" width="{w}" height="{CH}" rx="4" fill="{RAMP[step]}"/>', text(x + w / 2, y + CH / 2 + (2 if sub else 9), f'{v * 100:.0f}%', 26, ink, 'middle', 'bold')]
        if sub: out.append(text(x + w / 2, y + CH / 2 + 22, sub, 11.5, ink, 'middle'))
        return out

    n_loop, n_clean = r[('looping', 'base')][1], r[('clean', 'base')][1]
    y = Y0
    o += [text(X0 - 14, y + CH / 2 - 2, 'Looping history', 13, INK, 'end', 'bold'), text(X0 - 14, y + CH / 2 + 16, f'written by an adapter, {n_loop} cuts', 11.5, INK2, 'end')]
    for j, k in enumerate(('base', 'other', 'writer')): o += cell(X0 + j * (CW + GAP), y, CW, r[('looping', k)][0])
    y = Y0 + CH + GAP
    o += [text(X0 - 14, y + CH / 2 - 2, 'Clean history', 13, INK, 'end', 'bold'), text(X0 - 14, y + CH / 2 + 16, f'written by the 31B or the base, {n_clean} cuts', 11.5, INK2, 'end')]
    o += cell(X0, y, CW, r[('clean', 'base')][0]) + cell(X0 + CW + GAP, y, 2 * CW + GAP, r[('clean', 'adapter')][0], 'either adapter')
    ny = Y0 + 2 * (CH + GAP) + 26
    o += [text(24, ny, f'On looping histories, base minus the adapter that wrote them: {d["diff"] * 100:.0f} points '
                       f'(90% interval {d["lo"] * 100:.0f} to {d["hi"] * 100:.0f}, task bootstrap).', 12.5, INK),
          text(24, ny + 20, 'The base rarely continues an adapter\'s loop, an adapter continues its own far more than another\'s, and on clean ones all repeat on 3% or less.', 12.5, INK), '</svg>']
    return '\n'.join(o).replace('-46 points', '−46 points').replace('interval -65 to -27', 'interval −65 to −27') + '\n'


def hazard(run_id: str) -> dict:
    """For each trailing run of identical calls (1, 2, 3, 4, 5 or more): [next calls, of which repeats]."""
    out = {k: [0, 0] for k in range(1, 6)}
    for f in sorted((ROOT / 'results/kaggle' / run_id / 'traces').glob('*.json')):
        cs = calls(json.loads(f.read_text())); run = 1
        for i in range(1, len(cs)):
            rep = cs[i] == cs[i - 1]; b = min(run, 5); out[b][0] += 1; out[b][1] += rep
            run = run + 1 if rep else 1
    return out


def p3() -> str:
    acc = {c: json.loads((ROOT / f'results/lora/cycles/{c}/accept.json').read_text()) for c in ('c1', 'c2', 'c3', 'c5')}
    per = {c: hazard(acc[c]['adapter']['run_id']) for c in acc}; base = hazard(acc['c1']['base']['run_id'])
    pooled = {k: [sum(per[c][k][0] for c in per), sum(per[c][k][1] for c in per)] for k in range(1, 6)}
    W, H, X0, X1, Y0, Y1 = 860, 410, 130, 620, 124, 334
    xs = lambda k: X0 + (X1 - X0) * (k - 1) / 4; ys = lambda v: Y1 - (Y1 - Y0) * v
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         text(24, 32, 'Once a repeat starts it deepens', 17, weight='bold'),
         text(24, 52, 'Share of next calls that repeat the previous one, by how many identical calls already end the history. Calls within the budget, 16 held-out tasks.', 12.5, INK2)]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        o += [f'<line x1="{X0}" y1="{ys(v)}" x2="{X1}" y2="{ys(v)}" stroke="{GRID}" stroke-width="1"/>', text(X0 - 56, ys(v) + 4, f'{v * 100:.0f}%', 11.5, INK2, 'end')]
    for k, lab in zip(range(1, 6), ('1', '2', '3', '4', '5 or more')): o.append(text(xs(k), Y1 + 22, lab, 12, INK2, 'middle'))
    o.append(text((X0 + X1) / 2, Y1 + 44, 'identical calls ending the history', 12, INK2, 'middle'))
    for c in per:   # the three cycles, thin, behind the pooled line
        pts = ' '.join(f'{xs(k):.1f},{ys(per[c][k][1] / per[c][k][0]):.1f}' for k in range(1, 6))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{ORANGE}" stroke-width="1" stroke-opacity="0.35"/>')
    pts = ' '.join(f'{xs(k):.1f},{ys(pooled[k][1] / pooled[k][0]):.1f}' for k in range(1, 6))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{ORANGE}" stroke-width="2"/>')
    for k in range(1, 6):
        n, r = pooled[k]; x, y = xs(k), ys(r / n)
        o += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{ORANGE}" stroke="{SURFACE}" stroke-width="2"/>',
              text(x - 12, y - 2, f'{r / n * 100:.0f}%', 13, INK, 'end', 'bold') if k == 1 else
              text(x + (12 if k < 5 else 0), y + (20 if k < 5 else -14), f'{r / n * 100:.0f}%', 13, INK, 'start' if k < 5 else 'middle', 'bold'),
              text(x + 12, y + 35, f'{r} of {n}', 11, INK2) if 1 < k < 5 else (text(x, y - 30, f'{r} of {n}', 11, INK2, 'middle') if k == 5 else '')]
    bn, br = base[1]
    o += [f'<circle cx="{xs(1):.1f}" cy="{ys(br / bn):.1f}" r="5" fill="{BLUE}" stroke="{SURFACE}" stroke-width="2"/>',
          text(xs(1) - 12, ys(br / bn) + 14, f'{br / bn * 100:.0f}%', 13, INK, 'end', 'bold')]
    lx = 660; n1, r1 = pooled[1]
    o += [f'<line x1="{lx}" y1="120" x2="{lx + 22}" y2="120" stroke="{ORANGE}" stroke-width="2"/>', f'<circle cx="{lx + 11}" cy="120" r="4" fill="{ORANGE}"/>',
          text(lx + 30, 124, 'adapters of cycles 1, 2, 3, 5,', 12, INK2), text(lx + 30, 140, f'pooled ({r1} of {n1} at 1)', 12, INK2),
          f'<line x1="{lx}" y1="166" x2="{lx + 22}" y2="166" stroke="{ORANGE}" stroke-width="1" stroke-opacity="0.35"/>', text(lx + 30, 170, 'each cycle', 12, INK2),
          f'<circle cx="{lx + 11}" cy="196" r="4" fill="{BLUE}"/>', text(lx + 30, 200, 'base E4B, no adapter', 12, INK2),
          text(lx + 30, 216, f'({br} of {bn}; it rarely gets further)', 11, INK2), '</svg>']
    return '\n'.join(o) + '\n'


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(ROOT / 'docs/results/figures')); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, svg in (('P1-repeat-by-cycle.svg', p1()), ('P2-frontier-swap.svg', p2()), ('P3-repeat-hazard.svg', p3())):
        (out / name).write_text(svg); print(f'wrote {out / name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
