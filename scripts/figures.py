#!/usr/bin/env python3
"""Draw the paper's figures as plain SVG from docs/results/numbers.json (the output of scripts/regenerate.py).

No plotting library: every figure is a few hundred lines of SVG written by hand so the files diff cleanly and anyone can
regenerate them with the standard library. Each figure carries, under its title, the key path in numbers.json it was
drawn from, so a reader can check any point against the regenerated numbers.

  F1  pass rate against the clock, P(C): 4-bit at the cell and bf16 with the clock lifted; the 8-minute line; the
      observed cell pass rates marked at 8 minutes.
  F2  pass rate against calls, S(k): both precisions, the 28-call line; the 4-bit lifted subset drawn where it exists.
  F3  predicted against observed per task, four cells: rich (retrodiction) and fastapi (forecast, when its cell exists).
  F4  the three contrasts with their 90% intervals and the noise floor (API-lane pass rate, mixed-outcome tasks) as a footnote.

Usage: scripts/figures.py [--numbers docs/results/numbers.json] [--out docs/results/figures]
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W, H = 720, 420                       # canvas
L, R, T, B = 70, 30, 60, 60           # margins: left, right, top, bottom
FOUR_BIT, BF16, BF16_CELL, FOUR_BIT_LIFTED = 'bf18', 'bf18-bf16-calls', 'bf18-bf16', 'bf18-calls'
NAMES = {FOUR_BIT: '4-bit, at the cell (8 min)', BF16: 'bf16, clock lifted to 30 min', BF16_CELL: 'bf16, at the cell',
         FOUR_BIT_LIFTED: '4-bit, clock lifted (subset)'}
COLOR = {FOUR_BIT: '#1f5fbf', BF16: '#c8541b', BF16_CELL: '#e39a6d', FOUR_BIT_LIFTED: '#7fa6e0'}
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def esc(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class Fig:
    def __init__(self, title: str, source: str, w=W, h=H):
        self.w, self.h = w, h
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" {FONT} font-size="12">',
                      f'<rect width="{w}" height="{h}" fill="white"/>',
                      f'<text x="{L}" y="24" font-size="15" font-weight="bold">{esc(title)}</text>',
                      f'<text x="{L}" y="42" fill="#555" font-size="11">{esc(source)}</text>']

    def add(self, s: str): self.parts.append(s)

    def axes(self, xlab: str, ylab: str, xticks: list, yticks: list, xmax: float, ymax: float = 1.0):
        self.xmax, self.ymax = xmax, ymax
        x0, y0, x1, y1 = L, self.h - B, self.w - R, T
        self.add(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333"/><line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333"/>')
        for v, lab in xticks:
            x = self.px(v); self.add(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + 5}" stroke="#333"/><text x="{x:.1f}" y="{y0 + 18}" text-anchor="middle">{esc(lab)}</text>')
        for v, lab in yticks:
            y = self.py(v); self.add(f'<line x1="{x0 - 5}" y1="{y:.1f}" x2="{x0}" y2="{y:.1f}" stroke="#333"/><text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(lab)}</text>')
            if v > 0: self.add(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#eee"/>')
        self.add(f'<text x="{(x0 + x1) / 2:.1f}" y="{self.h - 14}" text-anchor="middle">{esc(xlab)}</text>')
        self.add(f'<text transform="translate(16,{(y0 + y1) / 2:.1f}) rotate(-90)" text-anchor="middle">{esc(ylab)}</text>')

    def px(self, v: float) -> float: return L + (self.w - L - R) * v / self.xmax
    def py(self, v: float) -> float: return (self.h - B) - (self.h - B - T) * v / self.ymax

    def step(self, pts: list, color: str, dash: str = '', width=2.0):
        """Right-continuous step curve through (x, y) points, starting at y = 0 at x = 0."""
        pts = sorted(pts); d = f'M{self.px(0):.1f},{self.py(0):.1f}'; prev = 0.0
        for x, y in pts:
            d += f' H{self.px(x):.1f} V{self.py(y):.1f}'; prev = y
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')

    def vline(self, x: float, label: str, color='#888', anchor='start'):
        X = self.px(x); dx = 4 if anchor == 'start' else -4
        self.add(f'<line x1="{X:.1f}" y1="{T}" x2="{X:.1f}" y2="{self.h - B}" stroke="{color}" stroke-dasharray="4 3"/>'
                 f'<text x="{X + dx:.1f}" y="{T + 12}" fill="{color}" font-size="11" text-anchor="{anchor}">{esc(label)}</text>')

    def point(self, x: float, y: float, color: str, label: str, dx=8, dy=4):
        X, Y = self.px(x), self.py(y)
        self.add(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="4.5" fill="white" stroke="{color}" stroke-width="2"/>'
                 f'<text x="{X + dx:.1f}" y="{Y + dy:.1f}" fill="{color}" font-size="11">{esc(label)}</text>')

    def legend(self, items: list, x=None, y=None):
        x = (self.w - R - 250) if x is None else x; y = (T + 10) if y is None else y
        for i, (color, label, dash) in enumerate(items):
            yy = y + 16 * i
            self.add(f'<line x1="{x}" y1="{yy}" x2="{x + 24}" y2="{yy}" stroke="{color}" stroke-width="2"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>'
                     f'<text x="{x + 30}" y="{yy + 4}">{esc(label)}</text>')

    def text(self, x, y, s, **kw):
        attrs = ' '.join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
        self.add(f'<text x="{x}" y="{y}" {attrs}>{esc(s)}</text>')

    def write(self, path: Path):
        self.add('</svg>'); path.write_text('\n'.join(self.parts) + '\n')


def curve_points(c: dict, key: str, scale=1.0, limit=None) -> list:
    return [(float(k) * scale, v) for k, v in c[key].items() if limit is None or float(k) <= limit]


def f1(n: dict, out: Path):
    c = n['curve']; b4, bf = c[FOUR_BIT], c[BF16]
    f = Fig('F1  Pass rate against the clock, P(C)', 'numbers.json: curve[bundle].P (fraction of tasks whose last successful edit came by C and that passed); cell pass rates from arms')
    f.axes('clock C (minutes)', 'P(C)', [(m, str(m)) for m in (0, 4, 8, 12, 16, 20, 24, 30)], [(v, f'{v:.1f}') for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0)], 30, 1.0)
    f.vline(8, 'the cell clock, 8 min')
    f.step(curve_points(b4, 'P', 1 / 60, 480), COLOR[FOUR_BIT])           # the 4-bit arm ran at the cell: nothing past 8 min
    f.step(curve_points(bf, 'P', 1 / 60), COLOR[BF16])
    p4 = b4['resolved'] / b4['tasks']; f.point(8, p4, COLOR[FOUR_BIT], f"4-bit cell pass rate {b4['resolved']}/{b4['tasks']}", dx=10, dy=-10)
    bc = c.get(BF16_CELL)
    if bc: pb = bc['resolved'] / bc['tasks']; f.point(8, pb, COLOR[BF16], f"bf16 cell pass rate {bc['resolved']}/{bc['tasks']} (rich)", dx=10, dy=16)
    f.legend([(COLOR[FOUR_BIT], f"{NAMES[FOUR_BIT]}, {b4['tasks']} tasks", ''), (COLOR[BF16], f"{NAMES[BF16]}, {bf['tasks']} tasks", '')], x=f.w - R - 290, y=T + 12)
    f.write(out / 'F1-p-of-c.svg')


def f2(n: dict, out: Path):
    c = n['curve']; b4, bf = c[FOUR_BIT], c[BF16]
    f = Fig('F2  Pass rate against calls, S(k)', 'numbers.json: curve[bundle].S (fraction of tasks whose last successful edit came by call k and that passed)')
    kmax = max(max(float(k) for k in v['S']) for v in c.values()); kmax = max(30, int(kmax) + 2)
    f.axes('tool calls k', 'S(k)', [(k, str(k)) for k in range(0, kmax + 1, 4)], [(v, f'{v:.1f}') for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0)], kmax, 1.0)
    f.vline(28, 'the call cap, 28', anchor='end')
    f.step(curve_points(b4, 'S'), COLOR[FOUR_BIT]); f.step(curve_points(bf, 'S'), COLOR[BF16])
    items = [(COLOR[FOUR_BIT], f"{NAMES[FOUR_BIT]}, {b4['tasks']} tasks", ''), (COLOR[BF16], f"{NAMES[BF16]}, {bf['tasks']} tasks", '')]
    bc = c.get(BF16_CELL)
    if bc: f.step(curve_points(bc, 'S'), COLOR[BF16_CELL], dash='5 3'); items.append((COLOR[BF16_CELL], f"{NAMES[BF16_CELL]}, {bc['tasks']} tasks (rich)", '5 3'))
    bl = c.get(FOUR_BIT_LIFTED)
    if bl: f.step(curve_points(bl, 'S'), COLOR[FOUR_BIT_LIFTED], dash='2 3'); items.append((COLOR[FOUR_BIT_LIFTED], f"{NAMES[FOUR_BIT_LIFTED]}, {bl['tasks']} tasks so far", '2 3'))
    f.legend(items, x=L + 20, y=T + 12)
    f.write(out / 'F2-s-of-k.svg')


def f3(n: dict, out: Path):
    rows = n['clock_prediction']['rows']
    panels = [('rich (retrodiction)', [r for r in rows if r['task'].startswith('rich_')]), ('fastapi (forecast)', [r for r in rows if r['task'].startswith('fastapi_')])]
    f = Fig('F3  Predicted against observed, per task, at the 8-minute clock', 'numbers.json: clock_prediction.rows (predicted = lifted bf16 resolved with solve time <= 480 s; observed = bf16 cell resolved)', w=W, h=460)
    pw = (W - 2 * 40) // 2
    for i, (name, rs) in enumerate(panels):
        x0 = 40 + i * pw; y0 = 70
        f.text(x0 + pw / 2, y0, f'{name}: {len(rs)} tasks' if rs else f'{name}: pending', text_anchor='middle', font_weight='bold')
        if not rs:
            f.text(x0 + pw / 2, y0 + 150, 'no bf16 cell outcomes yet;', text_anchor='middle', fill='#777', font_size='11')
            f.text(x0 + pw / 2, y0 + 166, 'the signed forecast is scored here when the cell run lands', text_anchor='middle', fill='#777', font_size='11'); continue
        cells = {(True, True): 'both', (True, False): 'predicted only', (False, True): 'observed only', (False, False): 'neither'}
        cw, ch = (pw - 60) / 2, 150
        for (pred, obs), label in cells.items():
            cx = x0 + 50 + (0 if obs else 1) * cw; cy = y0 + 30 + (0 if pred else 1) * ch
            ts = sorted(r['task'] for r in rs if r['predicted'] == pred and r['observed'] == obs)
            agree = pred == obs
            f.add(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw:.1f}" height="{ch}" fill="{"#eaf2e3" if agree else "#fbe9e4"}" stroke="#999"/>')
            f.text(cx + 8, cy + 18, f'{label}: {len(ts)}', font_weight='bold')
            line, lines = '', []; limit = int(cw / 5.8)   # characters that fit one line of 10-px text in the cell
            for t in ts:
                short = t.split('_', 1)[1]
                if line and len(line) + len(short) + 2 > limit: lines.append(line); line = ''
                line += (', ' if line else '') + short
            if line: lines.append(line)
            for j, ln in enumerate(lines[:9]): f.text(cx + 8, cy + 36 + 13 * j, ln, font_size='10', fill='#333')
            if len(lines) > 9: f.text(cx + 8, cy + 36 + 13 * 9, '...', font_size='10')
        f.text(x0 + 50 + cw / 2, y0 + 30 + 2 * ch + 16, 'observed pass', text_anchor='middle'); f.text(x0 + 50 + 1.5 * cw, y0 + 30 + 2 * ch + 16, 'observed fail', text_anchor='middle')
        f.add(f'<text transform="translate({x0 + 38},{y0 + 30 + ch / 2}) rotate(-90)" text-anchor="middle">predicted pass</text>')
        f.add(f'<text transform="translate({x0 + 38},{y0 + 30 + 1.5 * ch}) rotate(-90)" text-anchor="middle">predicted fail</text>')
        agree = sum(1 for r in rs if r['predicted'] == r['observed'])
        f.text(x0 + pw / 2, y0 + 30 + 2 * ch + 40, f"agreement {agree}/{len(rs)}; predicted {sum(r['predicted'] for r in rs)}, observed {sum(r['observed'] for r in rs)}", text_anchor='middle', font_size='11')
    f.write(out / 'F3-predicted-vs-observed.svg')


def f4(n: dict, out: Path):
    items = [('as scored\n(bf16 cell - 4-bit cell)', n['contrast_as_scored']), ('matched calls\n(bf16 at 4-bit\'s call count - 4-bit)', n['contrast_matched_calls']), ('D, the primary endpoint\n(bf16 lifted - 4-bit cell)', n['primary_endpoint_D'])]
    f = Fig('F4  The three contrasts with 90% intervals, and the noise floor', 'numbers.json: contrast_as_scored, contrast_matched_calls, primary_endpoint_D (point, lo, hi); noise_floor', w=W, h=380)
    x0, x1, yb = 200, W - 40, 300
    lo, hi = -0.4, 0.4
    def X(v): return x0 + (x1 - x0) * (v - lo) / (hi - lo)
    f.add(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="#333"/>')
    for v in (-0.4, -0.2, 0, 0.2, 0.4):
        f.add(f'<line x1="{X(v):.1f}" y1="{yb}" x2="{X(v):.1f}" y2="{yb + 5}" stroke="#333"/>'); f.text(X(v), yb + 18, f'{v:+.1f}' if v else '0', text_anchor='middle')
    f.add(f'<line x1="{X(0):.1f}" y1="70" x2="{X(0):.1f}" y2="{yb}" stroke="#bbb" stroke-dasharray="4 3"/>')
    f.text((x0 + x1) / 2, yb + 40, 'difference in pass rate (task-level, 90% cluster bootstrap over tasks)', text_anchor='middle')
    for i, (label, c) in enumerate(items):
        y = 90 + i * 60
        f.add(f'<line x1="{X(c["lo"]):.1f}" y1="{y}" x2="{X(c["hi"]):.1f}" y2="{y}" stroke="#1f5fbf" stroke-width="2"/>'
              f'<circle cx="{X(c["point"]):.1f}" cy="{y}" r="5" fill="#1f5fbf"/>')
        for j, ln in enumerate(label.split('\n')): f.text(x0 - 12, y - 4 + 14 * j, ln, text_anchor='end', font_size='11' if j else '12')
        f.text(X(c['point']), y + 20, f"{c['point']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}], n = {c['n_tasks']}", text_anchor='middle', font_size='11', fill='#333')
    nf = n['noise_floor']
    ceil = nf['all']['n_attempts'] - nf['without_ceiling_bound']['n_attempts']
    f.text(x0, yb + 58, f"Noise floor (cohort noise_floor_requests: the 4-bit bundle on the API lane, {nf['all']['n_tasks']} tasks, 5 repeats, no 8-min clock): pass rate {nf['all']['point']:.2f} [{nf['all']['lo']:.2f}, {nf['all']['hi']:.2f}]", font_size='10', fill='#555')
    f.text(x0, yb + 72, f"on {nf['all']['n_attempts']} attempts; {nf['without_ceiling_bound']['point']:.2f} [{nf['without_ceiling_bound']['lo']:.2f}, {nf['without_ceiling_bound']['hi']:.2f}] without the {ceil} attempts the lane's own 120-min ceiling ended; {nf['tasks_with_mixed_outcomes']} of {nf['tasks_with_repeats']} repeated tasks had mixed outcomes.", font_size='10', fill='#555')
    f.write(out / 'F4-contrasts.svg')


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--numbers', default=str(ROOT / 'docs/results/numbers.json')); ap.add_argument('--out', default=str(ROOT / 'docs/results/figures'))
    a = ap.parse_args(); n = json.loads(Path(a.numbers).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for fn in (f1, f2, f3, f4): fn(n, out)
    print(f"wrote {', '.join(sorted(p.name for p in out.glob('F*.svg')))} to {out} from {a.numbers} ({n['records']} records)" if isinstance(n.get('records'), int) else f"wrote figures to {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
