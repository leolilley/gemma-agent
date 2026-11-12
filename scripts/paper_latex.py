#!/usr/bin/env python3
"""Build the LaTeX and PDF form of the paper from docs/paper-draft-lora.md (pandoc, then pdflatex twice).

The Markdown is the source of record; this only reshapes it: title and subtitle become the title block, the abstract
becomes the abstract environment, the draft note is dropped, section numbers are left to LaTeX, the figures are the
SVGs of scripts/paper_figures.py converted to PDF, and tables are set small with rules.

Usage: scripts/paper_latex.py [--src docs/paper-draft-lora.md] [--out docs/paper]     writes paper.tex and paper.pdf
"""
import argparse, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = r"""\usepackage[T1]{fontenc}
\usepackage{libertinus}
\DeclareUnicodeCharacter{2212}{\textminus}
\usepackage{microtype}
\usepackage{xurl}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
\setlength{\LTcapwidth}{\textwidth}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\setlength{\emergencystretch}{2em}
\usepackage{float}
\floatplacement{figure}{H}
"""


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode: sys.exit(f'{cmd[0]} failed:\n{(r.stdout + r.stderr)[-3000:]}')
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--src', default='docs/paper-draft-lora.md'); ap.add_argument('--out', default='docs/paper'); a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    L = (ROOT / a.src).read_text().splitlines()
    title = L[0].lstrip('# ').strip()
    subtitle = next(l for l in L if l.startswith('**Subtitle:**')).split('**Subtitle:**', 1)[1].strip()
    i_abs, i_intro, i_refs = L.index('## Abstract'), next(i for i, l in enumerate(L) if l.startswith('## 1.')), L.index('## References')
    abstract = ' '.join(l for l in L[i_abs + 1:i_intro] if l.strip())
    body = '\n'.join(L[i_intro:i_refs]); refs = '\n'.join(L[i_refs + 1:])
    refs = re.sub(r'(https?://[^\s;]*[^\s;.])', r'<\1>', refs)                           # breakable URLs
    body = re.sub(r'(?m)^(#{2,3}) \d+(\.\d+)?\.? ', r'\1 ', body)                     # LaTeX numbers the sections
    body = body.replace('‖(alpha/r)·B·A‖', r'$\lVert(\alpha/r)\,BA\rVert$')
    # the cycle table: relative column widths from the separator row, so the long source column gets the room
    body = body.replace('|---|---|---|---|---|---|---|---|', '|-----|' + '-' * 34 + '|--------|--------|-------|------------|-----------|---------|')
    body = body.replace('| term | meaning |\n|---|---|', '| term | meaning |\n|' + '-' * 18 + '|' + '-' * 70 + '|')
    # figures: the SVGs as vector PDFs, with the italic caption line under each image as the caption
    for m in list(re.finditer(r'!\[Figure (\d+)\]\(results/figures/([^)]*)\.png\)\n\n\*Figure \1\. ([^*]*)\*', body)):
        stem = m.group(2)
        run(['rsvg-convert', '-f', 'pdf', '-o', str(out / f'{stem}.pdf'), str(ROOT / 'docs/results/figures' / f'{stem}.svg')], ROOT)
        body = body.replace(m.group(0), f'![{m.group(3)}]({stem}.pdf){{width=100%}}')
    if '![Figure' in body: sys.exit('a figure line was not converted')
    md = (f'---\ntitle: "{title}"\nsubtitle: "{subtitle}"\nauthor: "Leo Lilley"\ndate: "November 2026"\nabstract: |\n  {abstract}\n---\n\n'
          f'{body}\n\n## References {{-}}\n\n{refs}\n')
    (out / 'paper.md').write_text(md)
    (out / 'header.tex').write_text(HEADER)
    run(['pandoc', 'paper.md', '-o', 'paper.tex', '--standalone', '--number-sections', '--shift-heading-level-by=-1', '-V', 'documentclass=article', '-V', 'fontsize=11pt',
         '-V', 'papersize=a4', '-V', 'colorlinks=true', '-V', 'urlcolor=black', '-V', 'linkcolor=black', '--include-in-header=header.tex', '--pdf-engine=pdflatex'], out)
    for _ in range(2): run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'paper.tex'], out)
    for junk in ('paper.aux', 'paper.log', 'paper.out', 'paper.md', 'header.tex'): (out / junk).unlink(missing_ok=True)
    print(f'wrote {out / "paper.tex"} and {out / "paper.pdf"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
