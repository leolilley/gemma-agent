"""Reproduce every number in the paper from the released record set, on a CPU Kaggle notebook with no internet.

Input: the `gemma-agent-records` dataset, one tarball `gemma-agent-records-<stamp>.tar.gz` built by
scripts/release-snapshot.py from the project tree (records, traces, signed items, public key, analysis scripts) and its
MANIFEST.json (tarball sha256, file count). Steps, each printed with its command:
  1. check the tarball's sha256 against MANIFEST.json and extract it
  2. scripts/verify.py        every signature, ledger row, correction and forecast, from the shipped public key
  3. scripts/regenerate.py    tables.md and numbers.json from the records into a fresh directory
  4. diff                     the regenerated numbers.json and tables.md against the shipped copies (byte for byte)
  5. scripts/curve.py         S(k), P(C) and the prediction from the records; and once standalone, from two results
                              directories, with no records at all
  6. scripts/figures.py       the four figures from the regenerated numbers
  7. scripts/forecast.py evaluate  on every signed forecast whose cell outcomes exist
Exit status is non-zero if any step fails or the regenerated numbers differ from the shipped ones.

Locally: python3 notebooks/reproduce/reproduce.py --tarball release/gemma-agent-records-<stamp>.tar.gz [--work DIR]
"""
import argparse, hashlib, json, shutil, subprocess, sys, tarfile
from pathlib import Path

PY = sys.executable


def find_tarball() -> Path:
    for base in (Path('/kaggle/input'),):
        hits = sorted(base.rglob('gemma-agent-records-*.tar.gz'))
        if hits: return hits[-1]
    raise SystemExit('no gemma-agent-records-*.tar.gz under /kaggle/input; attach the gemma-agent-records dataset')


def run(step: str, cmd: list, cwd: Path, must=True) -> subprocess.CompletedProcess:
    print(f'\n== {step}\n$ {" ".join(str(c) for c in cmd)}', flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    print(out[-6000:] if len(out) > 6000 else out, flush=True)
    if must and r.returncode != 0: raise SystemExit(f'{step}: exit {r.returncode}')
    return r


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--tarball', default=None); ap.add_argument('--work', default='/kaggle/working/reproduce')
    a = ap.parse_args(); tb = Path(a.tarball) if a.tarball else find_tarball(); work = Path(a.work)
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    # 1. digest and extract
    digest = hashlib.sha256(tb.read_bytes()).hexdigest(); print(f'tarball {tb.name}: sha256 {digest}')
    mf = tb.with_name('MANIFEST.json')
    if mf.exists():
        m = json.loads(mf.read_text()); expect = m.get('tarballs', {}).get(tb.name, {}).get('sha256') or m.get('sha256')
        print('MANIFEST.json:', 'sha256 matches' if expect == digest else f'sha256 DIFFERS from manifest ({expect})')
        if expect != digest: return 1
    else: print('no MANIFEST.json beside the tarball; digest printed above, unchecked')
    with tarfile.open(tb) as t:
        names = t.getnames(); t.extractall(work, filter='data')
    roots = {n.split('/')[0] for n in names}; root = work / (roots.pop() if len(roots) == 1 else '.')
    print(f'extracted {len(names)} entries to {root}')
    # 2. verify
    run('verify', [PY, 'scripts/verify.py'], root)
    # 3. regenerate into a fresh directory, 4. compare with the shipped copies
    fresh = root / 'reproduced'; fresh.mkdir(exist_ok=True)
    run('regenerate', [PY, 'scripts/regenerate.py', '--out', fresh], root)
    same = True
    for name in ('numbers.json', 'tables.md'):
        a_, b_ = (root / 'docs/results' / name).read_bytes(), (fresh / name).read_bytes()
        print(f'{name}: {"identical to the shipped copy" if a_ == b_ else "DIFFERS from the shipped copy"} ({len(b_)} bytes)')
        same &= a_ == b_
    if not same:
        r = subprocess.run(['diff', str(root / 'docs/results/tables.md'), str(fresh / 'tables.md')], capture_output=True, text=True); print(r.stdout[-4000:])
    # 5. curve from the records, then standalone from two results directories
    run('curve (records)', [PY, 'scripts/curve.py'], root)
    cells = sorted((root / 'results/kaggle').glob('grid-bf16-cell-rich-*')); lifted = sorted((root / 'results/kaggle').glob('grid-bf16-calls-rich-c8-*'))
    if cells and lifted: run('curve (standalone)', [PY, 'scripts/curve.py', '--lifted-run', lifted[-1], '--cell-run', cells[-1]], root)
    # 6. figures
    run('figures', [PY, 'scripts/figures.py', '--numbers', fresh / 'numbers.json', '--out', fresh / 'figures'], root)
    # 7. forecasts
    for fc in sorted((root / '.ai/knowledge/gemma-agent').glob('forecast-*.md')):
        run(f'forecast evaluate {fc.name}', [PY, 'scripts/forecast.py', 'evaluate', fc], root, must=False)
    print('\nRESULT:', 'every check passed and the regenerated numbers are identical to the shipped ones' if same else 'regenerated numbers DIFFER from the shipped ones')
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main())
