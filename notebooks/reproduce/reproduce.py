"""Reproduce every number in the paper from the released record set, on a CPU Kaggle notebook with no internet.

Input: the `gemma-agent-records` dataset: the tarball `gemma-agent-records-<stamp>.tar.gz` built by
scripts/release-snapshot.py from the project tree (records, traces, signed items, public key, analysis scripts) and its
MANIFEST.json (tarball sha256 and every file's sha256). Kaggle unpacks an uploaded tarball into a folder of the same
name and keeps no copy of the archive, nested archives included, so the dataset ships the tarball under a .bin suffix,
which Kaggle keeps byte for byte; an unpacked folder beside MANIFEST.json is accepted as well. Steps, each printed with
its command:
  1. tarball: check its sha256 against MANIFEST.json and extract it; unpacked folder: copy it and check every file's
     sha256 against MANIFEST.json, refusing missing, extra or altered files
  2. scripts/verify.py        every signature, ledger row, correction and forecast, and the raw-file seal over the
                              traces, from the shipped public key
  3. scripts/regenerate.py    tables.md and numbers.json from the records into a fresh directory
  4. diff                     the regenerated numbers.json and tables.md against the shipped copies (byte for byte)
  5. scripts/curve.py         S(k), P(C) and the prediction from the records; and once standalone, from two results
                              directories, with no records at all
  6. scripts/figures.py       the four figures from the regenerated numbers
  7. scripts/forecast.py evaluate  on every signed forecast whose cell outcomes exist
Exit status is non-zero if any step fails or the regenerated numbers differ from the shipped ones.

Locally: python3 notebooks/reproduce/reproduce.py --tarball release/gemma-agent-records-<stamp>.tar.gz [--work DIR]
         (--tarball also accepts an unpacked snapshot folder beside its MANIFEST.json)
"""
import argparse, hashlib, json, shutil, subprocess, sys, tarfile
from pathlib import Path

PY = sys.executable


def find_input() -> Path:
    """The newest tarball under /kaggle/input, else the newest unpacked snapshot folder (what Kaggle serves)."""
    base = Path('/kaggle/input')
    # the dataset ships the tarball under a .bin suffix, which Kaggle keeps intact (it unpacks anything it recognises)
    hits = sorted(base.rglob('gemma-agent-records-*.tar.gz.bin')) + sorted(base.rglob('gemma-agent-records-*.tar.gz'))
    if hits: return hits[-1]
    # Kaggle unpacks <name>.tar.gz into a folder <name>/ which then holds the tarball's own top directory <name>/;
    # the snapshot root is the one that holds the scripts.
    dirs = sorted(p for p in base.rglob('gemma-agent-records-*') if p.is_dir() and (p / 'scripts/verify.py').is_file())
    if dirs: return dirs[-1]
    raise SystemExit('no gemma-agent-records-* tarball or folder under /kaggle/input; attach the gemma-agent-records dataset')


def manifest_entry(mf: Path, name: str) -> dict:
    if not mf.exists(): return {}
    m = json.loads(mf.read_text()); tbs = m.get('tarballs', {})
    name = name[:-4] if name.endswith('.bin') else name
    for k, v in tbs.items():
        if k == name or k == name + '.tar.gz' or v.get('top_dir') == name: return v
    return {}


def check_tree(src: Path, work: Path, entry: dict) -> Path:
    """Copy the unpacked snapshot into the work directory and check every file against the manifest's digests."""
    root = work / src.name; shutil.copytree(src, root)
    expect = entry.get('file_sha256')
    if not expect:
        print(f'copied {src.name}; MANIFEST.json has no per-file digests, so the tree is unchecked'); return root
    found = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    missing = sorted(set(expect) - set(found)); extra = sorted(set(found) - set(expect))
    altered = sorted(k for k in set(expect) & set(found) if expect[k] != found[k])
    print(f'unpacked snapshot {src.name}: {len(found)} files; manifest lists {len(expect)}; missing {len(missing)}, extra {len(extra)}, altered {len(altered)}')
    for label, items in (('missing', missing), ('extra', extra), ('altered', altered)):
        for k in items[:10]: print(f'  {label}: {k}')
    if missing or extra or altered: raise SystemExit('unpacked snapshot does not match MANIFEST.json')
    print('MANIFEST.json: every file digest matches'); return root


def run(step: str, cmd: list, cwd: Path, must=True) -> subprocess.CompletedProcess:
    print(f'\n== {step}\n$ {" ".join(str(c) for c in cmd)}', flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    print(out[-6000:] if len(out) > 6000 else out, flush=True)
    if must and r.returncode != 0: raise SystemExit(f'{step}: exit {r.returncode}')
    return r


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--tarball', default=None); ap.add_argument('--work', default='/kaggle/working/reproduce')
    a = ap.parse_args(); src = Path(a.tarball) if a.tarball else find_input(); work = Path(a.work)
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    # MANIFEST.json sits beside the tarball; on Kaggle it sits at the dataset root, two levels above the unpacked tree
    mf = next((d / 'MANIFEST.json' for d in (src.parent, *src.parents[1:3]) if (d / 'MANIFEST.json').is_file()), src.with_name('MANIFEST.json'))
    entry = manifest_entry(mf, src.name)
    if src.is_dir():
        # 1 (unpacked). copy and check file by file
        root = check_tree(src, work, entry)
    else:
        # 1 (tarball). digest and extract
        digest = hashlib.sha256(src.read_bytes()).hexdigest(); print(f'tarball {src.name}: sha256 {digest}')
        if mf.exists():
            expect = entry.get('sha256')
            print('MANIFEST.json:', 'sha256 matches' if expect == digest else f'sha256 DIFFERS from manifest ({expect})')
            if expect != digest: return 1
        else: print('no MANIFEST.json beside the tarball; digest printed above, unchecked')
        with tarfile.open(src) as t:
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
    # keep the output small: the regenerated numbers, tables and figures stay; the extracted snapshot (a copy of the input) goes
    keep = work / 'reproduced'
    if keep.exists(): shutil.rmtree(keep)
    shutil.move(str(fresh), str(keep)); shutil.rmtree(root)
    print(f'kept {keep} ({sum(1 for _ in keep.rglob("*") if _.is_file())} files); removed the extracted snapshot')
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main())
