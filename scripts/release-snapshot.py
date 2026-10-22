#!/usr/bin/env python3
"""Build the release snapshot: one tarball holding everything a stranger needs to rerun every number in the paper, and
nothing that is a secret or a build artifact.

Included, from the project root:
  records/                                 every attempt record, valid or not
  docs/                                    plan, decisions, status, draft, results (tables, numbers, curve.csv, soundness
                                           table, config history, public key, forecast-run evidence, figures)
  .ai/config/gemma-agent/                  the signed cohort and budget definitions
  .ai/knowledge/gemma-agent/               signed ledger entries, correction items, forecasts
  .ai/tools/gemma-agent/                   the attempt, ingest and dispatch tools and their library (validity, lane)
  submissions/                             the bundles whose digests the records carry
  scripts/, notebooks/, configs/           the analysis and launch scripts, the Kaggle kernels, the model configs
  results/adapter-log.jsonl                the API-lane adapter log (availability per attempt)
  results/<dir>/                           every results directory a record names (traces, task results, patches, logs)
  results/kaggle/lane-audit-*/             the two-lane soundness audit outputs
  results/replay/                          the 480-s replays (scripts/replay.py): report, patch and test output per task
Excluded: .env, .venv, __pycache__ and *.pyc, results/kaggle/incoming (downloaded archives), results/adapter.out and
.pid, datasets/ (the Kaggle upload staging), release/ itself. Before packing, every included text file is scanned for
key-like strings and the build refuses if one is found.

Usage: scripts/release-snapshot.py [--out DIR]       writes DIR/gemma-agent-records-<UTC stamp>.tar.gz and DIR/MANIFEST.json
"""
import argparse, hashlib, json, re, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TREES = ['records', 'docs', '.ai/config/gemma-agent', '.ai/knowledge/gemma-agent', '.ai/tools/gemma-agent', 'submissions', 'scripts', 'notebooks', 'configs']
FILES = ['results/adapter-log.jsonl', 'README.md', 'LICENSE']
SKIP_PARTS = {'__pycache__', '.venv', 'incoming', 'release', 'datasets', 'replay-work'}
SKIP_SUFFIX = ('.pyc', '.pid')
SECRET = re.compile(r'AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9]{20,}|LITELLM_API_KEY=[A-Za-z0-9_-]{8,}|api_key:\s*[A-Za-z0-9]{16,}|BEGIN (?:ENCRYPTED |EC |RSA )?PRIVATE KEY')


def wanted(p: Path) -> bool:
    return p.is_file() and not (set(p.parts) & SKIP_PARTS) and not p.name.endswith(SKIP_SUFFIX) and p.name != '.env'


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(ROOT / 'release')); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = []
    for t in TREES: files += [p for p in (ROOT / t).rglob('*') if wanted(p)]
    files += [ROOT / f for f in FILES if (ROOT / f).is_file()]
    for rp in sorted((ROOT / 'records').glob('*.json')):
        d = ROOT / json.loads(rp.read_text())['results_dir']
        if not d.is_dir(): sys.exit(f'refused: {rp.name} names a results directory that does not exist: {d}')
        files += [p for p in d.rglob('*') if wanted(p)]
    for d in sorted((ROOT / 'results/kaggle').glob('lane-audit-*')) + [ROOT / 'results/replay']: files += [p for p in d.rglob('*') if wanted(p)] if d.is_dir() else []
    files = sorted(set(files))
    # refuse on anything that looks like a credential
    for p in files:
        if p.suffix in ('.zip', '.gz', '.png', '.whl'): continue
        try: text = p.read_text(errors='ignore')
        except OSError: continue
        m = SECRET.search(text)
        if m: sys.exit(f'refused: {p.relative_to(ROOT)} contains a key-like string ({m.group(0)[:12]}...)')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); name = f'gemma-agent-records-{stamp}.tar.gz'; top = f'gemma-agent-records-{stamp}'
    tb = out / name
    with tarfile.open(tb, 'w:gz') as t:
        for p in files: t.add(p, arcname=f'{top}/{p.relative_to(ROOT)}', recursive=False)
    digest = hashlib.sha256(tb.read_bytes()).hexdigest()
    mf = out / 'MANIFEST.json'
    manifest = json.loads(mf.read_text()) if mf.exists() else {'tarballs': {}}
    manifest['tarballs'][name] = {'sha256': digest, 'bytes': tb.stat().st_size, 'files': len(files), 'built_at_utc': stamp,
                                  'records': len(list((ROOT / 'records').glob('*.json'))), 'top_dir': top}
    mf.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{tb}: {len(files)} files, {tb.stat().st_size / 1e6:.1f} MB, sha256 {digest}'); print(f'manifest: {mf}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
