#!/usr/bin/env python3
"""Write one signed seal over the raw files the analysis reads besides the records: every file in each record's results
directory (traces, task results, patches, logs), the two-lane audit outputs, the 480-s replays and the API-lane
adapter log.

The ledger entries sign the record files; the timing curves and the clock prediction read the traces, which no ledger
row covers. The seal closes that gap at release time: it lists each such file with its sha256 and is signed by the
same operator key, and scripts/verify.py checks every listed file against it. A seal authenticates the files as
released; it says nothing about when they were collected, which rests on the ledger entries and the forecast.

Entries live in .ai/knowledge/gemma-agent/seal-<UTC>.md and are never edited; a later seal supersedes an earlier one.

Usage: scripts/seal.py [--dry-run]      (needs the RyeOS daemon for the signing step)
"""
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE = ROOT / '.ai/knowledge/gemma-agent'
SKIP_SUFFIX = ('.pyc', '.pid', '.zip')   # the release ships no zips (scripts/release-snapshot.py), so none is sealed


def sealed_files(root: Path) -> list:
    """The raw files the seal covers, as paths relative to root, sorted."""
    dirs = {json.loads(rp.read_text())['results_dir'] for rp in (root / 'records').glob('*.json')}
    dirs |= {str(d.relative_to(root)) for d in (root / 'results/kaggle').glob('lane-audit-*') if d.is_dir()}
    if (root / 'results/replay').is_dir(): dirs.add('results/replay')
    files = {p for d in dirs for p in (root / d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith(SKIP_SUFFIX)}
    if (root / 'results/adapter-log.jsonl').is_file(): files.add(root / 'results/adapter-log.jsonl')
    return sorted(str(p.relative_to(root)) for p in files)


def main() -> int:
    dry = '--dry-run' in sys.argv
    files = sealed_files(ROOT)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    stamp = now.strftime('%Y%m%dT%H%M%SZ'); name = f'seal-{stamp}'
    body = [f'---\nkind: knowledge\nid: gemma-agent/{name}\nversion: "1.0.0"\n---\n',
            f'# Raw-file seal {now.isoformat()}\n',
            f'{len(files)} files the analysis reads besides the records (traces, task results, patches, logs, lane audits, replays, '
            'adapter log), each with its sha256 at signing time. Release-time authentication: it does not date the files.\n',
            '| path | sha256 |', '| --- | --- |']
    body += [f'| {f} | {hashlib.sha256((ROOT / f).read_bytes()).hexdigest()} |' for f in files]
    out = KNOWLEDGE / f'{name}.md'
    if dry: print(f'dry run: would write {out} with {len(files)} rows'); return 0
    out.write_text('\n'.join(body) + '\n')
    r = subprocess.run(['ryeos', 'sign', f'knowledge:gemma-agent/{name}', '--project', str(ROOT)], capture_output=True, text=True)
    try: res = json.loads(r.stdout)
    except Exception: res = {}
    if r.returncode != 0 or res.get('failed') or not res.get('signed'):
        out.unlink(); sys.exit(f'signing failed; seal removed:\n{(r.stdout + r.stderr)[-600:]}')
    print(f'seal: wrote and signed {out.relative_to(ROOT)} ({len(files)} files)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
