#!/usr/bin/env python3
"""Append one signed ledger entry covering every attempt record not yet in the ledger.

The ledger is the checkable half of the preregistration claim (PAPER.md G10): the cohort and budget configs are signed
before the data, each ledger entry lists the records that exist at signing time with their file digests, and the entry
is signed by the same operator key. Entries live in .ai/knowledge/gemma-agent/ledger-<UTC>.md, are never edited, and
every signed config version is copied verbatim to docs/results/config-history/ so the hashes an entry names stay
checkable after the config changes. scripts/verify.py checks all of it without RyeOS.

Usage: scripts/ledger.py [--dry-run]      (needs the RyeOS daemon for the signing step)
"""
import hashlib, json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER_DIR = ROOT / '.ai/knowledge/gemma-agent'
HISTORY = ROOT / 'docs/results/config-history'
CONFIGS = ['cohorts', 'budgets']
SIG = re.compile(r'ryeos:signed:(\S+?):([0-9a-f]{64}):([A-Za-z0-9+/=]+):([0-9a-f]{64})')


def header(path: Path) -> re.Match:
    m = SIG.search(path.read_text().splitlines()[0])
    if not m: sys.exit(f'{path} is not signed')
    return m


def archive_config(name: str) -> str:
    """Copy the signed config to config-history under <name>-v<version>-<signed-at>.yaml unless that signature is there."""
    src = ROOT / '.ai/config/gemma-agent' / f'{name}.yaml'
    m = header(src); text = src.read_text()
    version = re.search(r'^version:\s*"?([^"\n]+)"?', text, re.M).group(1)
    HISTORY.mkdir(parents=True, exist_ok=True)
    for p in HISTORY.glob(f'{name}-*.yaml'):
        if header(p).group(2) == m.group(2): break
    else:
        (HISTORY / f"{name}-v{version}-{m.group(1).replace(':', '').replace('-', '')}.yaml").write_text(text)
    return m.group(0)


def covered() -> set:
    ids = set()
    for p in LEDGER_DIR.glob('ledger-*.md'):
        for line in p.read_text().splitlines():
            if line.startswith('| ') and not line.startswith('| attempt_id') and not line.startswith('| ---'):
                ids.add(line.split('|')[1].strip())
    return ids


def main() -> int:
    dry = '--dry-run' in sys.argv
    done = covered(); rows = []
    for p in sorted((ROOT / 'records').glob('*.json')):
        r = json.loads(p.read_text())
        if r['attempt_id'] in done: continue
        if p.stem != r['attempt_id']: sys.exit(f'{p.name}: file name does not match attempt_id {r["attempt_id"]}')
        rows.append((r['started_at'], r['attempt_id'], hashlib.sha256(p.read_bytes()).hexdigest(), r['bundle'], r['config_digest'], r['site']))
    if not rows: print('ledger: nothing new'); return 0
    rows.sort()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    stamp = now.strftime('%Y%m%dT%H%M%SZ'); name = f'ledger-{stamp}'
    sigs = {c: archive_config(c) for c in CONFIGS}
    body = [f'---\nkind: knowledge\nid: gemma-agent/{name}\nversion: "1.0.0"\n---\n',
            f'# Record ledger entry {now.isoformat()}\n',
            f'{len(rows)} records not covered by an earlier entry. Each row: the attempt record file `records/<attempt_id>.json`, the '
            'sha256 of that file as written, the bundle and its directory digest from the record, the site, and when the attempt started. '
            'Signed configuration in force at signing time (signature lines verbatim; bodies in docs/results/config-history/):\n']
    body += [f'- {c}: `{s}`' for c, s in sigs.items()]
    body += ['', '| attempt_id | sha256 | bundle | config_digest | site | started_at |', '| --- | --- | --- | --- | --- | --- |']
    body += [f'| {a} | {h} | {b} | {d} | {s} | {t} |' for t, a, h, b, d, s in rows]
    out = LEDGER_DIR / f'{name}.md'
    if dry: print(f'dry run: would write {out} with {len(rows)} rows'); return 0
    LEDGER_DIR.mkdir(parents=True, exist_ok=True); out.write_text('\n'.join(body) + '\n')
    r = subprocess.run(['ryeos', 'sign', f'knowledge:gemma-agent/{name}', '--project', str(ROOT)], capture_output=True, text=True)
    try: res = json.loads(r.stdout)
    except Exception: res = {}
    if r.returncode != 0 or res.get('failed') or not res.get('signed'):
        out.unlink(); sys.exit(f'signing failed; entry removed:\n{(r.stdout + r.stderr)[-600:]}')
    print(f'ledger: wrote and signed {out.relative_to(ROOT)} ({len(rows)} records; {header(out).group(1)})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
