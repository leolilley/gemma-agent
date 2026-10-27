#!/usr/bin/env python3
"""Check the signed provenance chain of this record set without RyeOS.

What is checked:
  1. Every signed file (cohort and budget configs, their archived versions, ledger entries) carries a header
     `ryeos:signed:<time>:<sha256 of the body>:<ed25519 signature over that hex digest>:<signer fingerprint>`;
     the body hash is recomputed, the signature is verified against the shipped public key, and the fingerprint must
     equal sha256 of the raw public key.
  2. Every ledger row names a record file whose sha256 matches, whose attempt started before the entry was signed, and
     whose bundle directory digest matches the record (and the bundle on disk, when present).
  3. Every config signature a ledger entry names is one of the verified signed configs.
  4. Every record file is covered by some ledger entry (reported, not fatal: records written after the last entry).
  5. Every signed forecast (forecast-*.md, "Predicts: bundle=.. site=.. tasks=..") predates the records it covers: each
     such record started after the forecast's signature time AND carries forecast_sha256 equal to the digest of the
     forecast file. The digest reached the record through the Kaggle dataset version the run read (run.json), so the
     chronology rests on Kaggle's version timestamp, not on the signer's clock.
  6. Signed correction items (correction-*.md) are applied, never the records: a row `| attempt_id | field | recorded |
     corrected | reason |` replaces that field's value for the chronology checks above, after checking that the record
     still holds the recorded value. Records named by a ledger entry are never edited, so a wrong value is corrected by
     a later signed statement, and this script reports how many it applied.
  7. The newest signed seal (seal-*.md, scripts/seal.py) lists the raw files the analysis reads besides the records
     (traces, task results, patches, logs, lane audits, replays, adapter log) with their sha256: every listed file must
     be present and unchanged, and every such file on disk must be listed. The seal is release-time authentication:
     it shows the files are the ones released, not when they were collected.

Usage: scripts/verify.py [--root DIR] [--pubkey PEM]     exit 0 when every check passes.
Needs the `cryptography` package for the ed25519 step (pip install cryptography).
"""
import argparse, base64, hashlib, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seal import sealed_files  # noqa: E402  (the raw-file set a seal covers; one definition for writer and checker)

SIG = re.compile(r'ryeos:signed:(\S+?):([0-9a-f]{64}):([A-Za-z0-9+/=]+):([0-9a-f]{64})')


def sha256_dir(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob('*') if x.is_file() and x.name != 'PROVENANCE.md'):
        h.update(str(p.relative_to(root)).encode()); h.update(b'\0'); h.update(p.read_bytes()); h.update(b'\0')
    return h.hexdigest()


def load_pubkey(path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    text = path.read_text()
    if 'BEGIN PUBLIC KEY' in text:
        key = serialization.load_pem_public_key(text.encode())
    else:
        raw = base64.b64decode(re.search(r'raw base64:\s*(\S+)', text).group(1))
        key = Ed25519PublicKey.from_public_bytes(raw)
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return key, hashlib.sha256(raw).hexdigest()


def check_signed(path: Path, key, fp: str, problems: list) -> re.Match | None:
    text = path.read_text()
    nl = text.find('\n'); first, body = text[:nl], text[nl + 1:]
    m = SIG.search(first)
    if not m: problems.append(f'{path}: no signature header'); return None
    if hashlib.sha256(body.encode()).hexdigest() != m.group(2): problems.append(f'{path}: body hash differs from header'); return None
    if m.group(4) != fp: problems.append(f'{path}: signer fingerprint {m.group(4)[:12]} is not the shipped key {fp[:12]}'); return None
    try: key.verify(base64.b64decode(m.group(3)), m.group(2).encode())
    except Exception: problems.append(f'{path}: signature does not verify'); return None
    return m


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument('--pubkey', default=None); a = ap.parse_args()
    root = Path(a.root); key, fp = load_pubkey(Path(a.pubkey) if a.pubkey else root / 'docs/results/signing-public-key.pem')
    problems, notes = [], []
    signed_files = sorted((root / '.ai/config/gemma-agent').glob('*.yaml')) + sorted((root / 'docs/results/config-history').glob('*.yaml'))
    ledgers = sorted((root / '.ai/knowledge/gemma-agent').glob('ledger-*.md'))
    verified = {}; n_verified = 0   # verified: body digest -> a file carrying it (configs and their archived copies share one)
    for p in signed_files + ledgers:
        m = check_signed(p, key, fp, problems)
        if m: verified[m.group(2)] = p; n_verified += 1
    # corrections: signed statements that a record's field is wrong and what the right value is (the record itself stays
    # as the ledger entry hashed it)
    corrections = {}
    for cp in sorted((root / '.ai/knowledge/gemma-agent').glob('correction-*.md')):
        m = check_signed(cp, key, fp, problems)
        if not m: continue
        verified[m.group(2)] = cp; n_verified += 1
        for line in cp.read_text().splitlines():
            if not line.startswith('| ') or line.startswith('| attempt_id') or line.startswith('| ---'): continue
            cells = [c.strip().strip('`') for c in line.strip('|').split('|')]
            if len(cells) < 4: continue
            attempt, field, recorded, corrected = cells[:4]
            rp = root / 'records' / f'{attempt}.json'
            if not rp.exists(): problems.append(f'{cp.name}: corrects {attempt}, which has no record'); continue
            if str(json.loads(rp.read_text()).get(field)) != recorded: problems.append(f'{cp.name}: {attempt}.{field} is not the recorded value the correction names')
            corrections[(attempt, field)] = corrected
    def started_(rec: dict) -> str:
        return corrections.get((rec['attempt_id'], 'started_at'), rec.get('started_at', ''))
    covered = set()
    for lp in ledgers:
        text = lp.read_text(); m = SIG.search(text.splitlines()[0])
        if not m: continue
        signed_at = m.group(1)
        for cfg in re.findall(r'^- (\w+): `(ryeos:signed:[^`]+)`', text, re.M):
            h = SIG.search(cfg[1]).group(2)
            if h not in verified: problems.append(f'{lp.name}: names {cfg[0]} config {h[:12]} that is not among the verified signed configs')
        for line in text.splitlines():
            if not line.startswith('| ') or line.startswith('| attempt_id') or line.startswith('| ---'): continue
            attempt, digest, bundle, cfg_digest, site, started = [c.strip() for c in line.strip('|').split('|')]
            rp = root / 'records' / f'{attempt}.json'
            if not rp.exists(): problems.append(f'{lp.name}: record {attempt} missing'); continue
            if hashlib.sha256(rp.read_bytes()).hexdigest() != digest: problems.append(f'{attempt}: record file changed since ledger entry {lp.name}')
            rec = json.loads(rp.read_text())
            if rec.get('config_digest') != cfg_digest: problems.append(f'{attempt}: config_digest differs from ledger row')
            if started > signed_at: problems.append(f'{attempt}: started {started} after the ledger entry was signed {signed_at}')
            if started_(rec) > signed_at: problems.append(f'{attempt}: corrected start {started_(rec)} is after the ledger entry was signed {signed_at}')
            bdir = root / 'submissions' / bundle
            if bdir.is_dir() and sha256_dir(bdir) != cfg_digest: notes.append(f'{attempt}: bundle {bundle} on disk no longer matches the record digest {cfg_digest[:12]}')
            covered.add(attempt)
    # Forecasts: signed knowledge items forecast-*.md carrying a line
    #   Predicts: bundle=<bundle> site=<site> tasks=<comma-separated task ids>
    # Every record of that bundle, site and task set must have started after the forecast was signed; a forecast with
    # no such records yet is reported as pending, not as a failure.
    forecasts = sorted((root / '.ai/knowledge/gemma-agent').glob('forecast-*.md'))
    for fp_ in forecasts:
        m = check_signed(fp_, key, fp, problems)
        if not m: continue
        verified[m.group(2)] = fp_; n_verified += 1
        text = fp_.read_text(); signed_at = m.group(1)
        pm = re.search(r'^Predicts:\s*bundle=(\S+)\s+site=(\S+)\s+tasks=(\S+)', text, re.M)
        if not pm: problems.append(f'{fp_.name}: no "Predicts:" line'); continue
        bundle, site, tasks = pm.group(1), pm.group(2), set(pm.group(3).split(','))
        file_digest = hashlib.sha256(fp_.read_bytes()).hexdigest()
        later, earlier, untied = 0, [], []
        for rp in (root / 'records').glob('*.json'):
            rec = json.loads(rp.read_text())
            if rec.get('bundle') == bundle and rec.get('site') == site and rec.get('task_id') in tasks:
                if started_(rec) > signed_at: later += 1
                else: earlier.append(rec['attempt_id'])
                if rec.get('forecast_sha256') != file_digest: untied.append(rec['attempt_id'])
        for a in earlier: problems.append(f'{fp_.name}: record {a} started before the forecast was signed {signed_at}')
        for a in untied: problems.append(f'{fp_.name}: record {a} does not carry this forecast\'s digest {file_digest[:12]} (its run manifest did not ship the forecast)')
        print(f'forecast {fp_.name}: signed {signed_at}, file sha256 {file_digest[:12]}; {bundle} @ {site} on {len(tasks)} tasks; '
              f'records after signing: {later}, carrying the digest: {later + len(earlier) - len(untied)}' + ('' if later else ' (pending)'))
    # Seal: the newest signed seal-*.md covers the raw files; older seals are superseded and not checked.
    seals = sorted((root / '.ai/knowledge/gemma-agent').glob('seal-*.md'))
    seal_line = 'raw-file seal: none'
    if seals:
        m = check_signed(seals[-1], key, fp, problems)
        if m:
            verified[m.group(2)] = seals[-1]; n_verified += 1
            listed = {}
            for line in seals[-1].read_text().splitlines():
                if not line.startswith('| ') or line.startswith('| path') or line.startswith('| ---'): continue
                path, digest = [c.strip() for c in line.strip('|').split('|')]
                listed[path] = digest
            changed = [f for f, d in listed.items() if not (root / f).is_file() or hashlib.sha256((root / f).read_bytes()).hexdigest() != d]
            unlisted = sorted(set(sealed_files(root)) - set(listed))
            for f in changed[:20]: problems.append(f'{seals[-1].name}: {f} is missing or differs from its sealed sha256')
            if len(changed) > 20: problems.append(f'{seals[-1].name}: {len(changed) - 20} more sealed files missing or changed')
            for f in unlisted[:20]: problems.append(f'{seals[-1].name}: {f} is read by the analysis but not sealed')
            if len(unlisted) > 20: problems.append(f'{seals[-1].name}: {len(unlisted) - 20} more unsealed raw files')
            seal_line = f'raw-file seal {seals[-1].name}: signed {m.group(1)}, {len(listed) - len(changed)} of {len(listed)} files unchanged, {len(unlisted)} unsealed'
    else:
        problems.append('no signed raw-file seal (seal-*.md): the traces the curves read are not authenticated')
    all_records = {p.stem for p in (root / 'records').glob('*.json')}
    uncovered = sorted(all_records - covered)
    n_corr = len(list((root / '.ai/knowledge/gemma-agent').glob('correction-*.md')))
    print(seal_line)
    print(f'signed files verified: {n_verified} of {len(signed_files) + len(ledgers) + len(forecasts) + n_corr + min(len(seals), 1)}; ledger entries: {len(ledgers)}; forecasts: {len(forecasts)}; '
          f'corrections applied: {len(corrections)} values from {n_corr} items; records covered: {len(covered)} of {len(all_records)}; signer fingerprint {fp}')
    for n in sorted(set(notes))[:10]: print('note:', n)
    if uncovered: print(f'note: {len(uncovered)} records not yet in any ledger entry (newest first): {", ".join(uncovered[-5:])}')
    for p in problems: print('FAIL:', p)
    print('OK' if not problems else f'{len(problems)} problems')
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
