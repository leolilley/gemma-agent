#!/usr/bin/env python3
"""Checks a signed forecast file before a run that it predicts may be launched (used by scripts/push-run.py).

A forecast is acceptable for a run when: it lives under .ai/knowledge/gemma-agent/ as forecast-*.md; its header
`ryeos:signed:<time>:<sha256 body>:<ed25519 sig over the hex digest>:<signer fp>` is present; the body hash matches; the
signer fingerprint is the operator key; the signature verifies against the shipped public key
(docs/results/signing-public-key.pem, so the launcher refuses until that file exists); and its
`Predicts: bundle=<b> site=<s> tasks=<t1,t2,...>` line names the run's bundle and site and covers every task of the run.
Returns a dict (signed_at, body_digest, file_digest, bundle, site, tasks) or raises ForecastError with the reason.
"""
import base64, hashlib, re
from pathlib import Path

SIG = re.compile(r'ryeos:signed:(\S+?):([0-9a-f]{64}):([A-Za-z0-9+/=]+):([0-9a-f]{64})')
PREDICTS = re.compile(r'^Predicts:\s*bundle=(\S+)\s+site=(\S+)\s+tasks=(\S+)', re.M)
OPERATOR_FP = '82970d668dcc39d38a33264f59b1bc26760be487512a086be87bfd51e2d88b19'


class ForecastError(Exception): pass


def load_pubkey(path: Path):
    from cryptography.hazmat.primitives import serialization
    key = serialization.load_pem_public_key(path.read_bytes())
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return key, hashlib.sha256(raw).hexdigest()


def check_forecast(path: Path, bundle: str, site: str, task_ids, pubkey: Path, operator_fp: str = OPERATOR_FP, knowledge_dir: Path | None = None) -> dict:
    path = path.resolve()
    if knowledge_dir is not None and (path.parent != knowledge_dir.resolve() or not re.fullmatch(r'forecast-.*\.md', path.name)):
        raise ForecastError(f'{path} is not a knowledge item .ai/knowledge/gemma-agent/forecast-*.md (verify.py only checks those)')
    if not pubkey.is_file():
        raise ForecastError(f'public key {pubkey} is not shipped yet; the forecast run cannot be launched until the signature can be checked')
    text = path.read_text(); nl = text.find('\n'); first, body = text[:nl], text[nl + 1:]
    m = SIG.search(first)
    if not m: raise ForecastError(f'{path.name}: no signature header')
    signed_at, digest, sig, fp = m.groups()
    if hashlib.sha256(body.encode()).hexdigest() != digest: raise ForecastError(f'{path.name}: body hash differs from the header (edited after signing?)')
    key, key_fp = load_pubkey(pubkey)
    if fp != operator_fp or key_fp != operator_fp: raise ForecastError(f'{path.name}: signer {fp[:12]} / key {key_fp[:12]} is not the operator key {operator_fp[:12]}')
    try: key.verify(base64.b64decode(sig), digest.encode())
    except Exception: raise ForecastError(f'{path.name}: signature does not verify against {pubkey}')
    pm = PREDICTS.search(body)
    if not pm: raise ForecastError(f'{path.name}: no "Predicts: bundle=.. site=.. tasks=.." line')
    f_bundle, f_site, f_tasks = pm.group(1), pm.group(2), set(pm.group(3).split(','))
    if f_bundle != bundle or f_site != site: raise ForecastError(f'{path.name} predicts {f_bundle} @ {f_site}, the run is {bundle} @ {site}')
    missing = sorted(set(task_ids) - f_tasks)
    if missing: raise ForecastError(f'{path.name} does not cover {len(missing)} of the run\'s tasks: {", ".join(missing[:6])}')
    return {'signed_at': signed_at, 'body_digest': digest, 'file_digest': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bundle': f_bundle, 'site': f_site, 'tasks': sorted(f_tasks), 'signer': fp}
