#!/usr/bin/env python3
"""tool:gemma-agent/lora_register and tool:gemma-agent/lora_accept — the two bookkeeping steps of one LoRA cycle.

register: take one training kernel's downloaded output (identity.json, train.json, train-result.json, adapter/, bundle/)
and write the cycle's training record under results/lora/cycles/<cycle>/train.json: the configuration, the harness
and library versions, the digests of the dataset, the rendered token ids and the render report it trained on, the
adapter's digest, size and rank, the losses, and the bundle it produced. Install that bundle under
submissions/<bundle>/ with a PROVENANCE.md naming the record, so the evaluation arm runs the bytes the record names.

accept: apply the cycle's fixed acceptance rule to the two evaluation arms (base bundle and adapter bundle) on the
evaluation split of the dataset manifest: every task of the split must have a valid record for both arms; the adapter
is adopted when it resolves at least `margin` more tasks than the base. Writes results/lora/cycles/<cycle>/accept.json
and results/lora/selected.json (the bundle the campaign submits). Stdlib only."""
import argparse, hashlib, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_dir(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob('*') if x.is_file()):
        h.update(str(p.relative_to(d)).encode()); h.update(b'\0'); h.update(p.read_bytes()); h.update(b'\0')
    return h.hexdigest()


def register(root: Path, p: dict) -> dict:
    out = root / p['train_out']; cycle = p['cycle']
    need = ['identity.json', 'train.json', 'train-result.json', 'adapter/adapter_model.safetensors', 'adapter/adapter_config.json', 'bundle/agent.yaml']
    missing = [n for n in need if not (out / n).exists()]
    if missing: return {'ok': False, 'error': f'training output lacks {missing}'}
    ident = json.loads((out / 'identity.json').read_text()); cfg = json.loads((out / 'train.json').read_text()); res = json.loads((out / 'train-result.json').read_text())
    if not res.get('ok'): return {'ok': False, 'error': 'training result is not ok', 'worker_returncodes': res.get('worker_returncodes')}
    ds = root / p['dataset']; manifest = json.loads((ds / 'manifest.json').read_text())
    render = root / p['render_out']; rr = json.loads((render / 'render-report.json').read_text())
    if rr.get('examples_sha256') != manifest.get('examples_sha256'):
        return {'ok': False, 'error': 'render report does not name the dataset examples', 'render': rr.get('examples_sha256'), 'dataset': manifest.get('examples_sha256')}
    rendered_sha = sha(render / 'rendered.jsonl'); dagger = None
    if p.get('dagger'):
        # DAgger cycles train on the render output's rows plus labelled rows, combined by scripts/dagger_rows.py
        dd = root / p['dagger']; dm = json.loads((dd / 'manifest.json').read_text())
        if dm['teacher']['sha256'] != rendered_sha:
            return {'ok': False, 'error': 'the DAgger data was not built on the render output given', 'dagger_teacher': dm['teacher']['sha256'], 'render_out': rendered_sha}
        if sha(dd / 'rendered.jsonl') != dm['rendered_sha256']:
            return {'ok': False, 'error': 'the DAgger rendered.jsonl does not match its manifest'}
        dagger = {'dir': p['dagger'], 'manifest_sha256': sha(dd / 'manifest.json'), 'rows': dm['dagger'], 'rendered_sha256': dm['rendered_sha256']}
        rendered_sha = dm['rendered_sha256']
    if ident.get('rendered_sha256') != rendered_sha:
        return {'ok': False, 'error': 'the training identity names a different rendered.jsonl than the render output given', 'identity': ident.get('rendered_sha256'), 'render_out': rendered_sha}
    acfg = json.loads((out / 'adapter/adapter_config.json').read_text())
    bundle_name = p['bundle']; dest = root / 'submissions' / bundle_name
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(out / 'bundle', dest)
    status0 = (res.get('status') or {}).get('0') or {}
    rec = {'cycle': cycle, 'registered_at': now(), 'run_id': cfg['run_id'], 'config': cfg, 'versions': ident.get('versions'), 'model_path': ident.get('model_path'),
           'gpus': ident.get('gpus'), 'replicas': ident.get('replicas'),
           'dataset': {'id': manifest['dataset'], 'manifest_sha256': sha(ds / 'manifest.json'), 'examples_sha256': manifest['examples_sha256'], 'rule': manifest['rule'],
                       'evaluation_tasks': manifest['split']['evaluation_tasks'], 'seal': manifest.get('seal')},
           'rendering': {'rendered_sha256': rendered_sha, 'render_report_sha256': sha(render / 'render-report.json'), 'checks': rr.get('checks'), 'by_tier': rr.get('by_tier'),
                         'templates': {k: v.get('chat_template.jinja') for k, v in (rr.get('model_files') or {}).items()}},
           'dagger': dagger,
           'training': {k: status0.get(k) for k in ('rows_usable', 'val_attempts', 'val_examples', 'train_examples', 'shard', 'trainable_params', 'attn', 'train_seconds',
                                                      'examples_total', 'seen', 'skipped', 'updates', 'loss_first20', 'loss_last20', 'val_loss_base', 'val_loss_init', 'val_loss_trained')},
           'adapter': {'name': cfg['adapter_name'], 'sha256': sha(out / 'adapter/adapter_model.safetensors'), 'bytes': (out / 'adapter/adapter_model.safetensors').stat().st_size,
                       'r': acfg.get('r'), 'lora_alpha': acfg.get('lora_alpha'), 'target_modules': acfg.get('target_modules'), 'base_model_name_or_path': acfg.get('base_model_name_or_path')},
           'bundle': {'name': bundle_name, 'sha256': sha_dir(dest), 'parent': cfg.get('bundle'), 'scorer_discovery': res.get('bundle')}}
    cdir = root / 'results/lora/cycles' / cycle; cdir.mkdir(parents=True, exist_ok=True)
    (cdir / 'train.json').write_text(json.dumps(rec, indent=1))
    (dest / 'PROVENANCE.md').write_text(
        f'# {bundle_name}: cycle {cycle} adapter arm\n\nThe `{cfg.get("bundle")}` bundle plus `adapters/{cfg["adapter_name"]}` (LoRA r={acfg.get("r")}, '
        f'alpha={acfg.get("lora_alpha")}, sha256 {rec["adapter"]["sha256"]}) and the `adapter:` line in agent.yaml. Trained by run `{cfg["run_id"]}` on dataset '
        f'`{manifest["dataset"]}` (examples sha256 {manifest["examples_sha256"]}); the training record is results/lora/cycles/{cycle}/train.json.\n')
    return {'ok': True, 'record': str((cdir / 'train.json').relative_to(root)), 'bundle': bundle_name, 'bundle_sha256': rec['bundle']['sha256'], 'adapter_sha256': rec['adapter']['sha256'],
            'training': rec['training']}


def arm_records(root: Path, run_id: str, tasks: list) -> dict:
    out = {}
    for rf in sorted((root / 'records').glob(f'*-{run_id}.json')):
        r = json.loads(rf.read_text())
        if r.get('task') in tasks or r.get('task_id') in tasks:
            out[r.get('task') or r.get('task_id')] = {'file': str(rf.relative_to(root)), 'sha256': sha(rf), 'bundle': r.get('bundle'), 'validity': r.get('validity'),
                                                      'resolved': (r.get('outcome') or {}).get('resolved'), 'infrastructure_failure': r.get('infrastructure_failure')}
    return out


def accept(root: Path, p: dict) -> dict:
    cycle = p['cycle']; cdir = root / 'results/lora/cycles' / cycle
    train = json.loads((cdir / 'train.json').read_text())
    tasks = list(train['dataset']['evaluation_tasks']); margin = int(p.get('margin', 2))
    base = arm_records(root, p['base_run_id'], tasks); adapter = arm_records(root, p['adapter_run_id'], tasks)
    problems = []
    for name, arm, want in (('base', base, p.get('base_bundle')), ('adapter', adapter, train['bundle']['name'])):
        missing = [t for t in tasks if t not in arm]
        if missing: problems.append(f'{name} arm has no record for {missing}')
        bad = [t for t, r in arm.items() if r['validity'] != 'valid' or r['infrastructure_failure']]
        if bad: problems.append(f'{name} arm records not valid: {bad}')
        wrong = [t for t, r in arm.items() if want and r['bundle'] != want]
        if wrong: problems.append(f'{name} arm records name another bundle: {wrong}')
    if problems: return {'ok': False, 'error': problems}
    nb = sum(1 for t in tasks if base[t]['resolved'] is True); na = sum(1 for t in tasks if adapter[t]['resolved'] is True)
    adopted = na - nb >= margin
    rule = f'adopt the adapter bundle only if its arm resolves at least {margin} more of the {len(tasks)} evaluation tasks than the base arm, each arm one attempt per task at the same cell; otherwise the base stays'
    decision = {'adopted': train['bundle']['name'] if adopted else p.get('base_bundle'), 'bundle_sha256': train['bundle']['sha256'] if adopted else sha_dir(root / 'submissions' / p['base_bundle'])}
    rec = {'cycle': cycle, 'decided_at': now(), 'rule': rule, 'margin': margin, 'evaluation_tasks': tasks,
           'base': {'run_id': p['base_run_id'], 'bundle': p.get('base_bundle'), 'resolved': nb, 'records': base},
           'adapter': {'run_id': p['adapter_run_id'], 'bundle': train['bundle']['name'], 'resolved': na, 'records': adapter},
           'per_task': {t: {'base': base[t]['resolved'], 'adapter': adapter[t]['resolved']} for t in tasks},
           'train_record_sha256': sha(cdir / 'train.json'), **decision}
    (cdir / 'accept.json').write_text(json.dumps(rec, indent=1))
    sel = root / 'results/lora/selected.json'
    hist = json.loads(sel.read_text())['history'] if sel.exists() else []
    sel.write_text(json.dumps({'selected': decision, 'cycle': cycle, 'history': hist + [{k: v for k, v in rec.items() if k not in ('base', 'adapter', 'per_task')}]}, indent=1))
    return {'ok': True, 'cycle': cycle, 'base_resolved': nb, 'adapter_resolved': na, 'margin': margin, 'per_task': rec['per_task'], **decision}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['register', 'accept']); ap.add_argument('--project-path', required=True)
    a = ap.parse_args(); root = Path(a.project_path).resolve(); p = json.loads(sys.stdin.read() or '{}')
    print(json.dumps({'register': register, 'accept': accept}[a.mode](root, p)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
