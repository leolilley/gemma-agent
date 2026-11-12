"""LoRA training arm on Kaggle L4x4, driven by train.json from the gemma-agent-lora-run dataset. The kernel code is
the instrument and does not change between runs; each run is a new version of that dataset.

Data: rendered.jsonl from the gemma-agent-lora-render kernel (token ids rendered with the competition model's own
chat template and checked against the served token counts; only rows marked usable are trained on). Base: the
competition's gemma-4-e4b-it-qat-w4a16-ct checkpoint, decompressed to the bf16 values the scorer's vLLM serves
(compressed-tensors run_compressed=False), so the adapter is trained on the weights it will be served on.

Parallelism: `replicas` worker processes, each holding one copy of the model across its share of the GPUs
(device_map auto), each training its own shard; the LoRA gradients are averaged across workers (gloo, CPU) before
every optimizer update, so the result is one adapter trained on the whole shard set.

train.json: {"run_id", "tiers": ["qat", "bf16"], "max_total_tokens", "max_completion_tokens", "lora": {"r", "alpha",
  "dropout"}, "optim": {"lr", "warmup_updates", "accumulate", "epochs", "weight_decay"}, "seed", "replicas",
  "val_attempt_fraction", "val_max_examples", "adapter_name", "bundle": "<dir shipped in the dataset>"}
Outputs: /kaggle/working/lora-train/ (adapter/, bundle/, train-result.json, worker logs) and lora-train.zip.
"""
import glob, hashlib, importlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

os.environ.update({'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_OFFLINE': '1',
                   'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'TOKENIZERS_PARALLELISM': 'false'})
T0 = time.time(); OUT = Path('/kaggle/working/lora-train'); OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def find_input(*names):
    for n in names:
        for c in [Path('/kaggle/input') / n, *Path('/kaggle/input').glob(f'*/*/{Path(n).name}'), *Path('/kaggle/input').glob(Path(n).name)]:
            if c.exists(): return c
    for p in sorted(Path('/kaggle/input').rglob('*'))[:80]: print('input:', p)
    raise SystemExit(f'none of {names} mounted')


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# 0. environment: competition wheelhouse (as the organizer notebook) + peft
WH = find_input('datasets/metric/gemma-4-developer-agent-wheelhouse'); DATA = find_input('datasets/leolilley/gemma-agent-lora-data')
RUN = find_input('datasets/leolilley/gemma-agent-lora-run')
for pth in glob.glob('/usr/local/lib/python*/*-packages/*cutlass*.pth'):
    try: os.unlink(pth)
    except OSError: pass
tmp = Path('/tmp/wheelhouse'); tmp.mkdir(parents=True, exist_ok=True)
for w in WH.glob('*.whl'):
    if 'cutlass' in w.name.lower(): continue
    t = tmp / (w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name)
    if not t.exists():
        try: os.symlink(w, t)
        except OSError: shutil.copyfile(w, t)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *sorted(str(w) for w in tmp.glob('*.whl'))], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', str(next(DATA.glob('peft-*.whl')))], check=True)
importlib.invalidate_caches()
import importlib.metadata as md
versions = {d: md.version(d) for d in ('torch', 'transformers', 'peft', 'compressed-tensors', 'adk-submission', 'swegemma')}
cfg = json.loads((RUN / 'train.json').read_text())
rendered = next(iter(sorted(Path('/kaggle/input').rglob('rendered.jsonl'))), None)
if rendered is None: find_input('nothing-named-rendered.jsonl')
render_report = rendered.parent / 'render-report.json'
QAT = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2')
if not QAT.exists():
    c = [x.parent for x in Path('/kaggle/input').rglob('config.json') if 'e4b-it-qat' in str(x)]
    if c: QAT = c[0]
assert QAT.exists(), 'E4B QAT model not mounted'
import torch
gpus = torch.cuda.device_count(); R = int(cfg.get('replicas', 2))
# Kaggle hands a kernel the default T4x2 when the requested L4x4 is busy (one L4x4 session at a time); each replica
# needs two 22 GiB L4s, so anything else stops here before loading, at a cost of a couple of minutes.
names = [torch.cuda.get_device_name(i) for i in range(gpus)]; mem = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(gpus)]
if gpus != 2 * R or any(m < 20 for m in mem):
    (OUT / 'train-result.json').write_text(json.dumps({'ok': False, 'error': 'wrong machine', 'gpus': names, 'gib': mem, 'wanted': f'{2 * R} x L4'}))
    raise SystemExit(f'wrong machine for {R} replicas: {names} {mem}')
groups = [','.join(str(g) for g in range(i * (gpus // R), (i + 1) * (gpus // R))) for i in range(R)]
cfg['port'] = 29517; (OUT / 'train.json').write_text(json.dumps(cfg, indent=2))
ident = {'run_id': cfg['run_id'], 'versions': versions, 'model_path': str(QAT), 'rendered': str(rendered), 'rendered_sha256': sha(rendered),
         'render_report_sha256': sha(render_report) if render_report.exists() else None, 'train_json_sha256': sha(RUN / 'train.json'),
         'gpus': gpus, 'replicas': R, 'gpu_groups': groups, 'nvidia_smi': subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip().splitlines()}
(OUT / 'identity.json').write_text(json.dumps(ident, indent=2)); log('identity', json.dumps(ident)[:800])

# 1. workers
WORKER = r'''
import collections, json, os, sys, time, math, hashlib, random
import torch, torch.distributed as dist, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
rank, world = int(sys.argv[1]), int(sys.argv[2]); cfg = json.loads(open(sys.argv[3]).read()); data, out, model_path = sys.argv[4], sys.argv[5], sys.argv[6]
logf = open(f'{out}/worker-{rank}.log', 'a'); status = {}
def say(*a): print(f'[r{rank} {time.strftime("%H:%M:%S")}]', *a, file=logf, flush=True)
def note(**kw):
    status.update(kw); open(f'{out}/status-{rank}.json', 'w').write(json.dumps(status))
dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{cfg["port"]}', rank=rank, world_size=world, timeout=__import__('datetime').timedelta(hours=3))
say('process group up; visible GPUs', torch.cuda.device_count())

# data: usable rows of the chosen tiers within the length caps; validation = a seeded fraction of attempts; the rest
# shuffled (seeded) and cut into equal shards so every rank makes the same number of updates
rows = [json.loads(l) for l in open(data)]
rows = [r for r in rows if r['usable'] and r['tier'] in cfg['tiers'] and r['prompt_tokens'] + r['completion_tokens'] <= cfg['max_total_tokens'] and r['completion_tokens'] <= cfg['max_completion_tokens']]
def h(s): return int(hashlib.sha256(f'{cfg["seed"]}:{s}'.encode()).hexdigest()[:8], 16) / 0xffffffff
val_att = {a for a in sorted({r['attempt_id'] for r in rows}) if h(a) < cfg['val_attempt_fraction']}
val = [r for r in rows if r['attempt_id'] in val_att][:cfg.get('val_max_examples', 40)]; train = [r for r in rows if r['attempt_id'] not in val_att]
random.Random(cfg['seed']).shuffle(train)
n_per = len(train) // world; shard = train[rank * n_per:(rank + 1) * n_per]
note(rows_usable=len(rows), val_attempts=len(val_att), val_examples=len(val), train_examples=len(train), shard=len(shard),
     shard_tokens=sum(r['prompt_tokens'] + r['completion_tokens'] for r in shard))
say('data', json.dumps(status))

# model: the QAT checkpoint decompressed to the bf16 values vLLM serves; no fallback to another checkpoint
from transformers import AutoModelForImageTextToText, CompressedTensorsConfig
# Load onto the replica's first GPU, then spread the decoder layers over both GPUs with accelerate: device_map auto and
# balanced both ended with the whole 16 GiB model on one L4 after decompression (v3, v4), leaving no room to train.
model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map={'': 0}, quantization_config=CompressedTensorsConfig(run_compressed=False))
# half of each GPU; on a GPU over 40 GB (Modal A100) the first one is capped at 60% of the model's weights so the model
# splits as it does on an L4, where half the memory already forces the split
max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.5) for i in range(torch.cuda.device_count())}
if torch.cuda.get_device_properties(0).total_memory > 40 * 2**30:
    weights = sum(t.numel() * t.element_size() for t in [*model.parameters(), *model.buffers()])
    max_memory[0] = int(min(max_memory[0], weights * 0.6))
from accelerate import dispatch_model, infer_auto_device_map
no_split = sorted({type(m).__name__ for m in model.modules() if 'DecoderLayer' in type(m).__name__})
dmap = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split)
# skip_keys as from_pretrained passes it: the per-forward dict of shared key/value states must cross devices by reference, not as a copy (v5 KeyError)
model = dispatch_model(model, device_map=dmap, skip_keys=getattr(model, '_skip_keys_device_placement', None))
say('device map', no_split, collections.Counter(str(v) for v in dmap.values()))
q = [type(m).__name__ for n, m in model.named_modules() if n.endswith('language_model.layers.0.self_attn.q_proj')]
assert q == ['Linear'], f'q_proj is {q}, not a plain Linear after decompression'
model.config.use_cache = False
import peft.import_utils as _iu; _iu.is_torchao_available = lambda: False   # the image ships torchao 0.10; peft raises on it while choosing a layer type, and torchao is unused here
import peft.tuners.lora.torchao as _tq; _tq.is_torchao_available = lambda: False
from peft import LoraConfig, get_peft_model
PROJ = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
lcfg = LoraConfig(r=cfg['lora']['r'], lora_alpha=cfg['lora']['alpha'], lora_dropout=cfg['lora']['dropout'], bias='none', task_type='CAUSAL_LM',
                  target_modules=r'.*language_model\.layers\.\d+\.(self_attn|mlp)\.(' + '|'.join(PROJ) + r')')
model = get_peft_model(model, lcfg); model.gradient_checkpointing_enable(); model.enable_input_require_grads()
params = [p for p in model.parameters() if p.requires_grad]
dev = model.get_input_embeddings().weight.device
devs = collections.Counter(str(p.device) for p in model.parameters())
assert len(devs) == torch.cuda.device_count(), f'model not spread over the replica GPUs: {dict(devs)}'
note(trainable_params=sum(p.numel() for p in params), devices=dict(devs), attn=model.config._attn_implementation, max_memory=max_memory)
if len(devs) < torch.cuda.device_count(): say('WARNING: model is not spread over all visible GPUs', dict(devs))
say('model', json.dumps({k: status[k] for k in ('trainable_params', 'devices', 'attn')}))

def example_loss(r):
    # Logits for the target positions only (the full-sequence logits over a 262k vocabulary do not fit); the
    # cross-entropy runs in 1,024-token chunks under checkpointing so the float softmax is never kept for all positions.
    ids = r['prompt_token_ids'] + r['completion_token_ids']; k = len(r['completion_token_ids'])
    logits = model(input_ids=torch.tensor([ids], device=dev), logits_to_keep=k + 1).logits[0, :k]
    tgt = torch.tensor(r['completion_token_ids'], device=logits.device); total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for i in range(0, k, 1024):
        total = total + checkpoint(lambda lg, tg: F.cross_entropy(lg.float(), tg, reduction='sum'), logits[i:i + 1024], tgt[i:i + 1024], use_reentrant=False)
    return total / k, k

def val_loss():
    if not val: return None
    model.eval(); tot = n = 0
    with torch.no_grad():
        for r in val:
            l, k = example_loss(r); tot += l.item() * k; n += k
    model.train(); return tot / n

def sync_grads():
    # average the LoRA gradients across ranks: one flat float32 buffer on the CPU, one all-reduce
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).to('cpu', dtype=torch.float32) for p in params])
    dist.all_reduce(flat); flat /= world; i = 0
    for p in params:
        n = p.numel(); g = flat[i:i + n].view_as(p).to(p.device, dtype=p.dtype); i += n
        if p.grad is None: p.grad = g
        else: p.grad.copy_(g)

ACC = int(cfg['optim']['accumulate']); EPOCHS = int(cfg['optim']['epochs'])
updates_total = max(1, (len(shard) * EPOCHS) // ACC); warm = int(cfg['optim']['warmup_updates'])
opt = torch.optim.AdamW(params, lr=cfg['optim']['lr'], weight_decay=cfg['optim'].get('weight_decay', 0.0))
def lr_at(u): return (u + 1) / max(1, warm) if u < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (u - warm) / max(1, updates_total - warm))))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
if rank == 0:
    with model.disable_adapter(): note(val_loss_base=val_loss())
    note(val_loss_init=val_loss()); say('val', status.get('val_loss_base'), status.get('val_loss_init'))
dist.barrier()
model.train(); losses = []; skipped = 0; seen = 0; t0 = time.time(); upd = 0
for ep in range(EPOCHS):
    order = list(range(len(shard))); random.Random(cfg['seed'] + ep + 17 * rank).shuffle(order)
    for j, idx in enumerate(order):
        r = shard[idx]
        try:
            loss, k = example_loss(r); (loss / ACC).backward(); losses.append(loss.item()); seen += 1
        except torch.cuda.OutOfMemoryError:
            skipped += 1; say('OOM skipped', r['example_id'], r['prompt_tokens'], r['completion_tokens'])
            loss = None; import gc; gc.collect(); torch.cuda.empty_cache()
            if skipped > 3 + 0.05 * (j + 1):   # more than one in twenty: the memory plan is wrong, stop spending GPU time
                note(aborted='too many OOM skips'); raise SystemExit('too many OOM skips')
        if (j + 1) % ACC == 0:
            sync_grads(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); upd += 1
        if (j + 1) % 10 == 0:
            el = time.time() - t0; note(epoch=ep, examples=ep * len(shard) + j + 1, seen=seen, skipped=skipped, updates=upd, lr=sched.get_last_lr()[0],
                                        loss_last20=sum(losses[-20:]) / max(1, len(losses[-20:])), s_per_example=el / (ep * len(shard) + j + 1), elapsed_s=round(el))
            say(json.dumps({k: status[k] for k in ('epoch', 'examples', 'seen', 'skipped', 'updates', 'lr', 'loss_last20', 's_per_example')}))
note(train_seconds=round(time.time() - t0), examples_total=EPOCHS * len(shard), seen=seen, skipped=skipped, updates=upd,
     loss_first20=sum(losses[:20]) / max(1, len(losses[:20])), loss_last20=sum(losses[-20:]) / max(1, len(losses[-20:])), losses=losses)
dist.barrier()
if rank == 0:
    note(val_loss_trained=val_loss()); say('val trained', status['val_loss_trained'])
    ad = f'{out}/adapter'; model.save_pretrained(ad)
    c = json.loads(open(f'{ad}/adapter_config.json').read()); c['target_modules'] = PROJ; c['base_model_name_or_path'] = 'google/gemma-4-e4b-it-qat-w4a16-ct'
    open(f'{ad}/adapter_config.json', 'w').write(json.dumps(c, indent=2))
    from safetensors import safe_open
    with safe_open(f'{ad}/adapter_model.safetensors', 'pt') as f: keys = list(f.keys())
    note(adapter_tensors=len(keys), adapter_key_example=keys[:2], adapter_keys_all_language_model=all('.language_model.layers.' in k for k in keys), done=True)
dist.barrier(); dist.destroy_process_group(); say('done')
'''
Path('/tmp/train_worker.py').write_text(WORKER)
procs = []
for i, g in enumerate(groups):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': g}
    procs.append(subprocess.Popen([sys.executable, '/tmp/train_worker.py', str(i), str(R), str(OUT / 'train.json'), str(rendered), str(OUT), str(QAT)],
                                  env=env, stdout=open(OUT / f'worker-{i}.out', 'w'), stderr=subprocess.STDOUT))
log('workers started', [p.pid for p in procs], groups)
last = ''
while any(p.poll() is None for p in procs):
    time.sleep(60)
    st = {}
    for i in range(R):
        f = OUT / f'status-{i}.json'
        if f.exists():
            try: s = json.loads(f.read_text()); st[i] = {k: s[k] for k in ('examples', 'seen', 'skipped', 'updates', 'loss_last20', 's_per_example') if k in s}
            except Exception: pass
    line = json.dumps(st)
    if line != last: log('progress', line); last = line
    if time.time() - T0 > 10.5 * 3600: log('time limit, terminating'); [p.terminate() for p in procs]; break
rcs = [p.wait() for p in procs]; log('workers exited', rcs)
for i in range(R):
    o = OUT / f'worker-{i}.out'
    if o.exists() and rcs[i] != 0: print(f'--- worker {i} output tail ---\n' + o.read_text()[-3000:])
status = {i: json.loads((OUT / f'status-{i}.json').read_text()) for i in range(R) if (OUT / f'status-{i}.json').exists()}
result = {'identity': ident, 'config': cfg, 'worker_returncodes': rcs, 'status': {str(i): {k: v for k, v in s.items() if k != 'losses'} for i, s in status.items()},
          'losses': {str(i): s.get('losses') for i, s in status.items()}}
ok = all(rc == 0 for rc in rcs) and (OUT / 'adapter' / 'adapter_model.safetensors').exists()

# 2. the submission bundle: the base bundle shipped in the run dataset plus the adapter, checked with the scorer's discovery
if ok:
    AG = OUT / 'bundle'
    if AG.exists(): shutil.rmtree(AG)
    shutil.copytree(RUN / 'bundle', AG, ignore=shutil.ignore_patterns('PROVENANCE.md'))
    shutil.copytree(OUT / 'adapter', AG / 'adapters' / cfg['adapter_name'])
    for junk in ('README.md',):
        if (AG / 'adapters' / cfg['adapter_name'] / junk).exists(): (AG / 'adapters' / cfg['adapter_name'] / junk).unlink()
    ay = (AG / 'agent.yaml').read_text()
    assert 'adapter:' not in ay, ay
    (AG / 'agent.yaml').write_text(ay.replace('\ninstruction:', f'\nadapter: {cfg["adapter_name"]}\ninstruction:', 1))
    from adk_submission import discover_adapters
    from adk_submission.discovery import discover_declared_models
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    adapters = discover_adapters(str(AG), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    amap = getattr(adapters, 'adapters', None) or {}
    size = sum(p.stat().st_size for p in (AG / 'adapters').rglob('*') if p.is_file())
    result['bundle'] = {'declared_models': sorted(discover_declared_models(AG)), 'adapters': {k: {'rank': getattr(v, 'rank', None), 'path': str(getattr(v, 'path', ''))} for k, v in amap.items()},
                        'adapters_bytes': size, 'under_3GiB': size < 3 * 2**30, 'adapter_sha256': sha(AG / 'adapters' / cfg['adapter_name'] / 'adapter_model.safetensors'),
                        'agent_yaml': (AG / 'agent.yaml').read_text()}
    log('bundle', json.dumps(result['bundle'])[:600])
result['ok'] = ok; result['seconds'] = round(time.time() - T0, 1)
(OUT / 'train-result.json').write_text(json.dumps(result, indent=2))
with zipfile.ZipFile('/kaggle/working/lora-train.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file(): z.write(p, p.relative_to(OUT.parent))
log('done', ok, result['seconds'])
if not ok: raise SystemExit('training failed')
