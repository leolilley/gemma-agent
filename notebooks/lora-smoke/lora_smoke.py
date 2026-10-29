"""LoRA route smoke on Kaggle L4x4 (one session, no internet): can a PEFT LoRA be trained on the competition's E4B
checkpoint, served by the scorer's vLLM on the QAT W4A16 base, and change its output without breaking it?

1. Train (subprocess, so the GPUs are released): load gemma-4-e4b-it-qat-w4a16-ct decompressed to bf16 (falls back to
   the bf16 release, recorded), attach LoRA r=16 alpha=8 (scale 0.5) to the language model's seven projections, train
   on smoke-train.jsonl with the loss on target tokens only; held-out loss before and after on smoke-heldout.jsonl
   (repository-disjoint). Save adapter_config.json + adapter_model.safetensors in the verified module naming.
2. Serve: adk_submission VllmServer with the scorer's flags and enable_lora on the QAT base.
3. Probe: first-token top-20 for base, adapter, base repeat (effect vs drift); greedy generations from held-out prompts,
   base and adapter, with a repetition measure.
Outputs: /kaggle/working/lora-smoke/ (result.json, adapter/, logs) and lora-smoke.zip."""
import glob, importlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

os.environ.update({'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_OFFLINE': '1',
    'VLLM_NO_USAGE_STATS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_ENGINE_READY_TIMEOUT_S': '2400',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',   # as the eval kernel: without it the KV cache fills the GPU and cudagraph capture OOMs
    'NCCL_DEBUG': 'WARN', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OPENAI_API_KEY': 'EMPTY'})
T0 = time.time(); OUT = Path('/kaggle/working/lora-smoke'); OUT.mkdir(parents=True, exist_ok=True)
result = {'started_unix': T0, 'phases': {}}
def save():
    (OUT / 'result.json').write_text(json.dumps(result, indent=2))
def phase(name, **kw):
    result['phases'][name] = {'t_s': round(time.time() - T0, 1), **kw}; save(); print(f'[{time.time()-T0:7.1f}s] {name}', json.dumps(kw)[:600], flush=True)

def find_input(*names):
    for n in names:
        for c in [Path('/kaggle/input') / n, *Path('/kaggle/input').glob(f'*/*/{Path(n).name}'), *Path('/kaggle/input').glob(Path(n).name)]:
            if c.exists(): return c
    for p in sorted(Path('/kaggle/input').rglob('*'))[:60]: print('input:', p)
    raise SystemExit(f'none of {names} mounted')

# 0. environment: competition wheelhouse (as the organizer notebook) + peft
WH = find_input('datasets/metric/gemma-4-developer-agent-wheelhouse'); DS = find_input('datasets/leolilley/gemma-agent-lora-smoke')
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
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', str(next(DS.glob('peft-*.whl')))], check=True)
importlib.invalidate_caches()
import importlib.metadata as md
result['versions'] = {d: md.version(d) for d in ('torch', 'transformers', 'peft', 'vllm', 'compressed-tensors', 'adk-submission')}
QAT = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2')
BF16 = Path('/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/1')
for p, name in ((QAT, 'qat'), (BF16, 'bf16')):
    if not p.exists():
        c = [x.parent for x in Path('/kaggle/input').rglob('config.json') if ('e4b-it-qat' in str(x) if name == 'qat' else ('e4b-it' in str(x) and 'qat' not in str(x)))]
        if c: globals()['QAT' if name == 'qat' else 'BF16'] = c[0]
phase('environment', versions=result['versions'], qat=str(QAT), bf16=str(BF16), gpus=subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip().splitlines())

# 1. train in a subprocess, unless a trained smoke adapter is mounted (then serve and probe only)
ADAPTER_DS = next((c for c in [Path('/kaggle/input/datasets/leolilley/gemma-agent-lora-smoke-adapter'), *Path('/kaggle/input').glob('*/*/gemma-agent-lora-smoke-adapter'), *Path('/kaggle/input').glob('gemma-agent-lora-smoke-adapter')] if (c / 'adapter' / 'adapter_model.safetensors').exists()), None)
TRAIN = r'''
import json, sys, time, math, torch
from pathlib import Path
qat, bf16, ds, out = map(Path, sys.argv[1:5])
info = {}
from transformers import AutoModelForImageTextToText
def load(path, compressed):
    kw = dict(dtype=torch.bfloat16, device_map='auto')
    if compressed:
        from transformers import CompressedTensorsConfig
        kw['quantization_config'] = CompressedTensorsConfig(run_compressed=False)
    m = AutoModelForImageTextToText.from_pretrained(str(path), **kw)
    if compressed and hasattr(m, 'dequantize'):
        try: m.dequantize(); info['dequantize_called'] = True
        except Exception as e: info['dequantize_error'] = f'{type(e).__name__}: {str(e)[:300]}'
    return m
try:
    model = load(qat, True); info['base'] = 'qat-w4a16-decompressed'
    lin = [type(mod).__name__ for n, mod in model.named_modules() if n.endswith('language_model.layers.0.self_attn.q_proj')]
    info['q_proj_type'] = lin
    if not lin or lin[0] != 'Linear': raise RuntimeError(f'q_proj is {lin}, not a plain Linear after decompression')
except Exception as e:
    info['qat_load_error'] = f'{type(e).__name__}: {str(e)[:500]}'
    import gc; gc.collect(); torch.cuda.empty_cache()
    model = load(bf16, False); info['base'] = 'bf16-release'
model.config.use_cache = False
import peft.import_utils as _iu
_iu.is_torchao_available = lambda: False   # the image ships torchao 0.10; peft raises on it while choosing a layer type, and torchao is unused here
import peft.tuners.lora.torchao as _tq
_tq.is_torchao_available = lambda: False
from peft import LoraConfig, get_peft_model
info['torchao_disabled_for_peft'] = True
PROJ = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
cfg = LoraConfig(r=16, lora_alpha=8, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM',
                 target_modules=r'.*language_model\.layers\.\d+\.(self_attn|mlp)\.(' + '|'.join(PROJ) + r')')
model = get_peft_model(model, cfg); model.gradient_checkpointing_enable(); model.enable_input_require_grads()
info['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
dev = next(model.parameters()).device
def rows(name): return [json.loads(l) for l in (ds / name).read_text().splitlines() if l.strip()]
train, held = rows('smoke-train.jsonl'), rows('smoke-heldout.jsonl')
def target_loss(r):
    # Loss on the target tokens only, with logits computed for those positions alone (the full-sequence logits over a
    # 262k vocabulary do not fit beside the activations on one L4).
    ids = r['prompt_token_ids'] + r['completion_token_ids']; k = len(r['completion_token_ids'])
    out = model(input_ids=torch.tensor([ids], device=dev), logits_to_keep=k + 1)
    logits = out.logits[0, :k].float(); tgt = torch.tensor(r['completion_token_ids'], device=logits.device)
    return torch.nn.functional.cross_entropy(logits, tgt)
def note(**kw):
    info.update(kw); (out / 'train.json').write_text(json.dumps(info, indent=2))
def heldout_loss():
    model.eval(); tot = n = 0
    with torch.no_grad():
        for r in held:
            l = target_loss(r).item(); k = len(r['completion_token_ids']); tot += l * k; n += k
    model.train(); return tot / n
with model.disable_adapter(): note(heldout_loss_base=heldout_loss())
note(heldout_loss_init=heldout_loss())
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
ACC = 4; losses = []; t0 = time.time(); model.train()
for i, r in enumerate(train):
    loss = target_loss(r); (loss / ACC).backward(); losses.append(loss.item())
    if (i + 1) % ACC == 0: opt.step(); opt.zero_grad()
    if i % 20 == 0: print(f'step {i} loss {loss.item():.4f} {time.time()-t0:.0f}s', flush=True); note(progress_step=i, progress_loss=loss.item())
info.update({'train_steps': len(train), 'optimizer_updates': len(train) // ACC, 'train_seconds': round(time.time() - t0),
             'train_loss_first20': sum(losses[:20]) / 20, 'train_loss_last20': sum(losses[-20:]) / 20})
note(heldout_loss_trained=heldout_loss())
ad = out / 'adapter'; model.save_pretrained(str(ad))
c = json.loads((ad / 'adapter_config.json').read_text()); c['target_modules'] = PROJ
c['base_model_name_or_path'] = 'google/gemma-4-e4b-it-qat-w4a16-ct'; (ad / 'adapter_config.json').write_text(json.dumps(c, indent=2))
from safetensors import safe_open
with safe_open(str(ad / 'adapter_model.safetensors'), 'pt') as f: keys = list(f.keys())
info['adapter_tensors'] = len(keys); info['adapter_key_example'] = keys[:2]
info['adapter_keys_all_language_model'] = all('.language_model.layers.' in k for k in keys)
(out / 'train.json').write_text(json.dumps(info, indent=2)); print(json.dumps(info), flush=True)
'''
if ADAPTER_DS:
    shutil.copytree(ADAPTER_DS / 'adapter', OUT / 'adapter', dirs_exist_ok=True)
    import hashlib
    phase('train', reused_from=str(ADAPTER_DS), adapter_sha256=hashlib.sha256((OUT / 'adapter' / 'adapter_model.safetensors').read_bytes()).hexdigest(),
          **(json.loads((ADAPTER_DS / 'train.json').read_text()) if (ADAPTER_DS / 'train.json').exists() else {}))
else:
    (Path('/tmp/train.py')).write_text(TRAIN)
    p = subprocess.run([sys.executable, '/tmp/train.py', str(QAT), str(BF16), str(DS), str(OUT)], capture_output=True, text=True)
    (OUT / 'train.log').write_text(p.stdout + '\n----- stderr -----\n' + p.stderr)
    train_info = json.loads((OUT / 'train.json').read_text()) if (OUT / 'train.json').exists() else {}
    phase('train', returncode=p.returncode, **train_info, **({} if p.returncode == 0 else {'stderr_tail': p.stderr[-2500:]}))
    if p.returncode != 0:
        shutil.make_archive('/kaggle/working/lora-smoke', 'zip', OUT); raise SystemExit('training failed')

# 2. serve with the scorer's flags on the QAT base
from adk_submission import VllmConfig, VllmServer, discover_adapters
AG = Path('/kaggle/working/agent'); (AG / 'adapters').mkdir(parents=True, exist_ok=True)
shutil.copytree(OUT / 'adapter', AG / 'adapters' / 'smoke_lora', dirs_exist_ok=True)
(AG / 'agent.yaml').write_text('name: smoke\nmodel: gemma-4-e4b-it-qat-w4a16-ct\nadapter: smoke_lora\ninstruction: smoke\n')
adapters = discover_adapters(str(AG), adapter_extensions=('.safetensors',))
amap = getattr(adapters, 'adapters', None) or {}
import torch
tp = torch.cuda.device_count()
server = VllmServer(VllmConfig(model=str(QAT), port=8000, host='127.0.0.1', tool_call_parser='gemma4', enable_auto_tool_choice=True,
                               reasoning_parser='gemma4', max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90,
                               enable_lora=bool(amap), max_loras=max(1, len(amap)), max_lora_rank=16, scheduler_reserve_full_isl=False,
                               tensor_parallel_size=tp, startup_timeout=60 * 20), adapter_manifest=adapters)   # the scorer's flags, as the eval kernel
try:
    server.start(); phase('serve', base_url=server.base_url, adapters=sorted(amap), tp=tp)
except Exception as e:
    phase('serve', error=f'{type(e).__name__}: {str(e)[:1500]}'); shutil.make_archive('/kaggle/working/lora-smoke', 'zip', OUT); raise
import urllib.request
def post(path, payload):
    req = urllib.request.Request(server.base_url.rstrip('/') + path, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
    with urllib.request.urlopen(req, timeout=600) as r: return json.loads(r.read())
with urllib.request.urlopen(server.base_url.rstrip('/') + '/models', timeout=60) as r: models = [m['id'] for m in json.loads(r.read())['data']]
base_name = next((m for m in models if m != 'smoke_lora'), models[0]); ad_name = 'smoke_lora' if 'smoke_lora' in models else None

# 3. probe
PROMPT = 'Complete this Python expression with the numeric result only: len([1, 2, 3]) ='
def top20(model):
    d = post('/chat/completions', {'model': model, 'messages': [{'role': 'user', 'content': PROMPT}], 'temperature': 0, 'seed': 42, 'max_tokens': 8,
                                   'logprobs': True, 'top_logprobs': 20, 'chat_template_kwargs': {'enable_thinking': False}})
    return {t['token']: t['logprob'] for t in d['choices'][0]['logprobs']['content'][0]['top_logprobs']}
probe = {}
if ad_name:
    top20(base_name); top20(ad_name)
    b1, a1, b2 = top20(base_name), top20(ad_name), top20(base_name)
    drift = max((abs(b1[t] - b2[t]) for t in b1.keys() & b2.keys()), default=0.0)
    eff = max((abs(b1[t] - a1[t]) for t in b1.keys() & a1.keys()), default=0.0)
    probe = {'models': models, 'base_drift': drift, 'adapter_max_delta': eff, 'base_top1': max(b1, key=b1.get), 'adapter_top1': max(a1, key=a1.get),
             'effect_over_drift': eff > max(1e-5, 10 * drift)}
held = [json.loads(l) for l in (DS / 'smoke-heldout.jsonl').read_text().splitlines() if l.strip()]
def rep(ids):
    grams = [tuple(ids[i:i + 8]) for i in range(max(0, len(ids) - 7))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0
gens = []
for r in held[:4]:
    row = {'trajectory_id': r['trajectory_id'], 'turn': r['turn']}
    for label, m in (('base', base_name), ('adapter', ad_name)):
        if not m: continue
        d = post('/completions', {'model': m, 'prompt': r['prompt_token_ids'], 'max_tokens': 384, 'temperature': 0, 'seed': 42})
        txt = d['choices'][0]['text']; row[label] = {'text_head': txt[:600], 'chars': len(txt), 'finish': d['choices'][0].get('finish_reason'),
                                                    'repeat_8gram_frac_chars': rep(list(txt))}
    gens.append(row)
phase('probe', **probe, generations=gens)
server.stop()
result['finished_s'] = round(time.time() - T0, 1); save()
shutil.make_archive('/kaggle/working/lora-smoke', 'zip', OUT); print('done', result['finished_s'])
