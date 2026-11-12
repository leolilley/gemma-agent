"""Scale probe for the cycle 1 adapter (Kaggle L4x4, one session, no internet). The adapter arm looped: from the second turn
on, the agent repeated its previous reasoning and tool call verbatim. This kernel serves the QAT E4B base with the cycle 1
adapter at three scales (lora_alpha 16, 8, 4 over r = 16: scale 1.0, 0.5, 0.25; same weights) under the scorer's vLLM
flags, and generates greedily from held-out multi-turn prompts in exactly the served token form (the render kernel's
rendered.jsonl rows), base and the three scales, measuring length, 8-gram repetition and how much of the generation copies
the previous assistant turn that sits in the prompt. Inputs: the lora-train kernel output (adapter/), the lora-render
kernel output (rendered.jsonl), the wheelhouse, the E4B QAT mount. Output: /kaggle/working/lora-probe/{probe.json,
generations.jsonl} and lora-probe.zip."""
import difflib, glob, hashlib, importlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

os.environ.update({'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_OFFLINE': '1',
    'VLLM_NO_USAGE_STATS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_ENGINE_READY_TIMEOUT_S': '2400',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1', 'NCCL_DEBUG': 'WARN', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OPENAI_API_KEY': 'EMPTY'})
T0 = time.time(); OUT = Path('/kaggle/working/lora-probe'); OUT.mkdir(parents=True, exist_ok=True)
result = {'started_unix': T0, 'phases': {}}
def save(): (OUT / 'probe.json').write_text(json.dumps(result, indent=2))
def phase(name, **kw):
    result['phases'][name] = {'t_s': round(time.time() - T0, 1), **kw}; save(); print(f'[{time.time()-T0:7.1f}s] {name}', json.dumps(kw)[:800], flush=True)
def finish(msg):
    result['finished_s'] = round(time.time() - T0, 1); save(); shutil.make_archive('/kaggle/working/lora-probe', 'zip', OUT); print(msg, flush=True)

def find_input(*names):
    for n in names:
        for c in [Path('/kaggle/input') / n, *Path('/kaggle/input').glob(f'*/*/{Path(n).name}'), *Path('/kaggle/input').glob(Path(n).name)]:
            if c.exists(): return c
    for p in sorted(Path('/kaggle/input').rglob('*'))[:80]: print('input:', p)
    raise SystemExit(f'none of {names} mounted')

WH = find_input('datasets/metric/gemma-4-developer-agent-wheelhouse')
TRAIN_OUT = find_input('notebooks/leolilley/gemma-agent-lora-train/lora-train', 'gemma-agent-lora-train/lora-train')
RENDER_OUT = find_input('notebooks/leolilley/gemma-agent-lora-render/lora-render', 'gemma-agent-lora-render/lora-render')
QAT = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2')
if not QAT.exists(): QAT = next(x.parent for x in Path('/kaggle/input').rglob('config.json') if 'e4b-it-qat' in str(x))
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
importlib.invalidate_caches()
import importlib.metadata as md
result['versions'] = {d: md.version(d) for d in ('torch', 'transformers', 'vllm', 'compressed-tensors', 'adk-submission')}
adapter_src = TRAIN_OUT / 'adapter'
result['adapter_sha256'] = hashlib.sha256((adapter_src / 'adapter_model.safetensors').read_bytes()).hexdigest()
result['rendered_sha256'] = hashlib.sha256((RENDER_OUT / 'rendered.jsonl').read_bytes()).hexdigest()
phase('environment', versions=result['versions'], qat=str(QAT), adapter=str(adapter_src), adapter_sha256=result['adapter_sha256'], rendered_sha256=result['rendered_sha256'],
      gpus=subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip().splitlines())

# 1. the three scaled copies of the one adapter: same weights, lora_alpha 16 / 8 / 4
SCALES = {'c1_s100': 16, 'c1_s050': 8, 'c1_s025': 4}
AG = Path('/kaggle/working/agent'); (AG / 'adapters').mkdir(parents=True, exist_ok=True)
for name, alpha in SCALES.items():
    d = AG / 'adapters' / name; shutil.copytree(adapter_src, d, dirs_exist_ok=True)
    c = json.loads((d / 'adapter_config.json').read_text()); c['lora_alpha'] = alpha; (d / 'adapter_config.json').write_text(json.dumps(c, indent=2))
(AG / 'agent.yaml').write_text('name: probe\nmodel: gemma-4-e4b-it-qat-w4a16-ct\nadapter: c1_s100\ninstruction: probe\n')

# 2. prompts: held-out validation attempts (the training worker's split rule, seed 20261030, fraction 0.06) at calls 1, 2, 4, 8,
#    plus the first two 4-bit-tier training attempts at calls 1, 2, 4 (seen in training; labelled so)
SEED = 20261030
def h(s): return int(hashlib.sha256(f'{SEED}:{s}'.encode()).hexdigest()[:8], 16) / 0xffffffff
rows = [json.loads(l) for l in (RENDER_OUT / 'rendered.jsonl').read_text().splitlines() if l.strip()]
rows = [r for r in rows if r['usable'] and r['tier'] in ('qat', 'bf16') and len(r['prompt_token_ids']) + len(r['completion_token_ids']) <= 16384 and len(r['completion_token_ids']) <= 4096]
val_att = {a for a in sorted({r['attempt_id'] for r in rows}) if h(a) < 0.06}
qat_att = sorted({r['attempt_id'] for r in rows if r['tier'] == 'qat' and r['attempt_id'] not in val_att})[:2]
prompts = [dict(r, split='val') for r in rows if r['attempt_id'] in val_att and r['call_index'] in (1, 2, 4, 8)]
prompts += [dict(r, split='train') for r in rows if r['attempt_id'] in qat_att and r['call_index'] in (1, 2, 4)]
prompts.sort(key=lambda r: (r['split'], r['attempt_id'], r['call_index']))
from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(QAT / 'tokenizer.json'))
def dec(ids): return tok.decode(ids, skip_special_tokens=False)
def prev_turn(prompt_text):
    # the last assistant turn in the history: text between the last '<|turn>model\n' that is followed by content and its '<turn|>'
    parts = prompt_text.split('<|turn>model\n')
    for p in reversed(parts[1:]):
        body = p.split('<turn|>')[0]
        if body.strip(): return body
    return ''
for p in prompts:
    p['prompt_text'] = dec(p['prompt_token_ids']); p['prev_turn_text'] = prev_turn(p['prompt_text']); p['reference_text'] = dec(p['completion_token_ids'])
phase('prompts', n=len(prompts), val_attempts=sorted(val_att), train_attempts=qat_att,
      items=[{'attempt_id': p['attempt_id'], 'task_id': p['task_id'], 'tier': p['tier'], 'split': p['split'], 'call_index': p['call_index'], 'prompt_tokens': len(p['prompt_token_ids']), 'reference_tokens': len(p['completion_token_ids'])} for p in prompts])

# 3. serve with the scorer's flags on the QAT base, the three adapters registered
from adk_submission import VllmConfig, VllmServer, discover_adapters
adapters = discover_adapters(str(AG), adapter_extensions=('.safetensors',))
amap = getattr(adapters, 'adapters', None) or {}
import torch
tp = torch.cuda.device_count()
server = VllmServer(VllmConfig(model=str(QAT), port=8000, host='127.0.0.1', tool_call_parser='gemma4', enable_auto_tool_choice=True,
                               reasoning_parser='gemma4', max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90,
                               enable_lora=True, max_loras=max(1, len(amap)), max_lora_rank=16, scheduler_reserve_full_isl=False,
                               tensor_parallel_size=tp, startup_timeout=60 * 20), adapter_manifest=adapters)
try:
    server.start()
except Exception as e:
    phase('serve', error=f'{type(e).__name__}: {str(e)[:1500]}', adapters=sorted(amap)); finish('serve failed'); raise
import urllib.request
def post(path, payload):
    req = urllib.request.Request(server.base_url.rstrip('/') + path, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
    with urllib.request.urlopen(req, timeout=900) as r: return json.loads(r.read())
with urllib.request.urlopen(server.base_url.rstrip('/') + '/models', timeout=60) as r: models = [m['id'] for m in json.loads(r.read())['data']]
base_name = next((m for m in models if m not in SCALES), models[0])
phase('serve', base_url=server.base_url, models=models, adapters=sorted(amap), tp=tp)

# 4. generate: greedy, stop at <turn|> (106), up to 1536 tokens, raw text with the channel markers kept
def rep8(ids):
    grams = [tuple(ids[i:i + 8]) for i in range(max(0, len(ids) - 7))]
    return round(1 - len(set(grams)) / len(grams), 4) if grams else 0.0
def sim(a, b, n=800): return round(difflib.SequenceMatcher(None, a[:n], b[:n], autojunk=False).ratio(), 4)
ARMS = [('base', base_name)] + [(k, k) for k in SCALES if k in models]
gens = []; summary = {k: [] for k, _ in ARMS}
with (OUT / 'generations.jsonl').open('w') as f:
    for p in prompts:
        row = {k: p[k] for k in ('attempt_id', 'task_id', 'tier', 'split', 'call_index')}
        row['prompt_tokens'] = len(p['prompt_token_ids']); row['reference_tokens'] = len(p['completion_token_ids']); row['prev_turn_head'] = p['prev_turn_text'][:300]
        for label, m in ARMS:
            t1 = time.time()
            try:
                d = post('/completions', {'model': m, 'prompt': p['prompt_token_ids'], 'max_tokens': 1536, 'temperature': 0, 'seed': 42, 'stop_token_ids': [106], 'skip_special_tokens': False})
                ch = d['choices'][0]; txt = ch['text']
                ids = tok.encode(txt, add_special_tokens=False).ids
                g = {'text_head': txt[:1200], 'chars': len(txt), 'tokens': len(ids), 'finish': ch.get('finish_reason'), 'seconds': round(time.time() - t1, 1),
                     'rep8': rep8(ids), 'sim_prev_turn': sim(txt, p['prev_turn_text']), 'sim_reference': sim(txt, p['reference_text']),
                     'copies_prev_head': bool(p['prev_turn_text']) and txt[:150] != '' and txt[:150] in p['prev_turn_text'],
                     'has_tool_call': '<|tool_call>' in txt, 'thought_opened': '<|channel>thought' in txt, 'thought_closed': '<channel|>' in txt}
            except Exception as e:
                g = {'error': f'{type(e).__name__}: {str(e)[:400]}', 'seconds': round(time.time() - t1, 1)}
            row[label] = g; summary[label].append(g)
        f.write(json.dumps(row) + '\n'); f.flush(); gens.append(row)
        print(f"[{time.time()-T0:7.1f}s] {row['split']} {row['task_id']} call {row['call_index']}: " + ' | '.join(f"{k}: {row[k].get('tokens','ERR')}t rep{row[k].get('rep8','-')} prev{row[k].get('sim_prev_turn','-')} ref{row[k].get('sim_reference','-')} {row[k].get('finish','')}" for k, _ in ARMS), flush=True)
def agg(gs):
    ok = [g for g in gs if 'error' not in g]
    if not ok: return {'n': 0, 'errors': len(gs)}
    med = lambda xs: sorted(xs)[len(xs) // 2]
    return {'n': len(ok), 'errors': len(gs) - len(ok), 'tokens_median': med([g['tokens'] for g in ok]), 'hit_length_cap': sum(g['finish'] == 'length' for g in ok),
            'rep8_median': med([g['rep8'] for g in ok]), 'rep8_over_0.3': sum(g['rep8'] > 0.3 for g in ok), 'sim_prev_median': med([g['sim_prev_turn'] for g in ok]),
            'sim_prev_over_0.8': sum(g['sim_prev_turn'] > 0.8 for g in ok), 'copies_prev_head': sum(g['copies_prev_head'] for g in ok),
            'sim_ref_median': med([g['sim_reference'] for g in ok]), 'tool_call': sum(g['has_tool_call'] for g in ok), 'thought_closed': sum(g['thought_closed'] for g in ok)}
phase('probe', arms=[k for k, _ in ARMS], summary={k: agg(v) for k, v in summary.items()})
server.stop(); finish(f"done {round(time.time() - T0)}s")
