"""Loop-direction ablation of the c1 adapter (Kaggle L4x4, one session, no internet), as pre-registered in DECISIONS
(2026-11-02T21:20Z, amended 21:50Z). No tool runs and the harness is not involved.

Input: gemma-agent-ablate-data (attempts.jsonl and manifest.json from tool:gemma-agent/ablate_select; prefixes.jsonl from
tool:gemma-agent/swap_select, the swap v1 input; tools.json; adapters/lora_c1; the peft wheel), the wheelhouse, the E4B
QAT mount.

1. Render (the swap kernel's rule, unchanged). Contrast positions: the history before each labelled call of c1's 16
   attempts, kept when it has exactly the served prompt token count; every kept prompt of an attempt must be a token
   prefix of its longest. Swap prefixes: admitted as in swap v1.
2. Activations (one worker process, two GPUs, transformers + peft, the QAT weights decompressed to bf16 as in lora-train;
   base + c1 at full scale): one forward per attempt over its longest kept prompt; the output of every decoder layer at
   the last token of each kept prompt.
3. Direction: r_L = mean(repeat) - mean(non-repeat), unit norm, for every layer; L* = the layer in 10..31 with the largest
   standardised separation of the projections on r_L (difference of means over pooled s.d.); r = r_L*.
4. Edit: for every layer l, B of the adapter's o_proj and down_proj becomes (I - u u^T) B, u = normalise(g * r), g the
   weight of the RMSNorm after that module (post_attention_layernorm, post_feedforward_layernorm). Written as
   adapters/lora_c1_abl (same config, rank 16). Diagnostic: the forward of step 2 again with the edited adapter, the
   separation at L* reported before and after.
5. Swap check: base, c1 and c1abl on the admitted swap prefixes, served by one vLLM instance under the scorer's flags,
   one greedy and four sampled completions each (swap v1 settings); outcomes as in swap v1.

Full arm (DECISIONS 2026-11-03T00:45Z), when the dataset carries direction.pt: steps 3-4 are replaced by the given r and
L*, and every o_proj/down_proj write, base weight plus adapter, is projected off u: adapters/lora_c1_full (rank 17: c1's
B projected and [-u] appended, A with u^T W appended, other modules zero-padded) and adapters/lora_base_full (rank 1, the
base alone); the diagnostic forward runs the same edit in memory. Step 5's executors are base, c1, c1full and basefull.

MODE=render stops after step 1 (no GPU needed; a local rehearsal sets KAGGLE_ROOT to a folder laid out like /kaggle).
Output: /kaggle/working/ablate/{ablate.json, admission.jsonl, contrast.jsonl, direction.json, generations.jsonl,
adapters/lora_c1_abl/} and ablate.zip."""
import concurrent.futures, hashlib, importlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

KAGGLE = Path(os.environ.get('KAGGLE_ROOT', '/kaggle')); MODE = os.environ.get('MODE', 'full')
os.environ.update({'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_OFFLINE': '1',
    'VLLM_NO_USAGE_STATS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_ENGINE_READY_TIMEOUT_S': '2400',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1', 'NCCL_DEBUG': 'WARN', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OPENAI_API_KEY': 'EMPTY'})
T0 = time.time(); OUT = KAGGLE / 'working/ablate'; OUT.mkdir(parents=True, exist_ok=True)
result = {'started_unix': T0, 'mode': MODE, 'phases': {}}
def save(): (OUT / 'ablate.json').write_text(json.dumps(result, indent=2))
def phase(name, **kw):
    result['phases'][name] = {'t_s': round(time.time() - T0, 1), **kw}; save(); print(f'[{time.time()-T0:7.1f}s] {name}', json.dumps(kw)[:800], flush=True)
def finish(msg):
    result['finished_s'] = round(time.time() - T0, 1); save(); shutil.make_archive(str(KAGGLE / 'working/ablate'), 'zip', OUT); print(msg, flush=True)
def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def find_input(*names):
    for n in names:
        for c in [KAGGLE / 'input' / n, *(KAGGLE / 'input').glob(f'*/*/{Path(n).name}'), *(KAGGLE / 'input').glob(Path(n).name)]:
            if c.exists(): return c
    for p in sorted((KAGGLE / 'input').rglob('*'))[:80]: print('input:', p)
    raise SystemExit(f'none of {names} mounted')

DS = find_input('datasets/leolilley/gemma-agent-ablate-data')
QAT = KAGGLE / 'input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2'
if not QAT.exists(): QAT = next(x.parent for x in (KAGGLE / 'input').rglob('config.json') if 'e4b-it-qat' in str(x))
if MODE != 'render':
    WH = find_input('datasets/metric/gemma-4-developer-agent-wheelhouse')
    import glob
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
import jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
import importlib.metadata as md
result['versions'] = {d: md.version(d) for d in ('jinja2', 'tokenizers', *(('torch', 'transformers', 'vllm', 'adk-submission') if MODE != 'render' else ()))}
inputs = {n: sha(DS / n) for n in ('attempts.jsonl', 'manifest.json', 'prefixes.jsonl', 'tools.json')}
inputs['adapters/lora_c1'] = sha(DS / 'adapters/lora_c1/adapter_model.safetensors')
inputs.update({f'model/{n}': sha(QAT / n) for n in ('chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json')})
phase('environment', versions=result['versions'], qat=str(QAT), inputs=inputs)
result['inputs'] = inputs

# 1. render and admit (the lora-render kernel's rendering, unchanged)
IDENTITY = 'You are an agent. Your internal name is "swe_agent".'
def compile_template(text):
    def raise_exception(message): raise jinja2.exceptions.TemplateError(message)
    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    def strftime_now(fmt): return time.strftime(fmt)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters['tojson'] = tojson; env.globals['raise_exception'] = raise_exception; env.globals['strftime_now'] = strftime_now
    return env.from_string(text)
TEMPLATE = compile_template((QAT / 'chat_template.jinja').read_text())
TOK = Tokenizer.from_file(str(QAT / 'tokenizer.json'))
tcfg = json.loads((QAT / 'tokenizer_config.json').read_text())
BOS, EOS = tcfg.get('bos_token', '<bos>'), tcfg.get('eos_token', '<turn|>')
if isinstance(BOS, dict): BOS = BOS['content']
if isinstance(EOS, dict): EOS = EOS['content']
def encode(text): return TOK.encode(text, add_special_tokens=False).ids
OPENER = encode('<|channel>thought\n')
def as_mapping(tool_calls): return [{**tc, 'function': {**tc['function'], 'arguments': json.loads(tc['function']['arguments'])}} for tc in tool_calls]
def history(messages):
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if v is not None}
        if m['role'] == 'system': m['content'] = [{'type': 'text', 'text': m['content'].rstrip('\n') + '\n\n\n' + IDENTITY}]
        if m['role'] == 'assistant' and m.get('tool_calls'): m['tool_calls'] = as_mapping(m['tool_calls'])
        out.append(m)
    return out
def prompt_ids(msgs, tools):
    p = encode(TEMPLATE.render(messages=msgs, tools=tools, add_generation_prompt=True, enable_thinking=True, bos_token=BOS, eos_token=EOS))
    if p[-len(OPENER):] == OPENER: p = p[:-len(OPENER)]
    return p


TOOLS = json.loads((DS / 'tools.json').read_text()); tools = TOOLS['tools'] if isinstance(TOOLS, dict) and 'tools' in TOOLS else TOOLS
MAX_NEW = 4096; MAX_LEN = 32768

# contrast positions: per attempt the kept prompts, all token prefixes of the longest
attempts = [json.loads(l) for l in (DS / 'attempts.jsonl').read_text().splitlines() if l.strip()]
contrast = []; n_pos = n_kept = 0
with (OUT / 'admission.jsonl').open('w') as f:
    for a in attempts:
        kept = []
        for q in a['positions']:
            ids = prompt_ids(history(a['messages'][:q['messages_before']]), tools); d = len(ids) - (q['served_prompt_tokens'] or -1); n_pos += 1
            f.write(json.dumps({'set': 'contrast', 'attempt_id': a['attempt_id'], 'call_index': q['call_index'], 'rendered': len(ids), 'served': q['served_prompt_tokens'], 'delta': d, 'admitted': d == 0}) + '\n')
            if d == 0: kept.append((q, ids))
        if not kept: continue
        longest = max((ids for _, ids in kept), key=len)
        if not all(longest[:len(ids)] == ids for _, ids in kept): raise SystemExit(f"{a['attempt_id']}: a kept prompt is not a prefix of the longest")
        n_kept += len(kept)
        contrast.append({'attempt_id': a['attempt_id'], 'task_id': a['task_id'], 'ids': longest,
                         'positions': [{'call_index': q['call_index'], 'pos': len(ids) - 1, 'repeat': q['repeat']} for q, ids in kept]})
n_rep = sum(p['repeat'] for c in contrast for p in c['positions'])
with (OUT / 'contrast.jsonl').open('w') as f:
    for c in contrast: f.write(json.dumps({k: v for k, v in c.items() if k != 'ids'} | {'tokens': len(c['ids'])}) + '\n')

# swap prefixes: admitted as in swap v1
prefixes = [json.loads(l) for l in (DS / 'prefixes.jsonl').read_text().splitlines() if l.strip()]
admitted = []
with (OUT / 'admission.jsonl').open('a') as f:
    for x in prefixes:
        ids = prompt_ids(history(x['messages']), tools); d = len(ids) - (x['served_prompt_tokens'] or -1)
        ok = d == 0 and len(ids) + MAX_NEW <= MAX_LEN
        f.write(json.dumps({'set': 'swap', 'prefix_id': x['prefix_id'], 'source_bundle': x['source_bundle'], 'point': x['point'], 'rendered': len(ids),
                            'served': x['served_prompt_tokens'], 'delta': d, 'admitted': ok}) + '\n')
        if ok: x['prompt_token_ids'] = ids; admitted.append(x)
phase('admission', contrast_positions=n_pos, contrast_kept=n_kept, contrast_repeat=n_rep, contrast_attempts=len(contrast),
      longest_tokens=max(len(c['ids']) for c in contrast), swap_prefixes=len(prefixes), swap_admitted=len(admitted))
if MODE == 'render': finish('render-only rehearsal done'); sys.exit(0)

# 2-4. activations, direction and edit in one worker process (its GPU memory is released before vLLM starts)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', str(next(DS.glob('peft-*.whl')))], check=True)
WORK = OUT / 'work'; WORK.mkdir(exist_ok=True)
(WORK / 'contrast.json').write_text(json.dumps(contrast))
WORKER = r"""
import collections, json, re, shutil, sys, time
from pathlib import Path
import torch
work, model_path, c1_dir, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
note = {}
def save(): (work / 'worker.json').write_text(json.dumps(note, indent=1))
contrast = json.loads((work / 'contrast.json').read_text())
from transformers import AutoModelForImageTextToText, CompressedTensorsConfig
from accelerate import dispatch_model, infer_auto_device_map
max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.5) for i in range(torch.cuda.device_count())}
model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map={'': 0}, quantization_config=CompressedTensorsConfig(run_compressed=False))
no_split = sorted({type(m).__name__ for m in model.modules() if 'DecoderLayer' in type(m).__name__})
model = dispatch_model(model, device_map=infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split), skip_keys=getattr(model, '_skip_keys_device_placement', None))
model.config.use_cache = False; model.eval()
import peft.import_utils as _iu; _iu.is_torchao_available = lambda: False
import peft.tuners.lora.torchao as _tq; _tq.is_torchao_available = lambda: False
from peft import LoraConfig, PeftModel
# the saved config names bare module names, which also match the vision tower's projections; load with the language-model
# pattern the adapter was trained with (lora-train), so exactly the trained modules are wrapped
pcfg = LoraConfig.from_pretrained(str(c1_dir))
pcfg.target_modules = r'.*language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
model = PeftModel.from_pretrained(model, str(c1_dir), adapter_name='c1', config=pcfg); model.eval()
layers = {}
for n, m in model.named_modules():
    mm = re.fullmatch(r'.*language_model\.layers\.(\d+)', n)
    if mm: layers[int(mm.group(1))] = m
L = len(layers); note['layers'] = L; note['devices'] = dict(collections.Counter(str(p.device) for p in model.parameters()))
norms = torch.stack([torch.stack([layers[l].post_attention_layernorm.weight.detach().float().cpu(), layers[l].post_feedforward_layernorm.weight.detach().float().cpu()]) for l in range(L)])
grab = {}
def hook(l):
    def f(mod, inp, outp):
        h = outp[0] if isinstance(outp, tuple) else outp
        grab[l] = h[0, grab['pos']].detach().float().cpu()
    return f
handles = [layers[l].register_forward_hook(hook(l)) for l in range(L)]
dev = model.get_input_embeddings().weight.device
def activations():
    acts, labels, t0 = [], [], time.time()
    with torch.no_grad():
        for c in contrast:
            grab['pos'] = torch.tensor([p['pos'] for p in c['positions']])
            model(input_ids=torch.tensor([c['ids']], device=dev), logits_to_keep=1)
            acts.append(torch.stack([grab[l] for l in range(L)], 1)); labels += [p['repeat'] for p in c['positions']]
    return torch.cat(acts), torch.tensor(labels), round(time.time() - t0, 1)
A, y, secs = activations(); note['forward_seconds'] = secs; note['positions'] = len(y); note['repeat'] = int(y.sum()); save()
def separation(A, y, r):
    p = A @ r; a, b = p[y], p[~y]
    sd = (((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()) / (len(p) - 2)
    return float((a.mean() - b.mean()) / sd.sqrt())
if (work / 'given_direction.pt').exists():
    # full directional ablation (DECISIONS 2026-11-03T00:45Z): the given r; every o_proj/down_proj write, base weight plus
    # adapter, projected off u = normalise(g * r); shipped as rank-17 c1 and rank-1 base adapters
    from safetensors.torch import load_file, save_file
    given = torch.load(work / 'given_direction.pt'); r, Lstar = given['r'].float(), int(given['L_star'])
    assert torch.allclose(given['norms'], norms), 'the given direction was computed against other norm weights'
    note['L_star'] = Lstar
    def stats(A):
        return {'separation': round(separation(A[:, Lstar], y, r), 3), 'repeat': round(float((A[y, Lstar] @ r).mean()), 3), 'other': round(float((A[~y, Lstar] @ r).mean()), 3),
                'by_layer_mean_abs': [round(float((A[:, l] @ r).abs().mean()), 3) for l in range(L)]}
    note['c1'] = stats(A); save()
    sd = load_file(str(c1_dir / 'adapter_model.safetensors')); c1cfg = json.loads((c1_dir / 'adapter_config.json').read_text())
    assert c1cfg['r'] == 16 and c1cfg['lora_alpha'] == 16 and not c1cfg.get('use_rslora'), c1cfg
    mods = dict((n, m) for n, m in model.named_modules() if hasattr(m, 'lora_B') and re.search(r'language_model\.layers\.\d+\.(self_attn|mlp)\.\w+_proj$', n))
    full, basefull, checks = {}, {}, []
    for n, m in mods.items():
        A_, B_ = sd[n + '.lora_A.weight'].float(), sd[n + '.lora_B.weight'].float()
        mm = re.search(r'layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj)$', n)
        if mm:
            l = int(mm.group(1)); u = norms[l, 0 if 'o_proj' in mm.group(2) else 1] * r; u = u / u.norm()
            W = m.base_layer.weight.detach().float().cpu(); a = u @ W
            B17 = torch.cat([B_ - torch.outer(u, u @ B_), -u[:, None]], 1); A17 = torch.cat([A_, a[None, :]], 0)
            basefull[n + '.lora_A.weight'] = a[None, :].contiguous(); basefull[n + '.lora_B.weight'] = (-u[:, None]).contiguous()
            if len(checks) < 4:
                want = W + B_ @ A_; want = want - torch.outer(u, u @ want); got = W + B17 @ A17
                checks.append({'module': n.split('language_model.')[-1], 'max_abs_err': float((got - want).abs().max()), 'w_max_abs': float(W.abs().max()),
                               'residual_along_u': float((u @ got).abs().max())})
            with torch.no_grad():   # the same edit in memory, for the diagnostic forward
                wd = m.base_layer.weight; wd.copy_((W - torch.outer(u, a)).to(wd.device, wd.dtype))
                wb = m.lora_B['c1'].weight; wb.copy_((B_ - torch.outer(u, u @ B_)).to(wb.device, wb.dtype))
        else:
            B17 = torch.cat([B_, torch.zeros(B_.shape[0], 1)], 1); A17 = torch.cat([A_, torch.zeros(1, A_.shape[1])], 0)
        full[n + '.lora_A.weight'] = A17.contiguous(); full[n + '.lora_B.weight'] = B17.contiguous()
    assert len(full) == len(sd) and len(basefull) == 4 * L, (len(full), len(sd), len(basefull))
    note['exactness'] = checks; save()
    for name, tensors, rank, targets in (('lora_c1_full', full, 17, c1cfg['target_modules']), ('lora_base_full', basefull, 1, ['o_proj', 'down_proj'])):
        d = out / 'adapters' / name
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True); save_file(tensors, str(d / 'adapter_model.safetensors'), metadata={'format': 'pt'})
        (d / 'adapter_config.json').write_text(json.dumps({**c1cfg, 'r': rank, 'lora_alpha': rank, 'target_modules': targets}, indent=2))
    A2, _, _ = activations(); note['c1full'] = stats(A2)
    note['done'] = True; save(); sys.exit(0)
dirs, sep = [], []
for l in range(L):
    d = A[y, l].mean(0) - A[~y, l].mean(0); r = d / d.norm(); dirs.append(r); sep.append(separation(A[:, l], y, r))
Lstar = max(range(10, 32), key=lambda l: sep[l]); r = dirs[Lstar]
note['separation_by_layer'] = [round(s, 3) for s in sep]; note['L_star'] = Lstar; note['diff_norm_by_layer'] = [round(float((A[y, l].mean(0) - A[~y, l].mean(0)).norm()), 3) for l in range(L)]
torch.save({'r': r, 'L_star': Lstar, 'norms': norms}, work / 'direction.pt'); save()
# edit: B <- (I - u u^T) B for every layer's o_proj and down_proj, u = normalise(g * r)
from safetensors.torch import load_file, save_file
abl = out / 'adapters/lora_c1_abl'
if abl.exists(): shutil.rmtree(abl)
abl.mkdir(parents=True); shutil.copyfile(c1_dir / 'adapter_config.json', abl / 'adapter_config.json')   # copyfile: the input copy is read-only
sd = load_file(str(c1_dir / 'adapter_model.safetensors')); edited = {}
for k, B in sd.items():
    mm = re.search(r'layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj)\.lora_B\.weight$', k)
    if not mm: continue
    l = int(mm.group(1)); g = norms[l, 0 if 'o_proj' in mm.group(2) else 1]
    u = g * r; u = u / u.norm(); Bf = B.float(); Bn = Bf - torch.outer(u, u @ Bf)
    sd[k] = Bn.to(B.dtype); edited[k] = round(float((Bf - Bn).norm() / Bf.norm()), 4)
save_file(sd, str(abl / 'adapter_model.safetensors'), metadata={'format': 'pt'})
note['edited_modules'] = len(edited); note['removed_fraction_of_B_norm'] = {'mean': round(sum(edited.values()) / max(1, len(edited)), 4), 'max': max(edited.values(), default=0)}
assert len(edited) == 2 * L, f'edited {len(edited)} modules, expected {2 * L}'
save()
# diagnostic: the same forward with the edited adapter
# the edited B matrices into the loaded c1 modules (module path + '.lora_B.weight' is the saved key)
n_set = 0
with torch.no_grad():
    for n, m in model.named_modules():
        if re.search(r'language_model\.layers\.\d+\.(self_attn\.o_proj|mlp\.down_proj)$', n) and hasattr(m, 'lora_B'):
            w = m.lora_B['c1'].weight; w.copy_(sd[n + '.lora_B.weight'].to(w.device, w.dtype)); n_set += 1
assert n_set == 2 * L, f'set {n_set} modules, expected {2 * L}'
A2, _, _ = activations()
note['separation_at_L_star'] = {'c1': round(sep[Lstar], 3), 'c1abl': round(separation(A2[:, Lstar], y, r), 3)}
note['mean_projection_at_L_star'] = {'c1': {'repeat': round(float((A[y, Lstar] @ r).mean()), 3), 'other': round(float((A[~y, Lstar] @ r).mean()), 3)},
                                     'c1abl': {'repeat': round(float((A2[y, Lstar] @ r).mean()), 3), 'other': round(float((A2[~y, Lstar] @ r).mean()), 3)}}
note['done'] = True; save()
"""
FULL = (DS / 'direction.pt').exists()
if FULL: shutil.copyfile(DS / 'direction.pt', WORK / 'given_direction.pt'); result['direction_sha256'] = sha(DS / 'direction.pt')
result['arm'] = 'full' if FULL else 'abl'
(WORK / 'worker.py').write_text(WORKER)
rc = subprocess.run([sys.executable, str(WORK / 'worker.py'), str(WORK), str(QAT), str(DS / 'adapters/lora_c1'), str(OUT)],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1'}, stdout=open(OUT / 'worker.out', 'w'), stderr=subprocess.STDOUT).returncode
w = json.loads((WORK / 'worker.json').read_text()) if (WORK / 'worker.json').exists() else {}
(OUT / 'direction.json').write_text(json.dumps(w, indent=1))
if rc != 0 or not w.get('done'):
    phase('direction', error=f'worker exit {rc}', tail=(OUT / 'worker.out').read_text()[-3000:]); finish('worker failed'); raise SystemExit('worker failed')
NEW = ('lora_c1_full', 'lora_base_full') if FULL else ('lora_c1_abl',)
result['adapters_sha256'] = {a: sha(OUT / 'adapters' / a / 'adapter_model.safetensors') for a in NEW}
phase('direction', **{k: w.get(k) for k in (('positions', 'repeat', 'L_star', 'forward_seconds', 'exactness') + (('c1', 'c1full') if FULL else
      ('edited_modules', 'removed_fraction_of_B_norm', 'separation_at_L_star', 'mean_projection_at_L_star')))})

# 5. serve (the swap kernel's flags) and generate
AG = KAGGLE / 'working/agent'; (AG / 'adapters').mkdir(parents=True, exist_ok=True)
shutil.copytree(DS / 'adapters/lora_c1', AG / 'adapters/lora_c1', dirs_exist_ok=True)
for a in NEW: shutil.copytree(OUT / 'adapters' / a, AG / 'adapters' / a, dirs_exist_ok=True)
(AG / 'agent.yaml').write_text('name: ablate\nmodel: gemma-4-e4b-it-qat-w4a16-ct\nadapter: lora_c1\ninstruction: ablate\n')
from adk_submission import VllmConfig, VllmServer, discover_adapters
adapters = discover_adapters(str(AG), adapter_extensions=('.safetensors',)); amap = getattr(adapters, 'adapters', None) or {}
import torch
server = VllmServer(VllmConfig(model=str(QAT), port=8000, host='127.0.0.1', tool_call_parser='gemma4', enable_auto_tool_choice=True,
                               reasoning_parser='gemma4', max_model_len=MAX_LEN, dtype='bfloat16', gpu_memory_utilization=0.90,
                               enable_lora=True, max_loras=max(1, len(amap)), max_lora_rank=32 if FULL else 16, scheduler_reserve_full_isl=False,
                               tensor_parallel_size=torch.cuda.device_count(), startup_timeout=60 * 20), adapter_manifest=adapters)
try: server.start()
except Exception as e:
    phase('serve', error=f'{type(e).__name__}: {str(e)[:1500]}', adapters=sorted(amap)); finish('serve failed'); raise
import urllib.request
def post(path, payload):
    req = urllib.request.Request(server.base_url.rstrip('/') + path, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
    with urllib.request.urlopen(req, timeout=1800) as r: return json.loads(r.read())
with urllib.request.urlopen(server.base_url.rstrip('/') + '/models', timeout=60) as r: models = [m['id'] for m in json.loads(r.read())['data']]
EXECUTORS = {'base': next(m for m in models if m not in ('lora_c1', *NEW)), 'c1': 'lora_c1',
             **({'c1full': 'lora_c1_full', 'basefull': 'lora_base_full'} if FULL else {'c1abl': 'lora_c1_abl'})}
phase('serve', base_url=server.base_url, models=models, executors=EXECUTORS)

# 3. outcomes from the completion text
CALL = re.compile(r'<\|tool_call>call:([A-Za-z_][\w.-]*)\{')
class Unparsed(Exception): pass
def parse_value(s, i):
    if s.startswith('<|"|>', i):
        j = s.find('<|"|>', i + 5)
        if j < 0: raise Unparsed('string')
        return s[i + 5:j], j + 5
    if s[i] == '{':
        out, i = {}, i + 1
        while s[i] != '}':
            if s.startswith('<|"|>', i): k, i = parse_value(s, i)
            else:
                j = s.index(':', i); k, i = s[i:j], j
            if s[i] != ':': raise Unparsed('key')
            v, i = parse_value(s, i + 1); out[k] = v
            if s[i] == ',': i += 1
        return out, i + 1
    if s[i] == '[':
        out, i = [], i + 1
        while s[i] != ']':
            v, i = parse_value(s, i); out.append(v)
            if s[i] == ',': i += 1
        return out, i + 1
    m = re.compile(r'[^,}\]]*').match(s, i); tok = m.group(0).strip(); i = m.end()
    if tok in ('true', 'false'): return tok == 'true', i
    if tok == 'null': return None, i
    try: return (int(tok) if re.fullmatch(r'-?\d+', tok) else float(tok)), i
    except ValueError: return tok, i
def first_call(text):
    m = CALL.search(text)
    if not m: return None
    try:
        args, _ = parse_value(text, m.end() - 1)
        return {'name': m.group(1), 'arguments': args}
    except (Unparsed, IndexError, ValueError) as e:
        return {'name': m.group(1), 'arguments': None, 'unparsed': str(e) or type(e).__name__}
def canonical(name, args): return json.dumps([name, args or {}], sort_keys=True, separators=(',', ':'))
def outcome(text, finish, prev):
    c = first_call(text); prev_key = canonical(prev['function']['name'], json.loads(prev['function']['arguments'] or '{}'))
    return {'call': c, 'repeat_previous': bool(c and c.get('arguments') is not None and canonical(c['name'], c['arguments']) == prev_key),
            'edit': bool(c and c['name'] in ('edit_file', 'write_file')), 'final': c is None and finish == 'stop',
            'length_cap': finish == 'length', 'unparsed': bool(c and c.get('arguments') is None)}

SAMPLES = [('greedy', {'temperature': 0, 'seed': 42})] + [(f's{k}', {'temperature': 0.2, 'top_p': 0.95, 'seed': 1000 + k}) for k in range(4)]
def one(x, ex, label, kw):
    t1 = time.time()
    try:
        d = post('/completions', {'model': EXECUTORS[ex], 'prompt': x['prompt_token_ids'], 'max_tokens': MAX_NEW, 'stop_token_ids': [106], 'skip_special_tokens': False, **kw})
        ch = d['choices'][0]
        return {'prefix_id': x['prefix_id'], 'executor': ex, 'sample': label, 'finish': ch.get('finish_reason'), 'completion_tokens': (d.get('usage') or {}).get('completion_tokens'),
                'prompt_tokens_served': (d.get('usage') or {}).get('prompt_tokens'), 'seconds': round(time.time() - t1, 1), 'text': ch['text'],
                **outcome(ch['text'], ch.get('finish_reason'), x['previous_call'])}
    except Exception as e:
        return {'prefix_id': x['prefix_id'], 'executor': ex, 'sample': label, 'error': f'{type(e).__name__}: {str(e)[:400]}', 'seconds': round(time.time() - t1, 1)}
jobs = [(x, ex, lab, kw) for x in admitted for ex in EXECUTORS for lab, kw in SAMPLES]
done = 0; tally = {}
with (OUT / 'generations.jsonl').open('w') as f, concurrent.futures.ThreadPoolExecutor(16) as pool:
    for g in pool.map(lambda j: one(*j), jobs):
        f.write(json.dumps(g) + '\n'); done += 1
        t = tally.setdefault(g['executor'], {'n': 0, 'errors': 0, 'repeat_previous': 0, 'edit': 0, 'final': 0, 'length_cap': 0, 'unparsed': 0})
        t['n'] += 1
        if 'error' in g: t['errors'] += 1; continue
        for k in ('repeat_previous', 'edit', 'final', 'length_cap', 'unparsed'): t[k] += int(g[k])
        if done % 100 == 0: phase('progress', done=done, of=len(jobs), tally=tally)
phase('generate', done=done, of=len(jobs), tally=tally)
server.stop(); finish(f'done {round(time.time() - T0)}s')
