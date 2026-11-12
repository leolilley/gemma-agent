"""Frontier swap (Kaggle L4x4, one session, no internet): hand a recorded attempt's history, cut just before one model
call, to another executor and ask for that one next call. No tool runs and the harness is not involved. Executors: the
E4B QAT base and every cycle adapter the input dataset ships under adapters/ (lora_<cycle>, executor <cycle>), served by
one vLLM instance under the scorer's flags, the adapter chosen per request.

Input: gemma-agent-swap-data (prefixes.jsonl and manifest.json from tool:gemma-agent/swap_select, tools.json as the
harness declares the tools, adapters/lora_* from the installed cycle bundles), the wheelhouse, the
E4B QAT mount.

1. Render each prefix the way the harness's request reaches the template (the lora-render kernel's rule: ADK's identity
   line appended to the system instruction, earlier reasoning kept, tool-call arguments as mappings, thinking enabled,
   generation prompt ending at the tool response). A prefix is admitted only when its rendering has exactly the served
   prompt token count of the call it precedes; the others are listed with their difference and not generated from.
2. For every admitted prefix and executor: one greedy completion and four sampled at the bundle's temperature 0.2,
   top_p 0.95 (seeds fixed), up to 4,096 tokens, stopping at the end-of-turn token.
3. Outcomes per completion, from the text alone (kept, so they can be recomputed): the first tool call parsed from the
   Gemma call syntax; repeat_previous (its name and arguments equal the prefix's last call, canonical JSON as in
   scripts/episodes.py); edit (edit_file or write_file); final (no tool call, stopped); length_cap; unparsed (a call
   opener the parser could not read).

MODE=render stops after step 1 (no GPU needed; a local rehearsal sets KAGGLE_ROOT to a folder laid out like /kaggle).
Output: /kaggle/working/swap/{swap.json, admission.jsonl, generations.jsonl} and swap.zip."""
import concurrent.futures, hashlib, importlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

KAGGLE = Path(os.environ.get('KAGGLE_ROOT', '/kaggle')); MODE = os.environ.get('MODE', 'full')
os.environ.update({'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_OFFLINE': '1',
    'VLLM_NO_USAGE_STATS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_ENGINE_READY_TIMEOUT_S': '2400',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1', 'NCCL_DEBUG': 'WARN', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OPENAI_API_KEY': 'EMPTY'})
T0 = time.time(); OUT = KAGGLE / 'working/swap'; OUT.mkdir(parents=True, exist_ok=True)
result = {'started_unix': T0, 'mode': MODE, 'phases': {}}
def save(): (OUT / 'swap.json').write_text(json.dumps(result, indent=2))
def phase(name, **kw):
    result['phases'][name] = {'t_s': round(time.time() - T0, 1), **kw}; save(); print(f'[{time.time()-T0:7.1f}s] {name}', json.dumps(kw)[:800], flush=True)
def finish(msg):
    result['finished_s'] = round(time.time() - T0, 1); save(); shutil.make_archive(str(KAGGLE / 'working/swap'), 'zip', OUT); print(msg, flush=True)
def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def find_input(*names):
    for n in names:
        for c in [KAGGLE / 'input' / n, *(KAGGLE / 'input').glob(f'*/*/{Path(n).name}'), *(KAGGLE / 'input').glob(Path(n).name)]:
            if c.exists(): return c
    for p in sorted((KAGGLE / 'input').rglob('*'))[:80]: print('input:', p)
    raise SystemExit(f'none of {names} mounted')

DS = find_input('datasets/leolilley/gemma-agent-swap-data')
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
ADAPTERS = sorted(d.name for d in (DS / 'adapters').iterdir() if d.name.startswith('lora_'))
inputs = {n: sha(DS / n) for n in ('prefixes.jsonl', 'manifest.json', 'tools.json')}
inputs.update({f'adapters/{a}': sha(DS / 'adapters' / a / 'adapter_model.safetensors') for a in ADAPTERS})
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
prefixes = [json.loads(l) for l in (DS / 'prefixes.jsonl').read_text().splitlines() if l.strip()]
MAX_NEW = 4096; MAX_LEN = 32768
admitted = []
with (OUT / 'admission.jsonl').open('w') as f:
    for x in prefixes:
        ids = prompt_ids(history(x['messages']), tools); d = len(ids) - (x['served_prompt_tokens'] or -1)
        ok = d == 0 and len(ids) + MAX_NEW <= MAX_LEN
        f.write(json.dumps({'prefix_id': x['prefix_id'], 'source_bundle': x['source_bundle'], 'point': x['point'], 'rendered': len(ids),
                            'served': x['served_prompt_tokens'], 'delta': d, 'admitted': ok}) + '\n')
        if ok: x['prompt_token_ids'] = ids; admitted.append(x)
by = {}
for x in prefixes:
    k = f"{x['source_bundle']} {x['point']}"; by.setdefault(k, [0, 0]); by[k][1] += 1
for x in admitted: by[f"{x['source_bundle']} {x['point']}"][0] += 1
phase('admission', prefixes=len(prefixes), admitted=len(admitted), by_source_point={k: f'{a}/{n}' for k, (a, n) in sorted(by.items())})
if MODE == 'render': finish('render-only rehearsal done'); sys.exit(0)

# 2. serve (the lora-probe kernel's flags) and generate
AG = KAGGLE / 'working/agent'; (AG / 'adapters').mkdir(parents=True, exist_ok=True)
for a in ADAPTERS: shutil.copytree(DS / 'adapters' / a, AG / 'adapters' / a, dirs_exist_ok=True)
(AG / 'agent.yaml').write_text(f'name: swap\nmodel: gemma-4-e4b-it-qat-w4a16-ct\nadapter: {ADAPTERS[0]}\ninstruction: swap\n')
from adk_submission import VllmConfig, VllmServer, discover_adapters
adapters = discover_adapters(str(AG), adapter_extensions=('.safetensors',)); amap = getattr(adapters, 'adapters', None) or {}
import torch
server = VllmServer(VllmConfig(model=str(QAT), port=8000, host='127.0.0.1', tool_call_parser='gemma4', enable_auto_tool_choice=True,
                               reasoning_parser='gemma4', max_model_len=MAX_LEN, dtype='bfloat16', gpu_memory_utilization=0.90,
                               enable_lora=True, max_loras=max(1, len(amap)), max_lora_rank=16, scheduler_reserve_full_isl=False,
                               tensor_parallel_size=torch.cuda.device_count(), startup_timeout=60 * 20), adapter_manifest=adapters)
try: server.start()
except Exception as e:
    phase('serve', error=f'{type(e).__name__}: {str(e)[:1500]}', adapters=sorted(amap)); finish('serve failed'); raise
import urllib.request
def post(path, payload):
    req = urllib.request.Request(server.base_url.rstrip('/') + path, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
    with urllib.request.urlopen(req, timeout=1800) as r: return json.loads(r.read())
with urllib.request.urlopen(server.base_url.rstrip('/') + '/models', timeout=60) as r: models = [m['id'] for m in json.loads(r.read())['data']]
EXECUTORS = {'base': next(m for m in models if m not in ADAPTERS), **{a.removeprefix('lora_'): a for a in ADAPTERS}}
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
