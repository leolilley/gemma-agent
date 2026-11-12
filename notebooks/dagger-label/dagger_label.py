"""DAgger labels (Kaggle L4x4, one session, no internet), as pre-registered in DECISIONS (2026-11-03T05:50Z): the
learner's own states, labelled by the expert, written as training rows. No tool runs and the harness is not involved.

Input: gemma-agent-dagger-data (attempts.jsonl and manifest.json from tool:gemma-agent/dagger_select, tools.json), the
wheelhouse, the E4B QAT mount (the template and tokenizer the training rows are rendered with) and the 31B QAT mount
(the expert).

1. Render each state's history the way the harness's request reaches the template (the swap kernel's rule, unchanged);
   keep it when the rendering has exactly the served prompt token count.
2. The expert (31B 4-bit, the eval kernel's flags, no adapter) gives one greedy completion per kept state, up to 4,096
   tokens, stopping at the end-of-turn token.
3. Label: the completion parsed into an assistant message (reasoning from the thought channel; one tool call in the
   Gemma call syntax, or a final answer: no call, stopped), rendered back through the E4B template as the lora-render
   kernel renders training targets; kept when the rendered target is within 2 tokens of the generated length. Dropped
   labels are counted by reason; no label is filtered by what it does.

MODE=render stops after step 1 (no GPU; a local rehearsal sets KAGGLE_ROOT to a folder laid out like /kaggle).
Output: /kaggle/working/dagger-label/{label.json, admission.jsonl, labels.jsonl, rendered.jsonl} and dagger-label.zip:
rendered.jsonl holds the kept labels as training rows of tier `dagger`, the lora-render kernel's row format."""
import concurrent.futures, hashlib, importlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

KAGGLE = Path(os.environ.get('KAGGLE_ROOT', '/kaggle')); MODE = os.environ.get('MODE', 'full')
os.environ.update({'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_OFFLINE': '1',
    'VLLM_NO_USAGE_STATS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_ENGINE_READY_TIMEOUT_S': '2400',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1', 'NCCL_DEBUG': 'WARN', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OPENAI_API_KEY': 'EMPTY'})
T0 = time.time(); OUT = KAGGLE / 'working/dagger-label'; OUT.mkdir(parents=True, exist_ok=True)
result = {'started_unix': T0, 'mode': MODE, 'phases': {}}
def save(): (OUT / 'label.json').write_text(json.dumps(result, indent=2))
def phase(name, **kw):
    result['phases'][name] = {'t_s': round(time.time() - T0, 1), **kw}; save(); print(f'[{time.time()-T0:7.1f}s] {name}', json.dumps(kw)[:800], flush=True)
def finish(msg):
    result['finished_s'] = round(time.time() - T0, 1); save(); shutil.make_archive(str(KAGGLE / 'working/dagger-label'), 'zip', OUT); print(msg, flush=True)
def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def find_input(*names):
    for n in names:
        for c in [KAGGLE / 'input' / n, *(KAGGLE / 'input').glob(f'*/*/{Path(n).name}'), *(KAGGLE / 'input').glob(Path(n).name)]:
            if c.exists(): return c
    for p in sorted((KAGGLE / 'input').rglob('*'))[:80]: print('input:', p)
    raise SystemExit(f'none of {names} mounted')

DS = find_input('datasets/leolilley/gemma-agent-dagger-data')
QAT = KAGGLE / 'input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2'   # the template and tokenizer the training rows use
if not QAT.exists(): QAT = next(x.parent for x in (KAGGLE / 'input').rglob('config.json') if 'e4b-it-qat' in str(x))
EXPERT = KAGGLE / 'input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2'
if not EXPERT.exists(): EXPERT = next(x.parent for x in (KAGGLE / 'input').rglob('config.json') if '31b-it-qat' in str(x))
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
inputs = {n: sha(DS / n) for n in ('attempts.jsonl', 'manifest.json', 'tools.json')}
inputs.update({f'model/{n}': sha(QAT / n) for n in ('chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json')})
inputs.update({f'expert/{n}': sha(EXPERT / n) for n in ('chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json')})
phase('environment', versions=result['versions'], qat=str(QAT), expert=str(EXPERT), inputs=inputs)
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
attempts = [json.loads(l) for l in (DS / 'attempts.jsonl').read_text().splitlines() if l.strip()]
states = []
with (OUT / 'admission.jsonl').open('w') as f:
    for a in attempts:
        for q in a['states']:
            msgs = a['messages'][:q['messages_before']]; ids = prompt_ids(history(msgs), tools); d = len(ids) - (q['served_prompt_tokens'] or -1)
            ok = d == 0 and len(ids) + MAX_NEW <= MAX_LEN
            f.write(json.dumps({'attempt_id': a['attempt_id'], 'call_index': q['call_index'], 'rendered': len(ids), 'served': q['served_prompt_tokens'], 'delta': d, 'admitted': ok}) + '\n')
            if ok: states.append({'attempt': a, 'state': q, 'msgs': msgs, 'prompt_token_ids': ids})
phase('admission', attempts=len(attempts), states=sum(len(a['states']) for a in attempts), admitted=len(states),
      by_task={t: sum(1 for s in states if s['attempt']['task_id'] == t) for t in sorted({a['task_id'] for a in attempts})})
if MODE == 'render': finish('render-only rehearsal done'); sys.exit(0)

# 2. serve the expert (the eval kernel's flags, no adapter) and label
from adk_submission import VllmConfig, VllmServer
import torch
server = VllmServer(VllmConfig(model=str(EXPERT), port=8000, host='127.0.0.1', tool_call_parser='gemma4', enable_auto_tool_choice=True,
                               reasoning_parser='gemma4', max_model_len=MAX_LEN, dtype='bfloat16', gpu_memory_utilization=0.90,
                               enable_lora=False, scheduler_reserve_full_isl=False, tensor_parallel_size=torch.cuda.device_count(), startup_timeout=60 * 40))
try: server.start()
except Exception as e:
    phase('serve', error=f'{type(e).__name__}: {str(e)[:1500]}'); finish('serve failed'); raise
import urllib.request
def post(path, payload):
    req = urllib.request.Request(server.base_url.rstrip('/') + path, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
    with urllib.request.urlopen(req, timeout=1800) as r: return json.loads(r.read())
with urllib.request.urlopen(server.base_url.rstrip('/') + '/models', timeout=60) as r: models = [m['id'] for m in json.loads(r.read())['data']]
phase('serve', base_url=server.base_url, models=models)

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


# 3. labels: parse, render back as a training target, check the length
NL = encode('\n'); STOP_IDS = set(json.loads((QAT / 'generation_config.json').read_text()).get('eos_token_id', []))
THOUGHT = re.compile(r'^\s*<\|channel>thought\n?(.*?)<channel\|>', re.S)
def message_of(text, finish):
    m = THOUGHT.match(text); reasoning = m.group(1).strip() if m else None; rest = text[m.end():] if m else text
    calls = CALL.findall(rest)
    if len(calls) > 1: return None, 'multiple_calls'
    if calls:
        c = first_call(rest)
        if c.get('arguments') is None: return None, 'unparsed_call'
        return {'role': 'assistant', 'content': None, 'reasoning_content': reasoning,
                'tool_calls': [{'id': 'dagger', 'type': 'function', 'function': {'name': c['name'], 'arguments': json.dumps(c['arguments'])}}]}, None
    if finish != 'stop': return None, 'length_cap'
    content = rest.strip()
    if not content: return None, 'empty'
    return {'role': 'assistant', 'content': content, 'reasoning_content': reasoning}, None
def target_ids(msgs, msg, prompt):
    t = {k: v for k, v in msg.items() if v is not None}
    if t.get('tool_calls'): t['tool_calls'] = as_mapping(t['tool_calls'])
    full = encode(TEMPLATE.render(messages=history(msgs) + [t], tools=tools, add_generation_prompt=False, enable_thinking=True, bos_token=BOS, eos_token=EOS))
    if full[:len(prompt)] != prompt: return None
    comp = full[len(prompt):]
    if len(comp) >= 2 and comp[-2] in STOP_IDS and comp[-1:] == NL: comp = comp[:-1]
    return comp if comp and comp[-1] in STOP_IDS else None
def one(s):
    t1 = time.time(); a, q = s['attempt'], s['state']
    try:
        d = post('/completions', {'model': models[0], 'prompt': s['prompt_token_ids'], 'max_tokens': MAX_NEW, 'stop_token_ids': [106], 'skip_special_tokens': False, 'temperature': 0, 'seed': 42})
    except Exception as e:
        return {'attempt_id': a['attempt_id'], 'call_index': q['call_index'], 'error': f'{type(e).__name__}: {str(e)[:400]}'}, None
    ch = d['choices'][0]; text, finish = ch['text'], ch.get('finish_reason'); n_gen = (d.get('usage') or {}).get('completion_tokens')
    lab = {'attempt_id': a['attempt_id'], 'task_id': a['task_id'], 'call_index': q['call_index'], 'finish': finish, 'completion_tokens': n_gen,
           'seconds': round(time.time() - t1, 1), 'text': text}
    if q['previous_call']: lab.update({k: v for k, v in outcome(text, finish, q['previous_call']).items() if k != 'call'})
    learner = (q['learner_next'].get('tool_calls') or [None])[0]
    c = first_call(text)
    lab['learner_tool'] = learner['function']['name'] if learner else None; lab['expert_tool'] = c['name'] if c else None
    lab['agrees_with_learner'] = bool(c and learner and c.get('arguments') is not None and canonical(c['name'], c['arguments']) == canonical(learner['function']['name'], json.loads(learner['function']['arguments'] or '{}')))
    msg, why = message_of(text, finish)
    if msg is None: lab['dropped'] = why; return lab, None
    comp = target_ids(s['msgs'], msg, s['prompt_token_ids'])
    if comp is None: lab['dropped'] = 'render_boundary'; return lab, None
    delta = len(comp) - (n_gen or 0); lab['render_delta'] = delta
    if abs(delta) > 2: lab['dropped'] = 'render_length'; return lab, None
    kind = ('R' if msg.get('reasoning_content') else '-') + ('C' if msg.get('content') else '-') + ('T' if msg.get('tool_calls') else '-')
    row = {'example_id': f"{a['attempt_id']}#{q['call_index']}@dagger", 'attempt_id': a['attempt_id'], 'task_id': a['task_id'], 'repository': a['task_id'].split('_')[0],
           'call_index': q['call_index'], 'tier': 'dagger', 'kind': kind, 'usable': True, 'prompt_delta': 0, 'compaction_delta': 0, 'completion_delta': delta,
           'boundary': True, 'truncated': False, 'compacted': False, 'prompt_tokens': len(s['prompt_token_ids']), 'completion_tokens': len(comp),
           'provenance': {**a['provenance'], 'expert': 'gemma-4-31b-it-qat-w4a16-ct', 'label_sha256': hashlib.sha256(text.encode()).hexdigest()},
           'prompt_token_ids': s['prompt_token_ids'], 'completion_token_ids': comp}
    return lab, row
tally = {'labelled': 0, 'errors': 0, 'kept': 0, 'dropped': {}, 'repeat_previous': 0, 'edit': 0, 'final': 0, 'agrees_with_learner': 0}
with (OUT / 'labels.jsonl').open('w') as fl, (OUT / 'rendered.jsonl').open('w') as fr, concurrent.futures.ThreadPoolExecutor(16) as pool:
    for lab, row in pool.map(one, states):
        fl.write(json.dumps(lab) + '\n'); tally['labelled'] += 1
        if 'error' in lab: tally['errors'] += 1; continue
        for k in ('repeat_previous', 'edit', 'final', 'agrees_with_learner'): tally[k] += int(bool(lab.get(k)))
        if row: fr.write(json.dumps(row) + '\n'); tally['kept'] += 1
        else: tally['dropped'][lab['dropped']] = tally['dropped'].get(lab['dropped'], 0) + 1
        if tally['labelled'] % 50 == 0: phase('progress', of=len(states), tally=tally)
result['rendered_sha256'] = sha(OUT / 'rendered.jsonl')
phase('label', of=len(states), tally=tally)
server.stop(); finish(f'done {round(time.time() - T0)}s')
