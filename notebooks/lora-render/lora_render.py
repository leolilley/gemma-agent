"""Render the LoRA examples into token ids with the competition model's own chat template, and check the rendering
against what vLLM served (CPU kernel, no GPU hours).

Each example in gemma-agent-lora-data carries the OpenAI-style messages of one model call plus the prompt and completion
token counts vLLM reported for that call. The request the harness actually sent differs from the recorded messages in
one documented way: ADK's identity processor appends 'You are an agent. Your internal name is "swe_agent".' to the
system instruction (google.adk.flows.llm_flows.identity, after the instruction, joined with a blank line; the
instruction keeps the prompt file's final newline, and the message reaches the template as a text part, which the
template follows with a space; both confirmed against served token ids in the Gluzdov kit). History is
rendered with thinking enabled, earlier turns' reasoning kept, and tool-call arguments as mappings (the Gemma call
syntax); a search over those choices on a sample is recorded for the report. The target turn is rendered the way the
model generated it: reasoning in the thought channel, tool calls in the call syntax, the stop token included.

Two tiers, by the model that served the trace:
  qat   gemma-4-31b-it-qat-w4a16-ct (the scorer's template vintage): the rendered prompt must match the served prompt
        token count exactly, and the completion within one token;
  bf16  gemma-4-31b-it (the bf16 release, whose template renders history without reasoning): the prompt cannot be
        matched against the served count, so the history is rendered with the serving template of the adapter's base
        and only the completion is checked. Compacted histories (ADK event compaction) show as a large served shortfall
        and are excluded in both tiers.

Outputs under /kaggle/working/lora-render/: rendered.jsonl (prompt_token_ids, completion_token_ids, checks per example),
render-report.json, and the template, tokenizer config, generation config and config of both model mounts, with digests.
"""
import collections, hashlib, json, os, random, shutil, statistics, time
from pathlib import Path

import jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

T0 = time.time()
KAGGLE = Path(os.environ.get('KAGGLE_ROOT', '/kaggle'))   # a local rehearsal points this at a folder laid out like /kaggle
LIMIT = int(os.environ.get('LIMIT', '0'))                  # a local rehearsal renders only the first LIMIT examples
OUT = KAGGLE / 'working/lora-render'; OUT.mkdir(parents=True, exist_ok=True)
IDENTITY = 'You are an agent. Your internal name is "swe_agent".'
COMPACTION_SHORTFALL = 1000   # served prompt this many tokens shorter than the full history = compacted by ADK


def log(*a):
    print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)


def find_input(*names):
    for n in names:
        for c in [(KAGGLE / 'input') / n, *(KAGGLE / 'input').glob(f'*/*/{Path(n).name}'), *(KAGGLE / 'input').glob(Path(n).name)]:
            if c.exists(): return c
    for p in sorted((KAGGLE / 'input').rglob('*'))[:60]: print('input:', p)
    raise SystemExit(f'none of {names} mounted')


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


DS = find_input('datasets/leolilley/gemma-agent-lora-data')
MODELS = {'e4b': KAGGLE / 'input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2',
          '31b': KAGGLE / 'input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2'}
for tag, p in list(MODELS.items()):
    if not p.exists():
        c = [x.parent for x in (KAGGLE / 'input').rglob('chat_template.jinja') if f'{tag}-it-qat' in str(x)]
        if c: MODELS[tag] = c[0]
        else: raise SystemExit(f'{tag} model not mounted')
report = {'model_files': {}, 'identity_line': IDENTITY}
SMALL = ('chat_template.jinja', 'tokenizer_config.json', 'generation_config.json', 'config.json')
for tag, p in MODELS.items():
    d = OUT / 'model-files' / tag; d.mkdir(parents=True, exist_ok=True)
    for name in SMALL: shutil.copyfile(p / name, d / name)
    report['model_files'][tag] = {'path': str(p), **{name: sha(p / name) for name in SMALL}, 'tokenizer.json': sha(p / 'tokenizer.json')}
same_template = report['model_files']['e4b']['chat_template.jinja'] == report['model_files']['31b']['chat_template.jinja']
same_tokenizer = report['model_files']['e4b']['tokenizer.json'] == report['model_files']['31b']['tokenizer.json']
report['same_template_e4b_31b'] = same_template; report['same_tokenizer_e4b_31b'] = same_tokenizer
log('model files recorded; same template', same_template, 'same tokenizer', same_tokenizer)


# The jinja environment transformers (and so vLLM) renders chat templates with: sandboxed, trim/lstrip blocks,
# loop controls, tojson filter, raise_exception and strftime_now globals.
def compile_template(text):
    def raise_exception(message): raise jinja2.exceptions.TemplateError(message)
    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    def strftime_now(fmt): return time.strftime(fmt)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters['tojson'] = tojson; env.globals['raise_exception'] = raise_exception; env.globals['strftime_now'] = strftime_now
    return env.from_string(text)


TEMPLATES = {tag: compile_template((p / 'chat_template.jinja').read_text()) for tag, p in MODELS.items()}
TOK = Tokenizer.from_file(str(MODELS['e4b'] / 'tokenizer.json'))
tcfg = json.loads((MODELS['e4b'] / 'tokenizer_config.json').read_text())
BOS, EOS = tcfg.get('bos_token', '<bos>'), tcfg.get('eos_token', '<turn|>')
if isinstance(BOS, dict): BOS = BOS['content']
if isinstance(EOS, dict): EOS = EOS['content']
STOP_IDS = set(json.loads((MODELS['e4b'] / 'generation_config.json').read_text()).get('eos_token_id', []))


def encode(text):
    return TOK.encode(text, add_special_tokens=False).ids


NL = encode('\n'); OPENER = encode('<|channel>thought\n')
report['special'] = {'bos': BOS, 'eos': EOS, 'stop_ids': sorted(STOP_IDS), 'newline_ids': NL, 'thought_opener_ids': OPENER,
                     'ids': {s: TOK.token_to_id(s) for s in ('<bos>', '<turn|>', '<|turn>', '<|channel>', '<channel|>', '<|tool_call>', '<tool_call|>', '<|tool_response>', '<tool_response|>', '<|think|>')}}


def render(tag, messages, tools, add_generation_prompt, enable_thinking):
    return TEMPLATES[tag].render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                 enable_thinking=enable_thinking, bos_token=BOS, eos_token=EOS)


def as_mapping(tool_calls):
    return [{**tc, 'function': {**tc['function'], 'arguments': json.loads(tc['function']['arguments'])}} for tc in tool_calls]


def history(messages, hist='keep', args='map', identity=True):
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if v is not None}
        if m['role'] == 'system' and identity:
            # The request's system message: the instruction file text (the trace stores it without its final newline),
            # a blank line, the identity line, sent as one text part (the template then adds a trailing space).
            m['content'] = [{'type': 'text', 'text': m['content'].rstrip('\n') + '\n\n\n' + IDENTITY}]
        if m['role'] == 'assistant':
            if hist == 'drop': m.pop('reasoning_content', None)
            if m.get('tool_calls') and args == 'map': m['tool_calls'] = as_mapping(m['tool_calls'])
        out.append(m)
    return out


def target_message(t):
    # As generated: reasoning kept, arguments as a mapping so the template writes the Gemma call syntax.
    m = {k: v for k, v in t.items() if v is not None}
    if m.get('tool_calls'): m['tool_calls'] = as_mapping(m['tool_calls'])
    return m


def prompt_ids(tag, msgs, tools):
    # The served generation prompt ends at the tool response (the model opens its own thought channel); a template
    # that appends the opener after a tool response has it removed here, and the report says how often.
    p = encode(render(tag, msgs, tools, True, True)); stripped = False
    if p[-len(OPENER):] == OPENER: p = p[:-len(OPENER)]; stripped = True
    return p, stripped


TOOLS = json.loads((DS / 'tools.json').read_text())
tools = TOOLS['tools'] if isinstance(TOOLS, dict) and 'tools' in TOOLS else TOOLS
rows = [json.loads(l) for l in (DS / 'examples.jsonl').read_text().splitlines() if l.strip()]
if LIMIT: rows = rows[:LIMIT]
report['examples'] = len(rows); report['examples_sha256'] = sha(DS / 'examples.jsonl'); report['tools_sha256'] = sha(DS / 'tools.json')
tier_of = lambda r: 'qat' if 'qat-w4a16' in r['model_served'] else 'bf16'
report['tiers'] = dict(collections.Counter(tier_of(r) for r in rows))
log('examples', len(rows), 'tools', len(tools), report['tiers'])

# 1. the record of why the rendering is what it is: the history variants, with and without the identity line,
#    scored on a sample of the qat tier against the served prompt counts (the template that served those traces)
random.seed(20261030); qat_rows = [r for r in rows if tier_of(r) == 'qat']; sample = random.sample(qat_rows, min(120, len(qat_rows)))
matrix = {}
for identity in (True, False):
    for think in (True, False):
        for hist in ('keep', 'drop'):
            for args in ('str', 'map'):
                deltas = []
                for r in sample:
                    p = encode(render('31b', history(r['messages'], hist, args, identity), tools, True, think))
                    if p[-len(OPENER):] == OPENER: p = p[:-len(OPENER)]
                    deltas.append(len(p) - r['served_prompt_tokens'])
                key = '/'.join(('identity' if identity else 'noidentity', 'think' if think else 'nothink', hist, args))
                matrix[key] = {'exact': sum(d == 0 for d in deltas), 'of': len(sample), 'median_delta': statistics.median(deltas), 'min_delta': min(deltas), 'max_delta': max(deltas)}
                log('variant', key, matrix[key])
report['variant_matrix'] = matrix
report['variant'] = 'identity/think/keep/map (fixed by the harness: ADK appends the identity line, sends reasoning_content back, and the served counts match mapping-rendered arguments)'

# 2. render every example
TRAIN_TAG = 'e4b'
checks = collections.Counter(); kinds = collections.defaultdict(collections.Counter); tiers = collections.defaultdict(collections.Counter); mism = []
with (OUT / 'rendered.jsonl').open('w') as f:
    for i, r in enumerate(rows):
        tier = tier_of(r); msgs = history(r['messages'])
        prompt, stripped = prompt_ids(TRAIN_TAG, msgs, tools)
        if same_template: prompt_served = prompt
        else: prompt_served, _ = prompt_ids('31b', msgs, tools)
        full = encode(render(TRAIN_TAG, msgs + [target_message(r['target'])], tools, False, True))
        prompt_delta = len(prompt_served) - r['served_prompt_tokens']
        if tier == 'bf16':
            # the served template dropped history reasoning; the shortfall of the served count against a
            # reasoning-free rendering is the compaction signal here
            drop, _ = prompt_ids('31b', history(r['messages'], hist='drop'), tools); compaction_delta = len(drop) - r['served_prompt_tokens']
        else:
            compaction_delta = prompt_delta
        compacted = compaction_delta > COMPACTION_SHORTFALL
        boundary = full[:len(prompt)] == prompt
        completion = full[len(prompt):] if boundary else []
        if len(completion) >= 2 and completion[-2] in STOP_IDS and completion[-1:] == NL: completion = completion[:-1]
        ends_with_stop = bool(completion) and completion[-1] in STOP_IDS
        completion_delta = len(completion) - (r['served_completion_tokens'] or 0)
        truncated = (r['served_completion_tokens'] or 0) >= 8192
        t = r['target']; kind = ('R' if t.get('reasoning_content') else '-') + ('C' if t.get('content') else '-') + ('T' if t.get('tool_calls') else '-')
        prompt_ok = prompt_delta == 0 if tier == 'qat' else not compacted
        usable = prompt_ok and boundary and ends_with_stop and abs(completion_delta) <= 1 and not truncated and not compacted
        checks['prompt_exact'] += prompt_delta == 0; checks['boundary_ok'] += boundary; checks['ends_with_stop'] += ends_with_stop
        checks['completion_within_1'] += abs(completion_delta) <= 1; checks['truncated'] += truncated; checks['compacted'] += compacted
        checks['opener_stripped'] += stripped; checks['usable'] += usable
        kinds[kind]['n'] += 1; kinds[kind]['usable'] += usable; kinds[kind]['completion_delta_' + str(max(-3, min(3, completion_delta)))] += 1
        tiers[tier]['n'] += 1; tiers[tier]['usable'] += usable; tiers[tier]['prompt_exact'] += prompt_delta == 0; tiers[tier]['compacted'] += compacted
        tiers[tier]['prompt_delta_' + str(max(-5, min(5, prompt_delta)))] += 1
        if not usable and len(mism) < 60:
            mism.append({'example_id': r['example_id'], 'tier': tier, 'kind': kind, 'prompt_delta': prompt_delta, 'compaction_delta': compaction_delta, 'boundary': boundary,
                         'completion_delta': completion_delta, 'truncated': truncated, 'ends_with_stop': ends_with_stop,
                         'served': [r['served_prompt_tokens'], r['served_completion_tokens']],
                         'completion_tail': TOK.decode(completion[-12:], skip_special_tokens=False) if completion else None,
                         'full_tail': TOK.decode(full[-12:], skip_special_tokens=False)})
        f.write(json.dumps({'example_id': r['example_id'], 'attempt_id': r['attempt_id'], 'task_id': r['task_id'], 'repository': r['repository'], 'call_index': r['call_index'],
                            'tier': tier, 'kind': kind, 'usable': usable, 'prompt_delta': prompt_delta, 'compaction_delta': compaction_delta, 'completion_delta': completion_delta,
                            'boundary': boundary, 'truncated': truncated, 'compacted': compacted, 'prompt_tokens': len(prompt), 'completion_tokens': len(completion),
                            'provenance': r['provenance'], 'prompt_token_ids': prompt, 'completion_token_ids': completion}) + '\n')
        if i % 100 == 0: log('rendered', i, dict(checks))
report['checks'] = dict(checks); report['by_target_kind'] = {k: dict(v) for k, v in kinds.items()}; report['by_tier'] = {k: dict(v) for k, v in tiers.items()}
report['mismatches'] = mism
report['rendering'] = {'prompt_template': '31b (served) for the checks, e4b for training' if not same_template else 'e4b (identical to 31b)',
                       'history': 'thinking enabled, reasoning kept, arguments as mapping, identity line appended to the system instruction',
                       'generation_prompt': 'ends at the tool response; a template-appended thought opener is removed (count in checks.opener_stripped)',
                       'target': 'reasoning kept, arguments as mapping (Gemma call syntax), stop token kept, newline after <turn|> dropped',
                       'compaction_rule': f'served prompt more than {COMPACTION_SHORTFALL} tokens shorter than the rendered history'}
report['rendered_sha256'] = sha(OUT / 'rendered.jsonl'); report['seconds'] = round(time.time() - T0, 1)
(OUT / 'render-report.json').write_text(json.dumps(report, indent=2))
log('done', json.dumps(report['checks']), json.dumps(report['by_tier']))
