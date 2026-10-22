#!/usr/bin/env python3
"""Which LiteLLM route reaches Gemma 4 with tool calling? Tries the OpenAI-compatible endpoint and the
native Gemini provider, with and without tools, two model id forms. Prints one sanitized line per probe."""
import os, re, sys
from pathlib import Path
for line in Path('.env').read_text().splitlines() if Path('.env').exists() else []:
    if '=' in line and not line.startswith('#'):
        k, v = line.split('=', 1); os.environ.setdefault(k.strip(), v.strip())
key = os.environ.get('LITELLM_API_KEY') or sys.exit('set LITELLM_API_KEY')
os.environ.setdefault('GEMINI_API_KEY', key)
import litellm
litellm.drop_params = True; litellm.suppress_debug_info = True
TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file',
          'parameters': {'type': 'object', 'properties': {'filepath': {'type': 'string'}}, 'required': ['filepath']}}}]
MSG = [{'role': 'user', 'content': 'Read the file src/app.py using the tool.'}]
BASE = 'https://generativelanguage.googleapis.com/v1beta/openai'
def san(s): return re.sub(r'(AIza|key=)[A-Za-z0-9_-]+', r'\1***', str(s))[:260].replace('\n', ' ')
def probe(label, **kw):
    try:
        r = litellm.completion(messages=MSG, max_tokens=120, **kw)
        m = r.choices[0].message
        print(f'OK   {label}: tool_calls={bool(m.tool_calls)} text={san(m.content)[:60]!r}')
    except Exception as e:
        print(f'FAIL {label}: {type(e).__name__}: {san(e)}')
for mid in ('gemma-4-31b-it', 'models/gemma-4-31b-it'):
    probe(f'openai-compat no-tools {mid}', model=f'openai/{mid}', api_base=BASE, api_key=key)
    probe(f'openai-compat tools    {mid}', model=f'openai/{mid}', api_base=BASE, api_key=key, tools=TOOLS)
probe('gemini-native no-tools gemma-4-31b-it', model='gemini/gemma-4-31b-it', api_key=key)
probe('gemini-native tools    gemma-4-31b-it', model='gemini/gemma-4-31b-it', api_key=key, tools=TOOLS)
probe('openai-compat tools    gemma-4-26b-a4b-it', model='openai/gemma-4-26b-a4b-it', api_base=BASE, api_key=key, tools=TOOLS)
