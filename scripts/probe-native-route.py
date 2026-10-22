#!/usr/bin/env python3
"""Does LiteLLM's native gemini/ provider accept requests carrying the vLLM-only extra_body fields the
harness always sends? Tries the exact field set, with tools, with and without an api_base, as the
harness's registry would construct the model. Sanitized one-line results."""
import os, re, sys
from pathlib import Path
for line in Path('.env').read_text().splitlines() if Path('.env').exists() else []:
    if '=' in line and not line.startswith('#'):
        k, v = line.split('=', 1); os.environ.setdefault(k.strip(), v.strip())
key = os.environ['LITELLM_API_KEY']; os.environ.setdefault('GEMINI_API_KEY', key)
import litellm; litellm.drop_params = True; litellm.suppress_debug_info = True
TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file',
          'parameters': {'type': 'object', 'properties': {'filepath': {'type': 'string'},
          'start_line': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None}}, 'required': ['filepath']}}}]
MSG = [{'role': 'user', 'content': 'Read the file src/app.py using the tool.'}]
XB = {'chat_template_kwargs': {'enable_thinking': True}, 'thinking_token_budget': 4096}
def san(s): return re.sub(r'(AIza|key=)[A-Za-z0-9_-]+', r'\1***', str(s))[:240].replace('\n', ' ')
def probe(label, **kw):
    try:
        r = litellm.completion(messages=MSG, max_tokens=120, **kw); m = r.choices[0].message
        print(f'OK   {label}: tool_calls={bool(m.tool_calls)} text={san(m.content)[:50]!r}')
    except Exception as e: print(f'FAIL {label}: {type(e).__name__}: {san(e)}')
probe('gemini-native + extra_body + tools (no api_base)', model='gemini/gemma-4-31b-it', api_key=key, tools=TOOLS, extra_body=XB)
probe('gemini-native + extra_body + tools (api_base googleapis)', model='gemini/gemma-4-31b-it', api_key=key, api_base='https://generativelanguage.googleapis.com', tools=TOOLS, extra_body=XB)
probe('gemini-native + extra_body + tools + max_completion_tokens', model='gemini/gemma-4-31b-it', api_key=key, tools=TOOLS, extra_body=XB, max_completion_tokens=200)
probe('gemini-native + anyOf tool schema only', model='gemini/gemma-4-31b-it', api_key=key, tools=TOOLS)
probe('openai-compat + anyOf tool schema (no extra_body)', model='openai/gemma-4-31b-it', api_key=key, api_base='https://generativelanguage.googleapis.com/v1beta/openai', tools=TOOLS)
