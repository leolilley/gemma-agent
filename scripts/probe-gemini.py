#!/usr/bin/env python3
"""Day-one check 3: does the Gemini OpenAI-compatible endpoint serve Gemma 4 with tool calling,
and does it accept the vLLM-style thinking fields the harness sends? Prints one line per probe.
Requires LITELLM_API_KEY. Makes at most four small requests."""
import json, os, sys, urllib.request
from pathlib import Path
env = Path('.env')
if env.exists():
    for line in env.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1); os.environ.setdefault(k.strip(), v.strip())
key = os.environ.get('LITELLM_API_KEY') or sys.exit('set LITELLM_API_KEY')
base = os.environ.get('LITELLM_API_BASE', 'https://generativelanguage.googleapis.com/v1beta/openai').rstrip('/')
def call(path, body=None):
    req = urllib.request.Request(f'{base}{path}', data=json.dumps(body).encode() if body else None,
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}, method='POST' if body else 'GET')
    try:
        with urllib.request.urlopen(req, timeout=60) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, e.read().decode()[:300]
st, models = call('/models')
names = [m['id'] for m in models.get('data', [])] if isinstance(models, dict) else []
gemma = sorted(n for n in names if 'gemma' in n.lower())
print('models:', st, 'gemma ids:', gemma or '(none)')
tool = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file',
         'parameters': {'type': 'object', 'properties': {'filepath': {'type': 'string'}}, 'required': ['filepath']}}}]
msgs = [{'role': 'user', 'content': 'Read the file src/app.py using the tool.'}]
for mid in ['gemma-4-31b-it']:  # plain id; the 'models/' prefixed form intermittently returns 500 on this route
    st, r = call('/chat/completions', {'model': mid, 'messages': msgs, 'tools': tool, 'max_tokens': 200})
    tc = (r.get('choices', [{}])[0].get('message', {}).get('tool_calls') if isinstance(r, dict) else None)
    print('tool call plain:', mid, st, 'tool_calls=' + str(bool(tc)), '' if isinstance(r, dict) else r)
    st, r = call('/chat/completions', {'model': mid, 'messages': msgs, 'tools': tool, 'max_tokens': 200,
        'extra_body': {'chat_template_kwargs': {'enable_thinking': True}, 'thinking_token_budget': 1024}})
    print('with vLLM extra_body:', st, '' if isinstance(r, dict) else r)
    st, r = call('/chat/completions', {'model': mid, 'messages': msgs, 'tools': tool, 'max_tokens': 200, 'reasoning_effort': 'low'})
    print('with reasoning_effort:', st, '' if isinstance(r, dict) else r)
