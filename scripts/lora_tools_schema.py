#!/usr/bin/env python3
"""Write configs/lora/tools.json: the nine harness tools in the OpenAI tool format, built exactly as ADK sends them to the
model (swegemma.tools.create_tools -> google.adk FunctionTool declaration -> LiteLlm's _function_declaration_to_tool_param),
in the reference bundle's tool order. Run under the project venv."""
import json, sys, warnings
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
warnings.filterwarnings('ignore')
import yaml
from swegemma.tools import create_tools
from google.adk.tools.function_tool import FunctionTool
from google.adk.models.lite_llm import _function_declaration_to_tool_param

ROOT = Path(__file__).resolve().parent.parent
class _Loader(yaml.SafeLoader): pass
_Loader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))   # only the tools list is read here
order = yaml.load((ROOT / 'submissions/bf18/agent.yaml').read_text(), Loader=_Loader)['tools']
fns = create_tools(SimpleNamespace())
tools = [_function_declaration_to_tool_param(FunctionTool(fns[n])._get_declaration()) for n in order]
out = {'harness': {d: version(d) for d in ('swegemma', 'google-adk', 'adk-submission')}, 'order_from': 'submissions/bf18/agent.yaml', 'tools': tools}
(ROOT / 'configs/lora/tools.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
print(len(tools), [t['function']['name'] for t in tools])
