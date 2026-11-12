"""Modal image smoke: the versions the Kaggle kernels record, the GPUs, and the mounts they expect."""
import importlib.metadata as md, json, subprocess
from pathlib import Path
import torch
print(json.dumps({d: md.version(d) for d in ('torch', 'transformers', 'vllm', 'adk-submission', 'swegemma', 'google-adk', 'compressed-tensors', 'tokenizers', 'jinja2')}))
print('cuda', torch.cuda.is_available(), torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
import vllm, adk_submission, swegemma  # noqa: F401
for p in ('input/competitions/gemma-4-developer-agent/tasks.jsonl', 'input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2/model.safetensors',
          'input/datasets/metric/gemma-4-developer-agent-wheelhouse', 'input/datasets/leolilley/gemma-agent-bundles/run.json'):
    print(p, (Path('/kaggle') / p).exists())
print(subprocess.run(['git', '--version'], capture_output=True, text=True).stdout.strip())
