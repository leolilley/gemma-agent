"""Evaluation arm on Kaggle's free L4x4, driven entirely by run.json from the bundles dataset. The same script
serves both kernels (eval-4bit, eval-bf16); only kernel-metadata.json differs, in which model mounts it attaches.

Mirrors the organizer's Getting Started notebook cell for cell (wheelhouse install, dataset paths,
vLLM server with the scorer's flags, context cache and compaction config), then runs the official
Evaluator over the task ids in run.json. The kernel code never changes between runs; each run is a
new version of the `gemma-agent-bundles` dataset carrying a different run.json and bundle.

run.json: {"run_id": str, "bundle": "<dir under bundles/>", "task_ids": [..], "concurrency": int,
           "budgets": {"max_time_minutes": n, "max_tool_calls": n, "max_turns": n, "timeout_seconds": n}}
Outputs: /kaggle/working/results/<run_id>/ (harness artifacts) and results-<run_id>.zip.
"""
import glob, importlib, json, logging, os, re, shutil, subprocess, sys, zipfile
from pathlib import Path

os.environ.update({'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1',
    'VLLM_NO_USAGE_STATS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_ENGINE_READY_TIMEOUT_S': '2400',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1', 'NCCL_DEBUG': 'WARN', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
    'LITELLM_TELEMETRY': 'False', 'LITELLM_LOG': 'WARNING', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'OTEL_SDK_DISABLED': 'true', 'OTEL_PYTHON_DISABLED': 'true'})
os.environ.setdefault('OPENAI_API_KEY', 'EMPTY'); os.environ.setdefault('LOCAL_API_KEY', 'EMPTY')
for name in ('LiteLLM', 'litellm', 'httpx', 'httpcore', 'openai', 'urllib3', 'opentelemetry'):
    logging.getLogger(name).setLevel(logging.WARNING)

# 1. wheelhouse (identical to the organizer notebook)
def find_input(*names: str) -> Path:
    # Kaggle mounts inputs at /kaggle/input/<kind>/<owner>/<slug> or at /kaggle/input/<slug> depending on the kernel;
    # find the first existing candidate and print the input tree if none exists, so a mount problem is visible.
    for n in names:
        for c in [Path('/kaggle/input') / n, *Path('/kaggle/input').glob(f'*/*/{Path(n).name}'), *Path('/kaggle/input').glob(Path(n).name)]:
            if c.exists(): return c
    for p in sorted(Path('/kaggle/input').glob('*/*'))[:40]: print('input:', p)
    for p in sorted(Path('/kaggle/input').glob('*'))[:40]: print('input:', p)
    raise SystemExit(f'none of {names} mounted')

WHEELHOUSE_DIR = find_input('datasets/metric/gemma-4-developer-agent-wheelhouse')
print('wheelhouse:', WHEELHOUSE_DIR, len(list(WHEELHOUSE_DIR.glob('*.whl'))), 'wheels')
assert any(WHEELHOUSE_DIR.glob('*.whl')), 'wheelhouse mounted but empty'
for pth in glob.glob('/usr/local/lib/python*/*-packages/*cutlass*.pth'):
    try: os.unlink(pth)
    except OSError: pass
tmp_whl = Path('/tmp/wheelhouse'); tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower(): continue
    target = tmp_whl / (w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name)
    if not target.exists():
        try: os.symlink(w, target)
        except OSError: shutil.copyfile(w, target)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
                *sorted(str(w) for w in tmp_whl.glob('*.whl'))], check=True)
importlib.invalidate_caches()

# 2. run parameters and bundle
DATA_DIR = find_input('competitions/gemma-4-developer-agent')
BUNDLES = find_input('datasets/leolilley/gemma-agent-bundles')
WORK = Path('/kaggle/working'); WORK.mkdir(exist_ok=True)
run = json.loads((BUNDLES / 'run.json').read_text())
RUN_ID = run['run_id']; AGENT_DIR = WORK / 'bundle'
if AGENT_DIR.exists(): shutil.rmtree(AGENT_DIR)
shutil.copytree(BUNDLES / 'bundles' / run['bundle'], AGENT_DIR)
RESULTS = WORK / 'results' / RUN_ID; RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / 'run.json').write_text(json.dumps(run, indent=2))
# a forecast shipped with this dataset version (push-run.py --forecast) is returned with the outcomes it predicts
if (BUNDLES / 'forecast.md').exists(): shutil.copy(BUNDLES / 'forecast.md', RESULTS / 'forecast.md')
print(f'run {RUN_ID}: bundle={run["bundle"]} tasks={len(run["task_ids"])} concurrency={run.get("concurrency", 1)}')

from swegemma.models import load_tasks
tasks = load_tasks(DATA_DIR / 'tasks.jsonl')

# 3. vLLM with the scorer's flags
import litellm, torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from adk_submission.discovery import discover_declared_models
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
litellm.drop_params = True
MODELS = {  # alias in the bundle's agent.yaml -> Kaggle Models mount. The kernel metadata must attach the one the bundle names.
    'gemma-4-31b-it-qat-w4a16-ct': '/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2',
    'gemma-4-12b-it-qat-w4a16-ct': '/kaggle/input/models/google/gemma-4/other/gemma-4-12b-it-qat-w4a16-ct/2',
    'gemma-4-e4b-it-qat-w4a16-ct': '/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2',
    'gemma-4-e2b-it-qat-w4a16-ct': '/kaggle/input/models/google/gemma-4/other/gemma-4-e2b-it-qat-w4a16-ct/2',
    'gemma-4-31b-it': '/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1',   # bf16 weights, 62.6 GB, TP 4
}
# Memory headroom only, not numerics: bf16 31B weights leave 0.94 GiB of KV cache at 0.90 on 4×L4, below the 1.21 GiB one
# 32,768-token request needs (eval-bf16 v2, 2026-10-14). 4-bit models keep the scorer's 0.90.
GPU_MEM_UTIL = {'gemma-4-31b-it': 0.95}
EXTRA = {'gemma-4-12b-it-qat-w4a16-ct': ('--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')}
declared = discover_declared_models(AGENT_DIR); assert len(declared) == 1, declared
model = next(iter(declared)); assert model in MODELS, model
MODEL_PATH = Path(MODELS[model])
if not MODEL_PATH.exists():
    cands = [p for p in Path('/kaggle/input').rglob('config.json') if model.replace('-qat-w4a16-ct', '') in str(p).lower() and len(p.parts) < 12]
    print('model candidates:', cands[:5]); MODEL_PATH = cands[0].parent if cands else MODEL_PATH
assert MODEL_PATH.exists(), f'{MODEL_PATH} is not attached to this kernel'
print('model path:', MODEL_PATH)
adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
adapter_map = getattr(adapters, 'adapters', None) or {}
ranks = [r for i in adapter_map.values() if (r := getattr(i, 'rank', None))]
GPU_MEM_UTIL = GPU_MEM_UTIL.get(model, 0.90); print('gpu_memory_utilization', GPU_MEM_UTIL)
gpus = torch.cuda.device_count(); tp = 4 if gpus >= 4 else (2 if gpus >= 2 else 1)
server = VllmServer(VllmConfig(model=str(MODEL_PATH), port=8000, host='127.0.0.1', tool_call_parser='gemma4',
    reasoning_parser='gemma4', max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=GPU_MEM_UTIL,
    enable_auto_tool_choice=True, enable_lora=bool(adapter_map), max_loras=max(1, len(adapter_map)),
    max_lora_rank=max(ranks) if ranks else 128, scheduler_reserve_full_isl=False, tensor_parallel_size=tp,
    startup_timeout=60 * 40, extra_args=EXTRA.get(model, ())), adapter_manifest=adapters)   # 40 min: the model mount prefetch alone took 9 to 13 min on 2026-10-21
server.start(); print('vLLM up', server.base_url, 'tp', tp)
models = server.create_model_registry(aliases=[model], model_prefix='openai/', api_key='EMPTY')

# 4. evaluate the cohort with the scorer's context settings
import asyncio
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
limits, gen = build_submission_limits(); b = run['budgets']
cfg = EvalConfig(tasks_path=DATA_DIR / 'tasks.jsonl', snapshots_dir=DATA_DIR / 'snapshots', results_dir=RESULTS,
    submission_dir=AGENT_DIR, models=models, sandbox='subprocess', timeout_seconds=int(b['timeout_seconds']),
    verification_timeout_seconds=300, max_time_minutes=float(b['max_time_minutes']),
    max_tool_calls=int(b['max_tool_calls']), max_turns=int(b['max_turns']), task_ids=list(run['task_ids']),
    limits=limits, generation_constraints=gen, adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
    graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'), wheels_dir=DATA_DIR / 'wheels',
    concurrency=int(run.get('concurrency', 1)), display_mode='quiet', verbose=False)
result = asyncio.run(Evaluator(cfg).run())
print(f'{RUN_ID}: {result.resolved}/{result.total} resolved ({result.resolution_rate:.2%})')
server.stop()

# 5. identities and archive
ident = {'run_id': RUN_ID, 'model': model, 'model_path': str(MODEL_PATH), 'dtype': 'bfloat16', 'weights': 'w4a16' if 'w4a16' in model else 'bf16', 'tp': tp, 'gpus': gpus, 'gpu_memory_utilization': GPU_MEM_UTIL,
         'bundle': run['bundle'], 'adapters': sorted(adapter_map), 'wheelhouse': sorted(p.name for p in WHEELHOUSE_DIR.glob('*.whl'))}
(RESULTS / 'identity.json').write_text(json.dumps(ident, indent=2))
with zipfile.ZipFile(WORK / f'results-{RUN_ID}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in RESULTS.rglob('*'):
        if p.is_file(): z.write(p, p.relative_to(WORK))
print('archived', WORK / f'results-{RUN_ID}.zip')
