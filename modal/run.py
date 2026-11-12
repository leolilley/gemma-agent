"""Run one of the campaign's Kaggle kernels on Modal (A100-80GB by default; 4x L4, the Kaggle topology, did not schedule
on 2026-11-03), unchanged: the `gemma-agent`
volume holds /kaggle/input as a kernel sees it (modal/fetch.py for the models and wheelhouse; `modal volume put` for the
campaign's datasets), /kaggle is linked to it, and the kernel installs the wheelhouse itself as on Kaggle. Each run
starts from an empty /kaggle/working and leaves it at runs/<kernel>-<UTC>/ on the volume, with site.json.

  modal run modal/run.py --kernel-name dagger-label [--gpu A100-80GB:2]     # notebooks/<kernel>/<code_file> from kernel-metadata.json"""
import json
from pathlib import Path
import modal

app = modal.App('gemma-agent-gpu')
vol = modal.Volume.from_name('gemma-agent')
WH = '/vol/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse'


def install_wheelhouse():
    # the image: Kaggle's install (the wheelhouse with --no-deps, as the kernels do) over torch, then only the missing
    # dependencies, constrained to the wheelhouse's versions (Kaggle's base image supplies these there)
    import glob, re, subprocess, sys
    pip = [sys.executable, '-m', 'pip', 'install', '-q']
    torch = ['torch==2.10.0', 'torchvision==0.25.0', 'torchaudio==2.10.0']   # the trio vllm 0.19.1 pins
    subprocess.run(pip + torch + ['--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
    wheels = [w for w in sorted(glob.glob(f'{WH}/*.whl')) if 'cutlass' not in w.lower()]
    subprocess.run(pip + ['--no-deps', *wheels], check=True)
    pins = torch + [re.sub(r'_', '-', w.split('/')[-1].split('-')[0]) + '==' + w.split('/')[-1].split('-')[1] for w in wheels]
    open('/tmp/constraints.txt', 'w').write('\n'.join(pins) + '\n')
    for _ in range(6):
        out = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True).stdout
        missing = sorted({m.group(1) for m in re.finditer(r'requires ([A-Za-z0-9_.\-\[\]]+), which is not installed', out)})
        print('missing', missing)
        if not missing: break
        for name in missing:
            if subprocess.run(pip + ['-c', '/tmp/constraints.txt', '--extra-index-url', 'https://download.pytorch.org/whl/cu128', name]).returncode:
                subprocess.run(pip + ['--no-deps', name], check=True)
    # optional extras the kernels reach at run time (vLLM's server, the harness, litellm), which pip check does not flag:
    # import every entry point and install what is missing until they all import
    probe = ('import uvloop, vllm.entrypoints.openai.api_server, litellm, swegemma.evaluate, adk_submission, google.adk, peft, accelerate, '
             'jinja2, tokenizers, safetensors, yaml')
    alias = {'yaml': 'pyyaml', 'google': 'google-adk'}
    for _ in range(12):
        r = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
        m = re.search(r"No module named '([A-Za-z0-9_]+)", r.stderr)
        if r.returncode == 0: print('entry points import'); break
        if not m: raise SystemExit(r.stderr[-2000:])
        name = alias.get(m.group(1), m.group(1).replace('_', '-')); print('installing', name)
        subprocess.run(pip + ['-c', '/tmp/constraints.txt', name], check=True)
    else: raise SystemExit('entry points still do not import')
    print(subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True).stdout[-3000:])


image = (modal.Image.from_registry('nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04', add_python='3.12')
         .apt_install('git', 'build-essential').run_function(install_wheelhouse, volumes={'/vol': vol}))


def kernel(name: str, code: str, metadata: dict):
    import os, shutil, subprocess, sys, time
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    if not os.path.exists('/kaggle'): os.symlink('/vol/kaggle', '/kaggle')
    work = Path('/vol/kaggle/working')
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    script = Path('/tmp') / metadata['code_file']; script.write_text(code)
    site = {'site': 'modal-' + metadata['gpu'].lower().replace(':', 'x'), 'kernel': name, 'started_utc': stamp, 'metadata': metadata,
            'nvidia_smi': subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip().splitlines()}
    rc = subprocess.run([sys.executable, str(script)], cwd='/tmp').returncode
    site['returncode'] = rc; site['finished_utc'] = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    (work / 'site.json').write_text(json.dumps(site, indent=1))
    dest = Path('/vol/runs') / f'{name}-{stamp}'; dest.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(work), dest)
    vol.commit(); print('output', dest, 'returncode', rc)
    return str(dest)


# one function per GPU shape (this client cannot override a function's GPU per call)
OPTS = dict(image=image, cpu=16, memory=131072, volumes={'/vol': vol}, timeout=8 * 3600)


@app.function(gpu='A100-80GB', **OPTS)
def kernel_a100(name: str, code: str, metadata: dict): return kernel(name, code, metadata)


@app.function(gpu='A100-80GB:2', **OPTS)
def kernel_a100x2(name: str, code: str, metadata: dict): return kernel(name, code, metadata)


SHAPES = {'A100-80GB': kernel_a100, 'A100-80GB:2': kernel_a100x2}


@app.local_entrypoint()
def main(kernel_name: str, gpu: str = 'A100-80GB'):
    root = Path(__file__).resolve().parent.parent / 'notebooks' / kernel_name
    meta = json.loads((root / 'kernel-metadata.json').read_text())
    print(SHAPES[gpu].remote(kernel_name, (root / meta['code_file']).read_text(), {**meta, 'gpu': gpu}))
