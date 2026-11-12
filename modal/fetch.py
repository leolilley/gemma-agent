"""Modal side of the campaign's off-Kaggle steps (DAgger labels, training, swap gate). This file: fetch the competition's
inputs (the Gemma 4 QAT model instances and the wheelhouse) from Kaggle into the `gemma-agent` volume, laid out as a
kernel sees /kaggle/input, so the Kaggle kernels run unchanged with KAGGLE_ROOT pointing at it.

  modal run modal/fetch.py::check     # the credential works (lists the 31B instance's files)
  modal run modal/fetch.py::fetch     # download what is missing; prints sha256 of every file it wrote"""
import modal

app = modal.App('gemma-agent-fetch')
vol = modal.Volume.from_name('gemma-agent', create_if_missing=True)
image = modal.Image.debian_slim(python_version='3.12').pip_install('kaggle==2.2.4', 'six')
secret = modal.Secret.from_name('gemma-agent-kaggle')
ROOT = '/vol/kaggle/input'
MODELS = {'gemma-4-e4b-it-qat-w4a16-ct': 'google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2',
          'gemma-4-31b-it-qat-w4a16-ct': 'google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2'}
WHEELHOUSE = 'metric/gemma-4-developer-agent-wheelhouse'


def auth():
    # the secret holds a new-style token (KGAT_...), which the CLI reads as KAGGLE_API_TOKEN
    import os
    k = os.environ.get('KAGGLE_KEY', '')
    if k.startswith('KGAT_'): os.environ['KAGGLE_API_TOKEN'] = os.environ.pop('KAGGLE_KEY')


@app.function(image=image, secrets=[secret], timeout=300)
def check():
    auth()
    import subprocess
    r = subprocess.run(['kaggle', 'models', 'instances', 'versions', 'files', MODELS['gemma-4-31b-it-qat-w4a16-ct']], capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-800:])


@app.function(image=image, secrets=[secret], volumes={'/vol': vol}, timeout=3 * 3600, cpu=4, memory=8192)
def fetch():
    auth()
    import hashlib, subprocess
    from pathlib import Path
    def sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
        return h.hexdigest()
    jobs = [(['kaggle', 'models', 'instances', 'versions', 'download', ref, '--untar', '-p'], Path(ROOT) / 'models' / ref, name) for name, ref in MODELS.items()]
    jobs.append((['kaggle', 'datasets', 'download', WHEELHOUSE, '--unzip', '-p'], Path(ROOT) / 'datasets' / WHEELHOUSE, 'wheelhouse'))
    for cmd, dest, name in jobs:
        if dest.exists() and any(dest.iterdir()): print('present', name, dest); continue
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd + [str(dest)], check=True)
        vol.commit()
        for p in sorted(dest.rglob('*')):
            if p.is_file(): print(name, p.relative_to(dest), p.stat().st_size, sha(p))
    vol.commit()


@app.function(image=image, secrets=[secret], volumes={'/vol': vol}, timeout=3 * 3600, cpu=4, memory=8192)
def fetch_competition():
    auth()
    # the competition's data (tasks.jsonl, snapshots, wheels, graphs, embeddings), mounted on Kaggle as competitions/<slug>
    import hashlib, subprocess, zipfile
    from pathlib import Path
    dest = Path(ROOT) / 'competitions/gemma-4-developer-agent'
    if dest.exists() and (dest / 'tasks.jsonl').exists(): print('present', dest); return
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(['kaggle', 'competitions', 'download', '-c', 'gemma-4-developer-agent', '-p', str(dest)], check=True)
    for z in dest.glob('*.zip'):
        with zipfile.ZipFile(z) as f: f.extractall(dest)
        z.unlink()
    vol.commit()
    n = sum(1 for p in dest.rglob('*') if p.is_file()); print('files', n, 'tasks.jsonl sha256', hashlib.sha256((dest / 'tasks.jsonl').read_bytes()).hexdigest())
