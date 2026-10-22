"""CPU-only probe of the organizer's pinned Kaggle image: why `venv --with-pip` fails in the harness's subprocess sandbox,
and whether an offline repair makes the repository under test importable. No GPU, no model, no agent. Prints facts only.

Background (smoke run 2026-10-13): swegemma's SubprocessSandbox creates a venv with system_site_packages=True and
with_pip=True; ensurepip failed, the venv got no pip, the repository was never installed, and for src-layout repos
(requests) the tests imported /usr/local/lib/python3.12/dist-packages/requests instead of /workspace/src/requests."""
import glob, importlib.util, json, os, shutil, subprocess, sys, tarfile, tempfile, zipfile
from pathlib import Path

def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, (r.stdout + r.stderr)[-1500:]

out = {'python': sys.version, 'executable': sys.executable}
# 1. ensurepip state
try:
    import ensurepip
    bundled = Path(ensurepip.__file__).parent / '_bundled'
    out['ensurepip'] = {'file': ensurepip.__file__, 'pip_version': getattr(ensurepip, '_PIP_VERSION', None),
                        'setuptools_version': getattr(ensurepip, '_SETUPTOOLS_VERSION', None),
                        'bundled_dir_exists': bundled.exists(), 'bundled': sorted(p.name for p in bundled.glob('*')) if bundled.exists() else [],
                        'writable': os.access(bundled.parent, os.W_OK)}
except Exception as e:
    out['ensurepip'] = {'import_error': repr(e)}
# 2. system pip and the repos the image preinstalls
rc, txt = run([sys.executable, '-m', 'pip', '--version']); out['system_pip'] = txt.strip()
pre = {}
for mod in ('requests', 'rich', 'httpx', 'fastapi', 'pytest', 'pytest_httpbin', 'trustme', 'setuptools'):
    spec = importlib.util.find_spec(mod)
    pre[mod] = spec.origin if spec else None
out['preinstalled'] = pre
# 3. plain venv with pip: the harness's call
tmp = Path(tempfile.mkdtemp()); v1 = tmp / 'v1'
rc, txt = run([sys.executable, '-m', 'venv', '--system-site-packages', str(v1)]); out['venv_with_pip'] = {'rc': rc, 'tail': txt[-600:]}
# 4. offline repair attempt A: pip wheel built from the installed pip, named as ensurepip expects
repair = {}
try:
    import pip
    pipdir = Path(pip.__file__).parent; dist = next(pipdir.parent.glob('pip-*.dist-info'))
    want = out['ensurepip'].get('pip_version') or pip.__version__
    whl = tmp / f'pip-{want}-py3-none-any.whl'
    with zipfile.ZipFile(whl, 'w', zipfile.ZIP_DEFLATED) as z:
        for base in (pipdir, dist):
            for p in base.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts: z.write(p, p.relative_to(pipdir.parent))
    repair['wheel'] = whl.name; repair['wheel_bytes'] = whl.stat().st_size; repair['installed_pip_version'] = pip.__version__
    bundled = Path(ensurepip.__file__).parent / '_bundled'; bundled.mkdir(exist_ok=True)
    shutil.copy(whl, bundled / whl.name)
    v2 = tmp / 'v2'; rc, txt = run([sys.executable, '-m', 'venv', '--system-site-packages', str(v2)]); repair['venv_after_repair'] = {'rc': rc, 'tail': txt[-600:]}
    rc, txt = run([str(v2 / 'bin/python3'), '-m', 'pip', '--version']); repair['venv_pip'] = txt.strip()[-200:]
    # 5. install a src-layout repo editable, offline, from the competition wheels, the way the harness would
    data = Path('/kaggle/input/competitions/gemma-4-developer-agent'); snap = next(data.glob('snapshots/requests_*.tgz'), None)
    repair['snapshot'] = snap.name if snap else None
    if snap and rc == 0:
        ws = tmp / 'ws'; ws.mkdir()
        with tarfile.open(snap) as t: t.extractall(ws)
        root = next((p for p in [ws, *ws.iterdir()] if (p / 'pyproject.toml').exists() or (p / 'setup.py').exists()), ws)
        rc, txt = run([str(v2 / 'bin/python3'), '-m', 'pip', 'install', '--no-index', '--find-links', str(data / 'wheels'), '--no-build-isolation', '-e', str(root)], timeout=600)
        repair['editable_install'] = {'rc': rc, 'tail': txt[-900:]}
        rc, txt = run([str(v2 / 'bin/python3'), '-c', 'import requests, sys; print(requests.__file__)'], cwd=str(tmp)); repair['requests_import'] = txt.strip()[-300:]
        rc, txt = run([str(v2 / 'bin/python3'), '-m', 'pip', 'install', '--no-index', '--find-links', str(data / 'wheels'), 'pytest', 'pytest-httpbin', 'trustme'], timeout=600); repair['test_deps_install'] = {'rc': rc, 'tail': txt[-500:]}
except Exception as e:
    repair['error'] = repr(e)
out['repair'] = repair
out['competition_wheels'] = sorted(p.name for p in Path('/kaggle/input/competitions/gemma-4-developer-agent/wheels').glob('*.whl'))[:200] if Path('/kaggle/input/competitions/gemma-4-developer-agent/wheels').exists() else 'missing'
Path('/kaggle/working/env_probe.json').write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != 'competition_wheels'}, indent=2))
