from pathlib import Path
import hashlib, json, os, time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'outputs/contact_coordination_retargeting/20260907T063704Z'
ISAAC = '/home/jbnu/miniconda3/envs/isaaclab6/bin/python'
OFFLINE = '/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python'

def read(path):
    return json.loads(Path(path).read_text())

def record(path):
    p = Path(path).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())

def default(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)

def atomic_text(path, content):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.{os.getpid()}.incomplete')
    with t.open('w') as f:
        f.write(content); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)

def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=default, allow_nan=False)+'\n')

def atomic_npz(path, **arrays):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.incomplete')
    with t.open('wb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(t,p)

def fingerprint(paths):
    rows=[record(p) for p in sorted(set(map(Path,paths)))]
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(), rows
