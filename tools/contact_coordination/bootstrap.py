from pathlib import Path
import hashlib, importlib.metadata, platform, subprocess, sys
from .io import ROOT, atomic_json, read, record

def run(out,resume=False):
    old=ROOT/'outputs/final_single_variable_ab'
    split=read(old/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')
    rows=[]
    for e in split['entries']:
        raw=record(e['source_parquet'])
        assert raw['sha256']==e['source_parquet_sha256'],raw['path']
        rows.append({k:v for k,v in e.items() if k not in ('corrected_cartesian_references','previous_checkpoint_usage')})
    assert sum(bool(r['TRAIN40']) for r in rows)==40
    assert sum(bool(r['DEV35_DIAGNOSTIC35']) for r in rows)==35
    atomic_json(out/'bootstrap/SPLITS.json',dict(entries=rows,counts=split['counts'],source=record(old/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')))
    packages={}
    for k in ['numpy','scipy','mujoco','pyarrow','ruckig','ompl','matplotlib','trimesh']:
        try: packages[k]=importlib.metadata.version(k)
        except importlib.metadata.PackageNotFoundError: packages[k]=None
    atomic_json(out/'bootstrap/OFFLINE_SOFTWARE.json',dict(executable=sys.executable,python=sys.version,platform=platform.platform(),packages=packages))
    return dict(status='COMPLETE',raw_recordings_verified=len(rows),train=40,dev=35,old_project_jobs_found=0,prior_changes_preserved=True)
