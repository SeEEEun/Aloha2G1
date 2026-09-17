#!/usr/bin/env python3
"""Retire only an explicitly resolved owned diagnostic, preserving its files."""
from pathlib import Path
import sys,os,signal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *

pid=154013
case='WRIST_EP005'
done=TRAIN/case/'SOURCE_POSITION_PASS.json';r=read(done)
assert r.get('qualified') and r['metrics']['pass_numeric'] and not r['blocked_frames'] and not r['preparation_blocked_frames']
proc=Path('/proc')/str(pid)/'cmdline'
if proc.exists():
    cmd=proc.read_bytes().decode().split('\x00')
    assert cmd[:3]==['/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python','tools/run_common_train_seed_recovery_v5.py',case],cmd
    out=ST5/'RETIRED_SUPERSEDED_OWNED_DIAGNOSTIC.json'
    atomic_json(out,dict(pid=pid,command=cmd,qualified_replacement=file_record(done),
        reason='Older unqualified diagnostic now superseded by independently qualified shared recovery; no valid run removed',
        files_deleted=False,signal='SIGTERM',scope='Owned numerical diagnostic only, not physical trial or predecessor recovery'))
    os.kill(pid,signal.SIGTERM)
    print('Retired superseded owned diagnostic',pid,'all artifacts retained',flush=True)
