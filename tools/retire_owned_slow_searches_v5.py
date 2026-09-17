#!/usr/bin/env python3
"""Checkpoint-safe retirement of exactly two owned numerical trial processes."""
from pathlib import Path
import sys,os,signal,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *

rows=[]
for pid,case in [(157902,'WRIST_EP044'),(157909,'WRIST_EP032')]:
    proc=Path('/proc')/str(pid)/'cmdline'
    if not proc.exists():continue
    cmd=proc.read_bytes().decode().split('\x00')
    assert cmd[:3]==['/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python','tools/run_common_train_seed_recovery_v7.py',case],cmd
    saved=[]
    for p in (TRAIN/case/'common_posture_pool_recovery_v3').glob('*/GUIDE_PROGRESS.json'):
        r=read(p);z=p.with_suffix('.npz');q=np.load(z)['q'];frames=[x['frame'] for x in r['frames']]
        assert frames and np.isfinite(q[frames]).all()
        saved.append(dict(metadata=file_record(p),trajectory=file_record(z),completed_frames=len(frames),last_frame=frames[-1]))
    assert saved
    row=dict(pid=pid,case=case,command=cmd,checkpoints=saved,signal='SIGTERM',files_deleted=False,
        reason='Replace only exhaustive early-rejection work with bounded strict-witness sampling; unchanged complete final collision classifier',
        successor='tools/run_common_train_seed_recovery_v8.py',scope='Current-task owned numerical trial; not predecessor/hardware/physical rollout')
    atomic_json(ST5/f'OWNED_NUMERICAL_SEARCH_RESTART_{case}.json',row)
    os.kill(pid,signal.SIGTERM);rows.append(row)
    for _ in range(50):
        stat=Path('/proc')/str(pid)/'stat'
        if not stat.exists() or stat.read_text().split()[2]=='Z':break
        time.sleep(.1)
    else:raise RuntimeError(f'Owned process {pid} did not retire; do not launch duplicate')
print('OWNED_SEARCHES_RETIRED_WITH_CHECKPOINTS',[(r['case'],r['pid']) for r in rows],flush=True)
