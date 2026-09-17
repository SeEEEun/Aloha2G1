#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_local_redundancy_v5 import *

def main():
    verified_oracle_contract();result=read(ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json')
    assert result['status']=='SMOKE3_COMMON_EXECUTABLE_POSITION_QUALIFIED'
    folder=ST5/'common_training_seed_bank';out=folder/'MODEL_Q_SEEDS.npz'
    if out.exists():print('REUSE_COMMON_SEED_BANK',flush=True);return
    g,c,n=model();s=CommonPositionSolver(g,c,read(QUAL),n);qq=[];inputs=[]
    for row in result['rows']:
        p=Path(row['source_trajectory']['path']);assert file_record(p)==row['source_trajectory'];q=np.load(p)['q']
        ids=np.unique(np.r_[np.arange(0,len(q),8),len(q)-1]);qq.extend(q[ids]);inputs.append(dict(source=file_record(p),indices=ids.tolist()))
    bank=np.array(qq);opened=np.load(RUN/'startup/WRIST_EP000_COMMON_PREFIX.npz')['common_hand_q'][0]
    fk=np.array([s.pose_jacobian(q,opened)[0] for q in bank])
    atomic_npz(out,q=bank,model_wrist_position=fk)
    atomic_json(folder/'MANIFEST.json',dict(bank=file_record(out),seed_count=len(bank),inputs=inputs,
        shared_for_all_representations=True,numerical_file_has_no_method_label=True,
        selection='Same deterministic source-frame stride8 plus endpoint on all six qualified TRAIN smoke trajectories',
        online_inputs='Incoming wrist positions, model, previous q, common hands and joint/collision constraints; KD nearest8 in bilateral FK space',
        success_or_DEV35_inputs=False,interpretation='Common TRAIN-calibrated deterministic G1 posture seeds, not object/grasp rescue targets',
        implementation=file_record(ROOT/'tools/common_training_seed_bank_v5.py')))
    print('COMMON_SEED_BANK_CREATED',len(bank),flush=True)

if __name__=='__main__':main()
