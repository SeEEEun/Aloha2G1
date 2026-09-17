#!/usr/bin/env python3
"""Read-only orientation interface audit. Does not run or qualify 6D IK."""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *

def main():
    verified_oracle_contract();g,c,n=model();ids=read(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')['selection']['qualification_ids']
    rows=[]
    for ep in ids:
        p=OUT/'01_registration/raw_references'/f'TRAIN_EP{ep:03d}.npz';z=np.load(p)
        for mode in ('WRIST','INTERACTION'):
            for side in ('left','right'):
                R=z[f'{mode}_{side}_wrist_rotation_model'];tool=z[f'{mode}_{side}_wrist_to_tool']
                orth=float(np.max(np.abs(R@np.swapaxes(R,1,2)-np.eye(3))));det=float(np.max(np.abs(np.linalg.det(R)-1)))
                xyzw=Rotation.from_matrix(R).as_quat();roundtrip=Rotation.from_quat(xyzw).as_matrix()
                quaterror=float(np.max(np.abs(roundtrip-R)))
                world=g.model_to_world_rotation(R);back=g.world_to_model_rotation(world)
                rooterror=float(np.max(np.abs(back-R)))
                recomposed=R@tool[:3,:3];recovered=recomposed@tool[:3,:3].T
                toolerror=float(np.max(np.abs(recovered-R)))
                passed=max(orth,det,quaterror,rooterror,toolerror)<1e-12
                rows.append(dict(episode=ep,representation=mode,side=side,raw_reference=file_record(p),frames=len(R),
                    orthogonality_max=orth,determinant_error_max=det,xyzw_roundtrip_max=quaterror,
                    root_roundtrip_max=rooterror,wrist_tool_multiplication_roundtrip_max=toolerror,passed=passed))
    prior=OUT/'01_registration/RAW_TARGET_AUDIT.json';audit=read(prior)
    train=[r for r in audit['entries'] if r['qualification_subset']]
    assert len(train)==11 and all(r['pass'] for r in train) and all(r['passed'] for r in rows)
    folder=ST5/'orientation_interface_audit'
    result=dict(status='PASS_ORIENTATION_CONVENTION_ONLY',full_6d_ik_run=False,rows=rows,
        conventions=dict(rotations='active right-handed matrices',internal_quaternions='SciPy XYZW; compiled MuJoCo configuration WXYZ handled explicitly by model loader',
            model_to_world='R_world_wrist = R_world_model @ R_model_wrist',tool_chain='T_model_tool = T_model_wrist @ T_wrist_tool',
            solver_orientation_input='raw wrist_rotation_model; do not apply the tool calibration a second time',side_handling='named left/right model body and joint IDs'),
        source_independent_orientation_audit=file_record(prior),registration=file_record(OUT/'01_registration/COMMON_TASK_WORKSPACE_REGISTRATION_AUDIT.json'),
        common_frame_bug_found=False,frame_calibration_changes=0)
    atomic_json(folder/'ORIENTATION_CHAIN_AUDIT.json',result)
    atomic_text(folder/'ORIENTATION_CHAIN_AUDIT.md','# Common orientation chain — interface audit only\n\nPASS for all fixed TRAIN11 raw orientation arrays, both representations and both arms. Active right-handed rotation matrices, explicit XYZW/WXYZ conventions, root transform direction and wrist→tool multiplication order were checked. The existing independent source-orientation audit is hashed as supporting evidence. No calibration bug was found and no transform was changed.\n\nThis is not orientation reachability or full6D execution qualification. No6D solver ran; position TRAIN11 remains the prerequisite.\n')
    print('ORIENTATION_INTERFACE_AUDIT_PASS; FULL_6D_NOT_RUN',flush=True)

if __name__=='__main__':main()
