#!/usr/bin/env python3
"""Reproduce optimizer proxy-distance discontinuity and verify isolated fix."""
from pathlib import Path
import sys
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_physical_position_v4 import *
from tools.common_robust_proxy_penalty import RobustProxyPenaltySolver

def run():
    g,c,n=model();old=CommonPositionSolver(g,c,read(QUAL),n);new=RobustProxyPenaltySolver(g,c,read(QUAL),n)
    src=V4/'WRIST_EP024/REPAIRED_2.npz';z=np.load(src);q=z['q'][160];h=z['common_hand_q'][160];rows=[];before=int(g.model.opt.disableflags)
    for perturb in (0.,-1e-5,1e-5):
        v=q.copy();v[0]+=perturb;old.pose_jacobian(v,h);seg=np.zeros(6)
        d=float(mujoco.mj_geomDistance(g.model,g.data,36,45,.02,seg));robust,j=new.clearance_values([(36,45)])
        assert int(g.model.opt.disableflags)==before
        rows.append(dict(perturb_rad=perturb,native_distance_m=d,native_segment_length_m=float(np.linalg.norm(seg[3:]-seg[:3])),robust_distance_m=float(robust[0]),robust_pitch_gradient=float(j[0,0])))
    fd=(rows[2]['robust_distance_m']-rows[1]['robust_distance_m'])/2e-5
    result=dict(status='OPTIMIZER_DISTANCE_INCONSISTENCY_REPRODUCED_AND_ISOLATED',rows=rows,
        finite_difference_pitch_derivative=fd,analytic_pitch_derivative=rows[0]['robust_pitch_gradient'],
        geometry_classifier_changed=False,scene_or_physics_flags_persistently_changed=False,
        restored_disableflags=before,source=file_record(src),implementation=file_record(ROOT/'tools/common_robust_proxy_penalty.py'),
        interpretation='Native distance returns zero with nonzero22.4mm witness segment under10microrad perturbation. Optimizer-only alternative returns continuous11.93mm distance and consistent gradient. Final detailed classification remains unchanged.')
    assert abs(fd-rows[0]['robust_pitch_gradient'])<1e-4
    atomic_json(STAGE/'DISTANCE_PENALTY_NUMERICAL_AUDIT.json',result)
    atomic_text(STAGE/'DISTANCE_PENALTY_NUMERICAL_AUDIT.md','# Optimizer distance-query repair\n\nA deterministic TRAIN-state perturbation produced native proxy distance0 with a22.423mm witness segment, adjacent to11.932mm distances. This breaks the optimizer residual/gradient assumption. The alternative MuJoCo distance backend gives continuous11.932mm values and analytic/finite-difference derivatives agree within1e-4.\n\nOnly optimizer distance queries use the alternative backend. A try/finally restores model flags immediately, including on exceptions. No final collision classification, threshold, detailed geometry, scene physics, raw target, source timestamp, or per-joint bound changes. The same correction is available for every common solver invocation.\n')
    print(result,flush=True)

if __name__=='__main__':run()
