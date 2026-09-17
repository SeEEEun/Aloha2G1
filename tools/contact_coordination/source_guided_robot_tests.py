"""Replay fixed geometry witnesses with current G1 hull code, not task physics."""
from pathlib import Path
import numpy as np
from .io import read,record,atomic_json


def run(out):
    from .source_phase import COMMON
    from .planning_kinematics import G1Kinematics
    from .runtime_hulls import Checker
    from .joint_path_planner import Validator
    from .morphology_repair import world_wrist
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    old=Path(read(out/'SOURCE_GUIDED_RRT_REBUILD.json')['old_run']);cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg))
    cal=read(out/'target_repair/CONTACT_CALIBRATION.json');ch=Checker(g,out,cal['joint_names'])
    prior=read(old/'GOLDEN_PLANNING.json')['rows'][0];sid=prior['source_id']
    early=read(Path(prior['context'])/'prototype'/sid/'morphology_acquisition_v4/PLAN_RESULT.json')
    old_pre=np.array(next(p for p in early['phases'] if p['phase']=='PREGRASP')['connecting_q'])[-1]
    a=np.array(next(p for p in early['phases'] if p['phase']=='APPROACH_CLEARANCE')['connecting_q'])[-1]
    b=np.array(next(p for p in early['phases'] if p['phase']=='LEFT_ACQUISITION')['connecting_q'])[-1]
    x=np.array(read(out/'source_phase'/sid/'PHASE_RECORD.json')['initial_object_pose_world']);opened=np.array(cal['contacts']['pregrasp']['commanded_finger_q'])
    def protected(q):return not any(not h['allowed_contact'] for h in ch.query(np.r_[q,opened],x,())+ch.protected_object_contacts(np.r_[q,opened],x))
    guard=Validator(g.arm_limits[:,0],g.arm_limits[:,1],protected)
    assert guard.state(a) and not guard.edge(a,b)
    contact_hits=[h for h in ch.query(np.r_[b,opened],x,()) if 'hybrid_object' in h['geoms'] and not h['allowed_contact']]
    assert contact_hits
    # Reuse the earlier fixed table witness as a fixture, then query every
    # current robot-only and carried-object edge state again.
    witness=read(old/'CARRIED_OBJECT_REGRESSION.json')['witness'];a=np.asarray(witness['q_start']);b=np.asarray(witness['q_goal']);relation=np.asarray(witness['relation'])
    carry=read(out/'target_repair/CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json')['contacts']['right_carry'];finger=opened.copy();finger[7:]=carry['measured_finger_q'][7:]
    def hits(q):g.assign(q);obj=world_wrist(g,'right')@relation;return ch.query(np.r_[q,finger],obj,('right',),object_environment=True)
    robot=lambda q:not any(not h['allowed_contact'] and 'hybrid_object' not in h['geoms'] for h in hits(q))
    loaded=lambda q:not any(not h['allowed_contact'] for h in hits(q))
    rc=Validator(g.arm_limits[:,0],g.arm_limits[:,1],robot,max_checks=30000);lc=Validator(g.arm_limits[:,0],g.arm_limits[:,1],loaded,max_checks=30000)
    assert rc.edge(a,b) and not lc.edge(a,b)
    path=out/'SOURCE_GUIDED_ROBOT_GEOMETRY_TESTS.json';atomic_json(path,dict(status='PASS',
        protected_object=dict(status='PASS',approach_state_valid=True,contact_sweep_rejected=True,
            positive_separation_policy=True,old_pregrasp_near_contacts=ch.protected_object_contacts(np.r_[old_pre,opened],x),
            forbidden_links=contact_hits,checked_states=guard.calls),
        carried_object=dict(status='PASS',robot_only_edge_valid=True,carried_edge_rejected=True,current_checks=[rc.calls,lc.calls],fixture=record(old/'CARRIED_OBJECT_REGRESSION.json')),
        current_geometry=record(Path(__file__).parent/'runtime_hulls.py'),task_success_oracle=False))
    return path
