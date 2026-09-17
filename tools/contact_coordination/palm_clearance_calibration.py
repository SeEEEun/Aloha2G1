"""Small contact-offset bank from a calibrated palm/object separation normal."""
from pathlib import Path
import numpy as np
from .io import ROOT,read,record,atomic_json
from .source_phase import COMMON
from .morphology_repair import world_wrist


def build(out,plan):
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from .runtime_hulls import Checker
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg))
    calpath=out/'target_repair/CONTACT_CALIBRATION_COMMAND_PHASES.json';cal=read(calpath);c=cal['contacts']['right_carry_command_intent']
    selected=read(plan.parent/'SELECTED_CONTACTS.json');a=np.load(plan/'COMMANDS.npz');q=a['commanded_q_rad'][np.flatnonzero(a['stage']=='DUAL_SUPPORT')[0],:14]
    g.assign(q);x=world_wrist(g,'right')@np.asarray(c['T_wrist_H'])@np.asarray(c['T_HO'])
    f=np.r_[cal['contacts']['pregrasp']['commanded_finger_q'][:7],c['measured_finger_q'][7:]];ch=Checker(g,out,cal['joint_names']);ch.check(np.r_[q,f],x,('left','right'))
    contacts=[]
    for contact in ch.data.contact[:ch.data.ncon]:
        names=[ch.mj.mj_id2name(ch.model,ch.mj.mjtObj.mjOBJ_GEOM,int(v)) for v in [contact.geom1,contact.geom2]]
        if set(names)!={'collider_30','hybrid_object'} or contact.dist>=0:continue
        direction=g.root_pose[:3,:3]@contact.frame[:3]
        palm_id=ch.mj.mj_name2id(ch.model,ch.mj.mjtObj.mjOBJ_GEOM,'collider_30')
        palm=g.model_to_world_position(ch.data.geom_xpos[palm_id])
        if direction@(x[:3,3]-palm)<0:direction=-direction
        contacts.append((float(-contact.dist),direction))
    depth,direction=max(contacts,key=lambda r:r[0]);offset=read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')['object']['contact_offset_m']
    incoming=np.asarray(selected['objects']['left']);delta=incoming[:3,:3].T@direction*(depth+offset)
    result=dict(offset_object_frame_m=delta,maximum_offset_m=float(np.linalg.norm(delta)),fractions=[0.,.5,1.],
        penetration_m=depth,unchanged_contact_offset_m=offset,calibration=record(calpath),reference_plan=record(plan/'COMMANDS.npz'),
        rule='Translate receiver goal along the calibrated giver-palm/object separating normal by0,half,or full(overlap+existing contact offset). Preserve orientation and source contact axes. Geometry-only candidate, not a certified physical grasp.',
        coordinate_frame='incoming object',world_target_copied=False,runtime_geometry_or_threshold_changed=False,implementation=record(__file__))
    path=out/'target_repair/CONTACT_SEPARATION_CANDIDATES.json'
    if path.exists():raise FileExistsError(path)
    atomic_json(path,result);print(result['maximum_offset_m']);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();build(a.run_dir.resolve(),a.plan.resolve())
