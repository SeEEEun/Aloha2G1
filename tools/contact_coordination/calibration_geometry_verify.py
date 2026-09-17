"""Full-state measured geometry audit using explicitly cooked offline assets."""
from pathlib import Path
import shutil,time
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT,read,record,atomic_json
from .source_phase import COMMON,pose


def run(out):
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from .runtime_hulls import Checker
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id']
    prior=ROOT/'outputs/contact_coordination_hybrid/20260907T083735Z/prototype'/sid/'full_task_connection/ead64a3c8365/geometry_telemetry_diagnostic'
    sources=[('left_control',out/'common_control/left_full_state'),('right_control',out/'common_control/right_full_state'),
        ('source_acquisition',out/'prototype'/sid/'morphology_acquisition_v4/physics_attempt_01'),('prior_handoff_window',prior)]
    results=[]
    for label,folder in sources:
        a=dict(np.load(folder/'event_log.npz'));work=out/'geometry_verification'/label
        dest=work/'target_repair/runtime_bin150'
        if not dest.exists():shutil.copytree(out/'target_repair/runtime_bin150',dest)
        # Fresh source model per checker: mj_saveLastXML is process-global.
        c=load_common_config(COMMON);g=G1Kinematics(c,load_scene(c));ch=Checker(g,work,list(a['joint_names']))
        ids=np.arange(len(a['control_frame']))
        if label=='prior_handoff_window':
            beginning=int(np.min(a['control_frame'][a['stage']=='RECEIVER_APPROACH']))
            ids=ids[(a['control_frame'][ids]>=beginning)&(a['control_frame'][ids]<=1016)]
        saved=out/'geometry_verification'/f'{label}.json'
        if saved.exists():
            old=read(saved)
            if old['physics_rows_checked']==len(ids) and old['trace']==record(folder/'event_log.npz') and old['geometry']==record(out/'target_repair/runtime_bin150/RUNTIME_HULLS.json'):
                results.append(old);continue
        forbidden=[];contacts=[];start=time.monotonic()
        for i in ids:
            x=pose(Rotation.from_quat(a['object_quaternion_xyzw'][i]).as_matrix(),a['object_position_world_m'][i])
            hits=ch.check(a['MEASURED_Q'][i],x,('left','right'),dict(zip(a['all_joint_names'],a['all_measured_q_rad'][i],strict=True)))
            for h in hits:
                if h['allowed_contact']:continue
                row=dict(row=int(i),frame=int(a['control_frame'][i]),stage=str(a['stage'][i]),**h)
                robot_pair=all(not b.startswith('/') and b!='object' for b in h['bodies'])
                (forbidden if robot_pair or h['depth_m']>.003 else contacts).append(row)
        result=dict(label=label,trace=record(folder/'event_log.npz'),physics_rows_checked=len(ids),
            forbidden_count=len(forbidden),forbidden=forbidden[:100],noninvalidating_contacts=contacts[:100],
            maximum_noninvalidating_contact_depth_m=max([h['depth_m'] for h in contacts],default=0.),
            unchanged_measured_environment_penetration_limit_m=.003,full_named_articulation=True,
            geometry=record(out/'target_repair/runtime_bin150/RUNTIME_HULLS.json'),
            runtime_seconds=time.monotonic()-start,physical_state_or_collider_changes=False,
            limits='Discrete measured240Hz samples; same-hand legacy anatomical exclusions retained. Cooked replicas verified against actual contact points; not a direct native PxShape export.')
        atomic_json(out/'geometry_verification'/f'{label}.json',result);results.append(result)
        print(label,len(ids),len(forbidden),result['maximum_noninvalidating_contact_depth_m'],flush=True)
    atomic_json(out/'geometry_verification/RESULT.json',results)
    return results


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);run(p.parse_args().run_dir)
