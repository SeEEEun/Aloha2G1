"""Read saved Golden/TRAIN5 failures and run bounded endpoint-only IK diagnostics."""
from pathlib import Path
import csv, json, time
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT, read, record, atomic_json, atomic_text
from .source_phase import COMMON, mean_pose
from .prototype import INITIAL
from .morphology_repair import wrist_target, world_wrist


def audit(out, parent=None, instances=None, output_dir=None):
    from tools.doll_handoff_retargeting.common import load_common_config, load_scene
    from .planning_kinematics import G1Kinematics
    import mujoco
    cfg=load_common_config(COMMON);scene=load_scene(cfg);g=G1Kinematics(cfg,scene)
    cal=read(out/'target_repair/CONTACT_CALIBRATION.json')
    natural=np.asarray(read(INITIAL)['g1_14_arm_initial_q_rad']);g.assign(natural)
    shoulders={s:g.model_to_world_position(g.data.xpos[mujoco.mj_name2id(g.model,mujoco.mjtObj.mjOBJ_BODY,s+'_shoulder_pitch_link')]) for s in ('left','right')}
    selection=read(out/'bootstrap/SELECTION.json');ids=selection['train_source_ids']
    if instances is None:
        golden=read(parent/'golden_batch/closing_admission_v3/PLAN_RESULT.json')
        instances=[golden,*read(parent/'TRAIN_pilot/PILOT_LEDGER.json')['rows']]
    destination=Path(output_dir) if output_dir is not None else out
    scenes=[];rows=[];details=[]
    for sid in ids:
        p=read(out/'source_phase'/sid/'PHASE_RECORD.json');x=np.asarray(p['initial_object_pose_world']);b=np.eye(4);b[:3,3]=[*p['placement']['bin_center_xy_m'],.801]
        prior=dict(np.load(out/'source_phase'/sid/'SOURCE_PRIORS.npz'))
        grasp=wrist_target(x,cal['contacts']['acquisition_intent'])
        item=dict(source_id=sid,object_pose=x,object_quaternion_xyzw=Rotation.from_matrix(x[:3,:3]).as_quat(),object_yaw_deg=float(Rotation.from_matrix(x[:3,:3]).as_euler('xyz',degrees=True)[2]),bin_pose=b,object_to_bin=np.linalg.inv(x)@b,
            task_centroid=(x[:3,3]+b[:3,3])/2,base=np.asarray(scene['g1']['root_position_world_xyz_m']),
            base_to_object_m=float(np.linalg.norm(x[:3,3]-scene['g1']['root_position_world_xyz_m'])),
            left_shoulder_to_grasp_m=float(np.linalg.norm(grasp[:3,3]-shoulders['left'])),shoulders=shoulders,
            source_relations=p['source_functional_tool_object_relations'],registered_relations=p['registered_wrist_object_relations'],
            handoff_prior=mean_pose(prior['inferred_object_from_left'][p['handoff_sample_indices']]),placement=p['placement'],
            uncertainty=p['uncertainty'],registration_source=record(out/'source_phase'/sid/'PHASE_RECORD.json'))
        scenes.append(item)
        print('SCENE',sid,'xyz',np.round(x[:3,3],4),'yaw',round(item['object_yaw_deg'],2),'shoulder',round(item['left_shoulder_to_grasp_m'],4),flush=True)
    for inst in instances:
        sid=inst['source_id'];context=Path(inst['context']);base=context/'prototype'/sid/'morphology_acquisition_v4'
        groups=[(base,read(base/'GOALS.json')['goals'],read(base/'PLAN_RESULT.json'))]
        selected_chain=None
        if 'connection' in inst:
            connection_result=read(Path(inst['connection'])/'RESULT.json')
            selected_chain=next((r.get('subdirectory') for r in connection_result.get('results',[]) if r.get('all_admissible')),None)
            for folder in sorted(Path(inst['connection']).glob('contact_*')):
                if (folder/'PHASE_IK.json').exists():groups.append((folder,read(folder/'GOALS.json'),read(folder/'PHASE_IK.json')))
        for folder,goals,solution in groups:
            phases=solution['phases']
            for index,goal in enumerate(goals):
                phase=phases[index] if index<len(phases) else None
                candidates=[] if phase is None else phase['candidates'];bad=[h for c in candidates for h in c.get('connection_validation',{}).get('forbidden_contacts',[])]
                posegood=[c for c in candidates if c.get('pose_prior_within_tolerance',c.get('goal_satisfied',False))]
                row=dict(source_id=sid,condition=inst['condition'],phase=goal['name'],target_region=json.dumps(goal['wrist_pose_world']),generated_pose_goals=1,
                    candidate_chain=folder.name,instance_complete_plan=bool(inst.get('full_task_plan')),
                    selected_chain=folder==base or folder.name==selected_chain,
                    generated_joint_candidates=len(candidates),passing_full_SE3=len(posegood),passing_joint_limits=sum(bool(np.all(np.asarray(c['q'])>=g.arm_limits[:,0]) and np.all(np.asarray(c['q'])<=g.arm_limits[:,1])) for c in candidates),
                    passing_combined_geometry_edge=sum(c.get('admissible',False) for c in candidates),endpoint_geometry_count='NOT_SEPARATELY_LOGGED',
                    best_position_m=min((max(e['position_m'] for e in c['errors'].values()) for c in candidates),default=None),
                    best_orientation_rad=min((max(e['orientation_rad'] for e in c['errors'].values()) for c in candidates),default=None),
                    status='NOT_ATTEMPTED_UPSTREAM' if phase is None else ('PASS' if phase['admissible'] else 'REJECTED'),
                    first_rejection='NOT_ATTEMPTED_UPSTREAM' if phase is None else ('' if phase['admissible'] else json.dumps(bad[0]) if bad else 'BOUNDED_POSE_OR_REGION_FAILURE'),
                    evidence=str(folder))
                rows.append(row)
                details.append(dict(source_id=sid,condition=inst['condition'],phase=goal,solver=phase,evidence=str(folder)))
            # A failed free-preparation retry is separate evidence, not an unobserved phase.
        print('INSTANCE',sid,inst['condition'],'COMPLETE_VALID_PLAN' if inst.get('full_task_plan') else
            next(((r['phase'],r['first_rejection'][:150]) for r in rows if r['source_id']==sid and r['condition']==inst['condition'] and r['status']=='REJECTED'),inst.get('first_failure')),flush=True)
    atomic_json(destination/'diagnosis/SCENES_AND_RELATIONS.json',scenes)
    atomic_json(destination/'diagnosis/PHASE_FAILURE_DETAILS.json',details)
    with (destination/'GOLDEN_VS_TRAIN5_FIRST_FAILURE_MATRIX.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return scenes,details


def oracle(out, parent):
    from tools.doll_handoff_retargeting.common import load_common_config, load_scene
    from .planning_kinematics import G1Kinematics
    from .planner import realize_phase_goals
    from .runtime_hulls import Checker
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg))
    cal=read(out/'target_repair/CONTACT_CALIBRATION.json');checker=Checker(g,out,cal['joint_names'])
    q0=np.asarray(read(INITIAL)['g1_14_arm_initial_q_rad']);opened=np.asarray(cal['contacts']['pregrasp']['commanded_finger_q'])
    # Fixed robot-only seeds; no saved source trajectory or selected source solution.
    seeds=[np.asarray(cal['contacts'][k]['seed_q']) for k in ('acquisition_intent','handoff_left','handoff_right')]
    lo,hi=g.arm_limits.T
    for sign in (-1,1):
        seed=q0.copy();seed[[2,9]]+=sign*.6;seeds.append(np.clip(seed,lo+1e-6,hi-1e-6))
    config=dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,max_nfev_per_seed_per_goal=100,enable_collision_refinement=False)
    details=read(out/'diagnosis/PHASE_FAILURE_DETAILS.json');results=[]
    for d in details:
        if not d['solver'] or d['solver']['admissible']:continue
        if d['phase']['name'] not in ('APPROACH_CLEARANCE','PREGRASP','LEFT_ACQUISITION','LIFT'):continue
        sid=d['source_id'];x=np.asarray(read(out/'source_phase'/sid/'PHASE_RECORD.json')['initial_object_pose_world'])
        goal=d['phase'];hard=dict(goal,hard_pose_constraint=True)
        def endpoint(a,b,goal):
            hits=checker.query(np.r_[b,opened],x,() if goal['name']=='APPROACH_CLEARANCE' else ('left',))
            bad=[h for h in hits if not h['allowed_contact']]
            return dict(valid=not bad,forbidden_contacts=bad,forbidden_count=len(bad),endpoint_only=True)
        start=time.monotonic();result=realize_phase_goals(g,[hard],q0,config,endpoint,seeds);result.pop('q')
        valid=[c for c in result['phases'][0]['candidates'] if c['admissible']]
        row=dict(source_id=sid,condition=d['condition'],phase=goal['name'],goal=hard,result=result,
            status='REGION_HAS_VALID_G1_CONFIGURATION' if valid else 'REGION_HAS_NO_VALID_CONFIGURATION_WITHIN_BOUNDED_SEARCH',
            valid_configurations=len(valid),runtime_s=time.monotonic()-start,
            scope='Endpoint-only exact saved pose diagnostic. Does not certify connection, approximate Wrist prior, physical acquisition, or global infeasibility.')
        results.append(row);atomic_json(out/'diagnosis/REGION_ORACLE.json',results)
        print('ORACLE',sid,d['condition'],goal['name'],len(valid),round(row['runtime_s'],3),flush=True)
    return results


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--oracle',action='store_true');a=p.parse_args()
    out=a.run_dir.resolve();parent=Path(read(out/'PARENT_RUN.json')['path'])
    audit(out,parent)
    if a.oracle:oracle(out,parent)
