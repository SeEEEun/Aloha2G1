"""Short TRAIN diagnostic for a rejected empty-hand bin retreat.

Reuse the source-conditioned candidate up to placement, repair only its failed
connection, then revalidate every exported command. This is not a frozen-study
cache or an episode-specific rescue rule.
"""
from pathlib import Path
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_npz,fingerprint


def repair(context,previous,loaded_geometry):
    from .source_phase import COMMON
    from .morphology_repair import world_wrist
    from .planner import realize_phase_goals
    from .loaded_contact_geometry import object_pose
    from .runtime_hulls import Checker
    from .phase_clock_runtime import receiver_release_target
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from tools.common_execution_layer import Dex3Primitive,read_json
    from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
    result=read(previous/'RESULT.json');bank_record=result['contact_bank']
    assert record(bank_record['path'])==bank_record
    bank=read(bank_record['path']);cal=read(context/'target_repair/CONTACT_CALIBRATION.json')
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg));checker=Checker(g,context,cal['joint_names'])
    primitive=Dex3Primitive.from_frozen_dependencies(read_json(PHYSICS_CONFIG),read_json(PHYSICAL_ENVIRONMENT),read_json(COMMON_PHYSICAL_CONTROLLER))
    opened=np.asarray(cal['contacts']['pregrasp']['commanded_finger_q'])
    key=fingerprint([__file__,previous/'RESULT.json',loaded_geometry,ROOT/'tools/contact_coordination/planner.py',ROOT/'tools/contact_coordination/physical_attempt.py'])[0]
    folder=previous.parent/(key[:12]+'_cartesian_retreat')
    if folder.exists():raise FileExistsError(folder)
    rows=[]
    for row in result['results']:
        sub=previous/row['subdirectory'];ik=read(sub/'PHASE_IK.json')
        if len(ik['phases'])!=6 or not all(p['admissible'] for p in ik['phases'][:5]):continue
        candidate=next(c for c in bank if c['contact_candidate']==row['contact_candidate'] and c['seed']==row['candidate_seed'])
        model=dict(read(loaded_geometry),measured_T_HO=candidate['carry_contacts']['right']['measured_T_HO'])
        q=np.load(sub/'PHASE_Q.npz')['q'];goals=read(sub/'GOALS.json');goal=dict(goals[-1],cartesian_connection_steps=6)
        goal['object_pose']=object_pose(g,q[5],model)
        held=np.asarray(candidate['carry_contacts']['right']['measured_finger_q'])[7:]
        delayed=goal.get('final_release_delayed_digits',[])
        partial=receiver_release_target(held,opened[7:],primitive.release_frames,0,primitive.release_frames,delayed,10*primitive.release_frames)
        def validate(a,b,item):
            count=max(2,int(np.ceil(np.max(np.abs(b-a))/.02))+1);bad=[]
            for i,u in enumerate(np.linspace(0,1,count)):
                for hit in checker.check(np.r_[a+u*(b-a),opened[:7],partial],np.asarray(item['object_pose']),('right',)):
                    if not hit['allowed_contact']:bad.append(dict(sample=i,**hit))
            if item['name']=='POST_RELEASE_RETREAT':
                for step in range(primitive.release_frames):
                    fingers=receiver_release_target(held,opened[7:],step+primitive.release_frames,0,primitive.release_frames,delayed,primitive.release_frames)
                    for hit in checker.check(np.r_[b,opened[:7],fingers],np.asarray(item['object_pose']),('right',)):
                        if not hit['allowed_contact']:bad.append(dict(release_sample=step,**hit))
            return dict(valid=not bad,forbidden_count=len(bad),forbidden_contacts=bad[:15],samples=count)
        settings=dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,max_nfev_per_seed_per_goal=120)
        solved=realize_phase_goals(g,[goal],q[5],settings,validate,[candidate['q']]);lastq=solved.pop('q')
        updated=dict(ik,phases=ik['phases'][:5]+solved['phases'],failed_original_retreat=ik['phases'][5],short_retreat_solver=solved)
        goals[-1]=goal;q=np.vstack([q[:6],lastq[-1]])
        target=folder/row['subdirectory'];atomic_json(target/'PHASE_IK.json',updated);atomic_json(target/'GOALS.json',goals);atomic_npz(target/'PHASE_Q.npz',q=q)
        good=solved['phases'][0]['admissible'];rows.append(dict(row,all_admissible=good))
        if good:atomic_json(folder/'SELECTED_CONTACTS.json',candidate)
    atomic_json(folder/'RESULT.json',dict(result,status='FULL_PATH_BUILT' if any(r['all_admissible'] for r in rows) else 'NO_COMPLETE_PATH_WITHIN_BUDGET',results=rows,
        TRAIN_development_only=True,reused_candidate=record(previous/'RESULT.json'),implementation=record(__file__),
        complete_post_retime_revalidation_required=True))
    return folder


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--loaded-geometry',type=Path,required=True)
    a=p.parse_args();print(repair(a.context,a.previous,a.loaded_geometry))
