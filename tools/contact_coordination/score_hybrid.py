"""Measured stage scoring over the existing independent physics/integrity audit.

Legacy scripted stage names are retained as a diagnostic only. This task does
not require an additional upward right-arm lift after a high handoff.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT,read,record,atomic_json


def score(folder,out,geometry=True):
    from tools.finalize_common_dex3_grasp_qualification import full_task,CONTACT_THRESHOLD_N,TABLE_THRESHOLD_N,BIN_CENTER_XY_M,BIN_OPENING_XY_M,BIN_BOTTOM_Z_M,BIN_RIM_Z_M
    from tools.score_episode_registered_physical_eval35_run import longest_rows
    legacy=full_task(folder);atomic_json(folder/'LEGACY_CONTROL_SCORER.json',legacy)
    a=dict(np.load(folder/'event_log.npz'));su=read(folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json');events=su['events'];dt=1/240.
    forces={s:np.column_stack([a[f'{s}_{d}_force_n'] for d in ('thumb','index','middle')]) for s in ('left','right')}
    opposing={s:(v[:,0]>=CONTACT_THRESHOLD_N)&np.any(v[:,1:]>=CONTACT_THRESHOLD_N,axis=1) for s,v in forces.items()}
    support={s:np.maximum(np.max(v,axis=1),a[f'{s}_palm_force_n'])>=CONTACT_THRESHOLD_N for s,v in forces.items()}
    tablefree=a['table_contact_force_n']<=TABLE_THRESHOLD_N;pos=a['object_position_world_m'];z=pos[:,2];frames=a['control_frame'];initialz=float(np.median(z[frames==0]));duration=lambda m:float(longest_rows(m)*dt)
    left_confirm=events['grasp_confirmed_frames']['left'];lift_rows=np.flatnonzero(np.isin(a['stage'],['LIFT_5CM','LEFT_LIFT_5CM']))
    # An early table-contact bump is not the start of commanded lift. Nominal
    # phase timing defines only this observation window; measured contact and
    # confirmed enclosure, rather than the label, establish acquisition.
    cutoff=int(lift_rows[0]) if len(lift_rows) else len(z);pre=np.arange(len(z))<cutoff
    grasp=left_confirm is not None and duration(opposing['left']&pre)>=.1 and (not len(lift_rows) or left_confirm<int(frames[cutoff]))
    lift_duration=duration(opposing['left']&tablefree&(z>=initialz+.05));lift=grasp and lift_duration>=1.-1e-9
    dual_duration=duration(opposing['left']&opposing['right']&tablefree)
    right_mask=opposing['right']&~support['left']&tablefree;right_duration=duration(right_mask);right_owned=right_duration>=1.-1e-9
    handoff=lift and dual_duration>=.1-1e-9 and right_owned
    right_indices=np.flatnonzero(right_mask);transport_progress=0.
    if len(right_indices):
        first=int(right_indices[0]);distance=np.linalg.norm(pos[:, :2]-BIN_CENTER_XY_M,axis=1);after=right_mask&(np.arange(len(z))>=first)
        transport_progress=float(distance[first]-np.min(distance[after]))
    transport=handoff and transport_progress>=.05
    inside_xy=np.all(np.abs(pos[:,:2]-BIN_CENTER_XY_M)<=BIN_OPENING_XY_M/2,axis=1);inside=inside_xy&(z>BIN_BOTTOM_Z_M)&(z<BIN_RIM_Z_M);entry=bool(np.any(inside))
    speed=np.linalg.norm(a['object_linear_velocity_m_s'],axis=1);settle=bool(len(z)>=240 and np.all(inside[-240:]) and np.all(speed[-240:]<=.02) and np.max(a['doll_bin_contact_force_n'][-240:])>0)
    release=legacy['release_classification'] if right_owned else 'NO_RIGHT_OWNERSHIP' if grasp else 'NO_ACQUISITION'
    if right_owned and legacy.get('first_right_contact_loss_frame') is None:
        # A diagnostic that ends in retained support has not dropped the
        # object. The legacy full-script scorer defaults missing release to
        # premature drop; keep its raw output, but do not repeat that label.
        release='NOT_RELEASED'
    spec=read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json')['joint_specs'];q=a['MEASURED_Q'];raw_excess=np.maximum(np.maximum(np.asarray([r['minimum'] for r in spec])-q,q-np.asarray([r['maximum'] for r in spec])),0)
    forbidden=[]
    if geometry:
        from .runtime_hulls import Checker
        from .source_phase import COMMON,pose
        from tools.doll_handoff_retargeting.common import load_common_config,load_scene
        from tools.doll_handoff_retargeting.models import G1Kinematics
        c=load_common_config(COMMON);g=G1Kinematics(c,load_scene(c));checker=Checker(g,out,list(a['joint_names']))
        for i in range(len(q)):
            extra=dict(zip(a['all_joint_names'],a['all_measured_q_rad'][i],strict=True)) if 'all_measured_q_rad' in a else None
            x=pose(Rotation.from_quat(a['object_quaternion_xyzw'][i]).as_matrix(),pos[i]);hits=checker.check(q[i],x,('left','right'),all_joint_state=extra)
            # Runtime bin/table contacts use the frozen 3 mm penetration audit.
            # Unwanted inter-hand/arm intersections have no disabled-response exemption.
            for h in hits:
                if h['allowed_contact']:continue
                robot_pair=all(not b.startswith('/') and b!='object' for b in h['bodies'])
                if robot_pair or h['depth_m']>.003:forbidden.append(dict(row=i,frame=int(frames[i]),**h))
        atomic_json(folder/'MEASURED_GEOMETRY_AUDIT.json',dict(physics_rows_checked=len(q),forbidden_count=len(forbidden),first_forbidden=forbidden[:100],maximum_depth_m=max([r['depth_m'] for r in forbidden],default=0.),geometry=record(out/'target_repair/runtime_bin150/RUNTIME_HULLS.json')))
    validity=legacy['physical_validity'] and not forbidden
    stages=dict(GRASP=bool(grasp),LIFT=bool(lift),HANDOFF=bool(handoff),RIGHT_OWNERSHIP=bool(right_owned),TRANSPORT=bool(transport),BIN_ENTRY=entry,BIN_SETTLE=settle)
    success=validity and all(stages.values()) and release in {'CLEAN_COMMANDED_RELEASE','PREMATURE_DROP_INTO_BIN'}
    stages['FULL_TASK']=bool(success)
    terminal='VALID_PLAN_TASK_SUCCESS' if success else 'EXECUTION_ABORT_PHYSICAL_VALIDITY' if not validity else 'VALID_PLAN_TASK_FAILURE'
    result=dict(terminal=terminal,stages=stages,physical_validity=bool(validity),trace=record(folder/'event_log.npz'),source_conditioned_full_task='DEMONSTRATED' if success else 'NOT_DEMONSTRATED',
        release_classification=release,clean_release=release=='CLEAN_COMMANDED_RELEASE',left_opposing_elevated_s=lift_duration,dual_opposing_s=dual_duration,right_only_opposing_s=right_duration,right_transport_progress_toward_bin_m=transport_progress,
        first_failed_stage=next((k for k,v in stages.items() if not v),None),maximum_raw_joint_limit_excess_rad=float(raw_excess.max()),raw_excursion_samples=int(np.count_nonzero(raw_excess)),states_clipped=False,
        legacy_control_order_valid=legacy.get('event_order_valid'),legacy_script_order_not_applicable='No mandatory second upward right lift or legacy control stage-name equality in natural source task. Physical stage definitions require measured retention/motion.',
        geometry_checked=geometry,geometry_forbidden_count=len(forbidden),full_articulation_measured='all_measured_q_rad' in a,
        geometry_certification='UNRESOLVED_PHYSX_COOKING_AND_SAME_HAND_FILTER_SCOPE' if 'all_measured_q_rad' in a else 'UNRESOLVED_28D_TRACE_OMITS_LOADED_WAIST',
        numerical_validity_is_full_geometry_certificate=False,scorer=record(__file__),scoring_contract=dict(contact_N=CONTACT_THRESHOLD_N,table_N=TABLE_THRESHOLD_N,left_lift_m=.05,retention_s=1.,dual_s=.1,transport_progress_m=.05,settle_s=1.,settle_speed_m_s=.02))
    atomic_json(folder/'HYBRID_SCORE.json',result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--attempt-dir',type=Path,required=True);a=p.parse_args();print(score(a.attempt_dir,a.run_dir))
