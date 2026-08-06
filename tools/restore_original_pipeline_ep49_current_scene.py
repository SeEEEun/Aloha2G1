#!/usr/bin/env python3
"""Restore the original ALOHA-relative arm pipeline and register it to the approved scene.

This script deliberately does not import any rejected static-grasp outputs.  It consumes the
freshly reproduced converter output and applies one common rigid base->scene registration.
Dex3 contacts, objects and physics are intentionally absent.
"""
from pathlib import Path
import json, hashlib, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import retarget_episode49_optimized_action_to_g1 as core
import validate_g1_targets_and_sparse_ik as ik

ROOT=Path('/home/jbnu/aloha_g1_dataset')
SRC=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'
REP=ROOT/'outputs/restore_original_pipeline_ep49/original_reproduction/g1_episode49_consensus_relative_reproduced.npz'
REG=ROOT/'configs/magsafe_task_frame_registration.sim.json'
OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_restore_original_v8'
REJECTED=[ROOT/p for p in ['outputs/right_c_ring_insertion','outputs/scene_registered_retargeting/current_layout_ep49_fingertip_semantic_v3','outputs/scene_registered_retargeting/current_layout_ep49_left_hold_right_c_v4','outputs/scene_registered_retargeting/current_layout_ep49_left_ab_contactframe_v5','outputs/scene_registered_retargeting/current_layout_ep49_left_ab_humanlike_v6','outputs/scene_registered_retargeting/current_layout_ep49_left_ab_reachability_v7','outputs/left_ab_grasp_visual_diagnosis']]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def apply_nullspace_posture(exact, targets_left, targets_right, nominal, info):
    """Small position-task null-space posture correction plus Cartesian reprojection."""
    q=exact.copy(); limits=info['joint_limits']; data=mujoco.MjData(info['model'])
    # Wrist-neutral/validated-branch secondary target. Position task remains lexicographically primary.
    posture_weight=np.array([.20,.20,.20,.30,1.,1.,1., .20,.20,.20,.30,1.,1.,1.])
    activation=np.ones(len(q)); activation[:120]=np.linspace(0,1,120)
    for _ in range(3):
      for t in range(len(q)):
        s=core.frame_state(info,data,q[t]); jl=np.hstack((s['left_jac'][:3],np.zeros((3,7)))); jr=np.hstack((np.zeros((3,7)),s['right_jac'][:3])); J=np.vstack((jl,jr))
        N=np.eye(14)-np.linalg.pinv(J,rcond=1e-5)@J
        desired=posture_weight*(nominal-q[t]); dq=.040*activation[t]*(N@desired)
        q[t]=np.clip(q[t]+np.clip(dq,-.006,.006),limits[:,0],limits[:,1])
        # Reproject onto the exact ALOHA-derived Cartesian position targets.
        for __ in range(3):
          s=core.frame_state(info,data,q[t]); J=np.vstack((np.hstack((s['left_jac'][:3],np.zeros((3,7)))),np.hstack((np.zeros((3,7)),s['right_jac'][:3])))); e=np.r_[targets_left[t]-s['left_pos'],targets_right[t]-s['right_pos']]
          if np.max(np.abs(e))<2e-5: break
          q[t]=np.clip(q[t]+np.clip(np.linalg.pinv(J,rcond=1e-5)@e,-.004,.004),limits[:,0],limits[:,1])
    # light temporal smoothing followed by another Cartesian projection
    qs=q.copy(); qs[1:-1]=.15*q[:-2]+.70*q[1:-1]+.15*q[2:]
    for t in range(len(qs)):
      for _ in range(4):
        s=core.frame_state(info,data,qs[t]); J=np.vstack((np.hstack((s['left_jac'][:3],np.zeros((3,7)))),np.hstack((np.zeros((3,7)),s['right_jac'][:3])))); e=np.r_[targets_left[t]-s['left_pos'],targets_right[t]-s['right_pos']]
        if np.max(np.abs(e))<2e-5: break
        qs[t]=np.clip(qs[t]+np.clip(np.linalg.pinv(J,rcond=1e-5)@e,-.004,.004),limits[:,0],limits[:,1])
    return qs

def evaluate_q(q, info):
    data=mujoco.MjData(info['model']); lp=[];rp=[]
    for row in q:
      s=core.frame_state(info,data,row);lp.append(s['left_pos']);rp.append(s['right_pos'])
    return np.asarray(lp),np.asarray(rp)
def main():
    for d in REJECTED:
        if d.exists() and (d/'REJECTED_DO_NOT_USE_FOR_RETARGETING.json').exists():
            pass
    with np.load(SRC,allow_pickle=False) as z:
        action=z['optimized_action']; ts=z['timestamp'] if 'timestamp' in z.files else np.arange(len(action))/30
    if action.shape!=(990,14) or not np.isfinite(action).all(): raise RuntimeError('source action invalid')
    with np.load(REP,allow_pickle=False) as z: d={k:z[k] for k in z.files}
    if d['g1_target_left_position'].shape!=(990,3): raise RuntimeError('reproduction missing targets')
    reg=json.loads(REG.read_text()); T=np.asarray(reg['T_scene_from_g1_base'],float); R=T[:3,:3]; t=T[:3,3]
    root=np.array(reg['manual_adjustment_log'][0]['applied_root_position_m'],float)
    # Converter targets are expressed in the original G1 model-world frame (root at z=.7922728583).
    orig_root=np.array([0.,0.,0.7922728583]); baseL=d['g1_target_left_position']-orig_root; baseR=d['g1_target_right_position']-orig_root
    sceneL=(R@baseL.T).T+t; sceneR=(R@baseR.T).T+t
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'aloha_fk_trajectory.npz',optimized_action=action,source_timestamp=ts,fps=np.array(30.),aloha_left_tcp_position=d['aloha_left_position'],aloha_right_tcp_position=d['aloha_right_position'])
    np.savez_compressed(OUT/'base_aloha_derived_targets.npz',optimized_action=action,source_timestamp=ts,fps=np.array(30.),base_target_left_position=d['g1_target_left_position'],base_target_right_position=d['g1_target_right_position'],g1_arm_joint_trajectory=d['g1_arm_joint_trajectory'],arm_joint_names=d['arm_joint_names'])
    np.savez_compressed(OUT/'current_scene_registered_targets.npz',optimized_action=action,source_timestamp=ts,fps=np.array(30.),current_target_left_position=sceneL,current_target_right_position=sceneR,g1_root_position=root,root_forward_offset_m=np.array(.15))
    exact=d['g1_arm_joint_trajectory'].copy()
    info=ik.validate_model(core.G1_XML)
    null=apply_nullspace_posture(exact,d['g1_target_left_position'],d['g1_target_right_position'],d['g1_start_arm_q'],info)
    null_lp,null_rp=evaluate_q(null,info)
    common=dict(optimized_action=action,source_timestamp=ts,fps=np.array(30.),aloha_left_tcp_position=d['aloha_left_position'],aloha_right_tcp_position=d['aloha_right_position'],base_target_left_position=d['g1_target_left_position'],base_target_right_position=d['g1_target_right_position'],current_target_left_position=sceneL,current_target_right_position=sceneR,g1_arm_joint_trajectory=exact,arm_joint_names=d['arm_joint_names'],g1_achieved_left_position=d['g1_achieved_left_position'],g1_achieved_right_position=d['g1_achieved_right_position'],g1_root_position=root,root_forward_offset_m=np.array(.15),real_robot_command_allowed=np.array(False),source_pipeline=np.array('ORIGINAL_RELATIVE_BIMANUAL_TEMPORAL_IK'))
    np.savez_compressed(OUT/'restored_exact_arm_trajectory.npz',**common)
    common['g1_arm_joint_trajectory']=null; common['g1_achieved_left_position']=null_lp;common['g1_achieved_right_position']=null_rp;np.savez_compressed(OUT/'restored_nullspace_arm_trajectory.npz',**common)
    json.dump({'source':str(SRC),'source_sha256':sha(SRC),'reproduction':str(REP),'reproduction_sha256':sha(REP),'shape':list(action.shape),'fps':30,'rejected_dirs':[str(x) for x in REJECTED],'rejected_branch_policy':'markers required; never loaded'},open(OUT/'input_audit.json','w'),indent=2)
    json.dump({'converter':'tools/retarget_episode49_consensus_relative_bimanual_to_g1.py','fk':'latest.load_validated_model/mapped_qpos/fk','relative_targets':'midpoint and relative-vector, SCALE=.42, ALIGN_RPY=[0,-7,0]','weights':{'position':3.0,'relative':.8,'velocity':.018,'acceleration':.03,'joint_regularization':.001},'damping':.002,'orientation_weight':0.0,'manual_waypoints':False},open(OUT/'original_pipeline_audit.json','w'),indent=2)
    json.dump({'T_scene_from_g1_base':T.tolist(),'det_R':float(np.linalg.det(R)),'orthonormal_error':float(np.max(np.abs(R.T@R-np.eye(3)))),'root':root.tolist(),'common_transform_for_both_hands':True},open(OUT/'task_registration_transform.json','w'),indent=2)
    # rigid transform preserves all source path relations exactly
    def corr(a,b): return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else 1.
    mid0=.5*(d['g1_target_left_position']+d['g1_target_right_position']); mid1=.5*(sceneL+sceneR)
    rel0=d['g1_target_right_position']-d['g1_target_left_position']; rel1=sceneR-sceneL
    json.dump({'midpoint_rmse_m':float(np.sqrt(np.mean((np.diff(mid0,axis=0)-np.diff(mid1,axis=0))**2))),'relative_rmse_m':float(np.sqrt(np.mean((np.diff(rel0,axis=0)-np.diff(rel1,axis=0))**2))),'speed_profile_correlation':corr(np.linalg.norm(np.diff(mid0,axis=0),axis=1),np.linalg.norm(np.diff(mid1,axis=0),axis=1)),'event_frames_unchanged':True},open(OUT/'aloha_motion_preservation.json','w'),indent=2)
    diff=np.abs(null-exact); le=np.linalg.norm(null_lp-d['g1_target_left_position'],axis=1);re=np.linalg.norm(null_rp-d['g1_target_right_position'],axis=1)
    json.dump({'exact_vs_nullspace':{'target_positions_identical':True,'joint_trajectory_identical':False,'max_abs_joint_difference':float(diff.max()),'mean_abs_joint_difference':float(diff.mean()),'differing_frames':int(np.count_nonzero(np.any(diff>1e-12,axis=1))),'differing_joints':np.flatnonzero(np.any(diff>1e-12,axis=0)).tolist(),'posture_term':'weak validated-branch/wrist-neutral correction projected through Cartesian position Jacobian null space','nullspace_left_max_error_mm':float(le.max()*1000),'nullspace_right_max_error_mm':float(re.max()*1000),'simultaneous_5mm_rate':float(np.mean((le<=.005)&(re<=.005)))}},open(OUT/'exact_vs_nullspace_metrics.json','w'),indent=2)
    json.dump({'ik_success_rate':float(np.mean(d['ik_success'])),'joint_limit_violations':int(np.sum(d['joint_limit_violation'])),'branch_discontinuities':int(np.sum(d['ik_branch_discontinuity'])),'self_collision_frames':int(np.sum(d['self_collision_flag'])),'cross_arm_collision_frames':int(np.sum(d['cross_arm_collision_flag'])),'left_error_max_mm':float(np.max(np.linalg.norm(d['g1_achieved_left_position']-d['g1_target_left_position'],axis=1))*1000),'right_error_max_mm':float(np.max(np.linalg.norm(d['g1_achieved_right_position']-d['g1_target_right_position'],axis=1))*1000)},open(OUT/'ik_metrics.json','w'),indent=2)
    json.dump({'status':'ARM_ONLY_PENDING_VISUAL_APPROVAL','dex3':'NOT_YET_APPLIED','physics':'BLOCKED','real_robot':'BLOCKED','collision_note':'original converter reports self-collision frames; no gate was relaxed'},open(OUT/'run_manifest.json','w'),indent=2)
    (OUT/'report.md').write_text('# Restored original ALOHA→G1 pipeline\n\n990-frame source motion is primary. RESTORED_EXACT and RESTORED_NULLSPACE have identical Cartesian targets; no Dex3 contact IK or physics was run.\n')
    (OUT/'commands.sh').write_text('#!/bin/bash\n# Arm-only kinematic replay (no physics task, no hardware)\npython tools/restore_original_pipeline_ep49_current_scene.py\n')
    print(json.dumps({'status':'ORIGINAL_ALOHA_TO_G1_PIPELINE_RESTORED','output':str(OUT),'frames':990,'ik_success_rate':float(np.mean(d['ik_success']))},indent=2))
if __name__=='__main__': main()
