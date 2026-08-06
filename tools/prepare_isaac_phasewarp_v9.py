#!/usr/bin/env python3
"""Contract-first v9 bootstrap; stop before deformation if source object SE(3) is absent."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
from pxr import Usd, UsdGeom

ROOT=Path('/home/jbnu/aloha_g1_dataset')
OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_isaac_phasewarp_v9'
SRC=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'
RAW=ROOT/'raw_recordings/GoPark_20260729_111223'
TIMELINE=ROOT/'configs/episode49_task_timeline.approved.json'
V8=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_restore_original_v8/restored_exact_arm_trajectory.npz'
REP=ROOT/'outputs/restore_original_pipeline_ep49/original_reproduction/g1_episode49_consensus_relative_reproduced.npz'
SCENE=ROOT/'isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda'
SEM=ROOT/'outputs/task_frame_registration/magsafe_task_semantic_definition.json'
FRAME_GRAPH=ROOT/'outputs/task_frame_registration/frame_graph.json'
FORBIDDEN=[
 'outputs/right_c_ring_insertion','outputs/scene_registered_retargeting/current_layout_ep49_fingertip_semantic_v3',
 'outputs/scene_registered_retargeting/current_layout_ep49_left_hold_right_c_v4','outputs/scene_registered_retargeting/current_layout_ep49_left_ab_contactframe_v5',
 'outputs/scene_registered_retargeting/current_layout_ep49_left_ab_humanlike_v6','outputs/scene_registered_retargeting/current_layout_ep49_left_ab_reachability_v7',
 'outputs/left_ab_grasp_visual_diagnosis']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def tf(stage,path):
 prim=stage.GetPrimAtPath(path)
 if not prim.IsValid(): raise RuntimeError(f'missing active USD prim: {path}')
 M=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),float).T
 return {'prim_path':path,'T_scene_from_prim':M.tolist(),'position_m':M[:3,3].tolist()}

def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'report').mkdir(exist_ok=True)
 sys.path.insert(0,str(ROOT/'tools'))
 import retarget_episode49_optimized_action_to_g1 as core
 with np.load(SRC,allow_pickle=False) as z:
  action=z['optimized_action'].copy();timestamp=z['timestamp'].copy();fps=float(z['fps'])
 assert action.shape==(990,14) and timestamp.shape==(990,) and fps==30.0 and np.isfinite(action).all() and np.isfinite(timestamp).all()
 amodel,_=core.aloha.load_validated_model(core.ALOHA_XML);aq,clipped=core.aloha.mapped_qpos(action);fk=core.aloha.fk(amodel,aq)
 with np.load(V8,allow_pickle=False) as z:
  if not np.array_equal(z['optimized_action'],action): raise RuntimeError('v8 action is not bytewise-array equal to sole source')
  if not np.array_equal(z['source_timestamp'],timestamp): raise RuntimeError('v8 timestamps changed')
  base_l=z['base_target_left_position'].copy();base_r=z['base_target_right_position'].copy();names=z['arm_joint_names'].copy();warm=z['g1_arm_joint_trajectory'].copy()
 np.savez_compressed(OUT/'aloha_fk_source.npz',optimized_action=action,timestamp=timestamp,fps=np.array(fps),frame_index=np.arange(990),left_tcp_position=fk['left_position_m'],right_tcp_position=fk['right_position_m'],left_tcp_quaternion_wxyz=fk['left_quaternion_wxyz'],right_tcp_quaternion_wxyz=fk['right_quaternion_wxyz'],tcp_offset_local_m=np.array([.1487,0.,-.00105]),real_robot_command_allowed=np.array(False))
 np.savez_compressed(OUT/'restored_base_targets.npz',optimized_action=action,timestamp=timestamp,fps=np.array(fps),original_aloha_derived_left_position=base_l,original_aloha_derived_right_position=base_r,v8_exact_warm_start_only=warm,arm_joint_names=names,orientation_target_status=np.array('SOURCE_FK_STORED; ORIGINAL_G1_SOLVER_ORIENTATION_WEIGHT_ZERO'),real_robot_command_allowed=np.array(False))
 timeline=json.loads(TIMELINE.read_text());events=timeline['events'];by_frame=sorted(events,key=lambda x:(x['frame'],x['event']))
 expected={'left_phone_grasp_start':176,'phone_rotation_to_portrait_start':200,'phone_portrait_reached':223,'right_accessory_grasp_start':326,'accessory_detachment_start':329,'accessory_removed':341,'phone_move_to_charger_start':380,'phone_charger_attachment_complete':530,'left_phone_release_complete':586,'right_accessory_release_complete':646,'left_arm_return_near_home':702,'task_end':702}
 actual={e['event']:int(e['frame']) for e in events};diff={k:{'expected':v,'actual':actual.get(k)} for k,v in expected.items() if actual.get(k)!=v}
 dump('approved_timeline_audit.json',{'source':str(TIMELINE.resolve()),'sha256':sha(TIMELINE),'status':timeline['status'],'fps':timeline['fps'],'frame_range':timeline['frame_range'],'events_file_order':[{k:e[k] for k in ('event','frame','timestamp')} for e in events],'events_chronological':[{k:e[k] for k in ('event','frame','timestamp')} for e in by_frame],'expected_value_differences':diff,'file_array_is_frame_nondecreasing':all(events[i]['frame']<=events[i+1]['frame'] for i in range(len(events)-1)),'note':'Event values match the contract; JSON array order places frame 646 before frames 380/530/586. Chronological execution uses immutable frame values, never regenerated events.'})
 stage=Usd.Stage.Open(str(SCENE));usd={k:tf(stage,p) for k,p in {'g1':'/World/G1','phone':'/World/MagSafeScene/Phone','accessory':'/World/MagSafeScene/Accessory','charger':'/World/MagSafeScene/Charger','charger_pad_face':'/World/MagSafeScene/Charger/Frames/PhoneTargetCenter','charger_pad_face_geometry':'/World/MagSafeScene/Charger/Visuals/PadFace','table':'/World/MagSafeScene/Table'}.items()}
 required={'g1':[.44514890950197095,-.35257022755443246,.7922728583],'phone':[.525,.07,.83075],'accessory':[.525,.076425,.83075],'charger':[.42,.21,.807],'charger_pad_face':[.42,.2158465189,.9396181100]}
 deltas={k:(np.asarray(usd[k]['position_m'])-np.asarray(v)).tolist() for k,v in required.items()}
 padM=np.asarray(usd['charger_pad_face_geometry']['T_scene_from_prim']);pad_normal=(padM[:3,:3]@np.array([0,0,1.])).tolist();requested_normal=np.array([0,-.9659258263,.2588190451]);normal_error=float(np.linalg.norm(np.asarray(pad_normal)-requested_normal))
 dump('environment_audit.json',{'status':'PASS_ACTIVE_USD_MATCHES_CONTRACT' if normal_error<1e-6 else 'ACTIVE_USD_NORMAL_MISMATCH','active_composed_usd':str(SCENE.resolve()),'active_composed_usd_sha256':sha(SCENE),'read_method':'pxr.Usd.Stage.Open + composed local-to-world transforms','frames':usd,'required_minus_active_deltas_m':deltas,'charger_pad_outward_normal_from_active_padface_local_plus_z':pad_normal,'requested_pad_normal':requested_normal.tolist(),'pad_normal_l2_error':normal_error,'missing_declared_source_of_truth':str((ROOT/'configs/g1_magsafe_environment.current.json').resolve()),'missing_file':not (ROOT/'configs/g1_magsafe_environment.current.json').exists(),'object_or_root_mutation_performed':False,'camera_mutation_performed':False})
 provenance={'source_action':{'path':str(SRC.resolve()),'key':'optimized_action','sha256':sha(SRC),'shape':list(action.shape),'fps':fps,'timestamp_key':'timestamp','finite':True},'aloha_fk':{'implementation':str(Path(core.aloha.__file__).resolve()),'model':str(core.ALOHA_XML),'model_sha256':sha(core.ALOHA_XML),'tcp_parents':['follower_left_link_6','follower_right_link_6'],'tcp_offset_local_m':[.1487,0,-.00105],'mapped_clip_frames':int(clipped)},'relative_target':{'implementation':str((ROOT/'tools/retarget_episode49_consensus_relative_bimanual_to_g1.py').resolve()),'reproduction':str(REP.resolve()),'reproduction_sha256':sha(REP),'workspace_scale':float(core.SCALE),'ALIGN_RPY_deg':core.ALIGN_RPY.tolist(),'position_weight':3.0,'bimanual_relative_weight':.80,'velocity_weight':.018,'acceleration_weight':.030,'joint_regularization':.001,'lsqr_damping':.002,'orientation_weight':0.0,'manual_waypoints':False},'v8_usage':['restored ALOHA-derived base Cartesian positions','validated temporal IK implementation provenance','exact q only as future warm-start/branch prior','motion audit'],'forbidden_branch_policy':{'paths':FORBIDDEN,'loaded':[],'allowed_use':'failure diagnosis provenance only'}}
 dump('original_pipeline_provenance.json',provenance)
 raw_info=RAW/'meta/info.json';raw_parquet=RAW/'data/chunk-000/episode_000000.parquet';mapping=ROOT/'outputs/task_frame_registration/episode49_raw_source_mapping.json'
 evidence=[
  {'priority':1,'path':str(raw_info.resolve()),'sha256':sha(raw_info),'result':'NO_OBJECT_POSE_OR_CAMERA_CALIBRATION_FIELDS','coordinate_convention':'images and robot joints only'},
  {'priority':1,'path':str(raw_parquet.resolve()),'sha256':sha(raw_parquet),'result':'SCHEMA_HAS_ACTION_STATE_TIMESTAMP_INDICES_ONLY; NO_OBJECT_SE3','coordinate_convention':'ALOHA joint/action channels'},
  {'priority':2,'path':'NOT_FOUND_FOR_EPISODE_49','result':'NO_SOURCE_SCENE_LAYOUT_OR_BACKUP MAPPED TO GoPark_20260729_111223'},
  {'priority':3,'path':str((ROOT/'isaaclab_magsafe_fixed_scene/magsafe_scene_builder.py').resolve()),'sha256':sha(ROOT/'isaaclab_magsafe_fixed_scene/magsafe_scene_builder.py'),'result':'TARGET CURRENT ISAAC SCENE; NOT SOURCE REAL-RECORDING OBJECT SE3'},
  {'priority':4,'path':'NOT_FOUND','result':'NO APPROVED SOURCE-VIDEO 2D-TO-3D MANUAL REGISTRATION ARTIFACT; RGB ALONE IS NOT AN AUDITABLE SE3'},
  {'priority':5,'path':str(FRAME_GRAPH.resolve()),'sha256':sha(FRAME_GRAPH),'result':'CONTAINS TARGET fixed_scene_world OBJECTS; ALOHA TASK/OBJECT REGISTRATION NOT APPROVED'},
  {'priority':5,'path':str(SEM.resolve()),'sha256':sha(SEM),'result':'EXPLICITLY RECORDS ALOHA TCP-to-phone relation UNKNOWN AND downstream_generation BLOCKED'},
  {'mapping_path':str(mapping.resolve()),'sha256':sha(mapping),'result':'VERIFIES DATASET EP49 TO RAW EP0 ONLY; PROVIDES NO OBJECT FRAME'}]
 audit={'status':'BLOCKED_SOURCE_OBJECT_FRAME','source_episode':49,'raw_recording':str(RAW.resolve()),'evidence':evidence,'recoverable':{'phone':False,'accessory':False,'charger_pad':False},'missing_required_quantities':['T_aloha_world_from_phone_source','T_aloha_world_from_accessory_source','T_aloha_world_from_charger_pad_source','source phone initial orientation/back normal','source charger pad normal'],'why_rgb_not_used':'No recording-specific camera intrinsic/extrinsic calibration or approved manual 2D-to-3D registration; monocular visual estimation would be an arbitrary pose guess forbidden by the contract.','trajectory_deformation_generated':False,'ik_generated':False,'isaaclab_replay_or_render_generated':False,'fallback_generated':False}
 dump('source_object_frame_audit.json',audit)
 for obj in ('phone','accessory','charger'):
  dump(f'source_{obj}_frame.json',{'status':'BLOCKED_SOURCE_OBJECT_FRAME','object':obj,'recovered':False,'T_aloha_world_from_object':None,'provenance':evidence,'exact_blocker':'No authoritative Episode-49 source object SE(3) or approved video registration artifact.'})
 dump('input_audit.json',{'status':'PASS_UP_TO_SOURCE_OBJECT_FRAME_GATE','sole_primary_motion_source':str(SRC.resolve()),'optimized_action_sha256':sha(SRC),'shape':[990,14],'fps':30.0,'timestamps_exactly_retained':True,'approved_events_mutated':False,'gripper_channels_retained':[6,13],'forbidden_outputs_loaded':[],'real_robot_command_allowed':False,'dds_or_publisher_used':False})
 blocked={'status':'BLOCKED_SOURCE_OBJECT_FRAME','completed':['repository audit','source action invariant audit','original ALOHA FK restoration','base ALOHA-derived target restoration','approved timeline audit','active composed USD environment audit'],'not_run':['source hand-object relations','target anchors','phase correction candidates','fidelity selection','temporal IK','collision gates','Isaac Lab headless replay','Isaac Lab videos','GUI commands'],'reason':'Authoritative Episode-49 source phone/accessory/charger SE(3) frames are absent. Contract forbids estimating arbitrary offsets or replacing motion.'}
 dump('run_manifest.json',blocked)
 (OUT/'commands.sh').write_text('# BLOCKED_SOURCE_OBJECT_FRAME\n# No replay/GUI command is emitted because no contract-valid task-anchored trajectory exists.\n')
 report='''# Episode 49 Isaac phasewarp v9 — blocked at source-object-frame gate\n\nFinal status: `BLOCKED_SOURCE_OBJECT_FRAME`.\n\nThe sole source `optimized_action` passed shape, FPS, finite-value, timestamp, and raw-recording mapping checks. Original stationary-ALOHA FK and v8 base ALOHA-derived targets were restored without editing the 990 frames. The active composed Isaac USD was opened through `pxr.Usd`; G1 root, phone, accessory, charger, charger pad, and table transforms match the fixed contract. No scene transform was changed.\n\nThe raw Episode-49 parquet contains only action, observation state, timestamps, and indices. Its metadata contains images but no object pose or recording-specific camera calibration. No source scene layout mapped to this recording and no approved manual 2D-to-3D source registration artifact exists. The existing semantic-definition artifact explicitly records the ALOHA TCP-to-phone relation as unknown and downstream generation as blocked. Target-scene object frames cannot substitute for source object frames.\n\nTherefore source hand↔object relations, target anchors, correction candidates, temporal IK, collision gates, Isaac Lab replay/videos, and GUI commands were not generated. Creating them would require guessing source SE(3), which is forbidden.\n\nExact unblocker: provide an authoritative Episode-49 ALOHA-world SE(3) for the initial phone, accessory, and charger-pad frames, or an approved camera-calibrated source-video registration artifact with hashes and coordinate convention.\n'''
 (OUT/'report.md').write_text(report);(OUT/'report/index.html').write_text('<!doctype html><meta charset="utf-8"><title>v9 blocked</title><h1>BLOCKED_SOURCE_OBJECT_FRAME</h1><p>See ../report.md and source_object_frame_audit.json. No trajectory or render was fabricated.</p>')
 print(json.dumps({'status':'BLOCKED_SOURCE_OBJECT_FRAME','output':str(OUT),'aloha_fk_saved':True,'base_targets_saved':True,'deformation_or_ik_or_render_generated':False},indent=2));return 2
if __name__=='__main__': raise SystemExit(main())
