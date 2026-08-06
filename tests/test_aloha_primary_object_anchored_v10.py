from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import numpy as np

ROOT=Path('/home/jbnu/aloha_g1_dataset')
OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_aloha_primary_object_anchored_v10'
SOURCE=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'

def j(name):return json.loads((OUT/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def test_source_scene_declaration_and_target_immutability():
 d=j('user_approved_source_scene_declaration.json');s=j('source_scene_layout.json');e=j('environment_audit.json')
 assert d['status']=='USER_APPROVED_PROJECT_SCENE_EQUIVALENCE'
 assert s['phone']['bottom_left_xy'][1]==.255 and s['charger']['center_xy'][1]==.520
 assert e['phone_y_m']==.07 and e['charger_y_m']==.21 and e['g1_root_forward_offset_m']==.15
 assert e['hash_unchanged'] and e['source_layout_never_applied_to_target']

def test_sole_source_shape_time_and_timeline():
 z=np.load(SOURCE,allow_pickle=False);a=z['optimized_action'];t=z['timestamp'];fps=float(z['fps']);z.close()
 audit=j('input_audit.json');assert a.shape==(990,14) and np.isfinite(a).all() and t.shape==(990,) and fps==30
 assert audit['sole_source']==str(SOURCE.resolve()) and audit['timestamps_exact']
 expected={'left_phone_grasp_start':176,'phone_rotation_to_portrait_start':200,'phone_portrait_reached':223,'right_accessory_grasp_start':326,'accessory_detachment_start':329,'accessory_removed':341,'phone_move_to_charger_start':380,'phone_charger_attachment_complete':530,'left_phone_release_complete':586,'right_accessory_release_complete':646,'left_arm_return_near_home':702,'task_end':702}
 got={x['event']:x['frame'] for x in json.loads((ROOT/'configs/episode49_task_timeline.approved.json').read_text())['events']}
 assert all(got[k]==v for k,v in expected.items())

def test_source_relations_and_no_forbidden_generation():
 a=j('input_audit.json');s=j('source_scene_audit.json');r=j('source_hand_object_relations.json')
 assert s['camera_fitting_used'] is False and s['source_absolute_coordinates_used_for_target_generation'] is False
 assert r['source_absolute_coordinates_discarded_after_relation_extraction'] and r['camera_fitting_used'] is False
 assert a['forbidden_branches_loaded']==[] and not a['hand_written_waypoints'] and not a['per_frame_snapping'] and not a['static_grasp_first']
 assert not a['dex3_contact_ik'] and not a['physics'] and not a['dds_publisher_hardware']

def test_residual_knots_and_separate_base():
 z=np.load(OUT/'phase_residual_coefficients.npz',allow_pickle=False);assert np.array_equal(z['knot_frames'],[0,176,200,223,326,329,341,380,530,586,646,702,989]);assert z['left_translation_residual'].shape==(990,3);z.close()
 b=np.load(OUT/'restored_base_aloha_targets.npz',allow_pickle=False);c=np.load(OUT/'corrected_aloha_targets.npz',allow_pickle=False);assert b['base_left_position_scene'].shape==(990,3);assert c['corrected_left_position'].shape==(990,3);assert not np.array_equal(b['base_left_position_scene'],c['corrected_left_position']);assert np.array_equal(c['original_base_left_position'],b['base_left_position_scene']);b.close();c.close()

def test_exact_nullspace_targets_identical_but_q_different():
 e=np.load(OUT/'aloha_anchored_exact_arm_trajectory.npz',allow_pickle=False);n=np.load(OUT/'aloha_anchored_nullspace_arm_trajectory.npz',allow_pickle=False)
 for key in ('corrected_left_position_scene','corrected_right_position_scene','corrected_left_rotation_scene','corrected_right_rotation_scene'):assert np.array_equal(e[key],n[key])
 assert not np.array_equal(e['g1_arm_joint_trajectory'],n['g1_arm_joint_trajectory']);assert not bool(e['dex3_contact_ik_applied']) and not bool(n['dex3_contact_ik_applied']);assert not bool(e['physics_applied']) and not bool(n['physics_applied']);e.close();n.close()

def test_isaaclab_render_evidence_and_metadata():
 v=j('visual_validation_audit.json');h=j('isaaclab_headless_results.json');assert v['robot_render']=='ACTUAL_ISAAC_LAB_G1_AND_ALOHA';assert v['red_trajectory_plot_used'] is False;assert v['all_videos_990_decoded_frames'];assert v['exact_nullspace_video_sha256_different'];assert h['joint_mapping_all_name_based'] and h['missing_joints']==[] and h['max_mapped_joint_error_rad']==0 and h['physics_steps']==0
 for x in v['videos'].values():
  assert x['decoded_frames']==990 and x['fps']=='15/2' and x['comment']
  assert 'sha256' in x['comment'] and ('trajectory_npz' in x['comment'] or 'source_action_npz' in x['comment'] or 'exact_npz' in x['comment'])

def test_failure_is_not_mislabeled_as_success():
 ik=j('ik_metrics.json');fid=j('aloha_phase_fidelity_metrics.json');m=j('run_manifest.json');assert ik['status']=='BLOCKED_IK';assert fid['status']=='ALOHA_FIDELITY_WARNING';assert 'BLOCKED_IK' in m['status'] and 'BLOCKED_ALOHA_FIDELITY' in m['status'];assert m['dex3_applied'] is False and m['physics'] is False and m['dds'] is False and m['publisher'] is False and m['real_robot_commands'] is False
