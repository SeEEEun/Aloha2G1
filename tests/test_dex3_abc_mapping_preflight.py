import json
from pathlib import Path
ROOT=Path('/home/jbnu/aloha_g1_dataset')
OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_fingertip_semantic_v3'
def test_mapping_and_roles():
 d=json.loads((ROOT/'configs/dex3_abc_finger_mapping.sim.json').read_text())
 assert d['status']=='VERIFIED_FROM_ACTIVE_MODEL_GEOMETRY_FOR_SIMULATION'
 assert d['left']['A']['digit_chain']=='thumb' and d['left']['B']['digit_chain']=='index' and d['left']['C']['digit_chain']=='middle'
 assert d['right']['A']['digit_chain']=='index' and d['right']['B']['digit_chain']=='thumb' and d['right']['C']['digit_chain']=='middle'
 assert d['approved_roles']=={'left_phone_grasp':['A','B'],'left_noncontact':['C'],'right_accessory_removal':['C'],'right_noncontact':['A','B']}
def test_fingertips_are_collision_geometry():
 d=json.loads((ROOT/'configs/dex3_fingertip_frames.sim.json').read_text())
 assert all('collision mesh geom_id=' in x['source_geom'] for x in d['fingertips'].values())
def test_block_prevents_false_candidate():
 d=json.loads((OUT/'selected_candidate.json').read_text())
 assert d['classification']=='BLOCKED_RIGHT_C_INSERTION' and d['selected_candidate'] is None and not d['trajectory_generated']
 assert not d['real_robot_command_allowed'] and not d['authoritative_for_real_robot']
def test_manual_events_approved_without_invention():
 d=json.loads((OUT/'candidate_search_results.json').read_text())
 assert d['timeline_gate']['status']=='PASS'
 assert d['timeline_gate']['new_manual_events']['left_arm_return_near_home']['frame']==702
 assert d['timeline_gate']['new_manual_events']['task_end']['frame']==702
 assert d['timeline_gate']['chronology_validation']=='PASS_NON_DECREASING'
 assert d['timeline_gate']['invented_event_frames'] is False
