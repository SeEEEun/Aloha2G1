import json
from pathlib import Path
import numpy as np

R=Path('/home/jbnu/aloha_g1_dataset'); O=R/'outputs/scene_registered_retargeting/current_layout_ep49'
def j(name): return json.loads((O/name).read_text())
def test_current_scene_coordinates():
 d=j('current_scene_registration.json'); assert d['approved_values']=={'phone_y_m':.07,'charger_y_m':.21,'g1_root_forward_offset_m':.05}
def test_root_offset_exactly_once():
 d=j('current_scene_registration.json'); assert np.isclose(d['g1_root_position_m'],[.4267163031673236,-.45085674251814417,.7922728583]).all()
def test_stale_coordinates_not_used():
 d=j('run_manifest.json'); text=json.dumps(d); assert 'magsafe_magnetic_scene_v2.usda' not in d['authoritative_scene']; assert d['semantic_values_invented'] is False
def test_source_action():
 d=j('input_audit.json'); assert d['optimized_action_shape']==[990,14] and d['fps']==30 and d['finite']
def test_hashes_captured(): assert len(j('current_scene_registration.json')['evidence'])==4
def test_diagnostic_cannot_approve_final_or_robot():
 d=j('diagnostic_status.json'); assert d['candidate']=='NOT_FINAL' and not d['real_robot_command_allowed'] and not d['selected']
def test_strict_unresolved_gates_preserved(): assert set(j('approval_gate_audit.json')['unresolved_gates_preserved'])=={'semantic_frames','aloha_tool_axes'}
def test_shapes_joint_order_and_finite():
 with np.load(O/'g1_arm_scene_registered_trajectory.npz',allow_pickle=False) as z:
  assert z['g1_target_left_position'].shape==(990,3) and z['g1_arm_joint_trajectory'].shape==(990,14)
  assert len(z['arm_joint_names'])==14 and np.isfinite(z['g1_arm_joint_trajectory']).all()
def test_scene_hash_and_safety_metadata():
 with np.load(O/'g1_arm_scene_registered_trajectory.npz',allow_pickle=False) as z:
  assert str(z['scene_layout_hash']) and bool(z['diagnostic_only']) and not bool(z['real_robot_command_allowed'])
def test_dex3_sim_never_real():
 with np.load(O/'g1_arm_dex3_scene_registered_trajectory.npz',allow_pickle=False) as z:
  assert str(z['primitive_source'])=='simulation_placeholder' and not bool(z['authoritative_for_real_robot']) and not bool(z['real_robot_command_allowed'])
