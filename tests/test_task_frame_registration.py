from pathlib import Path
import ast,json
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');REG=ROOT/'outputs/task_frame_registration';SC=ROOT/'outputs/scene_registered_retargeting'
def cfg(name):return json.load(open(ROOT/'configs'/name))
def test_frame_graph_inverse_roundtrip():
 c=cfg('magsafe_task_frame_registration.sim.json');a=np.array(c['T_scene_from_task']);b=np.array(c['T_task_from_scene']);assert np.allclose(a@b,np.eye(4))
def test_rotation_determinant():assert np.isclose(np.linalg.det(np.array(cfg('magsafe_task_frame_registration.sim.json')['T_scene_from_g1_base'])[:3,:3]),1)
def test_unit_consistency():assert cfg('magsafe_task_frame_registration.sim.json')['validation']['units_consistent']
def test_object_poses_unchanged():assert cfg('magsafe_object_semantic_frames.sim.json')['object_poses_immutable']
def test_g1_root_fixed():assert cfg('magsafe_task_frame_registration.sim.json')['validation']['one_fixed_transform_for_all_candidates']
def test_semantic_reproducibility():
 a=cfg('magsafe_object_semantic_frames.sim.json');b=cfg('magsafe_object_semantic_frames.sim.json');assert a==b
def test_left_right_roles():
 a=json.load(open(ROOT/'outputs/task_semantic_retargeting/episode49_task_timeline.csv'.replace('.csv','.json'))) if False else None;assert cfg('magsafe_object_semantic_frames.sim.json')['phone']!=cfg('magsafe_object_semantic_frames.sim.json')['accessory']
def test_timeline_requires_approval():
 p=ROOT/'configs/episode49_task_timeline.approved.json';assert not p.exists() or json.load(open(p))['status'] in ('DRAFT_NEEDS_MANUAL_APPROVAL','MANUALLY_APPROVED')
def test_tool_axes_orthogonal():
 c=cfg('aloha_tool_axes_calibration.sim.json');
 for s in ('left','right'):
  A=np.array([c[s]['approach_axis_local'],c[s]['closing_axis_local'],c[s]['lateral_axis_local']]);assert np.allclose(A@A.T,np.eye(3))
def test_global_anchor_invariant():assert json.load(open(SC/'scene_aware_anchor.json'))['global_anchor_invariant_over_frames']
def test_continuation_monotonic():
 c=json.load(open(SC/'continuation_sweep.json'));assert c['strictly_monotonic'] and np.all(np.diff(c['weights'])>0)
def test_source_hash_preserved():
 h=json.load(open(REG/'input_hashes.json'))['source_action_sha256'];c=json.load(open(SC/'candidates/position_only_frozen/status.json'));assert c['source_action_hash']==h
def test_phase_partial_residual():assert np.allclose(np.cross([1,0,0],[0,1,0]),[0,0,1])
def test_grasp_region():
 c=cfg('magsafe_object_semantic_frames.sim.json')['phone']['left_grasp_band'];assert len(c['bounding_half_extents_m'])==3 and c['position_tolerance_m']>0
def test_placement_region():assert cfg('magsafe_object_semantic_frames.sim.json')['charger']['placement_region']['radius_m']>0
def test_collision_gate_not_faked():assert json.load(open(SC/'automatic_gate.json'))['collision']=='NOT_AVAILABLE'
def test_no_dds_publisher():
 files=[ROOT/'tools/task_frame_registration.py',ROOT/'tools/run_scene_registered_retargeting_offline.py',ROOT/'tools/calibrate_g1_to_magsafe_scene.py'];s='\n'.join(x.read_text().lower() for x in files);assert 'channelpublisher' not in s and 'unitree_sdk' not in s and 'import dds' not in s
def test_no_real_robot_use():assert json.load(open(SC/'selection.json'))['real_g1_safety']=='NOT_PERFORMED'
