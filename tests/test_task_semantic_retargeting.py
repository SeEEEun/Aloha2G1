from pathlib import Path
import ast,json
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/home/jbnu/aloha_g1_dataset')
OUT=ROOT/'outputs/task_semantic_retargeting'
ARM=ROOT/'converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz'

def test_frozen_exact_equality():
 a=np.load(ARM)['g1_arm_joint_trajectory'];b=np.load(OUT/'candidates/position_only_frozen/arm_trajectory.npz')['arm_qpos'];assert np.array_equal(a,b)
def test_source_hash_same():
 f=json.load(open(OUT/'authoritative_freeze.json'));z=np.load(OUT/'candidates/position_only_frozen/arm_trajectory.npz');assert str(z['input_action_hash'])==f['source_action_sha256']
def test_object_pose_shared_status():assert json.load(open(OUT/'scene_audit.json'))['warning'].startswith('Object poses were not moved')
def test_anchor_scale_not_rewritten():assert json.load(open(OUT/'selection.json'))['verdict']=='NO TASK-VALID CANDIDATE FOUND'
def test_palm_left_right_distinct():
 p=json.load(open(ROOT/'configs/g1_dex3_palm_frame_calibration.sim.json'));assert p['left']['parent_body']!=p['right']['parent_body']
def test_quaternion_sign():
 q=Rotation.random(random_state=1).as_quat();assert np.allclose(Rotation.from_quat(q).as_matrix(),Rotation.from_quat(-q).as_matrix())
def test_partial_axis_residual():
 a=np.array([1.,0,0]);b=np.array([0.,1,0]);assert np.allclose(np.cross(a,b),[0,0,1])
def test_phase_transition_input_complete():
 import csv
 r=list(csv.DictReader(open(ROOT/'outputs/magsafe_gripper_phases.csv')));assert len(r)==990 and all(x['left_phase'] and x['right_phase'] for x in r)
def test_config_reproducible():assert json.load(open(ROOT/'configs/g1_dex3_palm_frame_calibration.sim.json'))['simulation_only'] is True
def test_joint_limit_metric():assert json.load(open(OUT/'candidates/position_only_frozen/metrics.json'))['joint_limit_violations']==0
def test_no_branch_jump():assert json.load(open(OUT/'candidates/position_only_frozen/metrics.json'))['branch_discontinuity_count']==0
def test_roles():
 import csv
 r=next(csv.DictReader(open(OUT/'episode49_task_timeline.csv')));assert r['left_object_role']=='phone' and r['right_object_role']=='magsafe_accessory'
def test_no_hardware_imports():
 tree=ast.parse((ROOT/'tools/run_task_semantic_retargeting_offline.py').read_text());names=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):names.extend(x.name for x in n.names)
  elif isinstance(n,ast.ImportFrom):names.append(n.module or '')
 assert not any('unitree_sdk' in x or 'dds' in x for x in names)
def test_no_publisher_words():
 s=(ROOT/'tools/run_task_semantic_retargeting_offline.py').read_text().lower();assert 'channelpublisher' not in s and 'channel_publisher' not in s
def test_selected_gate():assert json.load(open(OUT/'selection.json'))['selected_candidate'] is None
