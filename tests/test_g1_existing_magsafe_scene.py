import json
from pathlib import Path
import numpy as np
R=Path('/home/jbnu/aloha_g1_dataset');O=R/'outputs/g1_in_existing_magsafe_scene';SC=R/'isaaclab_magsafe_fixed_scene'
def j(p):return json.load(open(p))
def test_01_authoritative_scene_reused():assert j(O/'audit/aloha_scene_replay_audit.json')['authoritative_scene'].endswith('magsafe_magnetic_scene_v2.usda')
def test_02_existing_root_pose():assert j(O/'audit/aloha_scene_replay_audit.json')['g1_root_pose']['position_xyz_m']==[.4175,-.5,.7922728583]
def test_03_runtime_name_mapping():assert j(O/'isaac_runtime_joint_mapping.json')['status']=='PASS'
def test_04_joint_counts():assert len(j(O/'isaac_runtime_joint_mapping.json')['requested_joint_names'])==28
def test_05_no_missing_joint():assert j(O/'isaac_runtime_joint_mapping.json')['missing']==[]
def test_06_frames():assert j(O/'audit/aloha_scene_replay_audit.json')['frames']==990
def test_07_fps():assert j(O/'audit/aloha_scene_replay_audit.json')['fps']==30
def test_08_objects_not_moved():assert j(O/'dry_run.json')['objects_moved_by_replay'] is False
def test_09_no_snap_or_attachment():
 s=(SC/'replay_magsafe_g1_trajectory.py').read_text().lower();assert 'palm-follow' not in s and 'object snap' not in s
def test_10_structural():assert j(O/'structural_metrics.json')['joint_limit_violations']==0
def test_11_placeholder_separate():assert j(O/'structural_metrics.json')['placeholder_status']=='PLACEHOLDER_HAND_DIAGNOSTIC_ONLY'
def test_12_gate_closed():assert j(O/'next_stage_gate.json')['ready_for_physics_grasp'] is False
def test_13_no_dds():assert 'ChannelPublisher' not in (SC/'replay_magsafe_g1_trajectory.py').read_text()
def test_14_no_publisher():assert j(O/'dry_run.json')['hardware_commands_sent'] is False
def test_15_no_real_robot():assert j(O/'dry_run.json')['dds_initialized'] is False
