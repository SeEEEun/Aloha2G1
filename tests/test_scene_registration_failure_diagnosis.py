import hashlib,json
from pathlib import Path
import numpy as np
R=Path('/home/jbnu/aloha_g1_dataset');O=R/'outputs/scene_registration_failure_diagnosis'
A=R/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz';F=R/'converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz'
def j(n):return json.load(open(O/n))
def test_01_source_unchanged():assert hashlib.sha256(A.read_bytes()).hexdigest()==j('source_vla_vs_gt_metrics.json')['action_hash']
def test_02_frozen_bitwise_unchanged():assert hashlib.sha256(F.read_bytes()).hexdigest()=='c58c8ee6f98e02d71e22abc721fcb92bb7e5c233963b0cb2d44b3fa6c4ad1f3e'
def test_03_events_unchanged():assert [x['frame'] for x in j('source_vla_vs_gt_metrics.json')['events']]==[176,223,326,341,530,586,646]
def test_04_objects_unchanged():assert j('anchor_invariant.json')['object_pose_changed'] is False
def test_05_root_unchanged():assert j('root_cause.json')['scene_or_robot_pose_changed'] is False
def test_06_scale():assert j('anchor_invariant.json')['scale']==.42
def test_07_anchor_invariant():assert j('anchor_invariant.json')['invariant_error_m']<1e-6
def test_08_palm_mapping():assert j('palm_wrist_frame_audit.json')['classification']=='PALM_WRIST_MAPPING_PASS'
def test_09_static_tested():assert len(j('static_reachability_summary.json')['best_by_frame'])==5
def test_10_temporal_conditional():assert j('temporal_window_diagnosis.json')['classification']=='NOT_RUN_STATIC_TARGETS_UNREACHABLE'
def test_11_bestfit_not_applied():assert j('semantic_waypoint_fit.json')['best_fit_applied'] is False
def test_12_no_stage_offset():assert 'stage offset' not in Path(R/'tools/diagnose_scene_registered_g1_failure.py').read_text().lower()
def test_13_no_right_offset():assert 'right_hand_offset' not in Path(R/'tools/diagnose_scene_registered_g1_failure.py').read_text()
def test_14_no_object_snap():assert 'snap' not in Path(R/'tools/diagnose_scene_registered_g1_failure.py').read_text().lower()
def test_15_seed_bug_regression():assert 'seed=core.position_seed' in Path(R/'tools/run_quick_scene_registered_g1_preview.py').read_text()
def test_16_no_fake_success():assert j('root_cause.json')['repaired_candidate_generated'] is False
def test_17_no_physics_grasp():assert not (O/'repaired_preview').exists()
def test_18_no_dds():assert 'ChannelPublisher' not in Path(R/'tools/diagnose_scene_registered_g1_failure.py').read_text()
def test_19_no_publisher():assert 'command client' not in Path(R/'tools/diagnose_scene_registered_g1_failure.py').read_text().lower()
def test_20_no_real_robot():assert j('root_cause.json')['status'].startswith('CURRENT G1 SCENE PLACEMENT IS UNREACHABLE')
