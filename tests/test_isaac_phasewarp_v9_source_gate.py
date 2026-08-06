from pathlib import Path
import json
import numpy as np

ROOT=Path('/home/jbnu/aloha_g1_dataset')
OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_isaac_phasewarp_v9'

def j(name): return json.loads((OUT/name).read_text())

def test_sole_source_action_and_timestamps_are_exact():
 with np.load(ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz',allow_pickle=False) as s, np.load(OUT/'aloha_fk_source.npz',allow_pickle=False) as f, np.load(OUT/'restored_base_targets.npz',allow_pickle=False) as b:
  assert s['optimized_action'].shape==(990,14)
  assert np.array_equal(s['optimized_action'],f['optimized_action'])
  assert np.array_equal(s['optimized_action'],b['optimized_action'])
  assert np.array_equal(s['timestamp'],f['timestamp'])
  assert np.array_equal(s['timestamp'],b['timestamp'])
  assert float(f['fps'])==30.0 and np.isfinite(f['optimized_action']).all()

def test_approved_event_values_unchanged_and_chronology_audited():
 a=j('approved_timeline_audit.json')
 assert a['expected_value_differences']=={}
 assert [x['frame'] for x in a['events_chronological']]==sorted(x['frame'] for x in a['events_chronological'])
 assert a['file_array_is_frame_nondecreasing'] is False

def test_active_usd_layout_and_root_unchanged():
 a=j('environment_audit.json')
 assert a['status']=='PASS_ACTIVE_USD_MATCHES_CONTRACT'
 assert max(abs(v) for d in a['required_minus_active_deltas_m'].values() for v in d)<1e-9
 assert a['pad_normal_l2_error']<1e-6
 assert a['object_or_root_mutation_performed'] is False
 assert a['camera_mutation_performed'] is False

def test_forbidden_branches_never_loaded_and_no_hardware_path():
 a=j('input_audit.json')
 assert a['forbidden_outputs_loaded']==[]
 assert a['real_robot_command_allowed'] is False
 assert a['dds_or_publisher_used'] is False

def test_source_object_gate_stops_before_fabrication():
 a=j('source_object_frame_audit.json');m=j('run_manifest.json')
 assert a['status']==m['status']=='BLOCKED_SOURCE_OBJECT_FRAME'
 assert not any(a['recoverable'].values())
 assert a['trajectory_deformation_generated'] is False
 assert a['ik_generated'] is False
 assert a['isaaclab_replay_or_render_generated'] is False
 assert a['fallback_generated'] is False
 forbidden=['phase_correction_coefficients.npz','corrected_aloha_targets.npz','task_anchored_exact_arm_trajectory.npz','task_anchored_nullspace_arm_trajectory.npz']
 assert not any((OUT/name).exists() for name in forbidden)
