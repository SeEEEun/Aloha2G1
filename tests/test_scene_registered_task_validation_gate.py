from pathlib import Path
import csv,hashlib,json
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');OUT=ROOT/'outputs/scene_registered_task_validation';ACTION=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz';FROZEN=ROOT/'converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_source_action_unchanged():assert sha(ACTION)=='a7f5543e07e315d59f52004dab48423a4ee52dfcbafb9b6d5d1a731fcbd3694c'
def test_frozen_unchanged():assert sha(FROZEN)=='c58c8ee6f98e02d71e22abc721fcb92bb7e5c233963b0cb2d44b3fa6c4ad1f3e'
def test_frame_count():assert np.load(ACTION)['optimized_action'].shape==(990,14)
def test_fps():assert float(np.load(ACTION)['fps'])==30
def test_semantic_order():
 d=json.load(open(ROOT/'outputs/task_frame_registration/magsafe_task_semantic_definition.json'));assert d['timeline_event_order'].index('left_phone_grasp_start')<d['timeline_event_order'].index('right_accessory_grasp_start')<d['timeline_event_order'].index('phone_charger_attachment_complete')
def test_automatic_phase_cannot_approve():assert json.load(open(ROOT/'configs/episode49_task_timeline.draft.json'))['automatic_events_saved'] is False
def test_timeline_gate():assert json.load(open(OUT/'automatic_gate.json'))['timeline_gate']=='FAIL'
def test_single_global_transform_not_generated():assert not (OUT/'registration/global_scene_anchor.json').exists()
def test_scale_not_tuned():assert json.load(open(ROOT/'outputs/task_frame_registration/magsafe_task_semantic_definition.json'))['downstream_generation']=='BLOCKED'
def test_no_stage_offset():assert not (OUT/'registration/global_scene_anchor.json').exists()
def test_no_candidate_object_motion():assert not (OUT/'candidates').exists()
def test_initial_vertical_landscape():assert json.load(open(ROOT/'outputs/task_frame_registration/magsafe_task_semantic_definition.json'))['poses']['initial_phone_pose']['semantic_orientation']=='VERTICAL_LANDSCAPE'
def test_portrait_hold():assert json.load(open(ROOT/'outputs/task_frame_registration/magsafe_task_semantic_definition.json'))['poses']['portrait_hold_pose']['semantic_orientation']=='VERTICAL_PORTRAIT'
def test_final_portrait():assert json.load(open(ROOT/'outputs/task_frame_registration/magsafe_task_semantic_definition.json'))['poses']['charger_attached_pose']['semantic_orientation']=='VERTICAL_PORTRAIT'
def test_charger_asset_values():
 c=json.load(open(ROOT/'outputs/task_frame_registration/magsafe_task_semantic_definition.json'))['charger_pad_verified_from_asset'];assert np.allclose(c['pad_face_center_scene_m'],[.42,.525846518946957,.9396181100184134]) and np.allclose(c['pad_outward_normal_scene'],[0,-.9659258262890683,.25881904510252074])
def test_phone_grasp_box_disabled():
 x=json.load(open(ROOT/'outputs/task_frame_registration/magsafe_task_semantic_definition.json'))['phone_grasp_region']['deprecated_box_volume'];assert not x['trajectory_gate_enabled'] and not x['success_metric_enabled']
def test_no_raw_metrics_before_gate():assert not (OUT/'metrics/task_semantic_raw_metrics.csv').exists()
def test_placeholder_separation_deferred():assert not (OUT/'collisions').exists()
def test_no_fake_task_success():assert json.load(open(OUT/'automatic_gate.json'))['task_success']=='NOT_AUTOMATICALLY_EVALUATED'
def test_no_fake_expert():assert 'expert' not in json.dumps(json.load(open(OUT/'manual_review_status.json'))).lower()
def test_no_dds():assert 'import dds' not in (ROOT/'tools/run_scene_registered_task_validation.py').read_text().lower()
def test_no_publisher():assert 'channelpublisher' not in (ROOT/'tools/run_scene_registered_task_validation.py').read_text().lower()
def test_no_command():assert 'command client' not in (ROOT/'tools/run_scene_registered_task_validation.py').read_text().lower()
def test_no_real_robot():assert json.load(open(OUT/'manual_review_status.json'))['real_g1_used'] is False
