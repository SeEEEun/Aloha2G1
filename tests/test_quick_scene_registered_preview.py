import csv, hashlib, json
from pathlib import Path
import numpy as np

R=Path('/home/jbnu/aloha_g1_dataset');O=R/'outputs/quick_scene_registered_preview'
A=R/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'
F=R/'converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz'
E={'left_phone_grasp_start':176,'phone_rotation_to_portrait_start':200,'phone_portrait_reached':223,'right_accessory_grasp_start':326,'accessory_detachment_start':329,'accessory_removed':341,'phone_move_to_charger_start':380,'phone_charger_attachment_complete':530,'left_phone_release_complete':586,'right_accessory_release_complete':646}
def audit(): return json.load(open(O/'audit.json'))
def test_01_action_hash(): assert hashlib.sha256(A.read_bytes()).hexdigest()==audit()['source_hash']
def test_02_frozen_hash(): assert hashlib.sha256(F.read_bytes()).hexdigest()==audit()['frozen_hash']
def test_03_events_exact(): assert audit()['approved_events']==E
def test_04_no_extra_event_gate(): assert json.load(open(O/'candidate_status.json'))['no_additional_events_required']
def test_05_coarse_phase(): assert audit()['coarse_phase']['frames']==[586,646]
def test_06_frames(): assert audit()['frames']==990
def test_07_fps(): assert audit()['fps']==30
def test_08_scale(): assert audit()['scale']==.42
def test_09_global_transform(): assert json.load(open(O/'global_scene_anchor.json'))['single_constant_transform']
def test_10_no_stage_offset(): assert json.load(open(O/'global_scene_anchor.json'))['stage_offsets'] is False
def test_11_no_right_offset(): assert json.load(open(O/'global_scene_anchor.json'))['right_hand_offset'] is False
def test_12_no_charger_snap(): assert 'snap' not in json.dumps(json.load(open(O/'global_scene_anchor.json'))).lower()
def test_13_no_table_snap(): assert np.allclose(np.load(O/'scene_position_only.npz')['accessory_pose'][646],np.load(O/'scene_position_only.npz')['accessory_pose'][647])
def test_14_phone_fixed_before_grasp():
 x=np.load(O/'scene_position_only.npz')['phone_pose'];assert np.allclose(x[:176],x[0])
def test_15_accessory_follows_through_645():
 x=np.load(O/'scene_position_only.npz')['accessory_pose'];assert not np.allclose(x[326],x[645]) and np.allclose(x[646],x[647])
def test_16_four_statuses():
 s=json.load(open(O/'candidate_status.json'));assert len(s['generated'])+len(s['failed'])==4
def test_17_no_fake_success(): assert 'NOT TASK SUCCESS' in (O/'report/table_task_semantic_raw_metrics.csv').read_text()
def test_18_not_physics(): assert 'KINEMATIC OBJECT REPLAY' in (R/'tools/run_quick_scene_registered_g1_preview.py').read_text()
def test_19_figures(): assert (O/'report/figure_candidate_keyframes.png').stat().st_size>100000
def test_20_tables(): assert (O/'report/table_retargeting_ablation.csv').exists()
def test_21_selection_null(): assert json.load(open(O/'selection.json'))['selected_candidate'] is None
def test_22_no_dds(): assert 'ChannelPublisher' not in (R/'tools/run_quick_scene_registered_g1_preview.py').read_text()
def test_23_no_publisher(): assert 'publisher' not in (R/'tools/run_quick_scene_registered_g1_preview.py').read_text().lower().replace('no dds or publisher was used','')
def test_24_no_hardware_command(): assert 'joint command' not in (R/'tools/run_quick_scene_registered_g1_preview.py').read_text().lower()
def test_25_no_real_robot(): assert audit()['scene_registration']=='PROVISIONAL_SIMULATION_ONLY'
