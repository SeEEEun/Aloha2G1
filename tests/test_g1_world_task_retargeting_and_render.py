import hashlib,json
from pathlib import Path
import numpy as np
R=Path('/home/jbnu/aloha_g1_dataset');O=R/'outputs/g1_world_task_retargeting';SC=R/'isaaclab_magsafe_fixed_scene'
def j(p):return json.load(open(p))
def test_01_scene_reused():assert j(O/'input_audit.json')['scene'].endswith('magsafe_magnetic_scene_v2.usda')
def test_02_original_usd_unchanged():assert hashlib.sha256((SC/'generated/magsafe_magnetic_scene_v2.usda').read_bytes()).hexdigest()==j(O/'input_audit.json')['scene_hash']
def test_03_objects_unchanged():assert j(O/'input_audit.json')['objects_immutable']
def test_04_cameras_unchanged():assert j(O/'input_audit.json')['camera_immutable']
def test_05_tcp_frames():assert np.load(O/'source/aloha_tcp_world_trajectory.npz')['left_tcp_world_position'].shape==(990,3)
def test_06_source_gate_detects_mismatch():assert j(O/'source/source_validation.json')['status']=='ALOHA_SCENE_REPLAY_MISMATCH'
def test_07_bounds_not_invented():assert j(O/'root_search/root_candidates.json')['bounds']=='NOT_DERIVED_SOURCE_GATE_FAILED'
def test_08_no_arbitrary_range():assert not j(O/'root_search/root_candidates.json')['arbitrary_bounds_used']
def test_09_no_invalid_root_candidates():assert j(O/'root_search/root_candidates.json')['candidates']==[]
def test_10_feet_gate_not_faked():assert j(O/'root_search/root_candidates.json')['status']=='ALOHA_SCENE_REPLAY_MISMATCH'
def test_11_static_not_faked():assert 'downstream not run' in (O/'root_search/root_candidate_metrics.csv').read_text()
def test_12_top5_explicitly_blocked():assert not j(O/'root_search/root_candidates.json')['candidates']
def test_13_temporal_not_faked():assert 'downstream not run' in (O/'metrics/temporal_candidate_metrics.csv').read_text()
def test_14_orientation_zero():assert j(O/'input_audit.json')['orientation_weight']==0
def test_15_no_object_snap():assert 'snap' not in (R/'tools/run_g1_world_task_retargeting_and_render.py').read_text().lower()
def test_16_no_fake_grasp_success():assert j(O/'selection.json')['selected_trajectory'] is None
def test_17_visual_copy_not_misrepresented():assert not (SC/'generated/magsafe_g1_headless_visualization.usda').exists()
def test_18_smoke_failure_recorded():assert j(O/'render_status.json')['smoke_mp4_generated'] is False
def test_19_full_failure_recorded():assert j(O/'render_status.json')['status']=='NOT_RUN_SOURCE_GATE_FAILED'
def test_20_sequential_render_plan():assert j(O/'render_status.json')['sequential_rendering_planned']
def test_21_figure_generated():assert (O/'report/root_candidate_keyframes.png').stat().st_size>50000
def test_22_tables_generated():assert (O/'report/table_temporal_candidate_metrics.csv').exists()
def test_23_no_auto_selection():assert j(O/'selection.json')['selected_root_candidate'] is None
def test_24_no_dds():assert 'ChannelPublisher' not in (R/'tools/run_g1_world_task_retargeting_and_render.py').read_text()
def test_25_no_publisher():assert j(O/'input_audit.json')['hardware_used'] is False
def test_26_no_real_robot():assert j(O/'selection.json')['status']=='BLOCKED_ALOHA_SCENE_REPLAY_MISMATCH'
