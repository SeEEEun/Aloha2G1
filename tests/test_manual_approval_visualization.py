from pathlib import Path
import json
ROOT=Path('/home/jbnu/aloha_g1_dataset');OUT=ROOT/'outputs/task_frame_registration';V=OUT/'approval_views';SC=ROOT/'outputs/scene_registered_retargeting'
def test_frame0_initial_posture():assert json.load(open(ROOT/'configs/episode49_task_timeline.approved.json'))['frame0']['approved_task_stage']=='INITIAL_POSTURE'
def test_frame0_auto_hold_allowed():
 import csv
 x=next(csv.DictReader(open(ROOT/'outputs/magsafe_gripper_phases.csv')));assert x['left_phase']=='HOLD' and x['right_phase']=='HOLD'
def test_auto_hold_not_semantic_hold():assert json.load(open(ROOT/'configs/episode49_task_timeline.approved.json'))['frame0']['approved_semantic_event'] is None
def test_episode49_mapping_verified():assert json.load(open(OUT/'episode49_raw_source_mapping.json'))['status']=='VERIFIED'
def test_g1_visual_mesh_rendered():assert all((V/f'g1_scene_{x}.png').stat().st_size>10000 for x in ('front','side','top','isometric'))
def test_object_visual_geometry_rendered():assert all((V/f'semantic_frames_{x}.png').stat().st_size>10000 for x in ('front','side','top','isometric'))
def test_regions_rendered():assert (V/'semantic_frames_phone_accessory_closeup.png').stat().st_size>10000 and (V/'semantic_frames_charger_closeup.png').stat().st_size>10000
def test_tool_axes_model():assert (V/'aloha_left_tool_axes_model_view_frame_000000.png').stat().st_size>10000
def test_tool_axes_video():assert (V/'aloha_left_tool_axes_video_view_frame_000000.png').stat().st_size>10000
def test_approval_defaults_false():assert not any(x['approved'] for x in json.load(open(OUT/'approval_status.json'))['flags'].values())
def test_downstream_blocked():assert json.load(open(SC/'selection.json'))['downstream_generation']=='BLOCKED'
def test_no_dds():assert 'import dds' not in '\n'.join((ROOT/'tools'/x).read_text().lower() for x in ('calibrate_g1_to_magsafe_scene.py','review_magsafe_semantic_frames.py','review_aloha_tool_axes.py','review_episode49_task_timeline.py'))
def test_no_publisher():assert 'channelpublisher' not in '\n'.join((ROOT/'tools'/x).read_text().lower() for x in ('calibrate_g1_to_magsafe_scene.py','review_magsafe_semantic_frames.py','review_aloha_tool_axes.py','review_episode49_task_timeline.py'))
def test_no_real_robot():assert json.load(open(SC/'selection.json'))['real_g1_safety']=='NOT_PERFORMED'
