import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/jbnu/aloha_g1_dataset')
OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_left_hold_right_c_v4'
def load(n):return json.loads((OUT/n).read_text())

def test_source_and_timeline_are_preserved():
    with np.load(OUT/'aloha_portrait_motion_prior.npz') as z:
        assert z['optimized_action'].shape==(990,14)
        assert np.isfinite(z['optimized_action']).all() and float(z['fps'])==30
        assert z['event_frames'].tolist()==[176,200,223,326,329,341,380,530,586,702,702]

def test_two_left_ab_assignments_were_searched():
    d=load('left_ab_candidate_results.json')
    assert d['candidate_count']==20 and d['assignment_count']==2
    assert {x['assignment'] for x in d['candidates']}=={'A_SCREEN_B_BACK','A_BACK_B_SCREEN'}

def test_left_gate_fails_closed():
    d=load('selected_left_ab_grasp.json')
    assert d['status']=='BLOCKED_LEFT_AB_PHONE_GRASP'
    assert d['selected_left_ab_grasp'] is None
    assert d['best_failed']['position_error_m']>.003

def test_no_portrait_or_right_c_is_invented():
    assert load('selected_portrait_hold_pose.json')['source_derived_pose'] is None
    assert load('right_c_axial_candidates.json')['candidate_count']==0
    assert load('right_c_radial_gap_candidates.json')['candidate_count']==0
    assert load('selected_right_c_candidate.json')['selected_candidate'] is None

def test_no_full_v3_or_real_robot_output():
    assert not load('coupled_feasibility_metrics.json')['full_990_frame_v3_resume']
    for name in ('g1_full_arm_dex3_fingertip_trajectory.npz','phone_pose_trajectory.npz'):
        assert not (OUT/name).exists()
    text=(ROOT/'tools/run_left_hold_right_c_v4.py').read_text().lower()
    for forbidden in ('import rclpy','unitree_sdk','create_publisher(','write_command('):assert forbidden not in text
