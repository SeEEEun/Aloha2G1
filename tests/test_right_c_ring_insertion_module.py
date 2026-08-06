import json
from pathlib import Path

ROOT=Path('/home/jbnu/aloha_g1_dataset')
OUT=ROOT/'outputs/right_c_ring_insertion'

def load(name): return json.loads((OUT/name).read_text())

def test_active_right_c_mapping_and_collision_geom():
    a=load('active_geometry_audit.json')
    assert a['right_C']['joint_chain']==['right_hand_middle_0_joint','right_hand_middle_1_joint']
    assert a['right_C']['distal_link']=='right_hand_middle_1_link'
    assert a['right_C']['collision_geom_id'] is not None

def test_active_ring_and_size_precheck():
    a=load('active_geometry_audit.json')
    assert a['ring']['inner_radius_m']==0.0225
    assert a['ring']['collision']=='12 convex box segments copied from active scene builder'
    assert a['hard_radial_clearance_m']>0

def test_full_required_preinsert_grid_was_evaluated():
    g=load('insertion_candidate_grid.json');r=load('insertion_candidate_results.json')
    assert len(g['radial_angle_deg'])>=16
    assert min(g['preinsert_distance_m'])==0.03 and max(g['preinsert_distance_m'])==0.08
    assert r['candidate_count']==len(g['radial_angle_deg'])*len(g['preinsert_distance_m'])*len(g['wrist_rpy_offset_deg'])

def test_failure_does_not_claim_continuous_collision_or_hook():
    s=load('selected_insertion_candidate.json');c=load('continuous_collision_results.json');h=load('hook_contact_metrics.json')
    assert s['status']=='BLOCKED_PREINSERT_IK' and s['selected_candidate'] is None
    assert not s['full_v3_resume']
    assert c['hard_geometry_gate']=='NOT_PASSED' and c['segments_checked']==[]
    assert not h['inner_rim_contact_confirmed']

def test_no_full_trajectory_or_hardware_path():
    assert not (OUT/'g1_full_arm_dex3_fingertip_trajectory.npz').exists()
    text=(ROOT/'tools/run_right_c_ring_insertion_feasibility.py').read_text().lower()
    for forbidden in ('import rclpy','unitree_sdk','create_publisher(','write_command('):
        assert forbidden not in text
