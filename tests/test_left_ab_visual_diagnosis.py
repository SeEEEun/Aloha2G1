import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/jbnu/aloha_g1_dataset');OUT=ROOT/'outputs/left_ab_grasp_visual_diagnosis'
def load(n):return json.loads((OUT/n).read_text())

def test_reproduction_matches_original():
    d=load('rerun/reproduction_audit.json');assert d['status']=='PASS' and d['reproduction_match']
    x=d['difference_from_previous'];assert abs(x['position_error_difference_m'])<=1e-4 and abs(x['orientation_error_difference_deg'])<=.1 and max(map(abs,x['error_vector_difference_m']))<=1e-4

def test_all_20_candidates_and_qpos_are_saved():
    with np.load(OUT/'rerun/all_candidates.npz') as z:
        assert z['solved_left_arm_qpos'].shape==(20,7) and z['full_qpos'].shape==(20,50)
        assert np.isfinite(z['full_qpos']).all()

def test_best_failed_payload_contract():
    required=('left_arm_qpos','left_dex3_qpos','controlled_joint_names','wrist_target_position','wrist_target_rotation','wrist_achieved_position','wrist_achieved_rotation','A_target_position','B_target_position','A_achieved_position','B_achieved_position','C_achieved_position','palm_position','palm_rotation','AB_midpoint_target','AB_midpoint_achieved','pinch_axis_target','pinch_axis_achieved','collision_geom_pairs','clearance_values')
    with np.load(OUT/'best_failed_candidate.npz') as z:assert all(k in z.files for k in required)

def test_transform_is_proper_and_classification_evidenced():
    t=load('transform_audit.json');d=load('diagnosis.json')
    assert t['rotation_determinant']==1 and t['orthonormal'] and not t['left_handed_reflection']
    assert d['classification']=='MULTIPLE_CAUSES'

def test_static_images_and_no_physics_or_trajectory():
    for n in ('overview_front.png','overview_side.png','overview_top.png','phone_closeup_front.png','phone_closeup_side.png','phone_closeup_top.png','palm_closeup.png','fingertip_target_vs_achieved.png','collision_closeup.png','transform_axes_comparison.png','pose_comparison.png'):assert (OUT/n).stat().st_size>0
    text=(ROOT/'tools/view_left_ab_best_failed_candidate.py').read_text()
    assert 'mujoco.mj_step(' not in text and not (OUT/'trajectory.npz').exists()
