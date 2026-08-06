import json
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_left_ab_humanlike_v6'
def load(n):return json.loads((OUT/n).read_text())
def test_direct_phone_contact_axis():
 d=load('hand_only_results.json');assert d['candidate_count']==18
 assert all(x['axis_error_deg']<=1 and abs(x['separation_m']-.00795)<1e-12 and x['inside_surface'] for x in d['candidates'])
def test_swivel_grid_and_fail_closed():
 g=load('elbow_swivel_candidates.json');assert g=={'range_deg':[-70,70],'step_deg':10,'candidate_count_per_assignment':15}
 s=load('selected_humanlike_left_grasp.json');assert s['status']=='BLOCKED_HUMANLIKE_ARM_CARRIER' and s['selected'] is None
def test_failed_pose_persisted_but_not_accepted():
 with np.load(OUT/'best_failed_humanlike_left_grasp.npz') as z:assert z['full_qpos'].shape==(50,) and not bool(z['accepted'])
 assert not (OUT/'selected_humanlike_left_grasp.npz').exists()
def test_images_and_no_trajectory():
 for n in ('front_view.png','side_view.png','top_view.png','phone_closeup.png','elbow_posture_front.png','forearm_alignment_side.png','wrist_neutral_closeup.png','comparison_with_previous_failed_pose.png'):assert (OUT/n).stat().st_size>0
 assert not any(OUT.glob('*trajectory*.npz'))
