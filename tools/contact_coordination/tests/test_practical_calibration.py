import copy,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
from tools.contact_coordination.io import ROOT,read,atomic_json
from tools.contact_coordination.practical_parameters import RECIPES,DEFAULT,parameters
from tools.contact_coordination.incidental_contact import permits,filter_hits,measured_assessment


class PracticalTests(unittest.TestCase):
    def config(self):
        return dict(enabled=True,maximum_depth_m=.001,maximum_displacement_m=.005,
            maximum_rotation_rad=.12,maximum_linear_speed_m_s=.5,maximum_angular_speed_rad_s=10.,
            patch_center_object_m=[.04,0,0],patch_radius_m=.035,patch_side_normal_object=[1,0,0],minimum_patch_side_projection_m=.005)

    def test_shallow_intended_palm_and_finger_only(self):
        c=self.config();x=np.eye(4)
        self.assertTrue(permits(c,'left_hand_thumb_2_link',.0005,[.04,0,0],x))
        self.assertTrue(permits(c,'left_palm_link',.0005,[.04,0,0],x))
        for link in ['left_elbow_link','left_wrist_yaw_link','left_hand_camera_base_link','left_shoulder_pitch_link','torso_link','right_hand_thumb_2_link']:
            self.assertFalse(permits(c,link,.0001,[.04,0,0],x))

    def test_authored_palm_alias_is_component_specific_and_matches_physx_name(self):
        from tools.contact_coordination.incidental_contact import policy
        base=ROOT/'outputs/source_guided_rrt_rebuild/20260911T035757Z'
        with tempfile.TemporaryDirectory() as directory:
            import shutil
            out=Path(directory);shutil.copytree(base/'target_repair',out/'target_repair')
            atomic_json(out/'PRACTICAL_PARAMETERS.json',dict(values={k:RECIPES['NEAR_APPROACH'][k] for k in DEFAULT}))
            c=policy(out);meta=read(out/'target_repair/runtime_bin150/RUNTIME_HULLS.json')
            point=c['patch_center_object_m'];x=np.eye(4)
            wrist_rows=[r for r in meta['rows'] if r.get('body')=='left_wrist_yaw_link']
            self.assertGreaterEqual(len(wrist_rows),3)
            for r in wrist_rows:
                hit=dict(bodies=['object',r['body']],geoms=['hybrid_object',r['key']],depth_m=.0005,
                    signed_distance_m=-.0005,point_world=point,allowed_contact=False)
                result=filter_hits([hit],c,x)[0]
                palm='/left_hand_palm_link/' in r['path']
                self.assertEqual(result['allowed_contact'],palm)
                if palm:
                    self.assertEqual(result['bodies'],hit['bodies'])
                    self.assertTrue(permits(c,'left_hand_palm_link',.0005,point,x))
                    hit['signed_distance_m']=-.0021
                    self.assertFalse(filter_hits([hit],c,x)[0]['allowed_contact'])

    def test_deep_far_and_wrong_side_contacts_rejected(self):
        c=self.config();x=np.eye(4)
        self.assertFalse(permits(c,'left_hand_middle_0_link',.0011,[.04,0,0],x))
        self.assertFalse(permits(c,'left_hand_middle_0_link',.0001,[.04,.05,0],x))
        self.assertFalse(permits(c,'left_hand_middle_0_link',.0001,[-.04,0,0],x))

    def test_table_and_self_collisions_never_relaxed(self):
        c=self.config();hits=[dict(bodies=['left_hand_thumb_0_link','right_hand_thumb_0_link'],depth_m=.0001,point_world=[.04,0,0],allowed_contact=False),
            dict(bodies=['left_hand_thumb_0_link','/World/Table'],depth_m=.0001,point_world=[.04,0,0],allowed_contact=False)]
        self.assertEqual(filter_hits(hits,c,np.eye(4)),hits)

    def test_displacement_not_contact_alone_determines_policy(self):
        c=self.config();q=np.tile([0.,0.,0.,1.],(3,1));position=np.zeros((3,3))
        trace=dict(object_position_world_m=position,object_quaternion_xyzw=q,
            object_linear_velocity_m_s=np.zeros((3,3)),object_angular_velocity_rad_s=np.zeros((3,3)),timestamp_s=np.arange(3)/240.)
        contact=dict(timestamp_s=0.,robot_link='left_hand_thumb_0_link',separation_m=-.0005,point_world_m=[.04,0,0],force_n=1.)
        self.assertTrue(measured_assessment(c,[contact],trace,2,[0,0,0,0,0,0,1])['accepted'])
        position[1,0]=.006
        self.assertFalse(measured_assessment(c,[],trace,2,[0,0,0,0,0,0,1])['accepted'])

    def test_exact_four_recipes_and_same_candidate_counts(self):
        self.assertEqual(len(RECIPES),4)
        from tools.contact_coordination.interaction_candidates import acquisition_bank
        base=ROOT/'outputs/source_guided_rrt_rebuild/20260911T035757Z'
        phase=read(base/'source_phase/GoPark_20260820_152058/PHASE_RECORD.json');cal=read(base/'target_repair/CONTACT_CALIBRATION.json')
        for recipe in RECIPES.values():
            bank=acquisition_bank(phase,cal,[.1175,.0725,.0775],{k:recipe[k] for k in DEFAULT})
            self.assertEqual(len(bank),5);self.assertEqual(len({np.asarray(v['task_space_target']).tobytes() for v in bank}),5)

    def test_current_default_matches_frozen_all40_targets(self):
        from tools.contact_coordination.interaction_candidates import acquisition_bank
        base=ROOT/'outputs/source_guided_rrt_rebuild/20260911T035757Z'
        path=base/'FROZEN_SOURCE_GUIDED_CODE/tools/contact_coordination/interaction_candidates.py'
        spec=importlib.util.spec_from_file_location('tools.contact_coordination._frozen_practical_fixture',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        cal=read(base/'target_repair/CONTACT_CALIBRATION.json')
        for sid in read(base/'SPLIT_CONTRACT.json')['authorized_training_source_ids']:
            phase=read(base/'source_phase'/sid/'PHASE_RECORD.json')
            before=old.acquisition_bank(phase,cal,[.1175,.0725,.0775]);after=acquisition_bank(phase,cal,[.1175,.0725,.0775])
            for a,b in zip(before,after,strict=True):
                self.assertEqual(a['candidate_id'],b['candidate_id']);self.assertEqual(a['score'],b['score'])
                for name in a['targets']:np.testing.assert_array_equal(a['targets'][name],b['targets'][name])

    def test_bounds_and_unknown_parameters_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);values=dict(DEFAULT,incidental_penetration_m=.0021)
            atomic_json(p/'PRACTICAL_PARAMETERS.json',dict(values=values))
            with self.assertRaises(ValueError):parameters(p)

    def test_real_hull_approach_patch_accepts_shallow_not_deep_contact(self):
        from tools.contact_coordination.runtime_hulls import Checker
        from tools.contact_coordination.planning_kinematics import G1Kinematics
        from tools.contact_coordination.source_phase import COMMON
        from tools.doll_handoff_retargeting.common import load_common_config,load_scene
        base=ROOT/'outputs/source_guided_rrt_rebuild/20260911T035757Z'
        plan=read(base/'GOLDEN_PLANNING.json')['rows'][0];sid=plan['source_id']
        result=read(Path(plan['context'])/'prototype'/sid/'morphology_acquisition_v4/PLAN_RESULT.json')
        a=np.asarray(next(v for v in result['phases'] if v['phase']=='PREGRASP')['connecting_q'])[-1]
        b=np.asarray(next(v for v in result['phases'] if v['phase']=='LEFT_ACQUISITION')['connecting_q'])[-1]
        cal=read(base/'target_repair/CONTACT_CALIBRATION.json');cfg=load_common_config(COMMON)
        with tempfile.TemporaryDirectory() as directory:
            import shutil
            out=Path(directory);shutil.copytree(base/'target_repair',out/'target_repair')
            atomic_json(out/'PRACTICAL_PARAMETERS.json',dict(values={k:RECIPES['NEAR_APPROACH'][k] for k in DEFAULT}))
            checker=Checker(G1Kinematics(cfg,load_scene(cfg)),out,cal['joint_names'])
            x=np.asarray(read(base/'source_phase'/sid/'PHASE_RECORD.json')['initial_object_pose_world'])
            fingers=cal['contacts']['pregrasp']['commanded_finger_q'];allowed=[];deep=[]
            for u in np.linspace(0,2,49):
                for hit in checker.protected_object_contacts(np.r_[a+u*(b-a),fingers],x):
                    if hit['allowed_contact']:allowed.append(hit)
                    if hit['signed_distance_m']<-.002:deep.append(hit)
            self.assertTrue(allowed,'Real intended acquisition-face contacts must be admitted')
            self.assertTrue(deep,'Fixture must actually encounter deep penetration')
            self.assertTrue(all(h['actual_penetration_m']<=.002+1e-9 for h in allowed))
            self.assertTrue(all(not h['allowed_contact'] for h in deep))


if __name__=='__main__':unittest.main()
