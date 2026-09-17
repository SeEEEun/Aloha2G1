"""Small regression/property checks for the repaired phase/frame integration."""
import ast,unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT,read
from .morphology_repair import wrist_target
from .phase_clock_runtime import PhaseClockDex3
from .phase_physics import instrument,lift_allowed
from tools import run_direct_physical_execution_isaac as engine
from tools.common_execution_layer import Dex3Primitive,ExecutionSnapshot,read_json
from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
from tools.direct_physical_execution_layer import authoritative_joint_limits


class RepairTests(unittest.TestCase):
    def test_unreachable_endpoint_is_pruned_before_geometry_refinement(self):
        from .planner import realize_goal_region
        class LimitedFK:
            arm_limits=np.tile([-.5,.5],(14,1))
            def assign(self,q):self.q=np.asarray(q)
            def wrist_pose(self,side):
                t=np.eye(4);t[0,3]=self.q[0];return t
            def model_to_world_position(self,v):return v
            def model_to_world_rotation(self,v):return v
        calls=[]
        def geometry(a,b,goal):calls.append(b.copy());return dict(valid=True)
        target=np.eye(4);target[0,3]=2.
        goal=dict(name='UNREACHABLE',active_hands=['left'],wrist_pose_world={'left':target},position_tolerance_m=.003,orientation_tolerance_rad=.05)
        config=dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,max_nfev_per_seed_per_goal=20)
        result=realize_goal_region(LimitedFK(),[goal],np.zeros(14),config,geometry)
        self.assertFalse(result['valid']);self.assertEqual(calls,[])
        self.assertEqual(result['attempts'][0]['result']['phases'][0]['causal_failure'],'NO_IK')

    def test_receiver_departure_precedes_sole_ownership_but_needs_current_support(self):
        from types import SimpleNamespace
        from .phase_physics import receiver_departure_allowed,right_transport_allowed
        primitive=SimpleNamespace(release_frames=45,right_verification_frames=3,right_retention_frames=30)
        controller=SimpleNamespace(primitive=primitive,right_support_frame=10,left_release_frame=10,
            giver_release_duration_frames=90,grasp_confirmed_frame={'right':5},right_three_counter=80,
            right_owned_frame=None,right_retention_counter=0)
        snapshot=SimpleNamespace(previous_control_frame_support={'right_two_table_free':True,'giver_contact_present':True})
        self.assertTrue(receiver_departure_allowed(controller,snapshot,100))
        self.assertFalse(right_transport_allowed(controller))
        self.assertFalse(receiver_departure_allowed(controller,snapshot,50))
        snapshot.previous_control_frame_support['right_two_table_free']=False
        self.assertFalse(receiver_departure_allowed(controller,snapshot,100))
        controller.right_owned_frame=130;controller.right_retention_counter=30
        self.assertTrue(right_transport_allowed(controller))

    def test_joint_search_never_admits_a_colliding_curved_kinematics_chord(self):
        from .planner import realize_phase_goals
        class CurvedFK:
            arm_limits=np.tile([-2.,2.],(14,1))
            def assign(self,q):self.q=np.asarray(q)
            def wrist_pose(self,side):
                t=np.eye(4);t[:2,3]=[self.q[0],self.q[1]+self.q[0]**2];return t
            def model_to_world_position(self,v):return v
            def model_to_world_rotation(self,v):return v
        def validate(a,b,goal):
            u=np.linspace(0,1,21);path=a+(b-a)*u[:,None]
            error=float(np.max(np.abs(path[:,1]+path[:,0]**2)))
            bad=[] if error<=.015 else [dict(geoms=['hand','wall'],depth_m=error-.015)]
            return dict(valid=not bad,forbidden_contacts=bad)
        target=np.eye(4);target[0,3]=1.
        goal=dict(name='FREE_RETREAT',active_hands=['left'],wrist_pose_world={'left':target},
                  position_tolerance_m=.003,orientation_tolerance_rad=.05,cartesian_connection_steps=6)
        cfg=dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,max_nfev_per_seed_per_goal=40)
        result=realize_phase_goals(CurvedFK(),[goal],np.zeros(14),cfg,validate)
        phase=result['phases'][0]
        # A very narrow nonlinear tube need not be solved by a finite RRT
        # budget. It must reject the chord and never fall back to unchecked IK.
        if phase['admissible']:
            knots=np.asarray(phase['connecting_q'])
            self.assertFalse(validate(knots[0],knots[-1],goal)['valid'])
            self.assertTrue(all(validate(a,b,goal)['valid'] for a,b in zip(knots[:-1],knots[1:])))
            self.assertEqual(phase['connection_method'],'COLLISION_AWARE_GLOBAL_PLANNER')
        else:self.assertEqual(phase['causal_failure'],'NO_CONNECTING_PATH')
        blocked=realize_phase_goals(CurvedFK(),[goal],np.zeros(14),cfg,lambda a,b,g:dict(valid=False))
        self.assertFalse(blocked['phases'][0]['admissible'])

    def test_soft_spatial_prior_still_optimizes_and_requires_valid_connection(self):
        from .planner import realize_phase_goals
        from .source_phase import COMMON
        from tools.doll_handoff_retargeting.common import load_common_config,load_scene
        from tools.doll_handoff_retargeting.models import G1Kinematics
        from .morphology_repair import world_wrist
        c=load_common_config(COMMON);g=G1Kinematics(c,load_scene(c))
        q=np.asarray(read(ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json')['g1_14_arm_initial_q_rad'])
        g.assign(q);target=world_wrist(g,'left');target[:3,3]+=[2.,0.,0.]
        goal=dict(name='SOFT_PRIOR_EXAMPLE',active_hands=['left'],wrist_pose_world={'left':target},
                  hard_pose_constraint=False,position_tolerance_m=.003,orientation_tolerance_rad=.05)
        cfg=dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,max_nfev_per_seed_per_goal=15)
        passed=realize_phase_goals(g,[goal],q,cfg,candidate_validator=lambda a,b,goal:dict(valid=True))
        ph=passed['phases'][0];self.assertTrue(ph['admissible']);self.assertFalse(ph['pose_prior_within_tolerance'])
        self.assertGreater(np.linalg.norm(passed['q'][-1]-q),.1)
        rejected=realize_phase_goals(g,[goal],q,cfg,candidate_validator=lambda a,b,goal:dict(valid=False))
        self.assertFalse(rejected['phases'][0]['admissible'])

    def test_carried_object_environment_collision_is_checked_explicitly(self):
        import tempfile,shutil
        from pathlib import Path
        from .runtime_hulls import Checker
        from .source_phase import COMMON
        from tools.doll_handoff_retargeting.common import load_common_config,load_scene
        from tools.doll_handoff_retargeting.models import G1Kinematics
        c=load_common_config(COMMON);g=G1Kinematics(c,load_scene(c))
        with tempfile.TemporaryDirectory(prefix='hybrid_collision_test_') as temporary:
            out=Path(temporary)
            shutil.copytree(self.out/'target_repair/runtime_bin150',out/'target_repair/runtime_bin150')
            checker=Checker(g,out,self.cal['joint_names'])
            arms=read(ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json')['g1_14_arm_initial_q_rad']
            q=np.r_[arms,self.cal['contacts']['pregrasp']['commanded_finger_q']]
            x=np.eye(4);x[:3,3]=[.827,.0998,.9]
            collisions=checker.check(q,x,('right',),object_environment=True)
            self.assertTrue(any('hybrid_object' in h['geoms'] and any('TrashBin/' in b for b in h['bodies']) and not h['allowed_contact'] for h in collisions))
            x[2,3]=1.2
            collisions=checker.check(q,x,('right',),object_environment=True)
            self.assertFalse(any('hybrid_object' in h['geoms'] and any('TrashBin/' in b for b in h['bodies']) for h in collisions))

    def test_release_delays_only_obstructed_nonopposing_digit_until_clearance(self):
        from .phase_clock_runtime import receiver_release_target
        start=np.array([.2,.8,.4,-.2,-1.1,-.3,-1.2]);opened=np.zeros(7)
        path=np.array([receiver_release_target(start,opened,t,0,45,['middle'],135) for t in range(180)])
        simultaneous=np.array([receiver_release_target(start,opened,t,0,45) for t in range(45)])
        np.testing.assert_array_equal(path[0],start)
        np.testing.assert_allclose(path[:45,[0,1,2,5,6]],simultaneous[:,[0,1,2,5,6]])
        np.testing.assert_array_equal(path[134,3:5],start[3:5])
        np.testing.assert_array_equal(path[134,[0,1,2,5,6]],opened[[0,1,2,5,6]])
        np.testing.assert_allclose(path[135:,3:5],simultaneous[:,3:5])
        np.testing.assert_array_equal(path[-1],opened)

    def setUp(self):
        self.out=ROOT/'outputs/contact_coordination_hybrid/20260907T083735Z'
        self.cal=read(self.out/'target_repair/CONTACT_CALIBRATION.json')

    def test_fixed_tool_applied_once_and_world_equivariance(self):
        c=self.cal['contacts']['pickup'];x=np.eye(4);x[:3,3]=[.2,.1,.9]
        t=wrist_target(x,c)
        np.testing.assert_allclose(t@c['T_wrist_H']@c['T_HO'],x,atol=1e-12)
        world=np.eye(4);world[:3,:3]=Rotation.from_rotvec([.2,-.4,.7]).as_matrix();world[:3,3]=[2,3,4]
        np.testing.assert_allclose(wrist_target(world@x,c),world@t,atol=1e-12)

    def test_loaded_named_fk_matches_independent_runtime_body_poses(self):
        from .morphology_repair import assign_measured,world_wrist
        from .source_phase import COMMON
        from tools.doll_handoff_retargeting.common import load_common_config,load_scene
        from tools.doll_handoff_retargeting.models import G1Kinematics
        p=self.out/'prototype/GoPark_20260820_152058/full_task_connection/ead64a3c8365/geometry_telemetry_diagnostic/event_log.npz'
        trace=dict(np.load(p));config=load_common_config(COMMON);g=G1Kinematics(config,load_scene(config))
        for frame in [0,300,900]:
            i=int(np.flatnonzero(trace['control_frame']==frame)[-1]);assign_measured(g,trace,i)
            for side in ['left','right']:
                body=list(trace['body_names']).index(side+'_wrist_yaw_link');actual=world_wrist(g,side)
                np.testing.assert_allclose(actual[:3,3],trace['body_position_world_m'][i,body],atol=1e-6)
                rotation=Rotation.from_quat(trace['body_quaternion_xyzw'][i,body]).as_matrix()
                self.assertLess(Rotation.from_matrix(actual[:3,:3].T@rotation).magnitude(),1e-5)

    def test_incomplete_calibration_is_rejected(self):
        from .morphology_repair import extract_calibration
        with self.assertRaisesRegex(ValueError,'CALIBRATION_RECAPTURE_REQUIRED'):
            extract_calibration(None)

    def test_current_cross_factor_nonredundancy_and_world_invariance(self):
        from .handoff_repair import coupling_residual
        a=np.eye(4);b=np.eye(4);b[:3,3]=[.04,-.02,.01];b[:3,:3]=Rotation.from_rotvec([.1,.2,-.3]).as_matrix()
        on=coupling_residual(a,b,True);off=coupling_residual(a,b,False)
        np.testing.assert_array_equal(off,np.zeros(6));self.assertGreater(np.linalg.norm(on),0)
        world=np.eye(4);world[:3,:3]=Rotation.from_euler('xyz',[.2,-.1,.4]).as_matrix();world[:3,3]=[1,2,3]
        self.assertAlmostEqual(float(on@on),float(coupling_residual(world@a,world@b,True)@coupling_residual(world@a,world@b,True)))
        np.testing.assert_allclose(coupling_residual(a,a,True),coupling_residual(a,a,False))

    def test_relevant_source_scene_pose_changes_target_reproducibly(self):
        c=self.cal['contacts']['acquisition_intent'];p=read(self.out/'source_phase/GoPark_20260820_152058/PHASE_RECORD.json');x=np.asarray(p['initial_object_pose_world']);a=wrist_target(x,c)
        np.testing.assert_array_equal(a,wrist_target(x,c))
        changed=x.copy();changed[:3,3]+=[.01,-.02,0]
        np.testing.assert_allclose(wrist_target(changed,c)[:3,3]-a[:3,3],[.01,-.02,0],atol=1e-12)
        changed=x.copy();changed[:3,:3]=Rotation.from_euler('z',.1).as_matrix()@x[:3,:3]
        self.assertGreater(np.linalg.norm(wrist_target(changed,c)-a),.01)

    def test_single_prelift_sample_keeps_uncertainty_explicit(self):
        p=read(self.out/'source_phase/GoPark_20260820_155934/PHASE_RECORD.json')
        self.assertEqual(p['prelift_relation_evidence']['sample_count'],1)
        self.assertFalse(p['prelift_relation_evidence']['spread_estimable'])
        self.assertEqual(p['observed_source_contacts'],'UNKNOWN')

    def test_bin_export_uses_exact_runtime_150mm_beveled_function(self):
        r=read(self.out/'target_repair/runtime_bin150/RUNTIME_HULLS.json')
        self.assertEqual(r['bin_geometry']['external_height_m'],.150)
        self.assertEqual(r['bin_geometry']['rim_bevel_m'],.003)
        v=dict(np.load(self.out/'target_repair/runtime_bin150/RUNTIME_HULL_VERTICES.npz'))
        walls=[x for x in r['rows'] if x['kind']=='environment_convex']
        self.assertEqual(len(walls),4)
        for wall in walls:self.assertAlmostEqual(v[wall['key']][:,2].max(),.945,places=6)

    def test_instrumentation_does_not_shadow_summary_writer(self):
        source,_=instrument(engine.ENGINE.read_text());tree=ast.parse(source)
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        imports=[a for n in ast.walk(main) if isinstance(n,ast.ImportFrom) for a in n.names]
        self.assertFalse(any(a.name=='atomic_json' and a.asname is None for a in imports))

    def test_giver_withdrawal_precedes_sole_ownership_gate(self):
        from types import SimpleNamespace
        from tools.contact_coordination.phase_physics import giver_clearance_allowed
        ctrl=SimpleNamespace(right_support_frame=10,left_release_frame=12,right_owned_frame=None,primitive=SimpleNamespace(release_frames=44))
        self.assertFalse(giver_clearance_allowed(ctrl,54))
        self.assertTrue(giver_clearance_allowed(ctrl,55))
        ctrl.right_support_frame=None
        self.assertFalse(giver_clearance_allowed(ctrl,100))

    def test_thumb_first_release_preserves_endpoints_and_rates(self):
        from .phase_clock_runtime import giver_release_target
        from .phase_physics import giver_clearance_allowed
        from types import SimpleNamespace
        start=np.array([.1,.8,.2,-.1,-1.2,-.3,-1.1]);opened=np.zeros(7);frames=45
        path=np.array([giver_release_target(start,opened,i,frames,'thumb_first') for i in range(90)])
        simultaneous=np.array([giver_release_target(start,opened,i,frames,'simultaneous') for i in range(45)])
        np.testing.assert_array_equal(path[0],start)
        np.testing.assert_array_equal(path[44,:3],opened[:3])
        np.testing.assert_array_equal(path[44,3:],start[3:])
        np.testing.assert_array_equal(path[-1],opened)
        np.testing.assert_allclose(path[:45,:3],simultaneous[:,:3])
        np.testing.assert_allclose(path[45:,3:],simultaneous[:,3:])
        ctrl=SimpleNamespace(right_support_frame=10,left_release_frame=12,primitive=SimpleNamespace(release_frames=45),giver_release_duration_frames=90)
        self.assertFalse(giver_clearance_allowed(ctrl,56))
        self.assertTrue(giver_clearance_allowed(ctrl,101))

    def test_giver_release_waits_for_phase_but_does_not_require_sole_ownership(self):
        from .execution_timing import common_primitive
        primitive=common_primitive();withdraw=50+primitive.release_frames+5;owned=withdraw+29
        lo,hi,names=authoritative_joint_limits(read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json'))
        q=np.zeros((owned+3,28));q[:,:14]=read(ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json')['g1_14_arm_initial_q_rad']
        ctrl=PhaseClockDex3(primitive,['HANDOFF_INTENT']*len(q),q,q,'ACT-A40',lo[:14],hi[:14],lo[14:],hi[14:],release_not_before_frame=50)
        ctrl.left_trigger=0;ctrl.right_trigger=0;ctrl.left_start_q=ctrl.left_full_close.copy();ctrl.right_start_q=ctrl.right_full_close.copy()
        ctrl.grasp_confirmed_frame={'left':0,'right':0};ctrl.grasp_state={'left':'HOLD','right':'HOLD'};ctrl.grasp_confirmed_object_z_m={'left':1.,'right':1.}
        state=q[0].copy();state[14:]=np.r_[ctrl.left_full_close,ctrl.right_full_close];x=np.eye(4);x[2,3]=1.
        for frame in range(owned+1):
            snapshot=ExecutionSnapshot(state,x,{'left':x,'right':x},{s:{d:1. for d in ['thumb','index','middle']} for s in ['left','right']},0.)
            if frame>=withdraw:
                snapshot=ExecutionSnapshot(state,x,{'left':x,'right':x},{s:{d:(0. if s=='left' else 1.) for d in ['thumb','index','middle']} for s in ['left','right']},0.)
            if frame==50:
                from .phase_physics import receiver_candidate_ready
                self.assertTrue(receiver_candidate_ready(ctrl,snapshot))
                missing=ExecutionSnapshot(state,x,{'left':x,'right':x},{s:{d:0. for d in ['thumb','index','middle']} for s in ['left','right']},0.)
                self.assertFalse(receiver_candidate_ready(ctrl,missing))
            ctrl.step(frame,snapshot)
            if frame<50:self.assertIsNone(ctrl.left_release_frame)
            if frame<owned:self.assertIsNone(ctrl.right_owned_frame)
        self.assertEqual(ctrl.left_release_frame,50)
        self.assertEqual(ctrl.right_owned_frame,owned)
        self.assertEqual(ctrl.primitive,primitive)
        from .phase_physics import right_transport_allowed
        self.assertTrue(right_transport_allowed(ctrl))
        lost=ExecutionSnapshot(state,x,{'left':x,'right':x},{s:{d:0. for d in ['thumb','index','middle']} for s in ['left','right']},0.)
        ctrl.step(owned+1,lost)
        self.assertEqual(ctrl.right_owned_frame,owned)
        self.assertFalse(right_transport_allowed(ctrl))
        recontact=ExecutionSnapshot(state,x,{'left':x,'right':x},{s:{d:1. for d in ['thumb','index','middle']} for s in ['left','right']},0.)
        ctrl.step(owned+2,recontact)
        self.assertEqual(ctrl.right_owned_frame,owned)
        self.assertFalse(right_transport_allowed(ctrl))

    def test_capture_to_carry_relation_is_preserved_across_contact_candidates(self):
        from .handoff_repair import carry_contact_for_candidate
        from .source_phase import pose
        reference=pose(Rotation.from_rotvec([.1,.2,-.3]).as_matrix(),[.04,-.02,.03])
        carry=pose(Rotation.from_rotvec([-.2,.1,.3]).as_matrix(),[.01,.05,.02])
        adapted=pose(Rotation.from_rotvec([.15,-.05,.1]).as_matrix(),[.003,.001,-.002])@reference
        calibration={'T_HO':carry,'measured_T_HO':carry.copy(),'acquisition_reference_T_HO':reference}
        result=carry_contact_for_candidate(calibration,{'T_HO':adapted})
        np.testing.assert_allclose(np.linalg.inv(adapted)@result['T_HO'],np.linalg.inv(reference)@carry,atol=1e-12)
        same=carry_contact_for_candidate(calibration,{'T_HO':reference})
        np.testing.assert_allclose(same['T_HO'],carry,atol=1e-12)
        np.testing.assert_array_equal(calibration['T_HO'],carry)

    def test_middle_unloading_keeps_opposing_giver_digits_until_second_phase(self):
        from .phase_clock_runtime import giver_release_target
        primitive=Dex3Primitive.from_frozen_dependencies(read_json(PHYSICS_CONFIG),read_json(PHYSICAL_ENVIRONMENT),read_json(COMMON_PHYSICAL_CONTROLLER))
        lo,hi,names=authoritative_joint_limits(read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json'))
        q=np.zeros((200,28));q[:,:14]=read(ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json')['g1_14_arm_initial_q_rad']
        ctrl=PhaseClockDex3(primitive,['HANDOFF_INTENT']*200,q,q,'ACT-A40',lo[:14],hi[:14],lo[14:],hi[14:],release_not_before_frame=0,giver_release_policy='middle_first')
        ctrl.right_support_frame=0;ctrl.left_release_frame=10;ctrl.left_release_start_q=ctrl.left_full_close.copy()
        n=ctrl.primitive.release_frames
        path=np.asarray([ctrl._left_target(10+i) for i in range(2*n)])
        np.testing.assert_allclose(path[n-1,[0,1,2,5,6]],ctrl.left_full_close[[0,1,2,5,6]])
        np.testing.assert_allclose(path[n-1,3:5],ctrl.left_open[3:5])
        np.testing.assert_allclose(path[-1],ctrl.left_open)
        common=np.asarray([giver_release_target(ctrl.left_full_close,ctrl.left_open,i,n,'simultaneous') for i in range(n)])
        self.assertLessEqual(float(np.max(np.abs(np.diff(path,axis=0)))),float(np.max(np.abs(np.diff(common,axis=0))))+1e-12)
        self.assertEqual(ctrl.giver_release_duration_frames,2*n)

    def test_release_geometry_uses_existing_finger_transition(self):
        from .phase_clock_runtime import release_geometry_fingers,giver_release_target
        start=np.linspace(-.3,.2,7);opened=np.zeros(7);right=np.ones(7)*.1
        for t in (0,11,22,44,60):
            fingers,fraction=release_geometry_fingers(start,right,opened,right,t,45)
            np.testing.assert_allclose(fingers[:7],giver_release_target(start,opened,t,45,'simultaneous'))
            np.testing.assert_allclose(fingers[7:],right)
            self.assertGreaterEqual(fraction,0);self.assertLessEqual(fraction,1)

    def test_coordinated_middle_release_has_stationary_unload_and_bounded_motion(self):
        from .phase_clock_runtime import release_arm_fraction,release_geometry_fingers
        n=145;base=45;u=np.asarray([release_arm_fraction(i,n,'middle_first',base) for i in range(n+1)])
        np.testing.assert_array_equal(u[:base+1],0.)
        self.assertEqual(u[-1],1.);self.assertTrue(np.all(np.diff(u)>=0))
        # Delaying the existing100-frame quintic does not increase its speed.
        expected=np.asarray([release_arm_fraction(i,n-base,'simultaneous',base) for i in range(n-base+1)])
        np.testing.assert_allclose(u[base:],expected)
        fingers,fraction=release_geometry_fingers(np.ones(7),np.ones(7),np.zeros(7),np.zeros(7),44,n,'middle_first',base)
        np.testing.assert_allclose(fingers[[0,1,2,5,6]],1.)
        np.testing.assert_allclose(fingers[3:5],0.);self.assertEqual(fraction,0.)

    def test_release_clearance_cannot_escape_the_declared_bin_region(self):
        from .full_task_plan import planar_release_clearance
        normals=[[-1.,0.,0.],[0.,1.,0.]];depths=[.006,.002]
        delta=planar_release_clearance(normals,depths,[-.015,-.02],[.03,.02],.0015)
        np.testing.assert_allclose(delta,[-.0075,.0035],atol=1e-9)
        self.assertIsNone(planar_release_clearance(normals,depths,[0.,-.02],[.03,.02],.0015))
        self.assertIsNone(planar_release_clearance([[0.,0.,1.]],[.006],[-.1,-.1],[.1,.1],.0015))

    def test_loaded_fk_goal_returns_commands_without_applying_bias_twice(self):
        import mujoco
        from .planner import realize_phase_goals
        from .source_phase import COMMON
        from .morphology_repair import world_wrist
        from tools.doll_handoff_retargeting.common import load_common_config,load_scene
        from tools.doll_handoff_retargeting.models import G1Kinematics
        cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg))
        q=np.asarray(read(ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json')['g1_14_arm_initial_q_rad'])
        bias=np.linspace(-.01,.01,14);g.assign(q+bias)
        joint=mujoco.mj_name2id(g.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_pitch_joint')
        g.data.qpos[g.model.jnt_qposadr[joint]]=.015;mujoco.mj_forward(g.model,g.data)
        goal=dict(name='RIGHT_CARRY',active_hands=['right'],wrist_pose_world={'right':world_wrist(g,'right')},preferred_q=q,
                  position_tolerance_m=.003,orientation_tolerance_rad=.05,
                  kinematic_calibration=dict(arm_joint_offset_rad=bias,uncommanded_joint_positions_rad={'waist_pitch_joint':.015}))
        result=realize_phase_goals(g,[goal],q,dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,max_nfev_per_seed_per_goal=120))
        self.assertTrue(result['phases'][0]['admissible'])
        np.testing.assert_array_equal(result['q'][-1],q)
        self.assertLess(result['phases'][0]['selected_errors']['right']['position_m'],1e-12)

    def test_loaded_geometry_keeps_raw_nonobject_safety_and_raw_commands(self):
        from .loaded_contact_geometry import check
        from types import SimpleNamespace
        class FakeChecker:
            def __init__(self):
                self.calls=[];self.g1=SimpleNamespace(arm_limits=np.tile([-10.,10.],(14,1)))
            def check(self,q,x,allowed,object_environment=False):
                assert object_environment
                self.calls.append(q.copy())
                return ([dict(geoms=['arm','torso'],allowed_contact=False),dict(geoms=['palm','hybrid_object'],allowed_contact=False)] if len(self.calls)==1 else [])
        ch=FakeChecker();q=np.zeros(28);audit=[]
        hits=check(ch,q,np.eye(4),('right',),dict(arm_measured_minus_command_rad=np.ones(14)*.001),verify_relation=False,audit=audit)
        np.testing.assert_array_equal(q,0.)
        self.assertEqual(hits[0]['geoms'],['arm','torso']);self.assertFalse(hits[0]['allowed_contact'])
        self.assertEqual(len(audit[0]['raw_command_object_overlaps']),1)
        self.assertEqual(len(ch.calls),2)

    def test_receiver_retiming_preserves_endpoints_and_reduces_command_rate(self):
        from .execution_timing import common_primitive
        primitive=common_primitive();base=primitive.preshape_frames+primitive.close_frames
        lo,hi,names=authoritative_joint_limits(read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json'))
        q=np.zeros((base*2+2,28));q[:,:14]=read(ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json')['g1_14_arm_initial_q_rad']
        paths=[]
        for frames in (base,base*2):
            ctrl=PhaseClockDex3(primitive,['HANDOFF_INTENT']*len(q),q,q,'ACT-A40',lo[:14],hi[:14],lo[14:],hi[14:],release_not_before_frame=len(q)-1,receiver_transition_frames=frames)
            x=np.eye(4);x[2,3]=1.;path=[]
            for frame in range(frames+1):
                snap=ExecutionSnapshot(q[0],x,{'left':x,'right':x},{s:{d:0. for d in ['thumb','index','middle']} for s in ['left','right']},0.)
                path.append(ctrl.step(frame,snap).executed_command[21:].copy())
            paths.append(np.asarray(path));self.assertEqual(ctrl.primitive,primitive)
        np.testing.assert_allclose(paths[0][0],paths[1][0]);np.testing.assert_allclose(paths[0][-1],paths[1][-1])
        self.assertLess(np.max(np.abs(np.diff(paths[1],axis=0))),.51*np.max(np.abs(np.diff(paths[0],axis=0))))


if __name__=='__main__':unittest.main()
