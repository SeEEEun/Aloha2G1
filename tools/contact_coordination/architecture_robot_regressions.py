"""Regression fixtures using the actual authored G1/Dex3/environment hulls."""
import numpy as np
from .io import read,atomic_json,atomic_npz,record
from .source_phase import COMMON
from .prototype import INITIAL
from .morphology_repair import world_wrist,wrist_target


def carried_object_test(out):
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from .planning_kinematics import G1Kinematics
    from .runtime_hulls import Checker
    from .planner import solve_endpoints
    from .joint_path_planner import Validator
    config=load_common_config(COMMON);scene=load_scene(config);g=G1Kinematics(config,scene)
    cal=read(out/'target_repair/CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json')
    contact=cal['contacts']['right_carry'];natural=np.asarray(read(INITIAL)['g1_14_arm_initial_q_rad'])
    g.assign(natural);relation=np.asarray(contact['T_wrist_H'])@np.asarray(contact['T_HO'])
    initial=world_wrist(g,'right')@relation
    fingers=np.asarray(cal['contacts']['pregrasp']['commanded_finger_q']);fingers[7:]=contact['measured_finger_q'][7:]
    ch=Checker(g,out,cal['joint_names']);tested=[];witness=None
    def object_pose(q):g.assign(q);return world_wrist(g,'right')@relation
    def robot_only(q):
        hits=ch.query(np.r_[q,fingers],object_pose(q),('right',),object_environment=True)
        return not any(not h['allowed_contact'] and 'hybrid_object' not in h['geoms'] for h in hits)
    def loaded(q):
        hits=ch.query(np.r_[q,fingers],object_pose(q),('right',),object_environment=True)
        return dict(valid=not any(not h['allowed_contact'] for h in hits),forbidden_contacts=[h for h in hits if not h['allowed_contact']])
    settings=dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,max_nfev_per_seed_per_goal=100)
    # Source-independent deterministic table/bin geometry fixture. Try the bin
    # side wall and tabletop center with the same actual right-carry relation.
    xy=[initial[:2,3],np.asarray(scene['bin']['center_world_xy_m']),np.array([.5,.18]),np.array([.45,.32])]
    for center in xy:
        if witness:break
        for z in (.84,.81,.79,.77,.74,.90):
            obj=initial.copy();obj[:2,3]=center;obj[2,3]=z
            goal=dict(name='CARRIED_OBJECT_REGRESSION',active_hands=['right'],wrist_pose_world={'right':wrist_target(obj,contact)},position_tolerance_m=.003,orientation_tolerance_rad=.05)
            result=solve_endpoints(g,goal,natural,settings,seed_postures=[contact['seed_q']])
            for candidate in result['phases'][0]['candidates']:
                if not candidate['goal_satisfied']:continue
                q=np.asarray(candidate['q']);rb=robot_only(q);full=loaded(q)
                object_env=[h for h in full['forbidden_contacts'] if 'hybrid_object' in h['geoms'] and any('environment_' in ch.rows.get(n,{}).get('kind','') for n in h['geoms'])]
                row=dict(target=obj,q=q,robot_valid=rb,loaded_valid=full['valid'],object_environment_hits=object_env)
                tested.append(row)
                if rb and object_env:
                    # Find a robot-only valid chord from natural towards the
                    # collision state. The held object remains FK-attached.
                    robot_checker=Validator(g.arm_limits[:,0],g.arm_limits[:,1],robot_only,max_checks=30000)
                    if robot_checker.edge(natural,q):
                        full_checker=Validator(g.arm_limits[:,0],g.arm_limits[:,1],loaded,max_checks=30000)
                        assert not full_checker.edge(natural,q)
                        witness=dict(row,q_start=natural,q_goal=q,
                            robot_only_edge_valid=True,carried_object_edge_rejected=True,
                            relation=relation,source='Actual calibrated right-carry transform; authored table/bin/object hulls',
                            robot_state_checks=robot_checker.calls,loaded_state_checks=full_checker.calls)
                        break
            if witness:break
    # A geometric regression is not limited by a prescribed wrist orientation.
    # Deterministic bounded configuration fixtures retain exactly the same
    # attached-object transform and require a valid robot-only connecting edge.
    if witness is None:
        rng=np.random.default_rng(4801)
        for index in range(2048):
            q=natural.copy();q[7:]=np.clip(natural[7:]+rng.normal(0,.7,7),g.arm_limits[7:,0]+1e-6,g.arm_limits[7:,1]-1e-6)
            full=loaded(q)
            object_env=[h for h in full['forbidden_contacts'] if 'hybrid_object' in h['geoms'] and any('environment_' in ch.rows.get(n,{}).get('kind','') for n in h['geoms'])]
            if not object_env or not robot_only(q):continue
            robot_checker=Validator(g.arm_limits[:,0],g.arm_limits[:,1],robot_only,max_checks=30000)
            if not robot_checker.edge(natural,q):continue
            full_checker=Validator(g.arm_limits[:,0],g.arm_limits[:,1],loaded,max_checks=30000)
            assert not full_checker.edge(natural,q)
            witness=dict(q_start=natural,q_goal=q,object_pose=object_pose(q),object_environment_hits=object_env,
                robot_only_edge_valid=True,carried_object_edge_rejected=True,relation=relation,
                source='Actual right-carry relation and authored hulls; fixed seed4801 configuration fixture',
                fixture_index=index,fixture_budget=2048,robot_state_checks=robot_checker.calls,loaded_state_checks=full_checker.calls)
            break
    path=out/'CARRIED_OBJECT_REGRESSION.json'
    atomic_json(path,dict(status='PASS' if witness else 'FAIL',witness=witness,tested=tested,
        checker=record(out/'target_repair/runtime_bin150/RUNTIME_HULLS.json'),physical_outcome_used=False))
    if witness is None:raise AssertionError('No deterministic actual-hull carried-object witness found')
    return [path]
