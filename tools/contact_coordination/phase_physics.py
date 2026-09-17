#!/usr/bin/env python3
"""Explicit shared phase-admission guard over the qualified physical runner.

No spatial correction. A failed pre-lift candidate stops before the arm lift.
The generated prefix already contains the common bounded close/retention wait.
"""
import argparse,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.contact_coordination import physics_capture
from tools import run_direct_physical_execution_isaac as engine


def lift_allowed(controller):
    return controller.grasp_confirmed_frame['left'] is not None and controller.grasp_state['left'] in {'PRELOAD','HOLD','LIFT'}


def giver_clearance_allowed(controller,frame):
    """Withdrawal completes giver release; sole ownership is checked after it."""
    return (controller.right_support_frame is not None and controller.left_release_frame is not None
        and frame-controller.left_release_frame>=getattr(controller,'giver_release_duration_frames',controller.primitive.release_frames)-1)


def right_transport_allowed(controller):
    return (controller.right_owned_frame is not None
            and controller.right_retention_counter>=controller.primitive.right_retention_frames)


def receiver_candidate_ready(controller,snapshot):
    """The same current support predicate used by the existing release logic."""
    p=controller.primitive
    support=(bool(snapshot.previous_control_frame_support['right_two_table_free'])
             if snapshot.previous_control_frame_support is not None
             else controller._two_digit(snapshot.digit_force_n['right'],p.force_threshold_n)
             and snapshot.table_force_n<=p.maximum_table_force_n)
    return (controller.grasp_confirmed_frame['right'] is not None and support
            and controller.right_three_counter>=p.right_verification_frames-1)


def receiver_departure_allowed(controller,snapshot,frame):
    """Planned clearance can establish sole ownership after giver opening.

    Require current sustained receiver support and completed giver opening.
    Residual giver contact does not prevent the already planned separating
    motion. Bin transport still requires verified right-only retention.
    """
    return giver_clearance_allowed(controller,frame) and receiver_candidate_ready(controller,snapshot)


def instrument(source):
    result,counts=physics_capture.instrument(source)
    old_import='from tools.direct_physical_execution_isaac_runtime import build_runtime'
    assert result.count(old_import)==1
    result=result.replace(old_import,'from tools.contact_coordination.phase_clock_runtime import build_runtime')
    anchor='    for control_frame, (command, label) in enumerate(zip(commands, stages, strict=True)):\n'
    assert result.count(anchor)==1
    addition='''        if label == "LIFT_5CM" and control_frame > 0 and stages[control_frame-1] != label:
            from tools.contact_coordination.phase_physics import lift_allowed
            if not lift_allowed(common_execution.controller):
                from tools.contact_coordination.io import atomic_json as write_phase_stop
                write_phase_stop(output_dir / "PHASE_STOP.json", {"phase": "LIFT_5CM", "reason": "NO_ACQUISITION_BEFORE_LIFT", "control_frame": control_frame, "arm_rescue": False})
                print("HYBRID_PHASE_STOP NO_ACQUISITION_BEFORE_LIFT", control_frame, flush=True)
                break
'''
    result=result.replace(anchor,anchor+addition)
    guard='''        if label == "GIVER_CLEARANCE" and control_frame > 0 and stages[control_frame-1] != label:
            from tools.contact_coordination.phase_physics import giver_clearance_allowed
            if not getattr(common_execution.controller, "coordinated_giver_release", False) and not giver_clearance_allowed(common_execution.controller, control_frame):
                from tools.contact_coordination.io import atomic_json as write_phase_stop
                write_phase_stop(output_dir / "PHASE_STOP.json", {"phase": "GIVER_CLEARANCE", "reason": "NO_RECEIVER_SUPPORT_OR_INCOMPLETE_GIVER_OPENING", "control_frame": control_frame, "arm_rescue": False})
                print("HYBRID_PHASE_STOP NO_RECEIVER_SUPPORT_OR_INCOMPLETE_GIVER_OPENING", control_frame, flush=True)
                break
        if label == "RIGHT_TRANSPORT" and control_frame > 0 and stages[control_frame-1] != label:
            from tools.contact_coordination.phase_physics import right_transport_allowed
            if not right_transport_allowed(common_execution.controller):
                from tools.contact_coordination.io import atomic_json as write_phase_stop
                write_phase_stop(output_dir / "PHASE_STOP.json", {"phase": "RIGHT_TRANSPORT", "reason": "NO_RIGHT_OWNERSHIP_AFTER_GIVER_WITHDRAWAL_AND_WAIT", "control_frame": control_frame, "arm_rescue": False})
                print("HYBRID_PHASE_STOP NO_RIGHT_OWNERSHIP_AFTER_GIVER_WITHDRAWAL", control_frame, flush=True)
                break
'''
    result=result.replace(anchor,anchor+guard)
    anchor='        command = common_execution.step(control_frame, common_snapshot)\n'
    assert result.count(anchor)==1
    result=result.replace(anchor,'''        if label == "GIVER_CLEARANCE" and control_frame > 0 and stages[control_frame-1] != label and getattr(common_execution.controller, "coordinated_giver_release", False):
            from tools.contact_coordination.phase_physics import receiver_candidate_ready
            if not receiver_candidate_ready(common_execution.controller, common_snapshot):
                from tools.contact_coordination.io import atomic_json as write_phase_stop
                write_phase_stop(output_dir / "PHASE_STOP.json", {"phase": "GIVER_CLEARANCE", "reason": "NO_CURRENT_RECEIVER_CANDIDATE_FOR_COORDINATED_RELEASE", "control_frame": control_frame, "arm_rescue": False})
                print("HYBRID_PHASE_STOP NO_CURRENT_RECEIVER_CANDIDATE", control_frame, flush=True)
                break
        if label == "RECEIVER_DEPARTURE" and control_frame > 0 and stages[control_frame-1] != label:
            from tools.contact_coordination.phase_physics import receiver_departure_allowed
            if not receiver_departure_allowed(common_execution.controller, common_snapshot, control_frame):
                from tools.contact_coordination.io import atomic_json as write_phase_stop
                write_phase_stop(output_dir / "PHASE_STOP.json", {"phase": "RECEIVER_DEPARTURE", "reason": "NO_CURRENT_RECEIVER_SUPPORT_OR_INCOMPLETE_GIVER_OPENING", "control_frame": control_frame, "arm_rescue": False})
                break
'''+anchor)
    # Read-only pose telemetry exposes any articulation/FK mismatch. Arm/hand
    # q alone cannot establish world geometry if an uncommanded joint moves.
    anchor='            "measured_q_rad",\n'
    assert result.count(anchor)==1
    result=result.replace(anchor,anchor+'            "all_measured_q_rad",\n            "body_position_world_m",\n            "body_quaternion_xyzw",\n')
    anchor='                "measured_q_rad": measured,\n'
    assert result.count(anchor)==1
    result=result.replace(anchor,anchor+'''                "all_measured_q_rad": numpy(robot.data.joint_pos)[0].astype(np.float64),
                "body_position_world_m": numpy(robot.data.body_pos_w)[0].astype(np.float64),
                "body_quaternion_xyzw": numpy(robot.data.body_quat_w)[0].astype(np.float64),
''')
    anchor='    arrays["joint_names"] = np.asarray(names)\n'
    assert result.count(anchor)==1
    result=result.replace(anchor,anchor+'    arrays["all_joint_names"] = np.asarray(isaac_names)\n    arrays["body_names"] = np.asarray(body_names)\n')
    compile(result,str(engine.ENGINE),'exec');counts['prelift_candidate_guard']=1
    counts['read_only_full_articulation_pose_capture']=1
    return result,counts


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--direct-freeze-manifest',type=Path,required=True);p.add_argument('--qualification-mode',action='store_true');p.add_argument('--validate-patch-only',action='store_true')
    args,remaining=p.parse_known_args();source,counts=instrument(engine.ENGINE.read_text())
    if args.validate_patch_only:print(counts);return
    os.environ['DIRECT_EVAL35_FREEZE_MANIFEST']=str(args.direct_freeze_manifest.resolve())
    if args.qualification_mode:os.environ['DIRECT_EXECUTION_QUALIFICATION_MODE']='1'
    sys.argv=[str(engine.ENGINE),*remaining,'--dex3-hard-limit-contract',str(engine.AUTHORITATIVE_JOINT_CONTRACT),'--dex3-hard-limit-inset-rad',engine.DEX3_JOINT_STOP_INSET_RAD]
    ns=dict(__name__='__main__',__file__=str(engine.ENGINE),__package__=None);exec(compile(source,str(engine.ENGINE),'exec'),ns,ns)


if __name__=='__main__':main()
