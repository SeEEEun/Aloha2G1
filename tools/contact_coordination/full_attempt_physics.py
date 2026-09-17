"""Explicit full-horizon recording adapter over the existing PhysX runner."""
import argparse
import ast
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.contact_coordination import phase_physics
from tools import run_direct_physical_execution_isaac as engine


def instrument(source):
    result,counts=phase_physics.instrument(source)
    # Move only the five phase-admission checks into the runtime adapter.
    # The engine's own physics/legacy numerical checks are left in place.
    tree=ast.parse(result);lines=result.splitlines(keepends=True);spans=[]
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For)
              and isinstance(n.target,ast.Tuple) and isinstance(n.target.elts[0],ast.Name)
              and n.target.elts[0].id=='control_frame')
    for node in loop.body:
        if isinstance(node,ast.If) and any(isinstance(n,ast.Name) and n.id=='write_phase_stop' for n in ast.walk(node)):
            spans.append((node.lineno-1,node.end_lineno))
    assert len(spans)==5,spans
    for first,last in reversed(spans):lines[first:last]=[]
    result=''.join(lines)
    old='from tools.contact_coordination.phase_clock_runtime import build_runtime'
    assert result.count(old)==1
    result=result.replace(old,'from tools.contact_coordination.full_attempt_runtime import build_runtime')
    anchor='    records: dict[str, list[Any]] = {'
    assert result.count(anchor)==1
    result=result.replace(anchor,'''    from tools.contact_coordination.io import atomic_json as record_reset
    record_reset(output_dir / "PRE_COMMAND_SCENE.json", {
        "requested_object_pose_xyzw": requested_initial_pose,
        "actual_object_pose_xyzw": actual_initial_pose,
        "named_joint_names": names,
        "actual_named_q_rad": numpy(robot.data.joint_pos)[0, joint_ids].astype(np.float64),
        "actual_all_q_rad": numpy(robot.data.joint_pos)[0].astype(np.float64),
        "all_joint_names": isaac_names,
        "natural_initial_q_rad": common_execution.initial_q_rad,
        "controller_initial_frame": common_execution.delegate.current_frame,
        "controller_trace_length": len(common_execution.controller.trace),
        "phase_stop": common_execution.stop,
        "bin": runtime_bin,
        "physics_steps_before_first_command": 0,
    })
'''+anchor)
    anchor='            measured_qd = numpy(robot.data.joint_vel)[0, joint_ids].astype(np.float64)\n'
    assert result.count(anchor)==1
    result=result.replace(anchor,anchor+'''            from tools.contact_coordination.full_attempt_runtime import state_validity_reason
            invalid_reason = state_validity_reason(pose, object_velocity, measured, measured_qd, records["object_position_world_m"][-1] if records["object_position_world_m"] else actual_initial_pose[:3], config["gates"])
            if invalid_reason is not None:
                from tools.contact_coordination.io import atomic_npz as record_invalid, atomic_json as record_abort
                record_invalid(output_dir / "NUMERICAL_INVALID_STATE.npz", pose=pose, object_velocity=object_velocity, measured_q=measured, measured_qd=measured_qd)
                record_abort(output_dir / "NUMERICAL_ABORT.json", {"reason": invalid_reason, "control_frame": control_frame, "physics_step": physics_step, "last_valid_rows": len(records["measured_q_rad"]), "physical_diagnostics_unavailable_after_this_state": True})
                break
''')
    anchor='        executed_control_frames = control_frame + 1\n'
    assert result.count(anchor)==1
    result=result.replace(anchor,'''        if (output_dir / "NUMERICAL_ABORT.json").exists():
            break
'''+anchor)
    anchor='    arrays["timestamp_s"] = arrays["physics_step"].astype(np.float64) * dt\n'
    assert result.count(anchor)==1
    result=result.replace(anchor,anchor+'''    from tools.contact_coordination.full_attempt_runtime import record_linear_speed_telemetry
    record_linear_speed_telemetry(arrays, config["gates"], output_dir)
''')
    from tools.contact_coordination.pregrasp_protection import instrument as instrument_contacts
    result=instrument_contacts(result)
    compile(result,str(engine.ENGINE),'exec')
    counts.update(full_attempt_admission_adapter=5,precommand_scene_readback=1,nonfinite_measurement_guard=1)
    counts['nonterminal_linear_speed_telemetry']=1
    return result,counts


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--direct-freeze-manifest',type=Path,required=True)
    p.add_argument('--qualification-mode',action='store_true')
    p.add_argument('--validate-patch-only',action='store_true')
    a,remaining=p.parse_known_args();source,counts=instrument(engine.ENGINE.read_text())
    if a.validate_patch_only:print(counts);return
    os.environ['DIRECT_EVAL35_FREEZE_MANIFEST']=str(a.direct_freeze_manifest.resolve())
    if a.qualification_mode:os.environ['DIRECT_EXECUTION_QUALIFICATION_MODE']='1'
    sys.argv=[str(engine.ENGINE),*remaining,'--dex3-hard-limit-contract',str(engine.AUTHORITATIVE_JOINT_CONTRACT),
              '--dex3-hard-limit-inset-rad',engine.DEX3_JOINT_STOP_INSET_RAD]
    ns=dict(__name__='__main__',__file__=str(engine.ENGINE),__package__=None)
    exec(compile(source,str(engine.ENGINE),'exec'),ns,ns)


if __name__=='__main__':main()
