"""Thin causal policy instrumentation of the existing dynamic PhysX runner."""
import argparse,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.contact_coordination import calibration_capture
from tools import run_direct_physical_execution_isaac as engine


def instrument(source):
    result,counts=calibration_capture.instrument(source)
    substitutions=[
        ('from tools.direct_physical_execution_isaac_runtime import build_runtime',
         'from tools.contact_coordination.act_policy_runtime import build_runtime'),
        ('    sim.reset()\n','    common_execution.create_camera()\n    sim.reset()\n'),
        ('    from tools.contact_coordination.calibration_capture import capture_cooked\n    capture_cooked(stage, output_dir)\n',
         '    common_execution.bind(sim, robot, doll)\n'),
        ('        command = common_execution.step(control_frame, common_snapshot)\n',
         '        command = common_execution.step(control_frame, common_snapshot)\n'
         '        if common_execution.aborted is not None:\n'
         '            print("ACT_POLICY_SAFETY_ABORT", common_execution.aborted, flush=True)\n'
         '            break\n'),
        ('    common_execution.write_summary(output_dir / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json")\n',
         '    common_execution.write_summary(output_dir / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json")\n'
         '    if common_execution.aborted is not None and executed_control_frames == 0:\n'
         '        atomic_json(output_dir / "ZERO_TRANSITION_POLICY_ABORT.json", {"status": "POLICY_SAFETY_ABORT", "executed_control_frames": 0, "reason": common_execution.aborted})\n'
         '        return 0\n'),
        ('    return 0 if passed else 2\n',
         '    return 0  # ACT task outcome is scored independently of scripted control stages.\n'),
    ]
    for old,new in substitutions:
        assert result.count(old)==1,old
        result=result.replace(old,new)
    result=result.replace('"policy_or_checkpoint_used": False,','"policy_or_checkpoint_used": True,')
    compile(result,str(engine.ENGINE),'exec');counts['causal_ACT_interface']=1
    return result,counts


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--direct-freeze-manifest',type=Path,required=True)
    p.add_argument('--qualification-mode',action='store_true');p.add_argument('--validate-patch-only',action='store_true')
    a,remaining=p.parse_known_args();source,counts=instrument(engine.ENGINE.read_text())
    if a.validate_patch_only:print(counts);return
    os.environ['DIRECT_EVAL35_FREEZE_MANIFEST']=str(a.direct_freeze_manifest.resolve())
    os.environ['DIRECT_EXECUTION_QUALIFICATION_MODE']='1'
    sys.argv=[str(engine.ENGINE),*remaining,'--enable_cameras','--dex3-hard-limit-contract',str(engine.AUTHORITATIVE_JOINT_CONTRACT),
        '--dex3-hard-limit-inset-rad',engine.DEX3_JOINT_STOP_INSET_RAD]
    ns=dict(__name__='__main__',__file__=str(engine.ENGINE),__package__=None)
    try:exec(compile(source,str(engine.ENGINE),'exec'),ns,ns)
    finally:
        # Engine writes the trace/summary on normal completion or a safety stop.
        # Any exception remains infrastructure-invalid and is not a task result.
        pass


if __name__=='__main__':main()
