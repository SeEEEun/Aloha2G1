"""Existing phase physics plus the common read-only target-domain camera."""
import argparse,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.contact_coordination import phase_physics
from tools import run_direct_physical_execution_isaac as engine


def instrument(source):
    result,counts=phase_physics.instrument(source)
    for old,new in [
        ('from tools.contact_coordination.phase_clock_runtime import build_runtime','from tools.contact_coordination.demonstration_observation import build_runtime'),
        ('    sim.reset()\n','    common_execution.create_camera()\n    sim.reset()\n'),
        ('    records: dict[str, list[Any]] = {','    common_execution.bind(sim, robot, doll)\n    records: dict[str, list[Any]] = {')]:
        assert result.count(old)==1,old
        result=result.replace(old,new)
    compile(result,str(engine.ENGINE),'exec');counts['read_only_pre_action_RGB_state_capture']=1
    return result,counts


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--direct-freeze-manifest',type=Path,required=True);p.add_argument('--qualification-mode',action='store_true');p.add_argument('--validate-patch-only',action='store_true');a,remaining=p.parse_known_args()
    source,counts=instrument(engine.ENGINE.read_text())
    if a.validate_patch_only:print(counts);return
    os.environ['DIRECT_EVAL35_FREEZE_MANIFEST']=str(a.direct_freeze_manifest.resolve())
    if a.qualification_mode:os.environ['DIRECT_EXECUTION_QUALIFICATION_MODE']='1'
    sys.argv=[str(engine.ENGINE),*remaining,'--enable_cameras','--dex3-hard-limit-contract',str(engine.AUTHORITATIVE_JOINT_CONTRACT),'--dex3-hard-limit-inset-rad',engine.DEX3_JOINT_STOP_INSET_RAD]
    ns=dict(__name__='__main__',__file__=str(engine.ENGINE),__package__=None);exec(compile(source,str(engine.ENGINE),'exec'),ns,ns)


if __name__=='__main__':main()
