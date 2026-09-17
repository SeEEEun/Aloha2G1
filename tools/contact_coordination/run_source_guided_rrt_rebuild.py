"""Locked, atomic, resumable repair orchestration. Never starts calibration.

Each stage runs in a fresh process so repaired imports cannot remain cached.
An ordinary failure waits for a dependency change; it is never blindly retried.
"""
import argparse
import ast
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from .io import ROOT, OFFLINE, atomic_json, atomic_text, read, record

STAGES = ('source_prior','structural_tests','golden_regression','coverage8_planning','planning_audit','small_physics_validation','full_video_validation','final_gate')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def log(out, event, **values):
    with (out/'RUN_LOG.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(timestamp=now(),event=event,**values),sort_keys=True)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def scientific_input_paths(out):
    """Outer stage receipts must notice changed inputs before skipping work."""
    from .source_phase import COMMON
    from .scientific_cache import CALIBRATION_FILES
    from tools.common_execution_isaac_runtime import PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
    fixed=read(out/'COVERAGE8.json');cfg=read(COMMON)
    paths=[COMMON,Path(cfg['scene_config']),PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER,
        ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json',
        ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json',
        ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json',
        ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json',out/'ABC_STUDY.json',out/'ARCHITECTURE_REPAIR.json']
    paths += [Path(cfg['models'][name]) for name in ('g1_xml','g1_usd','dex3_whole_hand_geometry')]
    for sid in [fixed['golden'],*fixed['source_ids']]:
        paths += [out/'source_phase'/sid/name for name in ('PHASE_RECORD.json','SOURCE_PRIORS.npz','FUNCTIONAL_WRIST_PRIORS.npz')]
    paths += [out/'target_repair'/name for name in (*CALIBRATION_FILES,'TASK_FRAME_NORMALIZATION.json')]
    paths += [out/'target_repair/runtime_bin150'/name for name in ('RUNTIME_HULLS.json','RUNTIME_HULL_VERTICES.npz','RUNTIME_CHECK_MODEL.json','runtime_check.xml')]
    return paths


def dependencies(out,stage):
    cache_tree=ast.parse((ROOT/'tools/contact_coordination/scientific_cache.py').read_text())
    names=list(ast.literal_eval(next(n.value for n in cache_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODE_FILES' for t in n.targets))))
    groups={
        'source_prior':['source_motion_prior.py','interaction_candidates.py','source_phase.py','path_quality.py'],
        'structural_tests':names+['source_guided_robot_tests.py','tests/test_source_guided_rrt.py'],
        'golden_regression':names+['source_guided_coverage.py'],'coverage8_planning':names+['source_guided_coverage.py'],
        'planning_audit':names+['source_guided_reports.py'],
        'small_physics_validation':names+['source_guided_physics.py','source_guided_physics_cache.py','source_guided_execution_watchdog.py','architecture_physics.py','full_attempt.py','full_attempt_runtime.py','full_attempt_physics.py','pregrasp_protection.py','physical_failure_evidence.py','contact_command_geometry.py','abc_score.py'],
        'full_video_validation':['source_guided_videos.py','architecture_videos.py','full_attempt_replay.py','paper_replays.py','physical_failure_evidence.py'],
        'final_gate':names+['source_guided_reports.py','source_guided_videos.py','pregrasp_protection.py']}
    paths=[ROOT/'tools/contact_coordination'/n for n in groups[stage]]
    if stage=='structural_tests':paths.extend((ROOT/'tools/contact_coordination/tests').glob('test_source_guided*.py'))
    paths += [out/'SOURCE_GUIDED_RRT_REBUILD.json',out/'SPLIT_CONTRACT.json',out/'CALIBRATION_PARAMETERS.json',out/'COVERAGE8.json']
    paths += scientific_input_paths(out)
    if stage=='full_video_validation':paths.append(out/'VISUAL_INSPECTION.json')
    if stage=='final_gate':paths.append(out/'ARCHITECTURE_DEFECT_REVIEW.json')
    result=[record(p) for p in sorted(set(paths)) if p.is_file()]
    stages=ROOT/'tools/contact_coordination/source_guided_stages.py';tree=ast.parse(stages.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==stage)
    result.append(dict(path=str(stages),function=stage,sha256=hashlib.sha256(ast.dump(function).encode()).hexdigest()))
    if stage in ('golden_regression','coverage8_planning'):
        helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_planning')
        result.append(dict(path=str(stages),function='_planning',sha256=hashlib.sha256(ast.dump(helper).encode()).hexdigest()))
    return result


def signature(deps):
    return hashlib.sha256(json.dumps(deps,sort_keys=True).encode()).hexdigest()


def valid_receipt(path, deps, predecessor):
    if not path.exists(): return False
    value = read(path)
    try:
        return (value['status']=='PASS' and value['dependency_hash']==signature(deps)
            and value['predecessor']==predecessor and bool(value['artifacts'])
            and all(record(r['path'])==r for r in value['artifacts']))
    except (OSError,KeyError): return False


def status(out,stage,last,blocker,next_action,worker=None,ready='NO'):
    active=None if stage=='COMPLETE' else (worker or os.getpid())
    heartbeat=dict(timestamp=now(),stage=stage,substage=next_action,
        source_id=None,method=None,completed_work=last,remaining_work=list(STAGES[STAGES.index(stage):]) if stage in STAGES else [],
        worker_pid=active,supervisor_pid=os.getpid() if active else None,last_supervisor_pid=os.getpid())
    progress=out/'WORK_PROGRESS.json'
    if progress.exists():
        detail=read(progress)
        if detail.get('stage')==stage:
            heartbeat.update(detail)
            engine=Path(detail['engine_log']) if detail.get('engine_log') else None
            if engine and engine.exists():
                progress_lines=[line for line in engine.read_text(errors='replace').splitlines() if line.startswith('HYBRID_MEASURED_PROGRESS ')]
                if progress_lines:
                    _,frame,horizon=progress_lines[-1].split()
                    heartbeat.update(physics_control_frame=int(frame),physics_command_frames=int(horizon),
                        substage=f'PhysX measured frame {frame}/{horizon}; full horizon recording',
                        engine_log_last_modified_utc=datetime.datetime.fromtimestamp(engine.stat().st_mtime,datetime.timezone.utc).isoformat())
    atomic_json(out/'HEARTBEAT.json',heartbeat)
    atomic_text(out/'CURRENT_STATUS.md', '\n'.join([
        'CURRENT_STAGE: '+stage,'LAST_COMPLETED_STAGE: '+str(last),
        'CURRENT_BLOCKER: '+str(blocker),'NEXT_ACTION: '+next_action,
        'SOURCE_GUIDED_RRT_READY_FOR_CALIBRATION: '+ready,
        'ACTIVE_PROCESS: '+str(active)+(' (supervisor '+str(os.getpid())+')' if active else ' (normal completion)'),
        'LAST_HEARTBEAT: '+heartbeat['timestamp'],'Calibration started: NO','']))


def execute_stage(out,stage,result_file):
    if stage == 'audit':
        from .architecture_audit import run
        artifacts=run(out)
    else:
        from . import source_guided_stages
        artifacts=getattr(source_guided_stages,stage)(out)
    if not artifacts: raise RuntimeError('Stage did not produce evidence')
    atomic_json(result_file,dict(status='PASS',artifacts=[record(p) for p in artifacts]))


def run(out,resume=False,once=False):
    if not (out/'SOURCE_GUIDED_RRT_REBUILD.json').exists(): raise ValueError('Explicit repair run required')
    lock=(out/'.source_guided_rrt.lock').open('a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('An existing worker owns this repair run; duplicate refused.',flush=True); return 0
    receipts=out/'STAGE_RECEIPTS';receipts.mkdir(exist_ok=True)
    if list(receipts.glob('*.json')) and not resume: raise ValueError('Use --resume for existing receipts')
    command=f'{OFFLINE} -B -m tools.contact_coordination.run_source_guided_rrt_rebuild --run-dir {out} --resume'
    atomic_text(out/'RESUME_COMMAND.txt',f'cd {ROOT}\n{OFFLINE} -B -m tools.contact_coordination.launch_source_guided_rrt_rebuild --run-dir {out}\n\n# Direct foreground resume:\n{command}\n')
    atomic_json(out/'ACTIVE_PROCESS.json',dict(pid=os.getpid(),started=now(),command=command))
    failures={};last=None;supervisor_implementation=record(__file__)['sha256']
    log(out,'supervisor_started',pid=os.getpid(),resume=resume)
    while True:
        if record(__file__)['sha256']!=supervisor_implementation:
            # Fresh stage processes are insufficient when the supervisor's
            # dependency groups themselves change. Reload at a safe boundary.
            log(out,'supervisor_reload_at_stage_boundary',pid=os.getpid())
            os.execv(sys.executable,[sys.executable,'-B','-m','tools.contact_coordination.run_source_guided_rrt_rebuild',
                '--run-dir',str(out),'--resume',*(['--once'] if once else [])])
        predecessor=None;pending=None
        for stage in STAGES:
            deps=dependencies(out,stage);receipt=receipts/(stage+'.json')
            if valid_receipt(receipt,deps,predecessor):
                predecessor=record(receipt);last=stage;continue
            pending=(stage,deps,predecessor,receipt);break
        if pending is None:
            gate=read(out/'SOURCE_GUIDED_RRT_READY_FOR_CALIBRATION.json')
            if gate['SOURCE_GUIDED_RRT_READY_FOR_CALIBRATION']!='YES': raise RuntimeError('Final gate contradicts receipts')
            status(out,'COMPLETE','final_gate',None,'Repair frozen; stop BEFORE calibration',ready='YES')
            atomic_json(out/'ACTIVE_PROCESS.json',dict(pid=None,last_pid=os.getpid(),finished=now(),normal_completion=True,calibration_started=False))
            log(out,'repair_complete',calibration_started=False);return 0
        stage,deps,predecessor,receipt=pending;key=signature(deps)
        if failures.get(stage)==key:
            status(out,stage,last,'Stage failure; see stage log','Waiting for general code repair; unchanged retry disabled')
            if once:return 1
            time.sleep(10);continue
        if receipt.exists():
            archived=out/'INVALIDATED_RECEIPTS'/(stage+'_'+record(receipt)['sha256'][:12]+'.json')
            atomic_json(archived,read(receipt))
        log(out,'stage_started',stage=stage,dependency_hash=key)
        atomic_json(out/'DEPENDENCIES.json',dict(stage=stage,hash=key,files=deps))
        result=out/'stage_results'/(stage+'_'+key[:12]+'.json');result.parent.mkdir(exist_ok=True)
        stage_log=out/'stage_logs'/(stage+'_'+key[:12]+'.log');stage_log.parent.mkdir(exist_ok=True)
        args=[OFFLINE,'-B','-m','tools.contact_coordination.run_source_guided_rrt_rebuild',
              '--run-dir',str(out),'--stage',stage,'--result-file',str(result)]
        with stage_log.open('a') as stream:
            proc=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl',
                    SOURCE_GUIDED_RRT_AUDIT_LOG=str(out/'planner_api_calls'/(stage+'_'+key[:12]+'.jsonl'))))
            while proc.poll() is None:
                status(out,stage,last,None,'Running '+str(stage_log),proc.pid)
                if signature(dependencies(out,stage))!=key:
                    log(out,'terminate_invalidated_stage',stage=stage,worker_pid=proc.pid)
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    break
                time.sleep(5)
        # Never certify work done across a code/dependency mutation.
        if signature(dependencies(out,stage))!=key:
            log(out,'stage_invalidated_during_execution',stage=stage);continue
        if proc.returncode==0 and result.exists():
            value=read(result)
            atomic_json(receipt,dict(value,stage=stage,dependency_hash=key,dependencies=deps,
                predecessor=predecessor,completed_at=now(),log=record(stage_log)))
            last=stage;log(out,'stage_completed',stage=stage,receipt=record(receipt))
        else:
            failures[stage]=key
            log(out,'stage_failed',stage=stage,returncode=proc.returncode,log=str(stage_log))
            with (out/'DECISIONS.md').open('a') as stream:
                stream.write(f'\n- {now()} `{stage}` failed under `{key[:12]}`; preserved `{stage_log}`. No unchanged automatic retry.\n')
            status(out,stage,last,'Stage failed: '+str(stage_log),'Diagnose and repair general implementation')
            if once:return 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--once',action='store_true',help='Run until first failed stage for orchestration regression')
    parser.add_argument('--stage',choices=STAGES)
    parser.add_argument('--result-file',type=Path)
    args=parser.parse_args();out=args.run_dir.resolve()
    if args.stage:
        if args.result_file is None:parser.error('--stage requires --result-file')
        execute_stage(out,args.stage,args.result_file);return
    sys.exit(run(out,args.resume,args.once))


if __name__=='__main__': main()
