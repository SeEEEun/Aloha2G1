"""Locked, detached-capable, receipt-based practical calibration supervisor."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from .io import ROOT,OFFLINE,read,record,atomic_json,atomic_text
from .practical_study import setup,now,assert_backend

STAGES=('planning_sweep','recipe_ranking','top_recipe_physics','final_selection','freeze',
        'paper_A_train40','paper_B_train40','metrics','figures','videos','final_report')
MODULES=dict(planning_sweep='practical_planning.py',recipe_ranking='practical_planning.py',
    top_recipe_physics='practical_physics.py',final_selection='practical_finalize.py',freeze='practical_finalize.py',
    paper_A_train40='practical_paper.py',paper_B_train40='practical_paper.py',metrics='practical_results.py',
    figures='practical_results.py',videos='practical_videos.py',final_report='practical_results.py')


def log(out,event,**kw):
    with (out/'RUN_LOG.jsonl').open('a') as f:f.write(json.dumps(dict(timestamp=now(),event=event,**kw),sort_keys=True)+'\n');f.flush()


def dependencies(out,stage):
    manifest=read(out/'IMPLEMENTATION_FREEZE.json')
    values=[record(x['path']) for x in manifest['files']]
    paths=[out/'PRACTICAL_STUDY.json',out/'CALIBRATION_SEARCH_SPACE.json',out/'PREFLIGHT_REGRESSION.json',
        out/'ARCHITECTURE_BASELINE_MANIFEST.json',Path(__file__).parent/'practical_study.py',
        Path(__file__).parent/MODULES[stage]]
    if stage=='recipe_ranking':paths.append(Path(__file__).parent/'practical_diagnostics.py')
    if stage=='final_report':paths.append(out/'TRAIN40_VISUAL_INSPECTION.json')
    values.extend(record(p) if p.exists() else dict(path=str(p.resolve()),missing=True) for p in paths)
    return values


def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def valid_receipt(path,deps,previous):
    if not path.exists():return False
    r=read(path)
    try:return r['status']=='PASS' and r['dependency_hash']==signature(deps) and r['predecessor']==previous and bool(r['artifacts']) and all(record(x['path'])==x for x in r['artifacts'])
    except (KeyError,OSError):return False


def status(out,stage,last,action,worker=None,blocker=None):
    active=None if stage in ('COMPLETE','PLANNING_CALIBRATION_INSUFFICIENT') else worker or os.getpid()
    h=dict(timestamp=now(),stage=stage,substage=action,worker_pid=active,supervisor_pid=os.getpid() if active else None,
        completed_work=last,remaining_work=list(STAGES[STAGES.index(stage):]) if stage in STAGES else [],source_id=None,method=None)
    progress=out/'WORK_PROGRESS.json'
    if progress.exists():
        detail=read(progress)
        if detail.get('stage')==stage:h.update(detail)
    if h.get('infrastructure_failure'):
        blocker='Recorded infrastructure failure: '+h['infrastructure_failure']['path']
        action=h['substage']
    if stage=='planning_sweep':
        # These are the existing inner planner's latest published substeps;
        # keep them distinct from the last completed immutable case.
        h['recipe_planner_progress']=[dict(recipe=p.parent.name,progress=read(p),
            last_update_unix_s=p.stat().st_mtime) for p in sorted((out/'recipes').glob('*/WORK_PROGRESS.json'))]
    if h.get('engine_log') and Path(h['engine_log']).exists():
        lines=[s for s in Path(h['engine_log']).read_text(errors='replace').splitlines() if s.startswith('HYBRID_MEASURED_PROGRESS ')]
        if lines:
            _,frame,horizon=lines[-1].split();h.update(physics_frame=int(frame),physics_horizon=int(horizon))
    atomic_json(out/'HEARTBEAT.json',h)
    atomic_text(out/'CURRENT_STATUS.md',f'CURRENT_STAGE: {stage}\nLAST_COMPLETED_STAGE: {last}\nCURRENT_BLOCKER: {blocker}\nNEXT_ACTION: {action}\nACTIVE_PROCESS: {active}\nLAST_HEARTBEAT: {h["timestamp"]}\nDEV35_STARTED: NO\nACT_STARTED: NO\n')


def dispatch(out,stage):
    if stage=='planning_sweep':
        from .practical_planning import run
    elif stage=='recipe_ranking':
        from .practical_planning import rank
        from .practical_diagnostics import run as explain
        return rank(out)+explain(out)
    elif stage=='top_recipe_physics':
        from .practical_physics import run
    elif stage in ('final_selection','freeze'):
        from . import practical_finalize
        run=getattr(practical_finalize,stage)
    elif stage in ('paper_A_train40','paper_B_train40'):
        from .practical_paper import run as paper
        return paper(out,'A_WRIST' if stage=='paper_A_train40' else 'C_COUPLED')
    elif stage=='videos':
        from .practical_videos import run
    else:
        from . import practical_results
        run=getattr(practical_results,stage)
    return run(out)


def run(out,resume):
    lock=(out/'.practical_calibration.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Existing valid worker owns lock; duplicate refused.',flush=True);return 0
    assert_backend(out)
    receipts=out/'STAGE_RECEIPTS';receipts.mkdir(exist_ok=True)
    if list(receipts.iterdir()) and not resume:raise ValueError('Use --resume')
    atomic_json(out/'ACTIVE_PROCESS.json',dict(pid=os.getpid(),started=now()))
    failures={};last=None;initial_code=record(__file__)['sha256'];log(out,'supervisor_started',pid=os.getpid())
    while True:
        if record(__file__)['sha256']!=initial_code:
            os.execv(sys.executable,[sys.executable,'-B','-m','tools.contact_coordination.run_train40_practical_calibration','--run-dir',str(out),'--resume'])
        frozen_inputs=read(out/'IMPLEMENTATION_FREEZE.json')['files']
        if read(out/'PREFLIGHT_REGRESSION.json')['status']!='PASS' or any(record(x['path'])!=x for x in frozen_inputs):
            status(out,'preflight_regression',last,'Wait for regression evidence matching the current scientific inputs',
                blocker='Authorized software consistency repair awaiting fresh preflight')
            time.sleep(5);continue
        previous=None;pending=None
        for stage in STAGES:
            deps=dependencies(out,stage);receipt=receipts/(stage+'.json')
            if valid_receipt(receipt,deps,previous):
                previous=record(receipt);last=stage
                if stage=='recipe_ranking' and not read(out/'RECIPE_RANKING.json')['planning_improved']:
                    status(out,'PLANNING_CALIBRATION_INSUFFICIENT',last,'All4 recipes evaluated; authorized early stop. No physics or final paper evaluation.')
                    atomic_json(out/'ACTIVE_PROCESS.json',dict(pid=None,last_pid=os.getpid(),normal_completion=True,terminal='PLANNING_CALIBRATION_INSUFFICIENT'))
                    log(out,'authorized_insufficient_planning_stop',physics_started=False,DEV35_started=False,ACT_started=False)
                    print((out/'PLANNING_CALIBRATION_INSUFFICIENT.md').read_text(),flush=True);return 0
                continue
            pending=(stage,deps,receipt,previous);break
        if pending is None:
            status(out,'COMPLETE',last,'Final TRAIN40 paper study complete; no DEV35/ACT.')
            atomic_json(out/'ACTIVE_PROCESS.json',dict(pid=None,last_pid=os.getpid(),normal_completion=True,finished=now()))
            log(out,'study_complete',DEV35_started=False,ACT_started=False);return 0
        stage,deps,receipt,previous=pending;key=signature(deps)
        if failures.get(stage)==key:
            status(out,stage,last,'Diagnose current stage log; unchanged retries disabled',blocker='Implementation failure or required agent visual review')
            time.sleep(10);continue
        if receipt.exists():atomic_json(out/'INVALIDATED_RECEIPTS'/(stage+'_'+record(receipt)['sha256'][:12]+'.json'),read(receipt))
        log(out,'stage_started',stage=stage,dependency_hash=key)
        atomic_json(out/'DEPENDENCIES.json',dict(stage=stage,files=deps,hash=key))
        result=out/'stage_results'/(stage+'_'+key[:12]+'.json');result.parent.mkdir(exist_ok=True)
        output=out/'stage_logs'/(stage+'_'+key[:12]+'.log');output.parent.mkdir(exist_ok=True)
        args=[OFFLINE,'-B','-m','tools.contact_coordination.run_train40_practical_calibration','--run-dir',str(out),'--stage',stage,'--result-file',str(result)]
        with output.open('a') as stream:
            child=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl'))
            while child.poll() is None:
                status(out,stage,last,'Running '+str(output),child.pid)
                if signature(dependencies(out,stage))!=key:
                    log(out,'stage_invalidated',stage=stage,worker_pid=child.pid);os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(5)
        if signature(dependencies(out,stage))!=key:continue
        if child.returncode==0 and result.exists():
            value=read(result);atomic_json(receipt,dict(value,stage=stage,dependency_hash=key,dependencies=deps,predecessor=previous,completed_at=now(),log=record(output)))
            log(out,'stage_completed',stage=stage,receipt=record(receipt));last=stage
        else:
            failures[stage]=key;log(out,'stage_failed',stage=stage,returncode=child.returncode,log=str(output))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--stage',choices=STAGES);p.add_argument('--result-file',type=Path);p.add_argument('--setup',action='store_true');p.add_argument('--detach',action='store_true')
    a=p.parse_args();out=a.run_dir.resolve()
    if a.setup:setup(out);return
    if a.stage:
        artifacts=dispatch(out,a.stage);assert artifacts
        atomic_json(a.result_file,dict(status='PASS',artifacts=[record(x) for x in artifacts]));return
    if a.detach:
        with (out/'.practical_calibration.lock').open('a+') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:print('Existing worker owns lock; duplicate launch refused');return
        command=[OFFLINE,'-B','-m','tools.contact_coordination.run_train40_practical_calibration','--run-dir',str(out),'--resume']
        atomic_text(out/'RESUME_COMMAND.txt','cd '+str(ROOT)+'\n'+' '.join(command)+' --detach\n\n# Foreground:\n'+' '.join(command)+'\n')
        with (out/'SUPERVISOR.log').open('a') as f:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        atomic_json(out/'LAUNCHER.json',dict(pid=child.pid,command=command,detached=True));print('Persistent supervisor PID',child.pid);return
    sys.exit(run(out,a.resume))


if __name__=='__main__':main()
