"""Measured task outcomes, explicit nonexecution/invalidity and release taxonomy."""
import subprocess,sys
import numpy as np
from .common import *
from tools.final_paper_position_run import model,inspect

STAGES=['EXECUTABLE_TRAJECTORY','APPROACH_VALID','LEFT_GRASP_SUCCESS','LIFT_SUCCESS','HANDOFF_SUCCESS','RIGHT_OWNERSHIP_SUCCESS','RIGHT_TRANSPORT_SUCCESS','BIN_ENTRY_SUCCESS','BIN_SETTLE_SUCCESS','FULL_TASK_SUCCESS']

def release_class(acquired,owned,commanded_release,loss,inside):
    if not acquired:return 'NO_ACQUISITION'
    if loss is None:return 'NO_RELEASE_OBSERVED'
    if owned and commanded_release is not None and loss>=commanded_release:return 'CLEAN_COMMANDED_RELEASE'
    return 'PREMATURE_DROP_INTO_BIN' if inside else 'PREMATURE_DROP_OUTSIDE_BIN'

def score(folder):
    folder=Path(folder);inv=read(folder/'INVOCATION_MANIFEST.json');row=inv['case']
    oldpath=folder/'EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.json'
    if not oldpath.exists():
        with (folder/'LEGACY_SCORER.log').open('w') as f:subprocess.run([sys.executable,str(ROOT/'tools/score_episode_registered_physical_eval35_run.py'),'--run-dir',str(folder)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    legacy=read(oldpath)
    with np.load(folder/'event_log.npz') as z:events={k:z[k].copy() for k in z.files}
    q=events['MEASURED_Q'];frames=events['control_frame'].astype(int);ev=events['DIRECT_COMMON_EXECUTION_EVENTS'].astype(str);positions=events['object_position_world_m']
    with np.load(inv['commands']['path']) as z:
        clock=dict(zip(z['source_event_names'].astype(str),z['execution_event_frames'].astype(int)));names=z['joint_names'].astype(str).tolist();count=len(z['commanded_q_rad'])
    complete=bool(np.array_equal(np.unique(frames),np.arange(count)))
    g,c,n=model();mnames=[*g.arm_joint_names,*g.hand_joint_names['left'],*g.hand_joint_names['right']];perm=[names.index(x) for x in mnames]
    gp=folder/'MEASURED_GEOMETRY.json'
    if gp.exists():geom=read(gp)
    else:
        # Every measured physics sample until first invalidity; no visual mesh
        # or runtime self-response is assumed to certify physical validity.
        examined=[];blocking=[]
        for i,v in enumerate(q):
            values=v[perm]
            try:recs=c.inspect(values[:14],values[14:21],values[21:])
            except Exception as exc:recs=[dict(classification='UNRESOLVED_GEOMETRY',reason=str(exc))]
            if recs:examined.append(dict(physics_sample=i,control_frame=int(frames[i]),records=recs))
            if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in recs):blocking.append(i);break
        geom=dict(nonclear_rows=examined,first_blocking_sample=blocking[0] if blocking else None,samples_checked=i+1,total_samples=len(q),complete_clear=not blocking,early_stop='first physical invalidity, never counted as clear remaining samples')
        save(gp,geom)
    control_rows=np.r_[np.flatnonzero(np.diff(frames)!=0),len(frames)-1];dist=[]
    for v in q[control_rows]:
        p=g.wrist_state(v[:14])['left_position'];world=g.model_to_world_position(p);dist.append(float(np.linalg.norm(world-positions[0])))
    dist=np.array(dist);cf=frames[control_rows];scope=(cf>=clock['APPROACH_START'])&(cf<=clock['LEFT_CLOSE_BEGIN']);approach=bool(scope.any() and dist[scope].min()<dist[0]-1e-6)
    def event(name):
        ix=np.flatnonzero(np.char.find(ev,name)>=0);return int(ix[0]) if len(ix) else None
    conf=event('LEFT_GRASP_CONFIRMED');liftrow=event('LEFT_GRASP_STATE_LIFT');owned=event('RIGHT_PHYSICAL_OWNERSHIP');rel=event('FINAL_RELEASE_INTENT_START')
    # Retained table-free acquisition is the existing common physical grasp
    # criterion. Pre-lift confirmation alone remains a candidate diagnostic.
    grasp=bool(legacy['outcomes']['LEFT_GRASP_SUCCESS']);lift=bool(grasp and liftrow is not None and conf is not None and positions[liftrow,2]-positions[conf,2]>=.005-1e-8)
    independent=[True,approach,grasp,lift,legacy['outcomes']['HANDOFF_SUCCESS'],legacy['outcomes']['RIGHT_OWNERSHIP_SUCCESS'],legacy['outcomes']['NO_DROP_TO_BIN'],legacy['outcomes']['BIN_ENTRY_SUCCESS'],legacy['outcomes']['BIN_SETTLE_SUCCESS'],legacy['outcomes']['FULL_TASK_SUCCESS']]
    # Explicitly repair the reproducible old default-else taxonomy bug.
    force_threshold=read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')['gates']['meaningful_digit_force_n']
    support=np.logical_or.reduce([events[f'{s}_{d}_force_n']>=force_threshold for s in ('left','right') for d in ('thumb','index','middle')]+[events[f'{s}_palm_force_n']>=force_threshold for s in ('left','right')])
    loss=None;run=0
    if grasp:
        for i in range(conf if conf is not None else 0,len(q)):
            run=0 if support[i] else run+1
            if run>=24:loss=i-23;break
    env=read(ROOT/'outputs/final_episode_registered_eval35/01_freeze/FINAL_PHYSICAL_ENVIRONMENT.json')['bin']
    inside=bool(loss is not None and np.all(np.abs(positions[loss,:2]-env['opening_center_world_xy_m'])<=np.array(env['opening_dimensions_xy_m'])/2))
    release=release_class(grasp,owned is not None,rel,loss,inside)
    valid=bool(complete and legacy['integrity']['hard_physical_validity'] and geom['complete_clear'])
    cumulative=dict(zip(STAGES,np.logical_and.accumulate(independent).tolist()))
    first=next((k for k,v in cumulative.items() if not v),'SUCCESS')
    result=dict(case=row,protocol=inv['freeze'],physical_trace=record(folder/'event_log.npz'),actual_physics_executed=True,complete_trace=complete,physical_valid=valid,
        outcome=('PHYSICAL_SUCCESS' if cumulative['FULL_TASK_SUCCESS'] else 'PHYSICAL_FAILURE') if valid else 'PHYSICAL_EXECUTION_INVALID_OUTCOME_UNKNOWN',
        physical_stages=cumulative if valid else {k:None for k in STAGES},cumulative_pipeline=cumulative if valid else {k:(True if k=='EXECUTABLE_TRAJECTORY' else None) for k in STAGES},
        measured_stage_diagnostics=dict(zip(STAGES,independent)),first_failure_stage=first if valid else 'PHYSICAL_EXECUTION_INVALID',release_classification=release,
        legacy_scoring=record(oldpath),measured_geometry=record(gp),integrity=legacy['integrity'],source_clock={k:int(v) for k,v in clock.items()},
        prelift_grasp_candidate_confirmed=conf is not None,prelift_candidate_before_nominal_lift=bool(conf is not None and frames[conf]<clock['LEFT_LIFT_BEGIN']),
        component_waits_and_interventions=record(folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json'),denominator_rules='Construction and cumulative:35; observed physical stages: valid actually executed only; invalid outcomes remain unknown')
    save(folder/'RESULT.json',result);print('MEASURED_RESULT',row['key'],result['outcome'],first,release,flush=True);return result

if __name__=='__main__':
    assert release_class(False,False,None,None,False)=='NO_ACQUISITION'
    assert release_class(True,True,100,110,True)=='CLEAN_COMMANDED_RELEASE'
    assert release_class(True,False,None,30,False)=='PREMATURE_DROP_OUTSIDE_BIN'
    score(Path(sys.argv[1]))
