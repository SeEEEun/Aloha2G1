"""Fixed common candidate portfolio; fidelity is not physical validity."""
import concurrent.futures, sys, time, traceback, os
import numpy as np
from .common import *
from .audit import assess, fk
from tools.final_paper_position_run import model, inputs, verify, preparation_path, inspect, RESET, load_common_config
from tools.final_paper_full6d_run import solve6
from tools.final_paper_reference_physics import nominal_from_source, JOINT
from tools.direct_physical_execution_layer import authoritative_joint_limits

CONTRACT=RUN/'freeze/CONSTRUCTION_PROTOCOL.json'
def npz(p,**values):
    p=Path(p);assert p.is_relative_to(RUN);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.incomplete')
    with tmp.open('wb') as f:np.savez_compressed(f,**values)
    os.replace(tmp,p)
def freeze():
    if CONTRACT.exists():return read(CONTRACT)
    old=verify();assert read(RUN/'audit/FULL6D_KNOWN_ANSWER_TESTS.json')['status']=='PASS'
    paths={Path(x['path']) for x in old['records']}
    paths.update(Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(m.__file__).startswith(str(ROOT/'tools')) and str(m.__file__).endswith('.py'))
    paths.update([Path(JOINT),RUN/'audit/FULL6D_KNOWN_ANSWER_TESTS.json'])
    deps=[record(p) for p in sorted(paths)]
    data=[]
    for row in old['cases']:
        prior=read(PRIOR/'position'/row['key']/'RESULT.json')
        data.append(dict(case=row,position_result=record(PRIOR/'position'/row['key']/'RESULT.json'),seeds=[x['trajectory'] for x in prior['families']]))
    value=dict(version='RECONCILED_CONSTRUCTION_V1',created_at=now(),files=deps,cases=data,
      scientific_change='Separate Layer A tracking diagnostics from Layer B physical command validity; no raw accuracy or loose lower-bound optimality requirement in Layer B.',
      position_generator='Unchanged frozen v9 forward/reverse generation and historical bounded early-stop rule; input/dependency-verified cached arrays, not selected historical local repairs.',
      full6d='Unchanged SharedTemporalIK and solve6: 2 seeds/frame,250 initial/40 subsequent iterations,9-frame cubic smoothing,25 reprojection iterations. One invocation for every complete finite position family. A numerical seed is not a physically executed trajectory.',
      invocation_wiring='Offline 6D construction is permitted from a finite limit-valid seed even if its complete temporal/collision validity fails. Only final independently Layer-B-qualified commands may enter physics. Same rule for both representations.',
      candidate_pool='For each existing frozen position family: unchanged seed and one full6D returned trajectory. No adaptive rescue, no added family, no outcome-selected budget. Seed is retained to prevent discarding a physically valid command when weighted 6D fitting worsens it; raw orientation error is always reported.',
      selection='Among complete Layer-B-valid candidates, minimize mean(3^2 * raw position residual squared + 0.04^2 * raw SO(3) residual squared), then fixed family/candidate order. No object/task outcome input.',
      layers=dict(A='Raw position/orientation diagnostics and 10mm/95% historical reporting, optional position-anchor FK explicitly distinguished',B='Finite complete commands; hard joint limits; per-joint arm qdot4.5/qddot130; adaptive branch .18/8; unchanged0.700s natural preparation including join; detailed geometry10um fail closed; final named P14 Dex3 command geometry',C='Measured physical task outcomes only'),
      no_change=['raw targets','registration','source event clock','natural q0','0.700s preparation','joint limits','geometry tolerance','existing numerical solver'],
      runtime_scope='Simulation only; no real hardware; later physics freeze and component qualification mandatory')
    save(CONTRACT,value);text(CONTRACT.with_suffix('.md'),'# Reconciled construction protocol\n\n'+json.dumps({k:v for k,v in value.items() if k not in ('files','cases')},indent=2)+'\n')
    return value

def run_case(key):
    contract=read(CONTRACT);entry=next(x for x in contract['cases'] if x['case']['key']==key);row=entry['case'];folder=RUN/'construction'/key;done=folder/'RESULT.json'
    signature=dict(protocol=record(CONTRACT),input=row['source'],position_result=entry['position_result'],seeds=entry['seeds'])
    if done.exists():
        result=read(done);assert result['signature']==signature
        if result['outcome']!='INFRASTRUCTURE_INVALID':return key,result['outcome']
    started=time.monotonic()
    try:
        for dep in contract['files']:assert record(dep['path'])==dep,dep['path']
        g,c,n=model();t,rot,reference_h,ts,opened=inputs(row,g,n);common=load_common_config(RESET/'config/common_config.json')
        old=read(PRIOR/'position'/key/'RESULT.json');natural=np.array(read(PRIOR.parent/'03_common_execution_freeze/COMMON_RETARGETING_FREEZE_MANIFEST.json')['acceptance']['natural_q0'])
        with np.load(row['source']['path']) as raw:hand,intent,events,names=nominal_from_source(raw)
        lo,hi,expected=authoritative_joint_limits(read(JOINT));assert list(names)==list(expected)
        mnames=[*g.arm_joint_names,*g.hand_joint_names['left'],*g.hand_joint_names['right']];permutation=[list(names).index(x) for x in mnames]
        hmodel=hand[:,np.array(permutation[14:])-14].reshape(-1,2,7)
        candidates=[]
        for family in old['families']:
            assert record(family['trajectory']['path'])==family['trajectory'];z=np.load(family['trajectory']['path']);seed=z['q'];anchor=z['EXECUTABLE_FK_POSITION']
            sub=folder/family['family'];output=sub/'SIX_Q.npz'
            if output.exists():q6=np.load(output)['q']
            else:
                historical=read(PRIOR/'full6d'/key/'RESULT.json')
                if historical.get('position_input')==family['trajectory'] and 'trajectory' in historical:
                    q6=np.load(historical['trajectory']['path'])['q'];save(sub/'CACHE_PROVENANCE.json',dict(source=historical['trajectory'],input=family['trajectory'],same_solve6=record(ROOT/'tools/final_paper_full6d_run.py')))
                else:q6=solve6(common,g,n,anchor,rot,seed,sub/'sixd_search')
                npz(output,q=q6)
            for label,q in [('POSITION_SEED',seed),('FULL6D_RETURN',q6)]:
                cp=sub/label;rp=cp/'CANDIDATE.json'
                if rp.exists():candidates.append(read(rp));continue
                prefix=preparation_path(natural,q[0],21);full=np.vstack((prefix[:-1],q));gp=cp/'GEOMETRY.json'
                if gp.exists():geom=read(gp)['rows']
                else:geom=inspect(c,full,hmodel);save(gp,dict(rows=geom))
                counts={k:sum(any(v['classification']==k for v in r['records']) for r in geom) for k in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY','PROXY_ONLY_OVERLAP')}
                result=assess(g,q,ts,t,rot,full,counts);values=np.hstack((full,hand));limits=int(np.count_nonzero((values<lo)|(values>hi)))
                result['complete_command_limit_violations']=limits
                if limits:result['layer_b_gates'].append('COMPLETE_COMMAND_LIMIT')
                if not np.array_equal(full[0],natural):result['layer_b_gates'].append('INITIAL_Q0')
                result['layer_b_valid']=not result['layer_b_gates'];p,r=fk(g,q)
                from scipy.spatial.transform import Rotation
                pe=np.linalg.norm(p-t,axis=2);ae=Rotation.from_matrix((rot@r.transpose(0,1,3,2)).reshape(-1,3,3)).magnitude().reshape(-1,2)
                cost=float(np.mean(9*pe**2+.0016*ae**2))
                path=cp/'TRAJECTORY.npz';npz(path,q=q,full_q=full,source_timestamp=ts,execution_timestamp=np.r_[np.arange(21)/30,ts-ts[0]+.7],
                    RAW_TARGET=t,RAW_ORIENTATION_TARGET=rot,OPTIONAL_COMMON_PROJECTED_TARGET=anchor,SOLVER_Q=q,FK_OF_SOLVER_Q=p,FK_ORIENTATION=r,POSITION_CORRECTION_MM=pe*1000,ORIENTATION_RESIDUAL_RAD=ae,
                    commanded_q_rad=values,raw_policy_command=values,policy_safe_command=values,common_source_nominal_dex3=hand,common_task_intent=intent,stage=intent,joint_names=np.asarray(names),control_fps_hz=np.asarray(30.),common_initial_q_rad=values[0],source_event_names=np.asarray(list(events)),execution_event_frames=np.asarray(list(events.values())),standardized_initial_grasp=np.asarray(False),runtime_right_three_digit_gate_required=np.asarray(False),policy_used=np.asarray(False),stable_episode_id=np.asarray(row['source_recording_id']))
                result.update(family=family['family'],candidate=label,tracking_cost=cost,trajectory=record(path),geometry=record(gp),full6d_invoked=True)
                save(rp,result);candidates.append(result)
        valid=[x for x in candidates if x['layer_b_valid']]
        selected=min(valid,key=lambda x:x['tracking_cost']) if valid else None
        result=dict(case=row,signature=signature,outcome='EXECUTABLE_COMMAND_CANDIDATE' if selected else 'NO_EXECUTABLE_TRAJECTORY_UNDER_FIXED_PROTOCOL',selected=selected,candidates=candidates,full6d_invocations=len(old['families']),runtime_s=time.monotonic()-started,physical_executed=False)
        save(done,result);return key,result['outcome']
    except Exception:
        save(done,dict(case=row,signature=signature,outcome='INFRASTRUCTURE_INVALID',traceback=traceback.format_exc()));raise

def main():
    if len(sys.argv)>1 and sys.argv[1]=='freeze':freeze();return
    contract=read(CONTRACT)
    keys=sys.argv[1:] or [x['case']['key'] for x in contract['cases']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(run_case,k):k for k in keys}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            try:print('CONSTRUCTION_COMPLETE',i,len(keys),f.result(),flush=True)
            except Exception as exc:print('CONSTRUCTION_INFRASTRUCTURE',futures[f],repr(exc),flush=True)

if __name__=='__main__':main()
