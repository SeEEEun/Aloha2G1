#!/usr/bin/env python3
"""Frozen, uniform full-cohort evaluation; labels only select raw input arrays.

No previous case-selected qualified trajectory is reused. The numerical search
is the latest TRAIN-side v9 posture-pool procedure, with its original budget.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, sys, time, traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import (
    OUT, RUN, QUAL, INITIAL, ST5, STAGE, RESET, read, file_record,
    atomic_json, atomic_npz, atomic_text, model, verified_oracle_contract,
    FramewiseReachabilityOracle, RobustProxyPenaltySolver, full_chain_enclosures,
    qualify_numeric, restore, preparation_path, temporal_metrics)
from tools.common_training_seed_bank_v9 import generate
from tools.doll_handoff_retargeting.common import load_common_config,load_scene

DEST=OUT/'paper_completion_v1'
FREEZE=OUT/'03_common_execution_freeze/COMMON_RETARGETING_FREEZE_MANIFEST.json'
SPLIT=OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json'
BLOCKING={'HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY'}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def cases():
    entries=read(SPLIT)['entries']; rows=[]
    for group,flag,index in [('TRAIN40','TRAIN40','final_dataset_index'),('DEV35','DEV35_DIAGNOSTIC35','dev_index')]:
        subset=sorted((x for x in entries if x[flag]),key=lambda x:x[index])
        assert len(subset)==(40 if group=='TRAIN40' else 35)
        for mode in ('WRIST','INTERACTION'):
            for e in subset:
                rows.append(dict(key=f'{group}_{mode}_{e[index]:03d}',group=group,representation_mode=mode,
                    index=e[index],source_recording_id=e['source_recording_id'],source=e['current_rebuild_raw_cartesian_reference']))
    return rows

def inputs(row,g,n):
    p=Path(row['source']['path']);assert file_record(p)==row['source']
    common=load_common_config(RESET/'config/common_config.json')
    primitives=g.derive_hand_primitives(load_scene(common),read(RESET/'config/proposed_config.json'),n)
    with np.load(p) as z:
        # The sole scientific switch selects the already frozen raw SE(3).
        mode=row['representation_mode']
        target=np.stack([z[f'{mode}_{side}_wrist_position_model'] for side in ('left','right')],axis=1)
        rotation=np.stack([z[f'{mode}_{side}_wrist_rotation_model'] for side in ('left','right')],axis=1)
        opened=np.array([primitives['states'][side]['OPEN'] for side in ('left','right')])
        hands=np.stack([opened[k]+z[f'common_{side}_close_fraction'][:,None]*(np.array(primitives['states'][side]['GRASP'])-opened[k]) for k,side in enumerate(('left','right'))],axis=1)
        ts=z['source_timestamp'].copy()
    assert np.isfinite(target).all() and np.isfinite(rotation).all() and np.isfinite(hands).all()
    assert np.all(np.diff(ts)>0) and abs(float(np.median(np.diff(ts)))-1/30)<1e-6
    return target,rotation,hands,ts,opened

def freeze():
    if FREEZE.exists():verify();return
    oc,_=verified_oracle_contract();g,c,n=model()
    rows=cases();bank=read(ST5/'common_training_seed_bank/MANIFEST.json')['bank']
    assert file_record(Path(bank['path']))==bank
    # Freeze all repository modules actually imported by this execution closure.
    paths={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(getattr(m,'__file__','')).startswith(str(ROOT/'tools')) and str(m.__file__).endswith('.py')}
    paths.add(Path(__file__).resolve())
    paths.update([QUAL,INITIAL,SPLIT,STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json',RUN/'orbit_certificate/REPEATABILITY_RESULT.json',ST5/'full_chain_certificate/MODEL_ONLY_CERTIFICATE.json',ST5/'common_training_seed_bank/MANIFEST.json',Path(bank['path']),Path(g.path),Path(g.mapping_path)])
    paths.update((RESET/'config').glob('*.json'))
    paths.update((ROOT/'configs').glob('common_*.json'))
    paths.update((OUT/'01_registration').glob('*.json'))
    paths.update(Path(r['source']['path']) for r in rows)
    # Mesh/include assets reside beside the authoritative XML model.
    paths.update(p for p in Path(g.path).parent.rglob('*') if p.is_file() and p.suffix.lower() in ('.xml','.stl','.obj','.ply','.json'))
    contract=dict(schema='frozen_common_retargeting_paper_v1',created_at=now(),cases=rows,
        only_intended_difference='raw WRIST versus INTERACTION SE(3) arrays; numerical functions receive no labels',
        selection='Latest completed TRAIN-side common posture-pool v9 core and v8 recovery budget; no DEV outcomes inspected; previous mixed recovery successes diagnostic only',
        diagnostic_train11={'WRIST':7,'INTERACTION':10,'denominator':11,'not_reused_as_new_frozen_results':True},
        budget=dict(startup_seeds=128,startup_max_nfev_per_seed=180,families=['forward','reverse'],
            seeds_per_frame=10,nearest_training_bank_seeds=8,max_fit_nfev=180,proxy_repair_nfev=80,
            restores_per_family=3,restore_max_nfev=400,restore_lsmr_maxiter=1000,
            wall_clock='No outcome-dependent wall-clock cutoff. Finite deterministic evaluation/search budget; measured wall time reported.',
            early_stop='First fully qualified family; frame seed early stopping unchanged v9'),
        acceptance=dict(raw_position_m=.01,certified_slack_m=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'],
            qdot_rad_s=4.5,qddot_rad_s2=130.,adaptive_branch=[.18,8.],aggregate_0179='internal only',
            geometry_tolerance_m=1e-5,unresolved_geometry='fail closed; classified no qualified solution, not certified impossibility',
            all_frames_required=True,hard_limits='authoritative model; no relaxation',natural_q0=read(INITIAL)['g1_14_arm_initial_q_rad'],
            prep_seconds=.7,prep_frames=21,prep_hand='OPEN',source_clock='unaltered; execution offset +0.700s'),
        outcome_precedence=['POSITION_NUMERICAL_INVALID','POSITION_HARD_COLLISION','POSITION_TEMPORAL_INVALID','POSITION_NO_SOLUTION_WITHIN_COMMON_BUDGET','POSITION_EXECUTABLE'],
        certification='Raw outer-envelope infeasibility is reported separately. It does not prove absence of an allowed closest-feasible executable trajectory. No global trajectory certificate is inferred from bounded failure.',
        failure_candidate_selection='Fewest Cartesian-failed frames, then max Cartesian excess, then physical temporal violation, then family order; all attempted family outcomes retained',
        rerun='Only infrastructure-invalid cases; numerical scientific outcomes immutable',
        records=[file_record(p) for p in sorted(paths) if p.is_file()],bank=bank,oracle=oc['oracle'],
        environment=dict(python=sys.executable,numpy=np.__version__,threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    for r in rows:assert file_record(Path(r['source']['path']))==r['source']
    for p in (OUT/'CURRENT_STATUS.md',OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md'):
        archive=DEST/'provenance'/p.name
        if p.exists() and not archive.exists():atomic_text(archive,p.read_text())
    atomic_json(FREEZE,contract)
    atomic_text(FREEZE.with_suffix('.md'),'# Frozen common retargeting experiment\n\n'+json.dumps({k:v for k,v in contract.items() if k not in ('records','cases')},indent=2)+'\n\nAll negative scientific outcomes are final data. Previous TRAIN11 7/11 versus10/11 is diagnostic provenance, not the new uniform run.\n')
    atomic_json(DEST/'FREEZE_SHA256.json',file_record(FREEZE))
    atomic_json(DEST/'CASE_MANIFEST.json',rows)
    update('COMMON_RETARGETING_FROZEN','Full TRAIN40 then DEV35 under identical finite budget; no individual rescue.')

def verify():
    contract=read(FREEZE)
    assert file_record(FREEZE)==read(DEST/'FREEZE_SHA256.json')
    for r in contract['records']:assert file_record(Path(r['path']))==r,r['path']
    return contract

def update(stage,action):
    record=dict(timestamp=now(),stage=stage,status='IN_PROGRESS',authoritative_inputs=[file_record(FREEZE)],
        observed_problem=None,root_cause_classification='SCIENTIFIC_FAILURES_ARE_OUTCOMES',action_taken=action,
        retries=0,artifacts_created=[],next_stage='FULL_COHORT_POSITION_6D_DEX3_REFERENCE_PHYSICS')
    p=DEST/'MASTER_RUN_LOG.jsonl';atomic_text(p,(p.read_text() if p.exists() else '')+json.dumps(record)+'\n')
    atomic_json(DEST/'CHECKPOINT_STATE.json',record)
    atomic_text(DEST/'CURRENT_STAGE.md',f'# Paper completion\n\n{stage}\n\n{action}\n')
    korean='공통 유한 탐색 절차를 동결했습니다. 이전 TRAIN11 결과는 진단 이력으로 보존하며, 모든 TRAIN40/DEV35를 동일 예산으로 평가합니다. 과학적 실패는 최종 결과로 기록하고 개별 구조 없이 다음 에피소드와 물리 평가를 계속합니다.\n'
    atomic_text(DEST/'CHATGPT_UPDATE.md',korean)
    atomic_text(OUT/'CURRENT_STATUS.md',f'# Current authoritative run\n\n{stage}\n\n{action}\n\nSee `{DEST}`. Prior reports remain diagnostic provenance.\n')
    print(korean,flush=True)

def inspect(c,q,h):
    result=[]
    for f,(v,hh) in enumerate(zip(q,h)):
        try:records=c.inspect(v,*hh)
        except Exception as exc:records=[dict(classification='UNRESOLVED_GEOMETRY',reason=str(exc))]
        result.append(dict(frame=f,records=records))
    return result

def realize(s,oracle,t,h,ts,opened,natural,bounds,bank,slack,folder):
    """Method-blind frozen evaluation. Folder is persistence only, never parsed."""
    start=time.monotonic();dt=float(np.median(np.diff(ts)))
    sp=folder/'STARTUP.json'
    if not sp.exists():atomic_json(sp,oracle.solve(t[0],opened))
    startup=read(sp);initial=np.array(startup['best']['q']);initial[[6,13]]=s.natural[[6,13]]
    allow=qualify_numeric(s,np.tile(initial,(len(t),1)),t,h,ts,bounds,slack)[-1]
    results=[]
    for reverse in (False,True):
        sub=folder/('reverse' if reverse else 'forward');qp=sub/'GUIDE.npz';rp=sub/'GUIDE.json'
        if qp.exists():q=np.load(qp)['q'].copy()
        else:
            checkpoint=sub/'PROGRESS.npz';report=sub/'PROGRESS.json';resume=None
            if checkpoint.exists() and report.exists():resume=dict(q=np.load(checkpoint)['q'],reports=read(report)['frames'])
            def progress(values,records):
                atomic_npz(checkpoint,q=values);atomic_json(report,dict(frames=records))
            q,records=generate(oracle,s,t,h,ts,initial,bank['q'],bank['model_wrist_position'],allow,reverse,progress,resume)
            q[0]=initial;atomic_npz(qp,q=q);atomic_json(rp,records)
        mask=np.zeros_like(q,bool);mask[0]=True;mask[:,[6,13]]=True;fixed=q.copy()
        for attempt in range(4):
            if attempt:q=np.load(sub/f'RESTORED_{attempt-1}.npz')['q'].copy()
            met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
            atomic_json(sub/f'CHECK_{attempt}.json',met)
            if met['pass_numeric'] or attempt==3:break
            output=sub/f'RESTORED_{attempt}.npz'
            if not output.exists():
                q,fit=restore(s,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=fixed)
                atomic_npz(output,q=q);atomic_json(sub/f'FIT_{attempt}.json',fit)
        prefix=preparation_path(natural,q[0],21);full=np.vstack((prefix[:-1],q))
        temporal=temporal_metrics(full,dt,s.config)
        # Final geometry is exhaustive even for numeric failures, so physical
        # diagnostics never confuse an unchecked trajectory with a clear one.
        gp=sub/'GEOMETRY.json'
        if gp.exists():geometry=read(gp)
        else:
            geometry=dict(source=inspect(s.collision,q,h),preparation=inspect(s.collision,prefix[:-1],np.repeat(opened[None],21,axis=0)))
            atomic_json(gp,geometry)
        allrows=geometry['source']+geometry['preparation']
        counts={name:sum(any(x['classification']==name for x in r['records']) for r in allrows) for name in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY','PROXY_ONLY_OVERLAP')}
        if not met['finite']:outcome='POSITION_NUMERICAL_INVALID'
        elif counts['HARD_SELF_COLLISION']:outcome='POSITION_HARD_COLLISION'
        elif not temporal['pass_temporal']:outcome='POSITION_TEMPORAL_INVALID'
        elif met['failed_frames'] or met['hard_limit_violations'] or counts['UNRESOLVED_GEOMETRY']:outcome='POSITION_NO_SOLUTION_WITHIN_COMMON_BUDGET'
        else:outcome='POSITION_EXECUTABLE'
        out=sub/'CANDIDATE.npz'
        atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,EXECUTABLE_FK_POSITION=act,
            POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),
            CERTIFIED_UNREACHABLE=cert,common_hand_q=h,source_timestamp=ts,
            full_q=full,full_hand_q=np.concatenate((np.repeat(opened[None],21,axis=0),h)),execution_timestamp=np.r_[np.arange(21)/30,ts-ts[0]+.7])
        r=dict(outcome=outcome,metrics=met,complete_temporal=temporal,geometry_counts=counts,
            max_cartesian_excess_m=float(np.maximum(res-allow,0).max()),trajectory=file_record(out),geometry=file_record(gp),
            family='reverse' if reverse else 'forward',runtime_s=time.monotonic()-start)
        atomic_json(sub/'RESULT.json',r);results.append(r)
        print('FROZEN_FAMILY_RESULT',folder.name,r['family'],outcome,flush=True)
        if outcome=='POSITION_EXECUTABLE':break
    selected=min(results,key=lambda r:(r['outcome']!='POSITION_EXECUTABLE',len(r['metrics']['failed_frames']),r['max_cartesian_excess_m'],not r['complete_temporal']['pass_temporal']))
    return dict(outcome=selected['outcome'],selected=selected,families=results,runtime_s=time.monotonic()-start,
        global_trajectory_infeasibility_proven=False,raw_targets_changed=False)

def run(key):
    contract=verify();row=next(x for x in contract['cases'] if x['key']==key);folder=DEST/'position'/key
    result=folder/'RESULT.json'
    if result.exists():
        r=read(result);assert r['freeze']==file_record(FREEZE)
        if r['outcome']!='INFRASTRUCTURE_INVALID':print('REUSE_FROZEN_RESULT',key,r['outcome'],flush=True);return
    try:
        g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=full_chain_enclosures(g)
        oracle=FramewiseReachabilityOracle(g,c,contract['oracle'],n)
        t,rot,h,ts,opened=inputs(row,g,n)
        r=realize(s,oracle,t,h,ts,opened,np.array(contract['acceptance']['natural_q0']),bounds,np.load(contract['bank']['path']),contract['acceptance']['certified_slack_m'],folder)
        atomic_npz(folder/'RAW_ORIENTATION.npz',rotation=rot)
        r.update(case=row,freeze=file_record(FREEZE),completed_at=now());atomic_json(result,r)
        print('POSITION_EPISODE_FINAL',key,r['outcome'],flush=True)
    except Exception:
        r=dict(outcome='INFRASTRUCTURE_INVALID',case=row,freeze=file_record(FREEZE),traceback=traceback.format_exc(),completed_at=now())
        atomic_json(result,r);print(r['traceback'],flush=True);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action');a=p.parse_args()
    freeze() if a.action=='freeze' else run(a.action)
