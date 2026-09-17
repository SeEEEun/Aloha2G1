#!/usr/bin/env python3
"""Same bounded collision repair and full acceptance for all six smoke cases."""
from pathlib import Path
import sys, time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_common_physical_position_v4 import *
from tools.common_physical_position_v4 import qualify_numeric
from tools.common_geometry_progress_penalty import GeometryProgressPenaltySolver
from tools.common_local_redundancy_v5 import solve

V5 = OUT/'02_common_execution_qualification/common_local_redundancy_v5'
ST5 = MASTER/'local_redundancy_v5'

def main():
    verified_oracle_contract()
    for p in [OUT/'CURRENT_STATUS.md', OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md', MASTER/'CURRENT_STAGE.md', MASTER/'CHATGPT_UPDATE.md', MASTER/'LATEST_BLOCKER.md', MASTER/'CHECKPOINT_STATE.json']:
        archive = ST5/'previous_status'/p.parent.name/p.name
        if p.exists() and not archive.exists(): atomic_text(archive, p.read_text())
    contract = dict(method_blind=True, window_padding_frames=[12,24,48], attempts_per_window=4,
                    maximum_iterations=300, targets_unchanged=True, preparation_unchanged=True,
                    acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),
                    numerical_search_goal='current proxy separation plus detailed penetration lower bound or geometry tolerance; final classifier unchanged',
                    code=file_record(ROOT/'tools/common_local_redundancy_v5.py'))
    cp = ST5/'LOCAL_REPAIR_CONTRACT.json'
    if not cp.exists(): atomic_json(cp, contract)
    log('LOCAL_REDUNDANCY_REPAIR','IN_PROGRESS',[cp], 'Two geometry-confirmed contacts remain in saved numeric-valid smoke candidate',
        'LOCAL_COLLISION_REDUNDANCY_SEARCH', 'Run fixed-target constrained local SQP on every blocked input using one common contract',
        [cp], 'POSITION_REQUALIFICATION', '상세 형상 충돌 두 프레임에 대해 목표·시간·물리 한계를 고정한 공통 국소 여유자유도 탐색을 시작합니다. A/B에 동일한 알고리즘을 적용하며 실행을 계속합니다.')
    atomic_text(OUT/'CURRENT_STATUS.md', '# Current state\n\nIN_PROGRESS: common local collision-aware redundancy repair. Previous A2/3 B3/3 retained as provenance; targets, preparation, physical gates unchanged.\n')
    g,c,n = model(); s = GeometryProgressPenaltySolver(g,c,read(QUAL),n)
    g.assign(n); bounds=orbit_enclosures(g)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    prior=read(STAGE/'final_diagnostics/POSITION_GEOMETRY_QUALIFICATION.json')
    atomic_json(ST5/'PRIOR_SCHEMA.json',dict(keys=list(prior)))
    results=[]
    for mode in ('WRIST','INTERACTION'):
        for ep in (0,24,49):
            case=f'{mode}_EP{ep:03d}'; folder=V5/case
            done=folder/'SOURCE_POSITION_PASS.json'
            if done.exists(): results.append(read(done)); continue
            old=V4/case/'SOURCE_POSITION_PASS.json'
            if old.exists(): src=Path(read(old)['trajectory']['path'])
            else:
                # Provenance selection only; repair receives no representation label.
                candidates=list((V4/case/'local_feasible_detailed_geometry_v1').glob('SEED_*/CHECK_3.json'))
                def rank(p):
                    geom=read(p.parent/'GEOMETRY_3.json')['geometry']
                    bad=[x for r in geom for x in r['records'] if x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')]
                    return len(read(p)['blocked_frames']), max((x.get('detailed_penetration_lower_bound_mm') or 0 for x in bad),default=0),str(p)
                src=min(candidates,key=rank).parent/'CANDIDATE_2.npz'
            t,h,ts,source=load_input(case,g,n); dt=float(np.median(np.diff(ts)))
            q=np.load(src)['q'].copy(); initial=q.copy(); goals={}; qualified=False
            atomic_json(folder/'INPUT.json',dict(trajectory=file_record(src),source=source,contract=file_record(cp)))
            for attempt in range(13):
                if attempt:
                    saved=folder/f'CANDIDATE_{attempt-1}.npz'
                    if saved.exists(): q=np.load(saved)['q'].copy()
                met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
                gp=folder/f'GEOMETRY_{attempt}.json'
                if gp.exists(): geom=read(gp)['geometry']
                else:
                    geom=[dict(frame=f,records=c.inspect(v,*hh)) for f,(v,hh) in enumerate(zip(q,h))]
                    atomic_json(gp,dict(geometry=geom))
                badrows=[r for r in geom if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
                atomic_json(folder/f'CHECK_{attempt}.json',dict(metrics=met,blocked_frames=[r['frame'] for r in badrows]))
                print('V5_CHECK',case,attempt,'numeric',met['pass_numeric'],'blocked',[r['frame'] for r in badrows],flush=True)
                if met['pass_numeric'] and not badrows:
                    out=folder/'QUALIFIED_SOURCE_Q.npz'
                    atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
                        EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,
                        SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
                    record=dict(case=case,metrics=met,trajectory=file_record(out),geometry=file_record(gp),source=source,
                        acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),preparation_join='RECHECK_BEFORE_FREEZE')
                    atomic_json(done,record); results.append(record); qualified=True; break
                if attempt==12: break
                # An invalid trial is never allowed to replace the saved feasible baseline.
                if not met['pass_numeric']: q=initial.copy(); met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
                influence=np.zeros(14)
                for r in badrows:
                    f=r['frame']
                    for x in r['records']:
                        if x['classification'] not in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY'): continue
                        pair=tuple(x['geom_pair']); s.pose_jacobian(q[f],h[f]); dist,jac=s.clearance_values([pair]); influence+=np.abs(jac).sum(axis=0)
                        depth=(x.get('detailed_penetration_lower_bound_mm') or 0)/1000
                        goal=float(dist[0]+max(depth,c.tolerance))
                        goals.setdefault(f,{})[pair]=max(goal,goals.get(f,{}).get(pair,-np.inf))
                padding=contract['window_padding_frames'][attempt//4]
                begin=max(0,min(goals)-padding); end=min(len(q),max(goals)+padding+1)
                pairs={f-begin:[(*p,v) for p,v in pp.items()] for f,pp in goals.items()}
                arms=[a for a in range(2) if influence[a*7:(a+1)*7].sum()>1e-12]
                out=folder/f'CANDIDATE_{attempt}.npz'
                if not out.exists():
                    fits=[]
                    for arm in arms:
                        start=time.monotonic(); q[begin:end],fit=solve(s,t[begin:end],h[begin:end],q[begin:end],allow[begin:end],dt,arm,pairs,300)
                        fit['runtime_s']=time.monotonic()-start; fits.append(fit)
                    atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
                    atomic_json(folder/f'FIT_{attempt}.json',dict(fits=fits,window=[begin,end],arms=arms,goals={str(f):v for f,v in pairs.items()}))
            if not qualified:
                record=dict(case=case,status='LOCAL_REDUNDANCY_COLLISION_INFEASIBILITY',meaning='No qualified witness in the declared bounded local search; not a global impossibility certificate',next='COMMON_CLOSEST_FEASIBLE_REALIZATION',metrics=met)
                atomic_json(folder/'BOUNDED_RESULT.json',record); results.append(record)
    atomic_json(ST5/'SMOKE3_RESULT.json',dict(results=results,position_pass_count=sum('trajectory' in r for r in results),next='PREPARATION_AND_TRAIN11' if all('trajectory' in r for r in results) else 'COMMON_CLOSEST_FEASIBLE_REALIZATION'))

if __name__=='__main__': main()
