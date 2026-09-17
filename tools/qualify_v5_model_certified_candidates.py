#!/usr/bin/env python3
"""Uniform candidate selection with target-blind tightened geometry certificate."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_local_redundancy_v5 import *
from tools.common_full_chain_certificate_v5 import full_chain_enclosures

def main():
    verified_oracle_contract();proof=ST5/'full_chain_certificate/VALIDATION.json'
    assert read(proof)['status']=='MODEL_ONLY_OUTER_CERTIFICATE_VALIDATED'
    g,c,n=model();s=CommonPositionSolver(g,c,read(QUAL),n);g.assign(n)
    oldbounds=orbit_enclosures(g);bounds=full_chain_enclosures(g)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'];rows=[]
    for mode in ('WRIST','INTERACTION'):
        for ep in (0,24,49):
            case=f'{mode}_EP{ep:03d}';base=V5/case;folder=base/'model_certified_selection_v1'
            t,h,ts,source=load_input(case,g,n);initial_path=Path(read(base/'INPUT.json')['trajectory']['path']);initial=np.load(initial_path)['q'].copy()
            paths={initial_path}
            if (base/'SOURCE_POSITION_PASS.json').exists():paths.add(Path(read(base/'SOURCE_POSITION_PASS.json')['trajectory']['path']))
            for gp in base.glob('**/GEOMETRY_*.json'):
                try:i=int(gp.stem.split('_')[-1])
                except ValueError:continue
                qp=gp.parent/f'CANDIDATE_{i-1}.npz'
                if i and qp.exists():
                    rr=read(gp).get('geometry',[])
                    if len(rr)==len(initial) and not any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in rr for x in r['records']):paths.add(qp)
            candidates=[]
            for qp in sorted(paths):
                q=np.load(qp)['q'].copy()
                met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
                if met['pass_numeric']:candidates.append((float(np.linalg.norm(q-initial)),str(qp),q,met,act,res,lb,cert))
            candidates.sort(key=lambda x:(x[0],x[1]));accepted=None
            for k,(deviation,qp,q,met,act,res,lb,cert) in enumerate(candidates):
                geometry=[dict(frame=f,records=c.inspect(v,*hh)) for f,(v,hh) in enumerate(zip(q,h))]
                bad=[r['frame'] for r in geometry if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
                gp=folder/f'GEOMETRY_CHECK_{k}.json';atomic_json(gp,dict(geometry=geometry,trajectory=file_record(Path(qp))))
                if bad:continue
                old=qualify_numeric(s,q,t,h,ts,oldbounds,slack)[0]
                assert old['raw_acceptance']==met['raw_acceptance']
                out=folder/'QUALIFIED_SOURCE_Q.npz';atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
                    EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
                accepted=dict(case=case,metrics=met,previous_loose_certificate_metrics=old,trajectory=file_record(out),geometry=file_record(gp),source=source,
                    selected_input=file_record(Path(qp)),joint_deviation_norm_rad=deviation,certificate=file_record(proof),
                    acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),
                    acceptance_thresholds_unchanged=True,raw_gate_m=.01,numerical_slack_m=slack,
                    certificate_uses_no_targets_or_method=True,preparation_join='RECHECK_BEFORE_FREEZE')
                atomic_json(folder/'SOURCE_POSITION_PASS.json',accepted)
                oldpass=base/'SOURCE_POSITION_PASS.json'
                if oldpass.exists() and not (folder/'PREVIOUS_SOURCE_POSITION_PASS.json').exists():atomic_json(folder/'PREVIOUS_SOURCE_POSITION_PASS.json',read(oldpass))
                atomic_json(oldpass,accepted);break
            if accepted is None:accepted=dict(case=case,qualified=False,candidates_tested=len(candidates))
            rows.append(accepted);print('MODEL_CERTIFIED_POSITION',case,'PASS' if 'trajectory' in accepted else 'FAIL',flush=True)
    result=dict(status='SOURCE_SMOKE3_POSITION_PASS' if all('trajectory' in r for r in rows) else 'SOURCE_POSITION_RECOVERY_REQUIRED',rows=rows,
        counts={m:sum('trajectory' in r for r in rows if r['case'].startswith(m)) for m in ('WRIST','INTERACTION')},
        certificate=file_record(proof),raw_gate_m=.01,numerical_slack_m=slack,all_thresholds_unchanged=True,next='FIXED_PREPARATION_AND_TRAIN11')
    rp=ST5/'MODEL_CERTIFIED_SMOKE_RESULT.json';atomic_json(rp,result)
    log('MODEL_CERTIFIED_SOURCE_POSITION',result['status'],[rp,proof], 'Old radius relaxation was 6.303 micrometres looser than a tighter model-only certified envelope',
        'GEOMETRIC_LOWER_BOUND_LOOSENESS_NOT_PHYSICAL_LIMIT',
        'Kept raw 10mm and numerical 10um gates unchanged; validated tighter coupled-chain certificate on both sides, 5000 FK samples and all602 saved witnesses; independently rechecked geometry',
        [rp],result['next'],f"기하 하한의 느슨함을 모델만으로 검증·개선했습니다(좌우 공통 6.303µm). 10mm 원시 게이트와 10µm 정밀도는 그대로입니다. 소스 위치 검증 A {result['counts']['WRIST']}/3, B {result['counts']['INTERACTION']}/3. 준비 구간과 TRAIN11 검증을 계속합니다.")

if __name__=='__main__':main()
