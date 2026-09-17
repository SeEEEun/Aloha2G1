#!/usr/bin/env python3
"""Disclose incompatible candidate gates without mixing their favorable metrics."""
from pathlib import Path
import sys,csv,io
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *

def main():
    verified_oracle_contract();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=full_chain_enclosures(g)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'];folder=OUT/'07_paper_artifacts/diagnostic_position_v5'
    ids=read(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')['selection']['qualification_ids'];rows=[]
    stream=io.StringIO();csvout=csv.writer(stream);csvout.writerow(['case','candidate_kind','frame','left_residual_mm','right_residual_mm','left_allowance_mm','right_allowance_mm','maximum_excess_mm','classification','trajectory'])
    for mode in ('WRIST','INTERACTION'):
        for ep in ids:
            case=f'{mode}_EP{ep:03d}';base=TRAIN/case
            if (base/'SOURCE_POSITION_PASS.json').exists():continue
            candidates=[]
            for p in base.rglob('FINAL_QUALIFICATION.json'):
                r=read(p);candidates.append((p,r))
            chosen=[]
            numeric=[x for x in candidates if x[1]['metrics']['pass_numeric']]
            clear=[x for x in candidates if not x[1]['blocked_frames'] and not x[1]['preparation_blocked_frames']]
            if numeric:chosen.append(('NUMERIC_VALID_GEOMETRY_REJECTED',min(numeric,key=lambda x:(len(x[1]['blocked_frames']),str(x[0])))))
            if clear:chosen.append(('GEOMETRY_CLEAR_CARTESIAN_REJECTED',min(clear,key=lambda x:(len(x[1]['metrics']['failed_frames']),str(x[0])))))
            if not chosen:
                path=base/'common_local_conic_v2/FINAL_CANDIDATE.npz';assert path.exists()
                chosen.append(('NUMERIC_ONLY_GEOMETRY_NOT_QUALIFIED',(None,dict(trajectory=file_record(path)))))
            t,h,ts,source=load_input(case,g,n)
            for kind,(p,r) in chosen:
                path=Path(r['trajectory']['path']);q=np.load(path)['q'];m,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
                bad=np.flatnonzero(np.any(res>allow,axis=1));details=[]
                for f in bad:
                    classification='CERTIFIED_OPTIMALITY_GAP' if np.any((res[f]>allow[f])&(lb[f]>.01)) else 'RAW_CARTESIAN_RESIDUAL'
                    item=dict(frame=int(f),residual_mm=(1000*res[f]).tolist(),allowance_mm=(1000*allow[f]).tolist(),
                        lower_bound_mm=(1000*lb[f]).tolist(),maximum_excess_mm=float(1000*np.max(res[f]-allow[f])),classification=classification)
                    details.append(item);csvout.writerow([case,kind,int(f),*(1000*res[f]),*(1000*allow[f]),item['maximum_excess_mm'],classification,str(path)])
                rows.append(dict(case=case,candidate_kind=kind,trajectory=file_record(path),qualification=None if p is None else file_record(p),
                    source=source,metrics=m,cartesian_failures=details,hard_collision_frames=r.get('hard_collision_frames'),
                    unresolved_geometry_frames=r.get('unresolved_geometry_frames'),blocked_frames=r.get('blocked_frames'),
                    maximum_cartesian_excess_mm=float(1000*np.maximum(res-allow,0).max()),global_infeasibility_proven=False))
    atomic_json(folder/'REMAINING_TRAIN_POSITION_FAILURES.json',dict(rows=rows,scope='OBSERVED_BOUNDED_CANDIDATE_FAILURES',
        warning='Alternative candidates are distinct q trajectories. Their favorable gates must never be combined into a pass. No global impossibility proof.'))
    atomic_text(folder/'REMAINING_TRAIN_POSITION_FAILURES.csv',stream.getvalue())
    lines=['# Remaining bounded TRAIN position failures\n','Each row is one complete saved candidate, not a mixture of solutions. This table does not prove global infeasibility.\n',
        '| Case | Candidate | Cartesian failed frames | Maximum excess (mm) | Geometry-invalid frames | Max qdot (rad/s) | Max qddot (rad/s²) | Branch flags |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['metrics'];tm=m['temporal'];geom='NOT QUALIFIED' if r['blocked_frames'] is None else len(r['blocked_frames'])
        lines.append(f"| {r['case']} | {r['candidate_kind']} | {len(r['cartesian_failures'])} | {r['maximum_cartesian_excess_mm']:.9f} | {geom} | {tm['maximum_velocity_rad_s']:.9f} | {tm['maximum_acceleration_rad_s2']:.9f} | {m['branch_discontinuities']} |")
    lines.extend(['','Exact failed-frame residuals, per-wrist allowances, model lower bounds, hashes and source paths are in the accompanying JSON/CSV. Reachable wrists retain the10mm raw gate; only previously certified wrists use the unchanged certified-bound plus10micrometre slack. Geometry NOT QUALIFIED is not a clear result.'])
    atomic_text(folder/'REMAINING_TRAIN_POSITION_FAILURES.md','\n'.join(lines)+'\n')
    print('REMAINING_TRAIN_FAILURE_REPORT',len(rows),'saved candidates',flush=True)

if __name__=='__main__':main()
