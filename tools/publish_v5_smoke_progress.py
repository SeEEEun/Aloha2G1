#!/usr/bin/env python3
"""Publish completed smoke evidence without implying later-stage completion."""
from pathlib import Path
import sys,csv,io,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_local_redundancy_v5 import *
from tools.common_full_chain_certificate_v5 import full_chain_enclosures

def main():
    verified_oracle_contract();result=read(ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json')
    assert result['status']=='SMOKE3_COMMON_EXECUTABLE_POSITION_QUALIFIED'
    folder=ST5/'smoke_diagnostics';table=[]
    for r in result['rows']:
        m=r['source_metrics'];tm=r['complete_temporal']
        table.append(dict(case=r['case'],qualified=r['qualified'],raw_10mm_fraction=m['raw_acceptance'],certified_unreachable=len(m['certified_unreachable_frames']),
            correction_mean_mm=m['correction_mm']['mean'],correction_p95_mm=m['correction_mm']['p95'],correction_max_mm=m['correction_mm']['max'],
            maximum_velocity_rad_s=tm['maximum_velocity_rad_s'],maximum_acceleration_rad_s2=tm['maximum_acceleration_rad_s2'],
            hard_collisions=len(r['hard_collision_frames']),unresolved_geometry=len(r['unresolved_geometry_frames']),
            hard_limit_violations=r['hard_limit_violations'],branch_discontinuities=len(tm['branch_discontinuity_frames']),
            proxy_only_frames=len(r['proxy_only_overlap_frames'])))
    textio=io.StringIO();writer=csv.DictWriter(textio,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    atomic_text(folder/'TABLE_SMOKE3_POSITION_QUALIFICATION.csv',textio.getvalue())
    md='# Common SMOKE3 executable-position qualification\n\nTRAIN-side kinematic qualification, not physical task success. Fixed natural q0 and0.7s preparation retained.\n\n| Case | Qualified | Raw≤10mm | Certified unreachable | Correction mean/p95/max mm | Hard / unresolved | qdot max | qddot max |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in table:
        md+=f"| {r['case']} | {r['qualified']} | {100*r['raw_10mm_fraction']:.2f}% | {r['certified_unreachable']} | {r['correction_mean_mm']:.3f}/{r['correction_p95_mm']:.3f}/{r['correction_max_mm']:.3f} | {r['hard_collisions']}/{r['unresolved_geometry']} | {r['maximum_velocity_rad_s']:.6f} | {r['maximum_acceleration_rad_s2']:.6f} |\n"
    atomic_text(folder/'TABLE_SMOKE3_POSITION_QUALIFICATION.md',md)
    g,c,n=model();s=CommonPositionSolver(g,c,read(QUAL),n);g.assign(n);bounds=full_chain_enclosures(g)
    record=read(V5/'WRIST_EP024/SOURCE_POSITION_PASS.json');q=np.load(record['trajectory']['path'])['q']
    oldpath=Path(read(V5/'WRIST_EP024/INPUT.json')['trajectory']['path']);old=np.load(oldpath)['q']
    t,h,ts,source=load_input('WRIST_EP024',g,n);slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    oldm,_,oldres,lb,cert,allow=qualify_numeric(s,old,t,h,ts,bounds,slack)
    newm,_,newres,_,_,_=qualify_numeric(s,q,t,h,ts,bounds,slack)
    changed=np.flatnonzero(np.any(q!=old,axis=1)).tolist()
    detail=[]
    for f in (166,167,171):
        detail.append(dict(frame=f,old_q=old[f].tolist(),new_q=q[f].tolist(),target=t[f].tolist(),
            old_residual_mm=(oldres[f]*1000).tolist(),new_residual_mm=(newres[f]*1000).tolist(),
            old_geometry=c.inspect(old[f],*h[f]),new_geometry=c.inspect(q[f],*h[f])))
    atomic_json(folder/'A24_LOCAL_REPAIR_DETAIL.json',dict(frames=detail,changed_frames=changed,
        input=file_record(oldpath),output=record['trajectory'],raw_targets_modified=False,source_event_timing_modified=False,
        new_model_certificate=file_record(ST5/'full_chain_certificate/MODEL_ONLY_CERTIFICATE.json')))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True,layout='constrained');f=np.arange(len(q));window=(f>=145)&(f<=185)
    axes[0].plot(f[window],1000*oldres[window,0],label='Before repair',color='#a55b31')
    axes[0].plot(f[window],1000*newres[window,0],label='Geometry-qualified candidate',color='#146782')
    axes[0].plot(f[window],1000*allow[window,0],'k--',label='Unchanged rule with tighter model certificate')
    axes[0].axhline(10,color='gray',lw=.7);axes[0].set_ylabel('Left residual (mm)');axes[0].legend(fontsize=8)
    axes[1].plot(f[window],1e6*(newres[window,0]-allow[window,0]),color='#146782');axes[1].axhline(0,color='black',ls='--')
    axes[1].set_ylabel('Residual minus allowance (µm)');axes[1].set_ylim(-20,2)
    axes[2].bar([166,167],[.13490747592090752,.23655911554461892],color='#a55b31',label='Before: certified penetration lower bound')
    axes[2].plot(f[window],np.zeros(window.sum()),color='#146782',label='After: zero hard / unresolved frames')
    axes[2].set_ylabel('Penetration lower bound (mm)');axes[2].set_xlabel('Original source frame');axes[2].legend(fontsize=8)
    fig.suptitle('A24 common local repair — TRAIN kinematic diagnostic (not PhysX task outcome)')
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'FIGURE_A24_COMMON_LOCAL_REPAIR.{ext}',dpi=220)
    plt.close(fig)
    modules=[ROOT/'tools'/name for name in ('common_local_redundancy_v5.py','common_geometry_boundary_v5.py','common_full_chain_certificate_v5.py',
        'common_conic_local_realization_v5.py','common_conic_trust_realization_v5.py','common_elastic_local_realization_v5.py','common_physical_position_v4.py')]
    forbidden=[]
    for p in modules:
        tree=ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value in ('WRIST','INTERACTION','A','B'):
                forbidden.append(dict(path=str(p),line=node.lineno,value=node.value))
    assert not forbidden
    parity='# Common pipeline parity — position repair v5\n\nA/B raw representation generation remains the only intended experimental difference. All geometry, redundancy, numerical recovery and final acceptance implementations consume targets/model/q/time, not method or outcome labels. Labels in orchestration locate frozen input/output files only. The same final evaluator ran all six smoke cases, including unchanged valid trajectories.\n\nThe 10mm raw gate,10µm closest-feasible numerical slack,10µm geometry tolerance,4.5rad/s and130rad/s² per-joint limits, adaptive branch test, natural q0 and0.7s preparation were not changed. A new target-blind full-chain outer certificate tightens the previous relaxed radius by6.303µm on BOTH sides. It has an analytic global enclosure proof and passed5000 FK consistency checks and all602 existing dense witnesses. This improves the certified lower bound; it does not move raw targets or manufacture a collision waiver. The old radius and all failed attempts remain provenance.\n\nScope: completed SMOKE3 position and preparation only. TRAIN11 in progress. Full6D, loaded Dex3, datasets/training parity and physical evaluation NOT_RUN. No final downstream freeze yet.\n'
    atomic_text(OUT/'02_common_execution_qualification/FINAL_COMMON_PIPELINE_PARITY_AUDIT_V5.md',parity)
    atomic_json(folder/'PARITY_AUDIT.json',dict(method_identity_constants_found=forbidden,common_core=[file_record(p) for p in modules],
        all_six_final_evaluations=file_record(ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json'),dataset_parity='NOT_AUDITED'))
    report=f'''# Final single-variable A/B rebuild — in progress

Completed: common SMOKE3 executable position and fixed preparation, A3/3 and B3/3. All six have zero geometry-confirmed hard collision, unresolved geometry, hard-limit violations and adaptive branch discontinuities. Per-joint velocity/acceleration gates pass. Natural q0 and the0.7s preparation duration are preserved.

A24's original frames166–167 are repaired through redundant joint motion with unchanged raw targets and source timestamps. Detailed geometry confirms clearance; no pair whitelist, tolerance waiver or task-success objective was used. A target-blind coupled-chain certificate also removes6.303µm of looseness from the prior outer-radius bound for both arms; raw10mm and numerical10µm acceptance thresholds remain unchanged. The global enclosure derivation and all602 saved-witness consistency checks are persisted.

Current next gate: fixed11-episode TRAIN position qualification, now running. This is not full common-execution qualification or a final freeze.

## Evidence

- Source/preparation qualification: `{ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json'}`
- Repair detail: `{folder/'A24_LOCAL_REPAIR_DETAIL.json'}`
- Model-only certificate: `{ST5/'full_chain_certificate/MODEL_ONLY_CERTIFICATE.json'}`
- Certificate validation: `{ST5/'full_chain_certificate/VALIDATION.json'}`
- Diagnostic table: `{folder/'TABLE_SMOKE3_POSITION_QUALIFICATION.md'}`
- Diagnostic figure: `{folder/'FIGURE_A24_COMMON_LOCAL_REPAIR.png'}`

## Unrun gates — no results claimed

Full6D; loaded Dex3; complete reference pipeline; common execution freeze; exact old/new action and dataset audit; regeneration/retraining decision; ACT training; policy sanity; DEV35 smoke/freeze;70 matched physical rollouts; stage success/TSR/paired statistics; final paper figures/tables; actual PhysX replay videos. These are NOT_RUN, not zero success rates. DEV35 remains development data, not untouched test data.

All earlier physical results remain PRE_FINAL_SINGLE_VARIABLE_REBUILD_DIAGNOSTIC_ONLY. Earlier blocker reports and every failed numerical trial are preserved. Work is continuing, not complete.
'''
    atomic_text(OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md',report)
    atomic_text(OUT/'CURRENT_STATUS.md','# Current state\n\nIN_PROGRESS: common SMOKE3 executable position and fixed0.7s preparation QUALIFIED, A3/3 B3/3. A24 local hard collision repaired. All thresholds/raw targets/event timing unchanged. Target-blind full-chain certificate validated. Fixed TRAIN11 position qualification running; later stages NOT_RUN.\n')
    artifacts=[file_record(p) for p in folder.iterdir() if p.is_file()]
    atomic_json(folder/'DIAGNOSTIC_MANIFEST.json',dict(status='SMOKE3_POSITION_ONLY',artifacts=artifacts,final_experiment_complete=False,
        inputs=[file_record(ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json'),file_record(ST5/'full_chain_certificate/VALIDATION.json')]))

if __name__=='__main__':main()
