#!/usr/bin/env python3
"""Publish the unresolved scientific temporal-gate discrepancy with diagnostics."""
from pathlib import Path
import sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_dual_position import *
from tools.run_reference_motion_scientific_reset import atomic_csv,atomic_text

def run():
    verified_oracle_contract();rows=[];inputs=[file_record(QUAL)]
    for case in ('WRIST_EP024','WRIST_EP049'):
        folder=RUN/case/'aggregate_step_semantics_diagnostic_v1'
        diagnostic=read(folder/'ATTEMPT_0.json')
        strict=read(RUN/case/'recovered_branch_strict_retry_v1/ATTEMPT_1.json')
        geom=read(folder/'DETAILED_GEOMETRY_0.json')
        classes=collections.defaultdict(set);pairs=collections.Counter()
        for f in geom['geometry']:
            for r in f['records']:
                classes[r['classification']].add(f['frame'])
                if r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY'):
                    pairs[str(r.get('pair',r.get('link_pair',r.get('links','see full record'))))]+=1
        inputs.extend([file_record(folder/'ATTEMPT_0.json'),file_record(folder/'DETAILED_GEOMETRY_0.json'),file_record(RUN/case/'recovered_branch_strict_retry_v1/ATTEMPT_1.json')])
        rows.append(dict(case=case,diagnostic_original_numeric_pass=diagnostic['original_evaluator_numeric_pass'],
            diagnostic_strict_pass=diagnostic['strict_metrics']['pass_numeric'],diagnostic_metrics=diagnostic['strict_metrics'],
            strict_recovery_metrics=strict['metrics'],detailed_classification_frame_counts={k:len(v) for k,v in classes.items()},
            detailed_blocked_frames=geom['blocked_frames'],collision_pair_counts=dict(pairs),geometry_candidate=file_record(Path(geom['trajectory']['path']))))
    folder=MASTER/'diagnostics/temporal_gate_semantics_v1'
    result=dict(status='SCIENTIFIC_CONTRACT_DECISION_REQUIRED',true_stop_condition=7,
        issue='Predeclared config and solver enforce a fixed aggregate L2 step cap; original evaluator uses per-joint bounds and adaptive branch discontinuity without that cap.',
        no_gate_changed_for_qualification=True,diagnostic_not_promoted=True,targets_unchanged=True,
        no_global_temporal_infeasibility_proof=True,
        decision_required='Is 0.179rad an independent hard experiment acceptance constraint, or only an internal solver trust radius superseded by unchanged per-joint velocity/acceleration and adaptive branch checks?',
        choices=['Keep0.179rad as hard acceptance: retain current failed status and require a new common strict trajectory method/certificate.',
                 'Explicitly amend common temporal acceptance to original evaluator semantics for BOTH methods; rerun position/geometry/preparation/TRAIN11. This is NOT an automatic pass.'],
        rows=rows,inputs=inputs,
        source_code_evidence=[file_record(ROOT/'tools'/p) for p in ('final_common_position_solver.py','final_single_variable_qualify_position.py','doll_handoff_retargeting/common.py','master_autonomous_common.py')])
    atomic_json(folder/'TEMPORAL_ACCEPTANCE_CONTRACT_AUDIT.json',result)
    text='# Temporal acceptance contract discrepancy\n\nTrue stop condition7: a material scientific acceptance choice is not resolved by repository evidence. No gate amendment has been applied.\n\n'
    text+='The predeclared POSITION_SOLVER_QUALIFICATION_CONTRACT.json contains maximum_step_norm_rad=0.179, and the common sequential solver clips to it. In contrast, final_single_variable_qualify_position.evaluate checks maximum scalar step0.15rad, per-joint velocity4.5rad/s, acceleration130rad/s² and branch_flags, which flags only an L2 jump above max(0.18rad,8×local median). It does not independently reject every aggregate step above0.179rad. The autonomous startup contract retained the old limits; this run therefore retained the stricter interpretation for qualification.\n\n'
    text+='A common non-promoting diagnostic removed only the aggregate trust cap from its search, replacing it by the redundant sqrt(14)×per-joint bound. It found Cartesian/per-joint/branch-valid source trajectories. Applying the unchanged strict rules again from those new seeds still failed. Detailed collision checks also found blocked frames in the diagnostic trajectories. Thus neither trajectory is promoted, and changing the cap alone is not a complete solution.\n\n'
    text+='| Case | Diagnostic Cartesian failures | qdot | qddot | L2 step | Adaptive branches | Detailed blocked | Strict retry Cartesian failures |\n|---|---:|---:|---:|---:|---:|---:|---|\n'
    for r in rows:
        m=r['diagnostic_metrics'];tm=m['temporal']
        text+=f"| {r['case']} | {len(m['failed_frames'])} | {tm['maximum_velocity_rad_s']:.6f} | {tm['maximum_acceleration_rad_s2']:.6f} | {tm['maximum_step_norm_rad']:.6f} | {m['branch_discontinuities']} | {len(r['detailed_blocked_frames'])} | {r['strict_recovery_metrics']['failed_frames']} |\n"
    text+='\nTRAIN evidence proves that the two rules are not equivalent; it does not establish which rule was scientifically intended. Choosing the more permissive rule after seeing failures without an explicit contract amendment would be outcome-conditioned gate selection. Retaining it is conservative, but finite strict-search failure is not a proof of global temporal infeasibility.\n\nDecision needed: retain0.179rad as an independent hard constraint, or explicitly adopt the original evaluator’s per-joint/adaptive-branch acceptance for both methods. If amended, repeat complete common geometry, preparation joins and TRAIN11 qualification before6D. No A-specific waiver, source-clock change, target movement or task-success tuning is authorized or applied.\n'
    atomic_text(folder/'TEMPORAL_ACCEPTANCE_CONTRACT_AUDIT.md',text)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,6),constrained_layout=True)
    for i,row in enumerate(rows):
        case=row['case'];left=axs[i,0];right=axs[i,1]
        for sub,name,color in [('recovered_branch_strict_retry_v1/ATTEMPT_1.npz','Strict-cap recovery','#b04c43'),('aggregate_step_semantics_diagnostic_v1/ATTEMPT_0.npz','Non-promoted diagnostic','#3976a0')]:
            z=np.load(RUN/case/sub);q=z['q'];dt=float(np.median(np.diff(z['source_timestamp'])));time=z['source_timestamp']-z['source_timestamp'][0]
            left.plot(time[1:],np.max(np.abs(np.diff(q,axis=0)),axis=1)/dt,lw=.8,label=name,color=color)
            right.plot(time[1:],np.linalg.norm(np.diff(q,axis=0),axis=1),lw=.8,label=name,color=color)
        left.axhline(4.5,color='black',ls='--',lw=.8);right.axhline(.179,color='black',ls='--',lw=.8)
        left.set_title(case+' — per-joint velocity');right.set_title(case+' — aggregate step')
        left.set_ylabel('rad/s');right.set_ylabel('rad');left.set_xlabel('Source time (s)');right.set_xlabel('Source time (s)')
    axs[0,0].legend(fontsize=8);fig.suptitle('TRAIN diagnostic: non-equivalent temporal rules\nNeither alternative candidate is collision-qualified; no DEV35 results')
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'FIGURE_TEMPORAL_GATE_DISCREPANCY.{ext}',dpi=180)
    plt.close(fig)
    atomic_json(folder/'MANIFEST.json',dict(status=result['status'],inputs=inputs,implementation=file_record(Path(__file__)),artifacts=[file_record(p) for p in sorted(folder.iterdir()) if p.is_file()]))
    print(text,flush=True)

if __name__=='__main__':run()
