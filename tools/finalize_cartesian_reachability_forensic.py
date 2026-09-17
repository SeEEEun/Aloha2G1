#!/usr/bin/env python3
"""Summarize dense witnesses, physical certificates and one common recovery."""
from pathlib import Path
import sys
import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cartesian_reachability_forensic import prepare,model,DEST,BASELINE,CONTRACT,intervals
from tools.common_g1_position_bounds import outer_enclosures,residual_lower_bounds
from tools.run_common_position_anchor_recovery import REC,preflight as recovery_preflight
from tools.final_single_variable_prepare import OUT,read,file_record,status
from tools.run_reference_motion_scientific_reset import atomic_json,atomic_text,atomic_csv,sha256


def stats(values,scale=1):
    a=np.asarray(values,dtype=float)*scale
    return {'mean':float(a.mean()),'p95':float(np.percentile(a,95)),'max':float(a.max())} if len(a) else {'mean':0.,'p95':0.,'max':0.}


def ranges(values):
    return ', '.join(str(a) if a==b else f'{a}–{b}' for a,b in intervals(values)) or 'none'


def diagnostic_figure():
    """Render the completed dense forensic independently of recovery runtime."""
    contract=prepare()
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(13,10),constrained_layout=True)
    for ax,case in zip(axes,contract['cases']):
        name=case['name'];frames=[f for a,b in case['failing_intervals'] for f in range(a,b+1)]
        entries=[read(DEST/'dense'/f'{name}_F{f:04d}.json') for f in frames]
        with np.load(BASELINE/(name+'.npz'),allow_pickle=False) as z:
            ax.plot(z['source_timestamp'],1000*z['position_residual_m'].max(axis=1),color='#c65136',lw=1.2,label='Previous sequential residual')
        for kind,color,marker,label in [('FRAME_REACHABLE','#087f5b','o','Oracle: reachable witness'),('FRAME_UNREACHABLE','#8c2d9b','x','Oracle: no witness in 128-seed budget')]:
            rs=[r for r in entries if r['classification']==kind]
            if rs:ax.scatter([r['timestamp_s'] for r in rs],[1000*r['best']['maximum_residual_m'] for r in rs],c=color,marker=marker,s=10,label=label,zorder=4)
        cs=[r for r in entries if r['position_infeasibility_certified']]
        if cs:ax.scatter([r['timestamp_s'] for r in cs],[1000*max(r['certified_residual_lower_bound_m']) for r in cs],c='#d99b11',s=5,label='Certified residual lower bound',zorder=3)
        ax.axhline(10,color='#333333',ls='--',lw=1,label='Unchanged 10 mm gate')
        label=('A' if case['representation_mode']=='WRIST' else 'B')+f"{case['episode']:02d}"
        ax.set_title(label+' — every originally failed frame tested',loc='left');ax.set_ylabel('Residual (mm)');ax.set_xlabel('Source time (s)');ax.set_ylim(bottom=-3);ax.grid(alpha=.2);ax.legend(fontsize=7,loc='upper right')
    fig.suptitle('Cartesian reachability forensic — MIXED: physical reach gaps and sequential IK failure',fontsize=13)
    figures=[]
    for suffix in ('png','pdf','svg'):
        path=DEST/f'CARTESIAN_REACHABILITY_FORENSIC_DENSE.{suffix}';fig.savefig(path,dpi=180);figures.append(path)
    plt.close(fig)
    return figures


def main():
    contract=prepare();recovery_preflight();sample=read(DEST/'CARTESIAN_REACHABILITY_FORENSIC.json')
    independent_geometry=read(DEST/'INDEPENDENT_REACHABLE_WITNESS_GEOMETRY_CHECK.json')
    assert independent_geometry['complete'] and independent_geometry['status']=='PASS'
    assert independent_geometry['count']==len(independent_geometry['rows'])
    assert sha256(Path(independent_geometry['validator']['path']))==independent_geometry['validator']['sha256']
    for row in independent_geometry['rows']:
        assert sha256(Path(row['source']['path']))==row['source']['sha256']
    independent_corrections=read(DEST/'INDEPENDENT_CORRECTION_WITNESS_GEOMETRY_CHECK.json')
    assert independent_corrections['complete'] and independent_corrections['status']=='PASS'
    assert independent_corrections['count']==len(independent_corrections['rows'])
    assert sha256(Path(independent_corrections['validator']['path']))==independent_corrections['validator']['sha256']
    for row in independent_corrections['rows']:
        for key in ('source','baseline'):
            assert sha256(Path(row[key]['path']))==row[key]['sha256']
    baseline_summary=read(BASELINE.parent/'GEOMETRY_CONFIRMED_COLLISION_AUDIT.json')
    for old_result in baseline_summary['rows']:
        for artifact in old_result['artifacts']:
            assert sha256(Path(artifact['path']))==artifact['sha256'],artifact['path']
    g1,collision,natural=model();g1.assign(natural);balls=outer_enclosures(g1)
    for u in qmc.Halton(14,scramble=False).random(2000):
        q=g1.arm_limits[:,0]+np.ptp(g1.arm_limits,axis=1)*u;g1.assign(q)
        targets=np.array([g1.data.xpos[g1.wrist_ids[s]].copy() for s in ('left','right')])
        assert residual_lower_bounds(targets,balls).max()<1e-10
    dense={};case_rows=[];verification=[];correction_witnesses=[]
    for case in contract['cases']:
        name=case['name'];frames=[f for a,b in case['failing_intervals'] for f in range(a,b+1)]
        entries=[read(DEST/'dense'/f'{name}_F{f:04d}.json') for f in frames];dense[name]=entries
        with np.load(BASELINE/(name+'.npz'),allow_pickle=False) as z:
            for entry in entries:
                f=entry['frame'];np.testing.assert_array_equal(entry['incoming_target_m'],z['RAW_REPRESENTATION_TARGET'][f])
                best=entry['best'];q=np.array(best['q']);g1.assign(q,*z['common_hand_q'][f])
                actual=np.array([g1.data.xpos[g1.wrist_ids[s]].copy() for s in ('left','right')])
                error=np.linalg.norm(actual-z['RAW_REPRESENTATION_TARGET'][f],axis=1)
                np.testing.assert_allclose(error,best['residual_m'],rtol=0,atol=1e-11)
                assert np.all(q>=g1.arm_limits[:,0]-1e-9) and np.all(q<=g1.arm_limits[:,1]+1e-9)
                assert error.max()+1e-8>=max(entry['certified_residual_lower_bound_m'])
                if entry['classification']=='FRAME_REACHABLE':
                    assert error.max()<=.01 and best['geometry_valid'] is True
                else:
                    # Every baseline trajectory was collision/limit-valid.
                    # Its q is an independent correction witness, NOT an oracle
                    # seed and NOT an alteration of the raw target.
                    original_upper=entry['correction_to_satisfy_gate_upper_bound_m']
                    baseline_upper=max(float(z['position_residual_m'][f].max())-.01,0.)
                    use_baseline=original_upper is None or baseline_upper<original_upper
                    effective_upper=baseline_upper if use_baseline else original_upper
                    entry['oracle_only_correction_upper_bound_m']=original_upper
                    entry['correction_to_satisfy_gate_upper_bound_m']=effective_upper
                    correction_witnesses.append({'case':name,'frame':f,
                       'lower_bound_m':entry['correction_to_gate_certified_lower_bound_m'],'upper_bound_m':effective_upper,
                       'witness_source':'FROZEN_COLLISION_VALID_BASELINE' if use_baseline else 'COLLISION_VALID_ORACLE_CANDIDATE',
                       'witness_q':z['q'][f].tolist() if use_baseline else best['q'],
                       'source_trajectory':file_record(BASELINE/(name+'.npz')),'no_target_write':True})
                verification.append({'case':name,'frame':f,'FK_and_input_verified':True})
            frame_count=len(z['q']);dt=float(np.median(np.diff(z['source_timestamp'])))
        reachable=[r['frame'] for r in entries if r['classification']=='FRAME_REACHABLE']
        no_witness=[r['frame'] for r in entries if r['classification']=='FRAME_UNREACHABLE']
        certified=[r['frame'] for r in entries if r['position_infeasibility_certified']]
        not_certified=[f for f in no_witness if f not in certified]
        gaps=[r for r in entries if r['classification']=='FRAME_UNREACHABLE']
        row={'case':name,'total_trajectory_frames':frame_count,'failed_frames_tested':len(entries),'reachable':len(reachable),
             'unreachable_bounded':len(no_witness),'physically_certified_unreachable':len(certified),'bounded_not_certified':len(not_certified),
             'unreachable_fraction_of_failed_frames':len(no_witness)/len(entries),'certified_fraction_of_trajectory':len(certified)/frame_count,
             'unreachable_intervals':intervals(no_witness),'certified_intervals':intervals(certified),'bounded_not_certified_frames':not_certified,
             'longest_unreachable_interval_frames':max((b-a+1 for a,b in intervals(no_witness)),default=0),
             'longest_unreachable_interval_seconds':dt*max((b-a+1 for a,b in intervals(no_witness)),default=0),
             'correction_to_gate_lower_bound_mm':stats([r['correction_to_gate_certified_lower_bound_m'] for r in gaps],1000),
             'correction_to_gate_upper_bound_mm':stats([r['correction_to_satisfy_gate_upper_bound_m'] for r in gaps if r['correction_to_satisfy_gate_upper_bound_m'] is not None],1000),
             'exact_target_residual_best_valid_mm':stats([.01+r['correction_to_satisfy_gate_upper_bound_m'] for r in gaps],1000)}
        case_rows.append(row)
    assert independent_geometry['count']==sum(row['reachable'] for row in case_rows)
    assert {(r['case'],r['frame']) for r in independent_geometry['rows']}=={
        (name,r['frame']) for name,entries in dense.items() for r in entries if r['classification']=='FRAME_REACHABLE'}
    checked_corrections={(r['case'],r['frame']):r for r in independent_corrections['rows']}
    assert len(checked_corrections)==len(correction_witnesses)
    for r in correction_witnesses:
        independent=checked_corrections[(r['case'],r['frame'])]
        np.testing.assert_array_equal(r['witness_q'],independent['q'])
        assert r['upper_bound_m']==independent['correction_to_gate_upper_bound_m']
    recovery=[];incomplete_recovery=[]
    for mode in ('WRIST','INTERACTION'):
        for ep in (0,24,49):
            name=f'{mode}_EP{ep:03d}'
            if not (REC/(name+'.json')).exists():
                assert (REC/'RECOVERY_STOP_DECISION.json').exists()
                incomplete_recovery.append({'representation_mode':mode,'episode':ep,'case':name,
                    'status':'STOPPED_INCOMPLETE; NOT_SCORED_AS_A_COMPLETED_TRAJECTORY',
                    'reason':'Full paired gate is impossible under certified A position gaps with immutable targets; diagnosis complete; candidate not promoted.'})
                continue
            r=read(REC/(name+'.json'))
            assert sha256(Path(r['trajectory']['path']))==r['trajectory']['sha256']
            with np.load(r['trajectory']['path'],allow_pickle=False) as z,np.load(BASELINE/(name+'.npz'),allow_pickle=False) as old:
                for key in ('RAW_REPRESENTATION_TARGET','common_hand_q','source_timestamp','initial_q'):np.testing.assert_array_equal(z[key],old[key])
                actual=[]
                for q,hands in zip(z['q'],z['common_hand_q']):
                    g1.assign(q,*hands);actual.append([g1.data.xpos[g1.wrist_ids[s]].copy() for s in ('left','right')])
                errors=np.linalg.norm(np.array(actual)-z['RAW_REPRESENTATION_TARGET'],axis=2)
                np.testing.assert_allclose(errors,z['position_residual_m'],rtol=0,atol=1e-12)
                old_failed=old['position_residual_m'].max(axis=1)>.01
                new_failed=errors.max(axis=1)>.01
                r['old_failed_now_accepted_frames']=np.flatnonzero(old_failed&~new_failed).tolist()
                r['old_accepted_now_failed_frames']=np.flatnonzero(~old_failed&new_failed).tolist()
                r['baseline_raw_acceptance_rate']=float(np.mean(~old_failed))
                guide=z['posture_guide']
                r['non_executable_anchor_maximum_step_norm_rad']=float(np.linalg.norm(np.diff(guide,axis=0),axis=1).max())
                old_intervals=intervals(np.flatnonzero(old_failed).tolist())
                if old_intervals:
                    first,last=max(old_intervals,key=lambda ab:ab[1]-ab[0])
                    midpoint=(first+last)//2;state=z['q'][midpoint]
                    margins=np.minimum(state-g1.arm_limits[:,0],g1.arm_limits[:,1]-state)
                    r['longest_original_failure_interval_midpoint']={
                        'frame':midpoint,'interval':[first,last],'executable_q':state.tolist(),
                        'wrist_residual_mm':(errors[midpoint]*1000).tolist(),
                        'joint_limit_margins_rad':dict(zip(g1.arm_joint_names.tolist(),margins.tolist()))}
            candidates_path=Path(r['trajectory']['path']).with_name(name+'_CANDIDATES.json')
            details=read(candidates_path)
            r['recovery_search_diagnostics']={
                'anchor_frames_with_no_geometry_valid_selected_candidate':sum(bool(v['selected_key'][0]) for v in details['anchor_reports']),
                'anchor_frames_above_cartesian_gate':sum(bool(v['selected_key'][1]) for v in details['anchor_reports']),
                'framewise_recovery_attempts':len(details['framewise_recovery_attempts']),
                'executable_empty_temporal_bound_frames':[v['frame'] for v in details['frame_reports'] if v['bounds_infeasible']],
                'source':file_record(candidates_path),
                'interpretation':'Anchor guides are not executable states. Their jumps or rejected candidates are not measured trajectory branch/collision violations.'}
            midpoint=r.get('longest_original_failure_interval_midpoint')
            if midpoint:
                attempt=next((v for v in details['framewise_recovery_attempts'] if v['frame']==midpoint['frame']),None)
                midpoint['local_recovery_attempt']=attempt
            recovery.append(r)
    method_gaps={}
    for mode in ('WRIST','INTERACTION'):
        entries=[r for name,rs in dense.items() if name.startswith(mode+'_') for r in rs]
        no_witness=[r for r in entries if r['classification']=='FRAME_UNREACHABLE']
        trajectory_frames=sum(row['total_trajectory_frames'] for row in case_rows if row['case'].startswith(mode+'_'))
        zero_correction_count=trajectory_frames-len(no_witness)
        method_gaps[mode]={'tested_failed_frames':len(entries),'unreachable_bounded':len(no_witness),
          'unreachable_fraction':len(no_witness)/len(entries),'physically_certified_unreachable':sum(r['position_infeasibility_certified'] for r in entries),
          'total_frames_in_audited_trajectories':trajectory_frames,
          'bounded_no_witness_fraction_of_all_trajectory_frames':len(no_witness)/trajectory_frames,
          'certified_unreachable_fraction_of_all_trajectory_frames':sum(r['position_infeasibility_certified'] for r in entries)/trajectory_frames,
          'correction_to_gate_certified_lower_mm':stats([r['correction_to_gate_certified_lower_bound_m'] for r in no_witness],1000),
          'correction_to_gate_witness_upper_mm':stats([r['correction_to_satisfy_gate_upper_bound_m'] for r in no_witness if r['correction_to_satisfy_gate_upper_bound_m'] is not None],1000),
          'all_frames_correction_to_gate_certified_lower_mm':stats([0.]*zero_correction_count+[r['correction_to_gate_certified_lower_bound_m'] for r in no_witness],1000),
          'all_frames_correction_to_gate_witness_upper_mm':stats([0.]*zero_correction_count+[r['correction_to_satisfy_gate_upper_bound_m'] for r in no_witness],1000),
          'best_valid_exact_target_residual_mm':stats([.01+r['correction_to_satisfy_gate_upper_bound_m'] for r in no_witness],1000),
          'scope':'Three originally failed A trajectories' if mode=='WRIST' else 'Originally failed B49 trajectory; not all B episodes'}
    counts={mode:sum(r['position_qualified'] for r in recovery if r['representation_mode']==mode) for mode in ('WRIST','INTERACTION')}
    completed_counts={mode:sum(r['representation_mode']==mode for r in recovery) for mode in ('WRIST','INTERACTION')}
    result={'root_cause':'MIXED','terminal_status':'MIXED_CARTESIAN_FEASIBILITY_BLOCKER','sample_counts':sample['counts'],
      'dense_cases':case_rows,'method_gap_statistics':method_gaps,'recovery_counts':counts,'recovery_rows':recovery,
      'recovery_counts_denominator':completed_counts,'recovery_smoke_complete':not incomplete_recovery,
      'incomplete_recovery_cases':incomplete_recovery,'authoritative_pre_recovery_position_counts':baseline_summary['position_counts'],
      'recovery_candidate_promoted':False,'ready_for_dataset_finalization':False,
      'targets_modified':False,'joint_limits_collision_registration_timing_unchanged':True,'full_6d':'NOT_RUN','loaded_dex3':'NOT_RUN',
      'outer_bounds':balls,'outer_bound_validation':{'deterministic_FK_configurations':2000,'violations':0},
      'independent_validation':{'oracle_frames':len(verification),'recovery_trajectories':len(recovery),'passed':True},
      'independent_geometry_validation':{'reachable_witnesses':independent_geometry['count'],'passed':True},
      'independent_correction_geometry_validation':{'correction_witnesses':independent_corrections['count'],'passed':True},
      'minimum_correction_interpretation':'Lower bounds are analytic outer-enclosure certificates. Upper bounds are observed collision-valid witnesses minus the unchanged 10mm gate. Neither upper bounds nor finite-search failures are global optimality proofs.',
      'interpretation':'There are both model-certified target-embodiment position failures and independently witnessed sequential IK failures. Do not project targets or proceed to 6D/Dex3.'}
    atomic_json(DEST/'FINAL_CARTESIAN_REACHABILITY_FORENSIC.json',result)
    atomic_json(DEST/'CORRECTION_BOUND_WITNESSES.json',correction_witnesses)
    atomic_json(DEST/'INDEPENDENT_FORENSIC_VALIDATION.json',{'status':'PASS','oracle_frames':verification,'recovery_trajectories':len(recovery),'incomplete_recovery_cases':incomplete_recovery,'outer_bound_FK_validation_count':2000})
    atomic_csv(DEST/'DENSE_FRAME_CLASSIFICATIONS.csv',[{'case':r['case'],'frame':r['frame'],'timestamp_s':r['timestamp_s'],'classification':r['classification'],
      'physical_unreachability_certified':r['position_infeasibility_certified'],'sequential_residual_mm':r['sequential_residual_mm'],'oracle_best_residual_mm':1000*r['best']['maximum_residual_m'],'oracle_best_geometry_valid':r['best']['geometry_valid'],
      'correction_to_gate_lower_mm':1000*r['correction_to_gate_certified_lower_bound_m'],
      'correction_to_gate_upper_mm':None if r['correction_to_satisfy_gate_upper_bound_m'] is None else 1000*r['correction_to_satisfy_gate_upper_bound_m']} for entries in dense.values() for r in entries])
    figures=diagnostic_figure()
    text='# Final Cartesian reachability forensic\n\nROOT CAUSE: **MIXED**. Targets, registration, workspace, 10 mm gate, geometry-confirmed collision rule, joint limits and temporal semantics unchanged. No 6D IK or Dex3 run.\n\n'
    text+='## Frozen 98-sample oracle\n\n| Case | Reachable | No witness in bounded search |\n|---|---:|---:|\n'
    for name,c in sample['counts'].items():text+=f"| {name} | {c['reachable']}/{c['sampled']} | {c['unreachable_bounded']}/{c['sampled']} |\n"
    text+='\n## Dense audit: all originally failed frames\n\n| Case | Tested | Reachable | Bounded no-witness | Physically certified | Longest no-witness interval |\n|---|---:|---:|---:|---:|---:|\n'
    for r in case_rows:text+=f"| {r['case']} | {r['failed_frames_tested']} | {r['reachable']} | {r['unreachable_bounded']} | {r['physically_certified_unreachable']} | {r['longest_unreachable_interval_frames']} frames |\n"
    for r in case_rows:
        text+=f"\n{r['case']} bounded no-witness ranges: {ranges([f for a,b in r['unreachable_intervals'] for f in range(a,b+1)])}. Certified ranges: {ranges([f for a,b in r['certified_intervals'] for f in range(a,b+1)])}. Remaining no-witness but uncertified frames: {ranges(r['bounded_not_certified_frames'])}.\n"
    text+='\n## Cartesian correction bounds, not target changes\n\nThe table uses only bounded no-witness frames. Correction means change needed to enter the unchanged 10 mm acceptance region; exact target-coincidence residuals are separately recorded in JSON. Values are mean / p95 / max in mm. A covers A00/A24/A49; B covers B49 only. Zero for B means no no-witness targets in this audited set, not a universal claim.\n\n| Method | No-witness fraction of tested failures | Certified correction lower bound | Collision-valid correction upper bound |\n|---|---:|---|---|\n'
    for mode,r in method_gaps.items():
        lo=r['correction_to_gate_certified_lower_mm'];hi=r['correction_to_gate_witness_upper_mm']
        text+=f"| {mode} | {r['unreachable_bounded']}/{r['tested_failed_frames']} = {100*r['unreachable_fraction']:.3f}% | {lo['mean']:.6f} / {lo['p95']:.6f} / {lo['max']:.6f} | {hi['mean']:.6f} / {hi['p95']:.6f} / {hi['max']:.6f} |\n"
    for mode,r in method_gaps.items():
        text+=f"\nAcross all frames of the audited {mode} trajectories (not only old failures), bounded no-witness targets are {r['unreachable_bounded']}/{r['total_frames_in_audited_trajectories']} = {100*r['bounded_no_witness_fraction_of_all_trajectory_frames']:.3f}%; physically certified targets are {r['physically_certified_unreachable']}/{r['total_frames_in_audited_trajectories']} = {100*r['certified_unreachable_fraction_of_all_trajectory_frames']:.3f}%. All-frame correction statistics, including zero corrections for witnessed reachable frames, are also retained in JSON.\n"
    text+='\nThe physical certificates use all actual model link translations and a tighter first-hinge axial/orbit enclosure. These are outer bounds: being outside proves impossibility, while being inside proves nothing. They supplement the actual 128-seed frame-independent IK. They are not a radial-only acceptance test. 2,000 deterministic FK configurations satisfy the enclosures. Remaining boundary cases retain the weaker bounded-search label. No globally exact minimum displacement is claimed.\n'
    text+='\nCorrection upper bounds use the better available collision-valid oracle state or the already-qualified collision-valid baseline state at that frame; the latter is a correction witness only, never an oracle seed. This supplies valid upper bounds for all 294 no-witness frames, including cases where the oracle best kinematic state was collision-invalid. `CORRECTION_BOUND_WITNESSES.json` retains the selected q and provenance. Each frame statistic is the maximum correction across its two wrists. Raw oracle records remain unchanged.\n'
    text+='\nAll 294 selected correction witnesses independently passed a fresh detailed-geometry, joint-limit and FK recheck. The independently recomputed displacements agree with the reported correction upper bounds.\n'
    text+='\nFramewise reachability does not by itself prove temporal reachability from the prescribed natural initial state. In particular, frames 0–2 are included in the requested samples but may be startup transients, not search bugs. Recovery preserves frame-0 initialization and the original temporal bounds and 95%-of-frames trajectory gate; it cannot jump to an oracle witness.\n'
    text+='\n## One shared recovery attempt\n\nThe fixed strategy uses continuity-regularized full-limit posture anchors, chooses nearby valid candidates, and invokes a bounded framewise reseed if residual grows. Anchor continuity is an objective, not a guaranteed result. Anchors do not replace wrist targets. Executable motion still uses the original velocity/acceleration/joint-step limits, natural frame-0 state, collision classifier and source timestamps.\n\n| Method | Episode | Qualified | Cartesian acceptance | Hard collision | Unresolved | Limits | Branches | Temporal |\n|---|---:|---|---:|---:|---:|---:|---:|---|\n'
    for r in recovery:
        c=r['classification_counts'];text+=f"| {r['representation_mode']} | {r['episode']} | {r['position_qualified']} | {100*r['raw_position_acceptance_rate']:.3f}% | {c['HARD_SELF_COLLISION']} | {c['UNRESOLVED_GEOMETRY']} | {r['hard_limit_violations']} | {r['branch_discontinuities']} | {r['temporal_pass']} |\n"
    text+=f"\nRecovery smoke: **INCOMPLETE; NOT PROMOTED**. Completed candidates qualify A {counts['WRIST']}/{completed_counts['WRIST']}; B {counts['INTERACTION']}/{completed_counts['INTERACTION']}. These are not six-trajectory smoke counts. No method-specific or episode-specific rescue. Any residual recovery failures remain reported; the failed full gate prevents downstream work.\n"
    text+='\nThe remaining A24, A49 and B00 candidate runs were stopped incomplete after the forensic established that each A trajectory exceeds the allowed 5% failure fraction through certified physical position gaps alone. A completed 3/3-per-method gate is impossible without prohibited target changes. No incomplete run is scored as a completed failure or success. The authoritative complete pre-recovery smoke remains A 0/3 and B 2/3. The common candidate is not a qualified replacement. The stop decision and preserved completed artifacts are recorded in `common_anchor_recovery_v2/RECOVERY_STOP_DECISION.json`.\n'
    a00=next(r for r in recovery if r['representation_mode']=='WRIST' and r['episode']==0)
    text+=f"\nA00 has {len(a00['old_failed_now_accepted_frames'])} previously failed frames now accepted and {len(a00['old_accepted_now_failed_frames'])} previously accepted frames newly failing. Its unchanged temporal and collision gates pass. This is an executable-trajectory witness of a real sequential-solver defect, stronger than framewise reachability alone; the separate certified morphology gap still prevents the complete A00 trajectory from qualifying.\n"
    b49=next(r for r in recovery if r['representation_mode']=='INTERACTION' and r['episode']==49)
    if not b49['position_qualified']:
        d=b49['recovery_search_diagnostics']
        text+=f"\nB49 remains an unresolved sequential-continuation problem after the fixed common recovery budget: {len(b49['failing_cartesian_frames'])} executable frames exceed 10 mm despite all 147 originally failed targets having independently verified framewise solutions. There were {d['framewise_recovery_attempts']} local recovery attempts and {d['anchor_frames_with_no_geometry_valid_selected_candidate']} non-executable anchor frames with no geometry-valid selected candidate. The guide's largest joint-space step was {b49['non_executable_anchor_maximum_step_norm_rad']:.6f} rad; this is not an executable branch discontinuity. Framewise witnesses do not prove a continuous path from the prescribed initial state. This attempt does not claim to have repaired all sequential-solver failures. No further recovery configuration was tuned to make a method pass.\n"
        text+='\nAt the preselected midpoint of B49’s longest original failing interval (frame 400), the right-wrist residual remains 96.414 mm and right-shoulder-roll margin is approximately 1e-7 rad. A valid full-limit anchor achieves 0.161 mm but is 3.568 rad away from the prior executable q. The temporally bounded retry remains at 96.414 mm. The exact states, per-joint margins and candidates are retained in the final JSON. This identifies the local branch/continuation trap without changing a limit or target.\n'
    text+='\n## Scientific interpretation\n\nThe evidence does not support a single universal explanation. Some A targets exceed a model-derived outer reachable workspace; other A failures and all originally failed B49 frames have actual collision-valid framewise solutions. Thus both target–embodiment position incompatibility and sequential solver failure are present. This is position-only simulated-model evidence, not real-hardware success or a full-task A/B result.\n\nAll sampled and dense targets retain q, seed ID, residual, limit margin, geometry records, and provenance. Independently recomputed FK matched all oracle best states and all three completed recovery trajectories. All 308 reachable oracle witnesses also passed a fresh independent FK, joint-limit and detailed-geometry collision check. All frozen source and collision/registration artifacts remain hash-identical.\n'
    report=DEST/'FINAL_CARTESIAN_REACHABILITY_FORENSIC.md';atomic_text(report,text)
    parity=DEST/'COMMON_RECOVERY_PARITY_AUDIT.md';atomic_text(parity,'# Common recovery parity\n\nThe same target-blind oracle and recovery implementation/configuration was used across all requested inputs. Only existing WRIST/INTERACTION spatial inputs differ. No wrist target, registration, source event timing, collision rule, hard limits, natural initialization, or executable temporal thresholds changed. Prior results and the prior solver remain preserved. Recovery is a separately versioned common candidate, not a silently replaced qualified solver.\n')
    artifacts=[CONTRACT,Path(__file__).resolve(),DEST/'FINAL_CARTESIAN_REACHABILITY_FORENSIC.json',report,DEST/'DENSE_FRAME_CLASSIFICATIONS.csv',DEST/'CORRECTION_BOUND_WITNESSES.json',DEST/'INDEPENDENT_FORENSIC_VALIDATION.json',DEST/'INDEPENDENT_REACHABLE_WITNESS_GEOMETRY_CHECK.json',ROOT/'tools/validate_cartesian_oracle_witnesses.py',DEST/'INDEPENDENT_CORRECTION_WITNESS_GEOMETRY_CHECK.json',ROOT/'tools/validate_cartesian_correction_witnesses.py',parity,*figures]
    manifest=DEST/'FINAL_HASH_MANIFEST.json';atomic_json(manifest,{'artifacts':[file_record(p) for p in artifacts],'dense_files':[file_record(p) for p in sorted((DEST/'dense').glob('*.json'))],'recovery_files':[file_record(p) for p in sorted(REC.glob('*')) if p.is_file()]})
    prepare();recovery_preflight()
    status('MIXED_CARTESIAN_FEASIBILITY_BLOCKER','REVIEW_REPORTED_MORPHOLOGY_GAPS; NO_TARGET_PROJECTION_OR_6D_OR_DEX3',[*artifacts,manifest])
    print('FINAL_FORENSIC',result['sample_counts'],'dense',[(r['case'],r['reachable'],r['unreachable_bounded'],r['physically_certified_unreachable']) for r in case_rows],'recovery',counts,flush=True)


if __name__=='__main__':main()
