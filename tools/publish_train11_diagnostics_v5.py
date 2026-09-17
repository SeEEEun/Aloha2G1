#!/usr/bin/env python3
"""Evidence-backed TRAIN diagnostics; never fabricates DEV35 physical results."""
from pathlib import Path
import sys,datetime,csv,io
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *

def main():
    verified_oracle_contract()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids=read(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')['selection']['qualification_ids']
    smoke={r['case']:r for r in read(ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json')['rows']}
    folder=OUT/'07_paper_artifacts/diagnostic_position_v5';folder.mkdir(parents=True,exist_ok=True)
    g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=full_chain_enclosures(g)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    rows=[];inputs=[]
    for mode in ('WRIST','INTERACTION'):
        for ep in ids:
            case=f'{mode}_EP{ep:03d}';base=TRAIN/case;passed=base/'SOURCE_POSITION_PASS.json'
            qualifications=[]
            for p in base.rglob('FINAL_QUALIFICATION.json'):
                r=read(p);m=r['metrics'];tm=r['complete_temporal']
                qualifications.append((p,r,(not m['pass_numeric'],not tm['pass_temporal'],len(r['blocked_frames']),len(m['failed_frames']),str(p))))
            if passed.exists():
                p=passed;r=read(p);assert file_record(Path(r['trajectory']['path']))['sha256']==r['trajectory']['sha256']
                if 'blocked_frames' not in r:
                    independent=smoke[case];assert independent['qualified']
                    r['hard_collision_frames']=independent['hard_collision_frames']
                    r['unresolved_geometry_frames']=independent['unresolved_geometry_frames']
                    r['blocked_frames']=sorted(set(r['hard_collision_frames']+r['unresolved_geometry_frames']))
            elif qualifications:p,r,_=min(qualifications,key=lambda x:x[2])
            else:
                candidate=base/'common_local_conic_v2/FINAL_CANDIDATE.npz'
                if candidate.exists():
                    t,h,ts,src=load_input(case,g,n)
                    m,*_=qualify_numeric(s,np.load(candidate)['q'],t,h,ts,bounds,slack)
                    p=folder/f'{case}_NUMERIC_ONLY.json'
                    r=dict(metrics=m,trajectory=file_record(candidate),complete_temporal=m['temporal'],blocked_frames=None,
                        preparation_blocked_frames=None,scope='SOURCE_NUMERIC_ONLY; geometry and preparation not qualified on this candidate')
                    atomic_json(p,r)
                else:p=None;r=None
            row=dict(case=case,mode=mode,episode=ep,qualified=passed.exists(),independent_candidates_tested=len(qualifications),
                bounded_fit_artifacts=len(list(base.rglob('*FIT*.json'))),
                diagnostic_selection='qualified artifact, else numeric-pass then complete-temporal-pass then least geometry-invalid frames; no task outcomes',
                selected_qualification=None if p is None else file_record(p),metrics=None if r is None else r['metrics'],
                hard_collision_frames=None if r is None else r.get('hard_collision_frames'),
                unresolved_geometry_frames=None if r is None else r.get('unresolved_geometry_frames'),
                blocked_frames=None if r is None else r['blocked_frames'],
                preparation_blocked_frames=None if r is None else r['preparation_blocked_frames'],
                complete_temporal=None if r is None else r['complete_temporal'])
            rows.append(row)
            if p:inputs.append(file_record(p))
    counts={mode:sum(r['qualified'] for r in rows if r['mode']==mode) for mode in ('WRIST','INTERACTION')}
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result=dict(timestamp=stamp,status='TRAIN_POSITION_DIAGNOSTIC_ONLY',fixed_train_ids=ids,rows=rows,counts=counts,
        smoke3_counts=read(ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json')['qualified_counts'],
        full_6d='NOT_RUN',loaded_dex3='NOT_RUN',dataset_action_diff='NOT_RUN',training='NOT_RUN',dev35_physics='NOT_RUN',
        warning='A diagnostic candidate is never promoted by combining different trajectories or ignoring any failed gate.',
        no_physical_tsr_or_significance_available=True,global_trajectory_infeasibility_proven=False)
    atomic_json(folder/'TRAIN11_POSITION_DIAGNOSTICS.json',result)
    textrows=['| Method / episode | Complete position qualified | Independent candidates audited | Selected candidate Cartesian failures | Selected candidate geometry-invalid frames |',
        '|---|---:|---:|---:|---:|']
    stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['case','qualified','independent_candidates','cartesian_failed_frames','geometry_invalid_frames','source'])
    for r in rows:
        cart='NOT AUDITED' if r['metrics'] is None else len(r['metrics']['failed_frames'])
        geom='NOT AUDITED' if r['blocked_frames'] is None else len(r['blocked_frames'])
        textrows.append(f"| {r['case']} | {'PASS' if r['qualified'] else 'NOT QUALIFIED'} | {r['independent_candidates_tested']} | {cart} | {geom} |")
        writer.writerow([r['case'],r['qualified'],r['independent_candidates_tested'],cart,geom,None if not r['selected_qualification'] else r['selected_qualification']['path']])
    atomic_text(folder/'TABLE_TRAIN11_POSITION_DIAGNOSTICS.md','# TRAIN qualification diagnostics — not DEV35 physical performance\n\n'+
        f"SMOKE3: A3/3, B3/3. Fixed TRAIN11: A{counts['WRIST']}/11, B{counts['INTERACTION']}/11.\n\n"+'\n'.join(textrows)+
        '\n\nEvery row references one saved candidate; no combination of favorable metrics from different candidates is treated as a pass. Complete qualification also requires limits, finite states, branch continuity, per-joint velocity/acceleration and the unchanged natural-start preparation. NOT RUN physical outcomes are not 0/35.\n')
    atomic_text(folder/'TABLE_TRAIN11_POSITION_DIAGNOSTICS.csv',stream.getvalue())
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    ax=axes[0,0];ax.axis('off');ax.text(0,1,'(a) Controlled target-realization pipeline',va='top',weight='bold')
    ax.text(.04,.80,'Same source / event clock / registration\n\nWRIST or INTERACTION raw target only\n\nCommon morphology / IK / detailed geometry\n\nFixed natural q0 + 0.7 s preparation\n\nTRAIN qualification precedes 6D / Dex3 / ACT / PhysX',va='top',linespacing=1.3)
    ax=axes[0,1];x=np.arange(2);a=[3,counts['WRIST']];b=[3,counts['INTERACTION']]
    ax.bar(x-.18,a,.36,label='A — WRIST',color='#3274a1');ax.bar(x+.18,b,.36,label='B — INTERACTION',color='#e1812c')
    ax.set_xticks(x,['SMOKE3','Fixed TRAIN11']);ax.set_ylim(0,12);ax.set_ylabel('Complete qualified trajectories');ax.set_title('(b) Position + preparation qualification');ax.legend()
    for i,den in enumerate((3,11)):
        ax.text(i-.18,a[i]+.2,f'{a[i]}/{den}',ha='center');ax.text(i+.18,b[i]+.2,f'{b[i]}/{den}',ha='center')
    for ax,case,label in ((axes[1,0],'WRIST_EP032','(c) A32: continuation diagnostic'),(axes[1,1],'INTERACTION_EP044','(d) B44: unresolved framewise feasibility')):
        path=TRAIN/case/'common_local_conic_v2/FINAL_CANDIDATE.npz';q=np.load(path)['q'];t,h,ts,src=load_input(case,g,n)
        m,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack);audit=path.parent/(path.stem+'_global_position_audit_v1')
        ax.plot(ts,1000*res.max(axis=1),label='Saved trajectory residual',lw=1.1,color='#3274a1')
        ax.plot(ts,1000*allow.max(axis=1),label='Certified-frame allowance',ls=':',color='#777777')
        ax.axhline(10,color='black',ls='--',lw=.9,label='Raw fidelity gate: 10 mm')
        for p in sorted(audit.glob('FRAME_*.json')):
            r=read(p);f=r['frame'];best=r['candidates'][0];w=r['witness']
            val=max((w or best)['residual_m'])*1000
            ax.scatter(ts[f],val,c='#18845b' if w else '#ba3434',marker='o' if w else 'x',s=35,zorder=4)
            inputs.append(file_record(p))
        failed=m['failed_frames']
        if failed:ax.set_xlim(ts[max(0,min(failed)-12)],ts[min(len(ts)-1,max(failed)+12)])
        ax.set_title(label);ax.set_xlabel('Unchanged source time (s)');ax.set_ylabel('Maximum wrist residual (mm)');ax.legend(fontsize=7)
    fig.suptitle('TRAIN-SIDE DIAGNOSTICS — NOT DEV35 PHYSICAL RESULTS',weight='bold')
    for suffix in ('png','pdf','svg'):fig.savefig(folder/f'FIGURE_TRAIN11_POSITION_DIAGNOSTICS.{suffix}',dpi=220)
    plt.close(fig)
    atomic_text(folder/'READ_ME.md','# Scope and interpretation\n\nThese are actual numerical/geometry qualification diagnostics, not final A/B task-success figures. Green dots are geometry-confirmed isolated witnesses under the common raw/certified-frame gate; red crosses are best kinematic results from a bounded search with no valid witness. A failed search does not prove global unreachability.\n\nThe same raw target, event clock, registration, natural q0, 0.7 s prefix, 10 mm raw gate, 10 micrometre numerical geometry rule and per-joint physical limits remain in force. No final 6D, Dex3, ACT or DEV35 result is inferred.\n')
    atomic_json(folder/'DIAGNOSTIC_ARTIFACT_MANIFEST.json',dict(timestamp=stamp,scope='TRAIN_POSITION_DIAGNOSTIC_ONLY',
        execution_pipeline_frozen=False,code=file_record(Path(__file__)),inputs=inputs,
        artifacts=[file_record(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='DIAGNOSTIC_ARTIFACT_MANIFEST.json']))
    print('TRAIN_DIAGNOSTICS',counts,str(folder),flush=True)

if __name__=='__main__':main()
