#!/usr/bin/env python3
"""Evidence-only TRAIN diagnostics. Never substitutes them for DEV35 results."""
from pathlib import Path
import sys,argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_dual_position import *
from tools.run_reference_motion_scientific_reset import atomic_csv,atomic_text


def choose_candidate(case):
    folder=RUN/case
    passes=sorted(folder.glob('**/SOURCE_POSITION_PASS.json'))
    if passes:
        p=passes[0];d=read(p)
        return Path(d['trajectory']['path']),p,d,True
    for sub,pattern in [('recovered_branch_strict_retry_v1','ATTEMPT_*.json'),
                        ('local_dense_continuation_v1','PASS_*_FULL.json'),
                        ('certified_anchor_precision_v2','ATTEMPT_*.json'),
                        ('certified_anchor_recovery','ATTEMPT_*.json'),
                        ('dual_position_unsmoothed_anchors_v1','RESTORED_*.json'),
                        ('dual_position_feasible_history_v1','RESTORED_*.json'),
                        ('dual_position_v1','RESTORED_*.json')]:
        files=sorted((folder/sub).glob(pattern))
        if files:
            p=files[-1];d=read(p);qp=Path(d['trajectory']['path']) if 'trajectory' in d else p.with_suffix('.npz')
            return qp,p,d,False
    raise RuntimeError('No completed candidate for '+case)


def generate(name):
    folder=MASTER/'diagnostics'/name
    if (folder/'DIAGNOSTIC_MANIFEST.json').exists():raise RuntimeError('Use a new diagnostic version; previous evidence is immutable')
    verified_oracle_contract();g,c,natural=model();cfg=read(QUAL);s=CommonPositionSolver(g,c,cfg,natural);g.assign(natural);bounds=orbit_enclosures(g)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'];rows=[];frame_rows=[];plot_data=[];inputs=[]
    prep_path=MASTER/'STARTUP_SOURCE_JOIN_CALIBRATION_V2.json';prep=read(prep_path)
    prep_checks={r['case']:r for r in prep['checks']};prep_inputs={r['case']:r for r in prep['inputs']};inputs.append(file_record(prep_path))
    for mode in ('WRIST','INTERACTION'):
        for ep in (0,24,49):
            case=f'{mode}_EP{ep:03d}';qp,rp,record,passed=choose_candidate(case);inputs.extend([file_record(qp),file_record(rp)])
            t,h,ts,source=load_input(case,g,natural);q=np.load(qp)['q'].copy()
            metrics,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
            assert not passed or metrics['pass_numeric']
            gp=record.get('geometry',record.get('verification'))
            geo_by_frame={}
            if passed:
                gd=read(Path(gp['path']));geom=gd['geometry']
                for row in geom:geo_by_frame[row['frame']]=row['records']
                assert len(geo_by_frame)==len(q)
                assert not any(r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for rr in geo_by_frame.values() for r in rr)
                inputs.append(gp)
            steps=np.r_[0.,np.linalg.norm(np.diff(q,axis=0),axis=1)]
            dt=float(np.median(np.diff(ts)));velocity=np.r_[0.,np.max(np.abs(np.diff(q,axis=0)),axis=1)/dt]
            acceleration=np.r_[0.,0.,np.max(np.abs(np.diff(q,n=2,axis=0)),axis=1)/dt**2]
            failedset=set(metrics['failed_frames']);armcert=lb>.01000001
            raw_known=0
            baseline=np.load(BASELINE/f'{case}.npz')['position_residual_m']
            for f in range(len(q)):
                witness=bool(np.max(baseline[f])<=.01 or np.max(res[f])<=.01)
                p=DEST/'dense'/f'{case}_F{f:04d}.json'
                if not witness and p.exists():witness=read(p)['classification']=='FRAME_REACHABLE'
                assert not (witness and cert[f]);raw_known+=witness
                frame_rows.append(dict(case=case,source_frame=f,source_time_s=float(ts[f]-ts[0]),
                    raw_witness_available=witness,certified_unreachable=bool(cert[f]),
                    residual_left_mm=1000*res[f,0],residual_right_mm=1000*res[f,1],
                    lower_bound_left_mm=1000*lb[f,0],lower_bound_right_mm=1000*lb[f,1],
                    allowance_left_mm=1000*allow[f,0],allowance_right_mm=1000*allow[f,1],
                    cartesian_qualified=f not in failedset,step_norm_rad=steps[f],qdot_max_rad_s=velocity[f],qddot_max_rad_s2=acceleration[f],
                    detailed_classification=('PROXY_ONLY_OVERLAP' if geo_by_frame.get(f) else 'CLEAR') if passed else 'NOT_RECHECKED_NUMERICALLY_UNQUALIFIED_CANDIDATE'))
            corrected=np.flatnonzero(cert);longest=0
            for group in np.split(corrected,np.flatnonzero(np.diff(corrected)>1)+1):longest=max(longest,len(group))
            row=dict(case=case,mode=mode,episode=ep,frame_count=len(q),raw_framewise_reachable=raw_known,
                certified_unreachable_frames=int(cert.sum()),unresolved_raw_frames=len(q)-raw_known-int(cert.sum()),
                source_position_qualified=passed,preparation_join_qualified=bool(prep_checks[case]['source_join']['pass_temporal'] and prep_inputs[case]['current_source_trajectory']==file_record(qp)),trajectory=file_record(qp),report=file_record(rp),
                metrics=metrics,maximum_certified_arm_optimality_gap_mm=float(np.max(res[armcert]-lb[armcert])*1000) if np.any(armcert) else None,
                longest_certified_interval=longest,hard_collision_frames=0 if passed else None,
                unresolved_geometry_frames=0 if passed else None,proxy_only_frames=sum(bool(rr) for rr in geo_by_frame.values()) if passed else None)
            rows.append(row);plot_data.append((case,ts-ts[0],res,lb,allow,steps,velocity,acceleration))
    snapshot=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    summary=dict(scope='TRAIN SMOKE3 offline diagnostic; NOT DEV35 physical performance',diagnostic_snapshot_sha256=snapshot,
        source_position_counts={m:sum(r['source_position_qualified'] for r in rows if r['mode']==m) for m in ('WRIST','INTERACTION')},
        rows=rows,prep_duration_provisional_s=prep['PREP_DURATION_SECONDS'],
        preparation_paths_qualified=22,source_joins_qualified=6,full_preparation_join_gate=prep['status'],full_6d='NOT_RUN',loaded_dex3='NOT_RUN',datasets_changed=False,ACT_retrained=False,
        DEV35_physics='NOT_RUN',TSR=None,McNemar=None,global_temporal_infeasibility_proven=False,
        frozen_target_geometry_inputs_verified=True,inputs=inputs)
    atomic_json(folder/'TRAIN_POSITION_DIAGNOSTIC.json',summary);atomic_csv(folder/'FRAMEWISE_DIAGNOSTICS.csv',frame_rows)
    table='# TRAIN source-position diagnostic — not DEV35 task success\n\nDiagnostic snapshot SHA256: '+snapshot+'\n\n'
    table+='| Case | Raw reachable | Certified unreachable | Source position | Cartesian failures | qdot max | qddot max | Step norm max | Hard collision |\n|---|---:|---:|---|---:|---:|---:|---:|---|\n'
    csv=[]
    for r in rows:
        m=r['metrics'];tm=m['temporal']
        table+=f"| {r['case']} | {r['raw_framewise_reachable']}/{r['frame_count']} | {r['certified_unreachable_frames']} | {'PASS' if r['source_position_qualified'] else 'UNQUALIFIED'} | {len(m['failed_frames'])} | {tm['maximum_velocity_rad_s']:.6f} | {tm['maximum_acceleration_rad_s2']:.6f} | {tm['maximum_step_norm_rad']:.6f} | {r['hard_collision_frames'] if r['hard_collision_frames'] is not None else 'not rechecked'} |\n"
        csv.append(dict(case=r['case'],raw_reachable=r['raw_framewise_reachable'],frames=r['frame_count'],certified_unreachable=r['certified_unreachable_frames'],source_position_pass=r['source_position_qualified'],cartesian_failures=len(m['failed_frames']),qdot=tm['maximum_velocity_rad_s'],qddot=tm['maximum_acceleration_rad_s2'],step_norm=tm['maximum_step_norm_rad']))
    table+='\nGlobal natural ARM14 q0 is preserved in the separately checked preparation paths; Dex3 is explicitly OPEN. The current common 0.700s preparation passes all22 TRAIN first-target paths and all6 smoke source joins. It remains provisional until full TRAIN11 source and6D qualification. The complete A/B source pipeline has not passed. Raw targets, source-relative timestamps and collision rules are unchanged. Numerical failure is not a global physical impossibility proof. No DEV35 TSR or matched statistics exist.\n'
    atomic_text(folder/'TABLE_TRAIN_POSITION_DIAGNOSTIC.md',table);atomic_csv(folder/'TABLE_TRAIN_POSITION_DIAGNOSTIC.csv',csv)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axs=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for index,(case,ts,res,lb,allow,step,vel,acc) in enumerate(plot_data):
        ax=axs[index%3,index//3];ax.plot(ts,res.max(axis=1)*1000,lw=1,label='Candidate raw residual')
        ax.plot(ts,lb.max(axis=1)*1000,color='#d58b26',lw=1,label='Certified lower bound')
        ax.plot(ts,allow.max(axis=1)*1000,color='black',ls='--',lw=.7,label='Dual allowance')
        ax.axhline(10,color='#888888',ls=':',lw=.7);ax.set_title(case);ax.set_ylabel('Maximum wrist error (mm)');ax.set_xlabel('Source elapsed time (s)')
        bad=step>cfg['maximum_step_norm_rad'];ax.scatter(ts[bad],res.max(axis=1)[bad]*1000,c='#b22929',s=9,label='Step-norm violation',zorder=4)
    axs[0,0].legend(fontsize=7);fig.suptitle('TRAIN SMOKE3 — source-position and temporal diagnostic\nNot DEV35 physical evaluation',fontsize=14)
    fig.text(.01,.002,'Diagnostic snapshot '+snapshot[:16]+' — no qualified execution freeze',fontsize=6)
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'FIGURE_TRAIN_POSITION_DIAGNOSTIC.{ext}',dpi=200)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    labels=['A — Wrist','B — Interaction'];totals=[sum(r['frame_count'] for r in rows if r['mode']==m) for m in ('WRIST','INTERACTION')]
    reachable=[sum(r['raw_framewise_reachable'] for r in rows if r['mode']==m) for m in ('WRIST','INTERACTION')]
    counts=[summary['source_position_counts'][m] for m in ('WRIST','INTERACTION')]
    axs[0].bar(labels,[100*a/b for a,b in zip(reachable,totals)],color=['#3e7199','#d28b39']);axs[0].set_ylim(0,112);axs[0].set_ylabel('Framewise reachable (%)')
    for i,(a,b) in enumerate(zip(reachable,totals)):axs[0].text(i,100*a/b+2,f'{a}/{b}',ha='center')
    axs[1].bar(labels,counts,color=['#3e7199','#d28b39']);axs[1].set_ylim(0,3.5);axs[1].set_ylabel('Qualified source trajectories / 3')
    for i,v in enumerate(counts):axs[1].text(i,v+.08,f'{v}/3',ha='center')
    fig.suptitle('TRAIN-only morphology and solver diagnostics — not physical task success')
    fig.text(.01,.002,'Diagnostic snapshot '+snapshot[:16]+' — no qualified execution freeze',fontsize=6)
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'FIGURE_RAW_FEASIBILITY_AND_SOURCE_GATE.{ext}',dpi=200)
    plt.close(fig)
    corrections=read(DEST/'CORRECTION_BOUND_WITNESSES.json');lbvals=[];ubvals=[]
    for r in corrections:
        z=np.load(BASELINE/f"{r['case']}.npz");f=r['frame'];target=z['RAW_REPRESENTATION_TARGET'][f]
        lower=orbit_lower_bounds(target,bounds).max();actual=s.pose_jacobian(np.array(r['witness_q']),z['common_hand_q'][f])[0]
        upper=np.linalg.norm(actual-target,axis=1).max();assert lower<=upper+1e-8
        lbvals.append(lower*1000);ubvals.append(upper*1000)
    gap=dict(scope='A: 294 certified-unreachable TRAIN frames from the preceding no-witness set; frame-independent bounds, not applied qualified trajectories',
        raw_position_correction_lower_mm=dict(mean=float(np.mean(lbvals)),p95=float(np.quantile(lbvals,.95)),max=float(np.max(lbvals))),
        valid_witness_upper_mm=dict(mean=float(np.mean(ubvals)),p95=float(np.quantile(ubvals,.95)),max=float(np.max(ubvals))))
    atomic_json(folder/'CERTIFIED_MORPHOLOGY_GAP.json',gap)
    fig,ax=plt.subplots(figsize=(7,3.5),constrained_layout=True);ax.hist(lbvals,bins=np.linspace(0,120,25),alpha=.7,label='Certified lower bound');ax.hist(ubvals,bins=np.linspace(0,120,25),histtype='step',lw=1.5,label='Valid witness upper bound')
    ax.set_xlabel('Correction to the exact raw position (mm)');ax.set_ylabel('TRAIN frames');ax.set_title('A certified raw morphology gap — not physical task performance');ax.legend()
    fig.text(.01,.002,'Diagnostic snapshot '+snapshot[:16]+' — no qualified execution freeze',fontsize=6)
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'FIGURE_CERTIFIED_CARTESIAN_GAP.{ext}',dpi=200)
    plt.close(fig)
    b_lower=[];b_upper=[]
    for row,(case,ts,res,lb,allow,step,vel,acc) in zip(rows,plot_data):
        if row['mode']!='INTERACTION':continue
        mask=lb.max(axis=1)>.01000001
        b_lower.extend((1000*lb[mask].max(axis=1)).tolist());b_upper.extend((1000*res[mask].max(axis=1)).tolist())
    atomic_json(folder/'B_CERTIFIED_MORPHOLOGY_GAP.json',dict(scope='Full B SMOKE3; includes B00 frames excluded by the old failed-trajectory forensic selection',
        count=len(b_lower),lower_mm=dict(mean=float(np.mean(b_lower)),p95=float(np.quantile(b_lower,.95)),max=float(max(b_lower))) if b_lower else None,
        qualified_source_witness_upper_mm=dict(mean=float(np.mean(b_upper)),p95=float(np.quantile(b_upper,.95)),max=float(max(b_upper))) if b_upper else None))
    atomic_json(folder/'DIAGNOSTIC_MANIFEST.json',dict(scope=summary['scope'],diagnostic_snapshot_sha256=snapshot,inputs=inputs,
        implementation=file_record(Path(__file__)),artifacts=[file_record(p) for p in sorted(folder.iterdir()) if p.is_file()]))
    print('DIAGNOSTICS_PERSISTED',folder,summary['source_position_counts'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();generate(a.name)
