#!/usr/bin/env python3
"""Resume-safe common TRAIN posture-pool recovery and complete qualification."""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.common_training_seed_bank_v8 import generate
from tools.common_fast_hard_witness_v5 import hard_witness

def run(case):
    oc=verified_oracle_contract()[0];g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=full_chain_enclosures(g)
    oracle=FramewiseReachabilityOracle(g,c,oc['oracle'],n)
    t,h,ts,source=load_input(case,g,n);dt=float(np.median(np.diff(ts)));folder=TRAIN/case/'common_posture_pool_recovery_v3'
    if (TRAIN/case/'SOURCE_POSITION_PASS.json').exists():print('REUSE_TRAIN11_PASS',case,flush=True);return
    manifest=read(ST5/'common_training_seed_bank/MANIFEST.json');bankpath=Path(manifest['bank']['path']);assert file_record(bankpath)==manifest['bank']
    bank=np.load(bankpath);slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    initial=np.array(read(RUN/'startup'/f'{case}.json')['first_task_q']);initial[[6,13]]=n[[6,13]]
    allow=qualify_numeric(s,np.tile(initial,(len(t),1)),t,h,ts,bounds,slack)[-1]
    atomic_json(folder/'CONTRACT.json',dict(source=source,bank=manifest['bank'],common_seed_count_per_frame=10,maximum_fit_evaluations=180,
        families=['forward','reverse'],maximum_restores=3,restore_evaluations=400,physical_thresholds_changed=False,
        implementation=file_record(ROOT/'tools/common_training_seed_bank_v8.py'),scope='Common TRAIN-calibrated posture seeds; no DEV35 or task-success input'))
    for reverse in (False,True):
        sub=folder/('reverse' if reverse else 'forward');ip=sub/'GEOMETRY_AWARE_GUIDE.npz';report=sub/'GUIDE_REPORT.json'
        if ip.exists():q=np.load(ip)['q'].copy()
        else:
            checkpoint=sub/'GUIDE_PROGRESS.npz';rp=sub/'GUIDE_PROGRESS.json';resume=None
            oldsub=TRAIN/case/'common_posture_pool_recovery_v2'/sub.name
            if not checkpoint.exists() and (oldsub/'GUIDE_PROGRESS.npz').exists() and (oldsub/'GUIDE_PROGRESS.json').exists():
                oldq=np.load(oldsub/'GUIDE_PROGRESS.npz')['q'];oldr=read(oldsub/'GUIDE_PROGRESS.json')
                atomic_npz(checkpoint,q=oldq,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
                atomic_json(rp,oldr)
                atomic_json(sub/'RESUME_PROVENANCE.json',dict(previous_checkpoint=file_record(oldsub/'GUIDE_PROGRESS.npz'),reason='Fail-closed handling of ambiguous early-rejection rays; existing final collision classifier unchanged'))
            if checkpoint.exists() and rp.exists():resume=dict(q=np.load(checkpoint)['q'],reports=read(rp)['frames'])
            def progress(values,records):
                atomic_npz(checkpoint,q=values,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
                atomic_json(rp,dict(frames=records,source=source,complete=len(records)==len(t)))
            q,records=generate(oracle,s,t,h,ts,initial,bank['q'],bank['model_wrist_position'],allow,reverse,progress,resume)
            q[0]=initial
            atomic_npz(ip,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts);atomic_json(report,records)
        mask=np.zeros_like(q,bool);mask[0]=True;mask[:,[6,13]]=True;fixed=q.copy()
        for attempt in range(4):
            qp=ip if not attempt else sub/f'RESTORED_{attempt-1}.npz'
            if attempt and qp.exists():q=np.load(qp)['q'].copy()
            met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
            atomic_json(sub/f'CHECK_{attempt}.json',dict(metrics=met,trajectory=file_record(qp)))
            print('POSTURE_POOL_NUMERIC',case,reverse,attempt,met['pass_numeric'],'failed',met['failed_frames'],
                'velocity',met['temporal']['maximum_velocity_rad_s'],flush=True)
            if met['pass_numeric'] or attempt==3:break
            out=sub/f'RESTORED_{attempt}.npz'
            if not out.exists():
                start=time.monotonic();q,fit=restore(s,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=fixed)
                atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
                atomic_json(sub/f'FIT_{attempt}.json',dict(fit=fit,runtime_s=time.monotonic()-start))
        if not met['pass_numeric']:continue
        # A hard witness can reject a trial early, never pass it. Deep contacts
        # need not incur an exhaustive triangle-distance audit of every frame.
        proxy=[]
        for f,(v,hh) in enumerate(zip(q,h)):
            rr=c.proxy._records(v,*hh);proxy.append((max((x['penetration_depth_m'] for x in rr),default=0),f))
        rejects=[]
        for depth,f in sorted(proxy,reverse=True)[:8]:
            if depth<=0:continue
            try:witness=hard_witness(c,q[f],h[f])
            except Exception:witness=None
            if witness:rejects.append(dict(frame=f,witness=witness))
        if rejects:
            atomic_json(sub/'HARD_GEOMETRY_REJECTION.json',dict(witnesses=rejects,trajectory=file_record(qp),complete_geometry=False));continue
        geometry=[];gf=sub/'frame_geometry'
        for f,(v,hh) in enumerate(zip(q,h)):
            p=gf/f'FRAME_{f:04d}.json'
            if p.exists():rr=read(p)
            else:rr=dict(frame=f,records=c.inspect(v,*hh));atomic_json(p,rr)
            geometry.append(rr)
        gp=sub/'COMPLETE_GEOMETRY.json';atomic_json(gp,dict(geometry=geometry,trajectory=file_record(qp)))
        bad=[r['frame'] for r in geometry if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
        prefix=preparation_path(np.array(read(INITIAL)['g1_14_arm_initial_q_rad']),q[0],21);full=np.vstack((prefix[:-1],q));temporal=temporal_metrics(full,dt,s.config)
        prefixgeom=[dict(frame=f,records=c.inspect(v,*h[0])) for f,v in enumerate(prefix[:-1])]
        prepbad=[r['frame'] for r in prefixgeom if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
        out=sub/'FINAL_CANDIDATE.npz';atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
            EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
        record=dict(case=case,metrics=met,trajectory=file_record(out),geometry=file_record(gp),blocked_frames=bad,
            preparation_blocked_frames=prepbad,preparation_geometry=prefixgeom,complete_temporal=temporal,source=source,
            contract=file_record(folder/'CONTRACT.json'))
        atomic_json(sub/'FINAL_QUALIFICATION.json',record)
        print('POSTURE_POOL_FINAL',case,reverse,'blocked',bad,'prep',prepbad,'temporal',temporal['pass_temporal'],flush=True)
        if not bad and not prepbad and temporal['pass_temporal']:
            atomic_json(TRAIN/case/'SOURCE_POSITION_PASS.json',record);return
    atomic_json(folder/'BOUNDED_RESULT.json',dict(status='COMMON_TRAIN_POSITION_RECOVERY_REQUIRED',global_infeasibility_proven=False))

if __name__=='__main__':run(sys.argv[1])

