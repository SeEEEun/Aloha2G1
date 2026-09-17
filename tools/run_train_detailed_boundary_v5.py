#!/usr/bin/env python3
"""Common local detailed-boundary repair of independently rejected trajectories."""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.common_geometry_boundary_v5 import GeometryBoundarySolver
from tools.common_conic_trust_realization_v5 import solve
from tools.qualify_train_candidate_geometry_v5 import run as qualify

def run(case,qualification_path):
    verified_oracle_contract();g,c,n=model();s=GeometryBoundarySolver(g,c,read(QUAL),n)
    g.assign(n);bounds=full_chain_enclosures(g);t,h,ts,source=load_input(case,g,n)
    dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    prior=read(Path(qualification_path));src=Path(prior['trajectory']['path']);q=np.load(src)['q'].copy()
    folder=src.parent/'common_detailed_boundary_repair_v1';atomic_json(folder/'INPUT.json',dict(qualification=file_record(Path(qualification_path)),
        source=source,passes=3,window_padding=24,restore_budget=400,conic_budget=300,targets_or_thresholds_changed=False))
    geometry=read(Path(prior['geometry']['path']))['geometry'];goals={};locals={}
    class Dispatch(GeometryBoundarySolver):
        def clearance_values(self,pairs):
            local=locals[int(pairs[0][2])];local.current_q=self.current_q
            return local.clearance_values([p[:2] for p in pairs])
    ss=Dispatch(g,c,read(QUAL),n)
    for attempt in range(3):
        bad=[(r['frame'],x) for r in geometry for x in r['records'] if x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')]
        if not bad:break
        for f,row in bad:
            pair=tuple(row['geom_pair']);local=locals.setdefault(f,GeometryBoundarySolver(g,c,read(QUAL),n))
            bp=folder/f'BOUNDARY_{attempt}_{f}_{pair[0]}_{pair[1]}.json'
            try:
                if bp.exists():boundary=read(bp)
                else:
                    boundary=local.discover(q[f],h[f],pair)
                    # Move the search plane one frozen geometry numerical unit
                    # into its clear side. This is not a collision waiver.
                    gradient=np.array(boundary['gradient']);point=np.array(boundary['point'])
                    point+=gradient*(1e-5/.04)/float(gradient@gradient)
                    boundary['point']=point.tolist();boundary['search_interior_offset_m']=1e-5
                    atomic_json(bp,boundary)
                local.boundaries[pair]=boundary
            except Exception as exc:
                atomic_json(bp.with_name(bp.stem+'_UNBRACKETED.json'),dict(frame=f,pair=pair,error=str(exc),fallback='unchanged conservative proxy search penalty; no automatic pass'))
            goals.setdefault(f,[])
            if (*pair,f) not in goals[f]:goals[f].append((*pair,f))
            print('TRAIN_DETAILED_BOUNDARY',case,attempt,f,pair,flush=True)
        met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        activeframes=sorted(set(goals)|set(met['failed_frames']));begin=max(0,min(activeframes)-24);end=min(len(q),max(activeframes)+25)
        mask=np.ones_like(q,bool);mask[begin+2:end-2,:]=False;mask[:,[6,13]]=True
        start=time.monotonic();q,fit=restore(ss,t,h,q,dt,allow,collision_pairs=goals,max_nfev=400,fixed_mask=mask,fixed_values=q)
        atomic_json(folder/f'SPARSE_FIT_{attempt}.json',dict(fit=fit,runtime_s=time.monotonic()-start))
        # Hard linear per-joint constraints are enforced by the conic subproblem;
        # original nonlinear position and detailed geometry remain final gates.
        for arm in range(2):
            pp={f-begin:pairs for f,pairs in goals.items() if begin<=f<end}
            q[begin:end],fit=solve(ss,t[begin:end],h[begin:end],q[begin:end],allow[begin:end],dt,arm,pp,trust_radius=.02,max_iterations=300)
            atomic_json(folder/f'CONIC_FIT_{attempt}_{arm}.json',fit)
        out=folder/f'CANDIDATE_{attempt}.npz';atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        qualify(case,str(out));record=read(out.parent/(out.stem+'_independent_qualification')/'FINAL_QUALIFICATION.json')
        if record['qualified']:return
        geometry=read(Path(record['geometry']['path']))['geometry']
    atomic_json(folder/'BOUNDED_RESULT.json',dict(status='COMMON_LOCAL_REPAIR_SEARCH_EXHAUSTED',global_infeasibility_proven=False))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
